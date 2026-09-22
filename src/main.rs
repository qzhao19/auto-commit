use std::process::ExitCode;

mod core;
mod infra;
mod shared;

use crate::core::git::types::BudgetPolicy;
use crate::core::inter::cycle::InteractiveLoop;
use crate::core::llm::client::LlmClient;
use crate::core::pipe::assembler::PromptAssembler;
use crate::core::pipe::context::AssemblyContext;
use crate::core::pipe::orchestrator::PipeOrchestrator;
use crate::core::pipe::validator::validate_and_normalize;
use crate::infra::config::ConfigLoader;
use crate::infra::git::GitRunner;
use crate::infra::terminal::{KeyListener, RawModeGuard, TerminalUi};
use crate::shared::exception::{
    AppError, GitError, GitErrorCode, LlmError, ProviderErrorType, ValidationError,
};

#[cfg(test)]
#[path = "../tests/unittest/mod.rs"]
mod unittest;

#[cfg(test)]
#[path = "../tests/integration/mod.rs"]
mod integration;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("auto-commit: {err}");
            ExitCode::from(err.exit_code())
        }
    }
}

async fn run() -> Result<(), AppError> {
    let invalid_message = |err: ValidationError| -> LlmError {
        LlmError::Provider(
            ProviderErrorType::Fatal,
            format!("invalid commit message: {err}"),
        )
    };

    // 1. Config: default < toml < env < CLI
    let config = ConfigLoader::load_from_defaults()
        .load()
        .map_err(AppError::Config)?;

    // 2. Git pipeline stages 0 - 4.1:
    // preflight -> operation state -> staged metadata -> classify/budget -> AssemblyContext
    let workdir = std::env::current_dir()
        .map_err(|err| AppError::Io(format!("cannot resolve working directory: {err}")))?;
    let runner = GitRunner::new(Some(workdir));
    let ctx = PipeOrchestrator::new(&runner, BudgetPolicy::default())
        .run()
        .await
        .map_err(AppError::Pipeline)?;

    // 3. Stage 4.2: assemble the final LlmMessage
    let prompt = PromptAssembler::assemble(&ctx);
    if prompt.user_message.trim().is_empty() {
        return Err(AppError::Pipeline(GitError::new(
            GitErrorCode::Other,
            "prompt assembly produced no content",
        )));
    }

    // 4. LLM build the client from config
    let client = LlmClient::new(&config).map_err(AppError::Llm)?;
    let raw = client.invoke(&prompt).await.map_err(AppError::Llm)?;
    let first_candidate = validate_and_normalize(&raw)
        .map(|msg| msg.into_string())
        .map_err(|err| AppError::Llm(invalid_message(err)))?;

    // 4.5 Headless automation entry (CI / scripts / e2e)
    if std::env::var("AUTOCOMMIT_ASSUME_YES").ok().as_deref() == Some("1") {
        println!("{first_candidate}");
        return commit_staged_changes(&runner, &first_candidate).await;
    }

    // 5. Interactive loop
    let repo = match &ctx {
        AssemblyContext::FromStaging { repo, .. } | AssemblyContext::FromOperation { repo, .. } => {
            repo.branch
                .clone()
                .unwrap_or_else(|| repo.worktree_root.display().to_string())
        }
    };

    let client_ref = &client;
    let prompt_ref = &prompt;

    let mut first_cache = Some(first_candidate);
    let generate = || {
        let cached = first_cache.take();
        async move {
            if let Some(cached) = cached {
                return Ok(cached);
            }
            let raw = client_ref.invoke(&prompt_ref).await?;
            validate_and_normalize(&raw)
                .map(|msg| msg.into_string())
                .map_err(invalid_message)
        }
    };

    let _raw = RawModeGuard::enter()
        .map_err(|err| AppError::Io(format!("failed to enable raw terminal mode: {err}")))?;
    let (stop_keys, key_rx) = KeyListener::spawn()
        .map_err(|err| AppError::Io(format!("failed to spawn key listener: {err}")))?
        .into_parts();
    let ui = TerminalUi::new(repo);

    let decision = InteractiveLoop::new(generate, key_rx, ui)
        .run()
        .await
        .map_err(AppError::Llm)?;

    drop(stop_keys);
    drop(_raw);

    let Some(candidate) = decision else {
        return Ok(()); // nothing to commit
    };

    commit_staged_changes(&runner, &candidate).await
}

async fn commit_staged_changes(runner: &GitRunner, message: &str) -> Result<(), AppError> {
    let result = runner
        .commit(message, None)
        .await
        .map_err(|err| AppError::Commit(err.to_string()))?;

    eprint!("{}", result.stdout_str());
    Ok(())
}
