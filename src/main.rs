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
use crate::infra::config::ConfigLoader;
use crate::infra::git::GitRunner;
use crate::infra::terminal::{KeyListener, RawModeGuard, TerminalUi};
use crate::shared::exception::{AppError, GitError, GitErrorCode, LlmError, ProviderErrorType};

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
    let raw = client.invoke(prompt).await.map_err(AppError::Llm)?;
    let message = raw.trim();
    if message.is_empty() {
        return Err(AppError::Llm(LlmError::Provider(
            ProviderErrorType::Fatal,
            "provider returned an empty commit message".to_string(),
        )));
    }

    // 5. Interactive loop
    let repo = match &ctx {
        AssemblyContext::FromStaging { repo, .. } | AssemblyContext::FromOperation { repo, .. } => {
            repo.branch
                .clone()
                .unwrap_or_else(|| repo.worktree_root.display().to_string())
        }
    };
    let generate = || async {
        let prompt = PromptAssembler::assemble(&ctx);
        let raw = client.invoke(prompt).await?;
        let candidate = raw.trim().to_owned();
        if candidate.is_empty() {
            return Err(LlmError::Provider(
                ProviderErrorType::Fatal,
                "provider returned an empty commit message".to_string(),
            ));
        }
        Ok(candidate)
    };

    let _raw = RawModeGuard::enter()
        .map_err(|err| AppError::Io(format!("failed to enable raw terminal mode: {err}")))?;
    let (stop_keys, key_rx) = KeyListener::spawn().into_parts();
    let ui = TerminalUi::new(repo);

    let decision = InteractiveLoop::new(generate, key_rx, ui)
        .run()
        .await
        .map_err(AppError::Llm)?;

    drop(stop_keys);
    drop(_raw);

    let Some(candidate) = decision else {
        return Ok(()); // user exited without accepting — nothing to commit
    };

    commit_staged_changes(&runner, &candidate).await
}

async fn commit_staged_changes(runner: &GitRunner, message: &str) -> Result<(), AppError> {
    let result = runner
        .commit(message, None)
        .await
        .map_err(|err| AppError::Commit(err.to_string()))?;

    print!("{}", result.stdout_str());
    Ok(())
}
