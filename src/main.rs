use std::process::ExitCode;

mod core;
mod infra;
mod shared;

use crate::core::git::types::BudgetPolicy;
use crate::core::llm::client::LlmClient;
use crate::core::pipe::assembler::PromptAssembler;
use crate::core::pipe::orchestrator::PipeOrchestrator;
use crate::infra::config::ConfigLoader;
use crate::infra::git::GitRunner;
use crate::shared::exception::{AppError, GitError, GitErrorCode, LlmError, ProviderErrorType};

#[cfg(test)]
#[path = "../test/unittest/mod.rs"]
mod unittest;

#[cfg(test)]
#[path = "../test/integration/mod.rs"]
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

    Ok(())
}
