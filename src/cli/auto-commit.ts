import type { ReadStream } from "node:tty";
import { GitRunner } from "../core/git/runner/index";
import { GitContextPipeline } from "../core/git/pipeline/index";
import { PromptAssembler } from "../core/llm/prompt/index";
import { createProvider } from "../core/llm/provider/index";
import type { BaseProvider } from "../core/llm/provider/index";
import { GitCode, GitError } from "../shared/exceptions/index";
import type { GitPipelineResult } from "../shared/types/index";

const MAX_REGENERATIONS = 3;

export class AutoCommit {
  private readonly runner: GitRunner;

  constructor(options?: { cwd?: string }) {
    this.runner = new GitRunner(options);
  }

  public async run(): Promise<void> {
    const pipelineResult = await new GitContextPipeline(this.runner).execute();

    // createProvider() reads Bun.argv via ConfigLoader — --temperature etc. already work
    const assembler = new PromptAssembler();
    const provider = await createProvider();

    let ttyIn: ReadStream | null = null;
    if (process.stdin.isTTY) {
      ttyIn = process.stdin as ReadStream;
    }

    try {
      if (pipelineResult.route === "interrupted") {
        await this.handleInterrupted(pipelineResult, ttyIn, assembler, provider);
      } else {
        await this.handleFull(pipelineResult, ttyIn, assembler, provider);
      }
    } finally {
      // Make sure clean raw mode when terminal exits
      if (ttyIn && ttyIn.isTTY && ttyIn.isRaw) {
        ttyIn.setRawMode(false);
      }

    }
  }

  // ── Interrupted route (merge / cherry-pick / rebase / etc.) ──

  private async handleInterrupted(
    pipelineResult: Extract<GitPipelineResult, { route: "interrupted" }>,
    ttyIn: ReadStream | null,
    assembler: PromptAssembler,
    provider: BaseProvider,
  ): Promise<void> {
    // Call LLM to update interrupted info
    const assembled = assembler.assemble(pipelineResult);
    const message = await provider.invoke(assembled.messages);

    if (message.trim().length === 0) {
      console.error("LLM returned empty message. Cannot commit.");
      return;
    }

    printCommitMessage(message);
    process.stdout.write("\n[ENTER] accept  [any other key] cancel\n");

    const key = await readKey(ttyIn);
    if (key === "\r" || key === "\n") {
      await this.commit(message);
      console.log("Committed.");
    } else {
      console.log("Cancelled.");
    }
  }

  // ── Full route (normal staged changes → LLM-generated message) ──

  private async handleFull(
    pipelineResult: Extract<GitPipelineResult, { route: "full" }>,
    ttyIn: ReadStream | null,
    assembler: PromptAssembler,
    provider: BaseProvider,
  ): Promise<void> {
    for (let attempt = 1; attempt <= MAX_REGENERATIONS; attempt++) {
      const assembled = assembler.assemble(pipelineResult);
      const message = await provider.invoke(assembled.messages);

      if (message.trim().length === 0) {
        console.error("LLM returned empty message. Regenerating…");
        continue;
      }

      printCommitMessage(message);

      const lastAttempt = attempt === MAX_REGENERATIONS;
      if (lastAttempt) {
        process.stdout.write("\n[ENTER] accept  [any other key] cancel\n");
      } else {
        process.stdout.write(
          "\n[ENTER] accept [TAB] regenerate  [any other key] cancel\n",
        );
      }

      const key = await readKey(ttyIn);

      if (key === "\t") {
        // Tab: regenerate commit message
        if (!lastAttempt) {
           console.log("Regenerating…");
           continue;
         }
         // The last press of the Tab key is treated as a cancel
         console.log("Cancelled.");
         return;
      }

      if (key === "\r" || key === "\n") {
        await this.commit(message);
        console.log("Committed.");
        return;
      }

      console.log("Cancelled.");
      return;
    }
  }

  private async commit(message: string): Promise<void> {
    try {
      await this.runner.run(["commit", "-m", message]);
    } catch (error) {
      if (error instanceof GitError) {
        throw new GitError({
          code: GitCode.COMMIT_FAILED,
          message: "Failed to create commit",
          cause: error,
        });
      }
      throw error;
    }
  }
}

// ── Helpers ──

function printCommitMessage(message: string): void {
  const separator = "─".repeat(10);
  console.log(`\n${separator}`);
  console.log(message);
  console.log(separator);
}

async function readKey(ttyIn: ReadStream | null): Promise<string> {
  if (!ttyIn || !ttyIn.isTTY) {
    throw new Error("Run in an interactive terminal");
  }

  ttyIn.setRawMode(true);
  try {
    const reader = Bun.stdin.stream().getReader();
    const result = await reader.read();
    reader.releaseLock();

    if (result.done || !result.value || result.value.length === 0) {
      return "";
    }
    return String.fromCharCode(result.value[0]!);
  } finally {
    ttyIn.setRawMode(false);
  }
}
