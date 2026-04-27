import { GitCode, GitError } from "../../../shared/exceptions/index";
import { type GitRunOptions, type GitRunResult } from "../../../shared/types";


// Injected into every git invocation to prevent interactive prompts
// from blocking a non-TTY CLI process.
const NON_INTERACTIVE_ENV: Record<string, string> = {
  GIT_TERMINAL_PROMPT: "0",
  GIT_ASKPASS: "true", // "true" binary exits 0 with empty output
};

export class GitRunner {
  private readonly defaultCwd: string;

  constructor(options?: { cwd?: string }) {
    this.defaultCwd = options?.cwd ?? process.cwd();
  }

  /**
   * Low-level execution:
   * - Runs git and always returns exit code/output.
   * - Throws GitError(COMMAND_FAILED) only when spawn itself fails.
   */
  private async runRaw(args: string[], options?: GitRunOptions): Promise<GitRunResult> {
    const cwd = options?.cwd ?? this.defaultCwd;
    const commandArgs = ["git", ...args];
    const startedAt = Date.now();

    let proc: ReturnType<typeof Bun.spawn>;
    try {
      proc = Bun.spawn(commandArgs, {
        cwd,
        env: {
          ...process.env,
          ...NON_INTERACTIVE_ENV,
          ...options?.env,
        },
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
      });
    } catch (error) {
      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message: "Failed to spawn git command: " + commandArgs.join(" "),
        details: { args, cwd },
        cause: error,
      });
    }

    let stdout: string;
    let stderr: string;
    let exitCode: number;

    try {
      [stdout, stderr, exitCode] = await Promise.all([
        this.readText(proc.stdout, "stdout"),
        this.readText(proc.stderr, "stderr"),
        proc.exited,
      ]);
    } catch (error) {
      proc.kill();
      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message: "Failed to read output of git command: " + commandArgs.join(" "),
        details: { args, cwd },
        cause: error,
      });
    }

    return {
      args: [...args],
      command: commandArgs.join(" "),
      cwd,
      exitCode,
      stdout: stdout.trim(),
      stderr: stderr.trim(),
      durationMs: Date.now() - startedAt,
    };
  }

  /**
   * High-level execution:
   * - Accepts only exit code 0 by default.
   * - Pass allowedExitCodes for commands with semantic non-zero exits
   *   (e.g. `git diff --cached --quiet` returns 1 when staging is non-empty).
   */
  public async run(args: string[], options?: GitRunOptions): Promise<GitRunResult> {
    const allowedExitCodes = options?.allowedExitCodes ?? [0];
    const result = await this.runRaw(args, options);

    if (!allowedExitCodes.includes(result.exitCode)) {
      const reason = result.stderr || result.stdout || "unknown git error";
      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message:
          `Git command failed with exit code ${result.exitCode}: ${result.command}\n${reason}`,
        details: {
          args: result.args,
          cwd: result.cwd,
          command: result.command,
          exitCode: result.exitCode,
          stdout: result.stdout,
          stderr: result.stderr,
          allowedExitCodes: [...allowedExitCodes],
        },
      });
    }

    return result;
  }

  private async readText(
    stream: number | ReadableStream<Uint8Array> | null | undefined,
    streamName: "stdout" | "stderr",
  ): Promise<string> {
    if (stream === null || stream === undefined) {
      return "";
    }

    // With stdout/stderr set to "pipe", number here indicates misconfigured spawn stdio mode.
    if (typeof stream === "number") {
      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message: `Git subprocess ${streamName} is not piped`,
        details: { streamName, fd: stream },
      });
    }

    return new Response(stream).text();
  }
}