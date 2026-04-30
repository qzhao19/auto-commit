import { join } from "node:path";
import { GitCode, GitError } from "../../../shared/exceptions/index";
import {
  type GitRunResult,
  type GitPrecheckStep,
  type GitRepoPrecheckContext,
  type GitRepoPrecheckResult,
} from "../../../shared/types/index";
import { type GitRunner } from "../runner/index";

export class RepoChecker {
  private readonly runner: GitRunner;

  constructor(runner: GitRunner) {
    this.runner = runner;
  }

  /**
   * Execute all prechecks in Phase 0; if successful, return a `GitRepoPrecheckResult`.
   * If a fatal issue is encountered in steps 1, 2, or 3, throw a `GitError` directly; do not return a result.
   * If a detached HEAD is encountered in step 6, log a `console.warn` but do not interrupt the process.
   */
  public async check(): Promise<GitRepoPrecheckResult> {
    const completedSteps: GitPrecheckStep[] = [];
    let currentStep: GitPrecheckStep = "is-repo";

    const runPrecheckStep = async <T>(
      step: GitPrecheckStep,
      fn: () => Promise<T>,
    ): Promise<T> => {
      currentStep = step;
      const result: T = await fn();
      completedSteps.push(step);
      return result;
    };

    try {
      // 1. Check if it is a Git repository
      const gitDir = await runPrecheckStep("is-repo", () => this.checkIsRepo());

      // 2. Check if other Git processes are occupying index.lock
      await runPrecheckStep("lock-check", () => this.checkLockFile(gitDir));

      // 3. Staging area + working directory state
      await runPrecheckStep("staging-check", () => this.resolveStagingState());

      // 4. Get the repository root path
      const workTree = await runPrecheckStep("resolve-worktree", () => this.resolveWorkTree());

      // 5. Check if this is the initial commit
      const isInitialCommit = await runPrecheckStep("initial-commit-check", () => this.detectInitialCommit());

      // 6. Check if HEAD is detached (warn but do not interrupt)
      const { isDetachedHead, currentBranch } = await runPrecheckStep(
        "detached-head-check",
        () => this.detectHeadState(),
      );

      const context: GitRepoPrecheckContext = {
        gitDir,
        workTree,
        isInitialCommit,
        isDetachedHead,
        currentBranch,
      };

      return {
        success: true,
        finalStep: currentStep,
        completedSteps,
        context,
      };
    } catch (error) {
      const stepDetails = {
        step: currentStep,
        completedSteps: [...completedSteps],
      };

      if (error instanceof GitError) {
        throw new GitError({
          code: error.code,
          message: error.message,
          details: { ...(error.details ?? {}), ...stepDetails },
          cause: error.cause, // cause
        });
      }
      // For unexpected runtime errors
      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message: error instanceof Error ? error.message : String(error),
        details: { step: currentStep, completedSteps: [...completedSteps] },
        cause: error,
      });
    }
  }

  // ── 1. Determine whether it is a Git repository ──

  private async checkIsRepo(): Promise<string> {
    const result: GitRunResult = await this.runner.run(
      ["rev-parse", "--git-dir"],
      { allowedExitCodes: [0, 128] },
    );

    if (result.exitCode !== 0) {
      throw new GitError({
        code: GitCode.NOT_A_REPO,
        message: "The current directory is not a Git repository",
        details: { cwd: result.cwd, stderr: result.stderr }
      });
    }

    // --git-dir may return relative paths within the worktree or submodule; 
    // convert them all to absolute paths
    const stdout: string = result.stdout;
    return stdout.startsWith("/") ? stdout : join(result.cwd, stdout);
  }

  // ── 2. Check if other Git processes are running ──

  private async checkLockFile(gitDir: string): Promise<void> {
    const lockPath: string = join(gitDir, "index.lock");

    if (await Bun.file(lockPath).exists()) {
      throw new GitError({
        code: GitCode.LOCK_FILE_EXISTS,
        message: "Another Git process is currently running. Please try again later.",
        details: { lockPath },
      });
    }
  }

  // ── 3. Check if the staging area is empty ──

  /**
   * git diff --cached --quiet semantics:
   *   exit 0 → Staging area is empty; check the working directory further
   *   exit 1 → Changes in the staging area (normal commit path)
   *
   * Throws GitError if the staging area is empty or there is nothing to commit.
   */
  private async resolveStagingState(): Promise<void> {
    const cached: GitRunResult = await this.runner.run(
      ["diff", "--cached", "--quiet"],
      { allowedExitCodes: [0, 1] },
    );

    if (cached.exitCode === 1) {
      return; // has-staged-changes: normal commit path
    }

    // Staging area is empty; check the working directory
    const workDir: GitRunResult = await this.runner.run(
      ["diff", "--quiet"],
      { allowedExitCodes: [0, 1] },
    );

    if (workDir.exitCode === 1) {
      const nameOnly: GitRunResult = await this.runner.run(
        ["diff", "--name-only"],
        { allowedExitCodes: [0] },
      );
      const unstagedFiles: string[] = nameOnly.stdout
        .split("\n")
        .filter((line) => line.length > 0);

      throw new GitError({
        code: GitCode.STAGING_EMPTY,
        message: "The staging area is empty. Please run `git add` first.",
        details: { unstagedFiles },
      });
    }

    throw new GitError({
      code: GitCode.NOTHING_TO_COMMIT,
      message: "There are no changes to commit.",
    });
  }

    // ── 4. Get the root path of the repository ──

  private async resolveWorkTree(): Promise<string> {
    const isBare = await this.runner.run(["rev-parse", "--is-bare-repository"]);
    if (isBare.stdout === "true") {
      throw new GitError({
        code: GitCode.BARE_REPO_UNSUPPORTED,
        message: "Bare repositories are not supported. A working tree is required.",
      });
    }
    const worktree = await this.runner.run(["rev-parse", "--show-toplevel"]);
    return worktree.stdout;
  }

  // ── 5. Determine whether this is the initial commit ──

  private async detectInitialCommit(): Promise<boolean> {
    const isInitialCommit: GitRunResult = await this.runner.run(
      ["rev-parse", "HEAD"],
      { allowedExitCodes: [0, 128] },
    );
    return isInitialCommit.exitCode !== 0;
  }

  // ── 6. Determine whether the HEAD is detached ──

  private async detectHeadState(): Promise<{
    isDetachedHead: boolean;
    currentBranch: string | null;
  }> {
    const result = await this.runner.run(
      ["symbolic-ref", "-q", "HEAD"],
      { allowedExitCodes: [0, 1] },
    );

    if (result.exitCode !== 0) {
      console.warn(
        "[auto-commit] Warning: You are currently in the “Detached HEAD” state. " +
        "This commit will not be part of any branch. Please confirm before continuing.",
      );
      return { isDetachedHead: true, currentBranch: null };
    }

    // "refs/heads/main" → "main"
    const currentBranch = result.stdout.replace(/^refs\/heads\//, "");
    return { isDetachedHead: false, currentBranch };
  }
}
