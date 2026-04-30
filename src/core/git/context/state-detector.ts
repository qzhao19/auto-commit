import { join } from "node:path";
import { stat } from "node:fs/promises";
import { GitCode, GitError } from "../../../shared/exceptions/index";
import {
  type GitInternalOpDetectResult,
  type GitInternalOpState,
  type GitStateDetectStep,
} from "../../../shared/types/index";
import { type GitRunner } from "../runner/index";

export class StateDetector {
  private readonly gitDir: string;
  private readonly runner: GitRunner;

  constructor(gitDir: string, runner: GitRunner) {
    this.gitDir = gitDir;
    this.runner = runner;
  }

  public async detect(): Promise<GitInternalOpDetectResult> {
    const completedSteps: GitStateDetectStep[] = [];
    let currentStep: GitStateDetectStep = "bisect-detect";

    const runDetectStep = async <T>(
      step: GitStateDetectStep,
      fn: () => Promise<T>,
    ): Promise<T> => {
      currentStep = step;
      const result: T = await fn();
      completedSteps.push(step);
      return result;
    };

    try {
      // 6. bisect — hard exit; must run first to block all other paths
      await runDetectStep("bisect-detect", () => this.detectBisect());

      // 5. rebase — takes priority over merge (rebase internally uses merge machinery)
      const rebaseState = await runDetectStep("rebase-detect", () => this.detectRebase());
      if (rebaseState !== null) {
        return { success: true, finalStep: currentStep, completedSteps, state: rebaseState };
      }

      // 1. merge
      const mergeState = await runDetectStep("merge-detect", () => this.detectMerge());
      if (mergeState !== null) {
        return { success: true, finalStep: currentStep,  completedSteps, state: mergeState};
      }

      // 2. squash-merge
      const squashState = await runDetectStep("squash-detect", () => this.detectSquashMerge());
      if (squashState !== null) {
        return { success: true, finalStep: currentStep, completedSteps, state: squashState };
      }

      // 3. cherry-pick
      const cherryPickState = await runDetectStep("cherry-pick-detect", () => this.detectCherryPick());
      if (cherryPickState !== null) {
        return { success: true, finalStep: currentStep, completedSteps, state: cherryPickState };
      }

      // 4. revert
      const revertState = await runDetectStep("revert-detect", () => this.detectRevert());
      if (revertState !== null) {
        return { success: true, finalStep: currentStep, completedSteps, state: revertState };
      }

      // All checks passed with no special state found
      const cleanState: GitInternalOpState = { status: "clean" };
      return { success: true, finalStep: currentStep, completedSteps, state: cleanState };

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
          cause: error.cause,
        });
      }

      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message: error instanceof Error ? error.message : String(error),
        details: stepDetails,
        cause: error,
      });
    }
  }

  // ── 1. Detect merge ──

  private async detectMerge(): Promise<GitInternalOpState | null> {
    const mergeHead = await this.readGitFile("MERGE_HEAD");
    if (mergeHead === null) return null;

    const mergeMessage = await this.readGitFile("MERGE_MSG");

    return { status: "merge", mergeHead, mergeMessage };
  }

  // ── 2. Detect squash-merge ──
  /**
   * SQUASH_MSG is created by `git merge --squash`.
   * This step only runs after confirming MERGE_HEAD is absent,
   * so there is no ambiguity with a regular merge.
   */
  private async detectSquashMerge(): Promise<GitInternalOpState | null> {
    const squashMessage = await this.readGitFile("SQUASH_MSG");
    if (squashMessage === null) return null;

    return { status: "squash-merge", squashMessage };
  }

  // ── 3. Detect cherry-pick ──

  private async detectCherryPick(): Promise<GitInternalOpState | null> {
    const cherryPickHead = await this.readGitFile("CHERRY_PICK_HEAD");
    if (cherryPickHead === null) return null;

    const originalTitle = await this.resolveCommitTitle(cherryPickHead);

    return { status: "cherry-pick", cherryPickHead, originalTitle };
  }

  // ── 4. Detect revert ──

  private async detectRevert(): Promise<GitInternalOpState | null> {
    const revertHead = await this.readGitFile("REVERT_HEAD");
    if (revertHead === null) return null;

    const originalTitle = await this.resolveCommitTitle(revertHead);

    return { status: "revert", revertHead, originalTitle };
  }

  // ── 5. Detect rebase ──

  private async detectRebase(): Promise<GitInternalOpState | null> {
    const rebaseMergeDir: string = join(this.gitDir, "rebase-merge");
    const rebaseApplyDir: string = join(this.gitDir, "rebase-apply");

    const [isMerge, isApply] = await Promise.all([
      this.directoryExists(rebaseMergeDir),
      this.directoryExists(rebaseApplyDir),
    ]);

    if (!isMerge && !isApply) return null;

    if (isMerge && isApply) {
      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message: 
          "Invalid git state: both rebase-merge and rebase-apply exist",
        details: { rebaseMergeDir, rebaseApplyDir },
      });
    }

    const rebaseType: "merge" | "apply" = isMerge ? "merge" : "apply";
    const rebaseDir: string  = isMerge ? rebaseMergeDir : rebaseApplyDir;
    const originalMessage = await this.resolveRebaseMessage(rebaseType, rebaseDir);

    return { status: "rebase", rebaseType, originalMessage };
  }

  // ── 6. Detect bisect ──

  private async detectBisect(): Promise<void> {
    const bisectLog = await this.readGitFile("BISECT_LOG");
    if (bisectLog === null) return;

    const bisectLogPath = join(this.gitDir, "BISECT_LOG");
    throw new GitError({
      code: GitCode.BISECT_IN_PROGRESS,
      message:
        "You are currently in a git bisect session. " +
        "Please complete or abort the bisect (`git bisect reset`) before committing.",
      details: { bisectLogPath },
    });
  }

  // ── Primitives ──

  /**
   * Reads a file directly inside .git/ by name (e.g. "MERGE_HEAD").
   */
  private async readGitFile(filename: string): Promise<string | null> {
    return this.readFile(join(this.gitDir, filename));
  }

  /**
   * Reads any file by absolute path, returns trimmed content, or null.
   */
  private async readFile(filePath: string): Promise<string | null> {
    const file = Bun.file(filePath);
    if (!(await file.exists())) return null;
    const content = (await file.text()).trim();
    return content.length > 0 ? content : null;
  }

  /**
  * Reads the first file in the list that exists and has non-empty content.
   */
  private async readFirstExistingFile(filePaths: readonly string[]): Promise<string | null> {
    for (const filePath of filePaths) {
      const content: string | null = await this.readFile(filePath);
      if (content !== null) {
        return content;
      }
    }
    return null;
  }

  /**
   * Checks whether a path exists and is a directory.
   */
  private async directoryExists(dirPath: string): Promise<boolean> {
    return stat(dirPath)
      .then((s) => s.isDirectory())
      .catch(() => false);
  }

  /**
   * Resolves the single-line subject of a commit by its hash.
   * Returns null if the hash is unreachable or the git command fails.
   */
  private async resolveCommitTitle(hash: string): Promise<string | null> {
    const result = await this.runner.run(
      ["log", "-1", "--format=%s", hash],
      { allowedExitCodes: [0, 128] },
    );

    if (result.exitCode !== 0 || result.stdout.length === 0) return null;
    return result.stdout;
  }

  /**
   * Resolves the original commit message for a rebase in progress.
   */
  private async resolveRebaseMessage(
    rebaseType: "merge" | "apply",
    rebaseDir: string,
  ): Promise<string | null> {
    if (rebaseType === "merge") {
      return this.readFile(join(rebaseDir, "message"));
    }

    return this.readFirstExistingFile([
      join(rebaseDir, "final-commit"), // present after rebase --continue finishes
      join(rebaseDir, "msg-clean"),    // present mid-apply (no comment lines)
      join(rebaseDir, "msg"),          // present mid-apply (raw message)
    ]);
  }

}