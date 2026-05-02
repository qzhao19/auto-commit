/**
 * E2E tests for StateDetector — exercises the full state-detection pipeline
 * against real git repositories on disk, using the real GitRunner.
 *
 * No git internals are mocked. Every state is created by running actual git commands.
 *
 * Requires: git >= 2.28 (for `git init -b <branch>`, `git rebase --merge`)
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GitCode, GitError } from "../../src/shared/exceptions/index";
import { StateDetector } from "../../src/core/git/context/state-detector";
import { GitRunner } from "../../src/core/git/runner/git-runner";

// ── Helpers 

/**
 * Runs a git command for test setup. Throws on non-zero exit so setup failures
 * are immediately distinguishable from SUT failures.
 */
async function git(args: string[], cwd: string): Promise<string> {
  const proc = Bun.spawn(["git", ...args], {
    cwd,
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    proc.exited,
  ]);
  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    throw new Error(`git ${args.join(" ")} failed (exit ${exitCode}): ${stderr.trim()}`);
  }
  return stdout.trim();
}

/**
 * Runs a git command that is expected to exit non-zero (e.g. rebase with conflict).
 * Both stdout and stderr are consumed to prevent pipe-buffer blocking.
 */
async function gitExpectFail(args: string[], cwd: string): Promise<void> {
  const proc = Bun.spawn(["git", ...args], {
    cwd,
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  });
  await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
}

/**
 * Initialises a real git repo with a deterministic branch name and
 * a test identity so commits can be created without a global git config.
 */
async function initRepo(dir: string, branch = "main"): Promise<void> {
  await git(["init", "-b", branch], dir);
  await git(["config", "user.email", "test@example.com"], dir);
  await git(["config", "user.name", "Test User"], dir);
}

/**
 * Writes a file, stages it, commits it, and returns the new HEAD hash.
 */
async function makeCommit(
  gitRoot: string,
  relPath: string,
  content: string,
  message = "chore: test commit",
): Promise<string> {
  await writeFile(join(gitRoot, relPath), content);
  await git(["add", relPath], gitRoot);
  await git(["commit", "-m", message], gitRoot);
  return git(["rev-parse", "HEAD"], gitRoot);
}

/**
 * Creates a bisect session with non-empty BISECT_LOG.
 * Marks HEAD as bad and HEAD~2 as good so git checkouts a middle commit to test.
 */
async function setupBisect(repoDir: string): Promise<void> {
  await makeCommit(repoDir, "file.txt", "v1", "chore: v1");
  await makeCommit(repoDir, "file.txt", "v2", "chore: v2");
  await makeCommit(repoDir, "file.txt", "v3", "chore: v3");
  const good = await git(["rev-parse", "HEAD~2"], repoDir);
  await git(["bisect", "start"], repoDir);
  await git(["bisect", "bad", "HEAD"], repoDir);
  await git(["bisect", "good", good], repoDir);
  // git now checks out the midpoint commit — BISECT_LOG has non-empty content
}

/**
 * Creates a rebase-in-progress state by rebasing a feature branch onto main
 * where both branches modify the same file, causing a conflict.
 */
async function setupRebaseConflict(repoDir: string): Promise<void> {
  await makeCommit(repoDir, "file.txt", "initial", "chore: initial");
  await git(["checkout", "-b", "feature"], repoDir);
  await makeCommit(repoDir, "file.txt", "feature content", "feat: feature change");
  await git(["checkout", "main"], repoDir);
  await makeCommit(repoDir, "file.txt", "main content", "chore: main change");
  await git(["checkout", "feature"], repoDir);
  // --merge forces the merge backend → creates rebase-merge/ on conflict
  await gitExpectFail(["rebase", "--merge", "main"], repoDir);
}

async function captureError(fn: () => Promise<unknown>): Promise<unknown> {
  try {
    await fn();
  } catch (e) {
    return e;
  }
  throw new Error("Expected function to throw, but it resolved");
}

function asGitError(value: unknown): GitError {
  expect(value).toBeInstanceOf(GitError);
  return value as GitError;
}

// ── Fixtures ───

let repoDir: string;
let gitDir: string;

beforeEach(async () => {
  const tmp = await mkdtemp(join(tmpdir(), "state-detector-e2e-"));
  repoDir = await realpath(tmp);  // resolve macOS /var → /private/var symlink
  gitDir = join(repoDir, ".git");
});

afterEach(async () => {
  await rm(repoDir, { recursive: true, force: true });
});

// ── Helpers for building the detector 

function makeDetector(): StateDetector {
  return new StateDetector(gitDir, new GitRunner({ cwd: repoDir }));
}

// ── Tests ───

describe("StateDetector e2e", () => {

  // ── Clean state 

  describe("detect() — clean state", () => {
    test("returns { status: 'clean' } on a normal repo with no operations in progress", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "content");

      const result = await makeDetector().detect();

      expect(result.success).toBe(true);
      expect(result.state).toEqual({ status: "clean" });
    });

    test("completedSteps contains all 6 steps in execution order", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "content");

      const result = await makeDetector().detect();

      expect(result.completedSteps).toEqual([
        "bisect-detect",
        "rebase-detect",
        "merge-detect",
        "squash-detect",
        "cherry-pick-detect",
        "revert-detect",
      ]);
    });

    test("finalStep is 'revert-detect' when all steps complete", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "content");

      expect((await makeDetector().detect()).finalStep).toBe("revert-detect");
    });
  });

  // ── Bisect 

  describe("detect() — bisect (hard exit)", () => {
    test("throws GitError(BISECT_IN_PROGRESS) when a bisect session is active", async () => {
      await initRepo(repoDir);
      await setupBisect(repoDir);

      await expect(makeDetector().detect()).rejects.toMatchObject({
        code: GitCode.BISECT_IN_PROGRESS,
      });
    });

    test("thrown error is a GitError instance", async () => {
      await initRepo(repoDir);
      await setupBisect(repoDir);

      await expect(makeDetector().detect()).rejects.toBeInstanceOf(GitError);
    });

    test("error details include bisectLogPath pointing to .git/BISECT_LOG", async () => {
      await initRepo(repoDir);
      await setupBisect(repoDir);

      const error = asGitError(await captureError(() => makeDetector().detect()));

      expect(error.details?.bisectLogPath).toBe(join(gitDir, "BISECT_LOG"));
    });

    test("error details include step='bisect-detect' and completedSteps=[]", async () => {
      await initRepo(repoDir);
      await setupBisect(repoDir);

      const error = asGitError(await captureError(() => makeDetector().detect()));

      expect(error.details?.step).toBe("bisect-detect");
      expect(error.details?.completedSteps).toEqual([]);
    });

    test("detect() returns clean state after bisect is reset", async () => {
      await initRepo(repoDir);
      await setupBisect(repoDir);
      await expect(makeDetector().detect()).rejects.toMatchObject({
        code: GitCode.BISECT_IN_PROGRESS,
      });

      await git(["bisect", "reset"], repoDir);

      await expect(makeDetector().detect()).resolves.toMatchObject({
        state: { status: "clean" },
      });
    });
  });

  // ── Merge ──

  describe("detect() — merge in progress", () => {
    test("detects merge state when MERGE_HEAD is present", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      await makeCommit(repoDir, "feature.txt", "feature content", "feat: add feature");
      await git(["checkout", "main"], repoDir);
      // --no-commit leaves MERGE_HEAD set even without conflicts
      await git(["merge", "--no-commit", "--no-ff", "feature"], repoDir);

      const result = await makeDetector().detect();

      expect(result.state.status).toBe("merge");
    });

    test("mergeHead matches the tip commit hash of the merged branch", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      const featureHash = await makeCommit(repoDir, "feature.txt", "feature content", "feat: add feature");
      await git(["checkout", "main"], repoDir);
      await git(["merge", "--no-commit", "--no-ff", "feature"], repoDir);

      const result = await makeDetector().detect();
      if (result.state.status !== "merge") throw new Error("Expected merge state");

      expect(result.state.mergeHead).toBe(featureHash);
    });

    test("mergeMessage is populated with MERGE_MSG content", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      await makeCommit(repoDir, "feature.txt", "feature content", "feat: add feature");
      await git(["checkout", "main"], repoDir);
      await git(["merge", "--no-commit", "--no-ff", "feature"], repoDir);

      const result = await makeDetector().detect();
      if (result.state.status !== "merge") throw new Error("Expected merge state");

      expect(result.state.mergeMessage).not.toBeNull();
      expect(typeof result.state.mergeMessage).toBe("string");
    });

    test("short-circuits: completedSteps ends at 'merge-detect'", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      await makeCommit(repoDir, "feature.txt", "feature content", "feat: add feature");
      await git(["checkout", "main"], repoDir);
      await git(["merge", "--no-commit", "--no-ff", "feature"], repoDir);

      const result = await makeDetector().detect();

      expect(result.completedSteps).toEqual(["bisect-detect", "rebase-detect", "merge-detect"]);
      expect(result.finalStep).toBe("merge-detect");
    });

    test("detect() returns clean state after merge is aborted", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      await makeCommit(repoDir, "feature.txt", "feature content", "feat: add feature");
      await git(["checkout", "main"], repoDir);
      await git(["merge", "--no-commit", "--no-ff", "feature"], repoDir);
      await expect(makeDetector().detect()).resolves.toMatchObject({ state: { status: "merge" } });

      await git(["merge", "--abort"], repoDir);

      await expect(makeDetector().detect()).resolves.toMatchObject({ state: { status: "clean" } });
    });
  });

  // ── Squash-merge 

  describe("detect() — squash-merge", () => {
    test("detects squash-merge state: SQUASH_MSG present, MERGE_HEAD absent", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      await makeCommit(repoDir, "feature.txt", "feature content", "feat: squash target");
      await git(["checkout", "main"], repoDir);
      // --squash creates SQUASH_MSG but NOT MERGE_HEAD
      await git(["merge", "--squash", "feature"], repoDir);

      const result = await makeDetector().detect();

      expect(result.state.status).toBe("squash-merge");
    });

    test("squashMessage contains the squashed commit's message", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      await makeCommit(repoDir, "feature.txt", "feature content", "feat: squash target");
      await git(["checkout", "main"], repoDir);
      await git(["merge", "--squash", "feature"], repoDir);

      const result = await makeDetector().detect();
      if (result.state.status !== "squash-merge") throw new Error("Expected squash-merge state");

      expect(result.state.squashMessage.length).toBeGreaterThan(0);
      expect(result.state.squashMessage).toContain("feat: squash target");
    });

    test("short-circuits: completedSteps ends at 'squash-detect'", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      await makeCommit(repoDir, "feature.txt", "feature content", "feat: squash target");
      await git(["checkout", "main"], repoDir);
      await git(["merge", "--squash", "feature"], repoDir);

      const result = await makeDetector().detect();

      expect(result.completedSteps).toEqual([
        "bisect-detect", "rebase-detect", "merge-detect", "squash-detect",
      ]);
      expect(result.finalStep).toBe("squash-detect");
    });
  });

  // ── Cherry-pick 

  describe("detect() — cherry-pick in progress", () => {
    // Sets up a cherry-pick that conflicts, leaving CHERRY_PICK_HEAD in place.
    // Conflict is required: --no-commit alone does NOT create CHERRY_PICK_HEAD
    // unless the apply fails.
    async function setupCherryPickConflict(): Promise<string> {
      await makeCommit(repoDir, "file.txt", "initial content", "chore: base");
      await git(["checkout", "-b", "feature"], repoDir);
      const pickHash = await makeCommit(repoDir, "file.txt", "feature content", "feat: cherry");
      await git(["checkout", "main"], repoDir);
      // Modify the same file so cherry-pick will conflict and leave CHERRY_PICK_HEAD
      await makeCommit(repoDir, "file.txt", "main content", "chore: main change");
      await gitExpectFail(["cherry-pick", pickHash], repoDir);
      return pickHash;
    }

    test("detects cherry-pick state from CHERRY_PICK_HEAD", async () => {
      await initRepo(repoDir);
      await setupCherryPickConflict();

      const result = await makeDetector().detect();

      expect(result.state.status).toBe("cherry-pick");
    });

    test("cherryPickHead matches the picked commit hash", async () => {
      await initRepo(repoDir);
      const pickHash = await setupCherryPickConflict();

      const result = await makeDetector().detect();
      if (result.state.status !== "cherry-pick") throw new Error("Expected cherry-pick state");

      expect(result.state.cherryPickHead).toBe(pickHash);
    });

    test("originalTitle is resolved from the real commit via git log", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "initial content", "chore: base");
      await git(["checkout", "-b", "feature"], repoDir);
      const pickHash = await makeCommit(repoDir, "file.txt", "feature content", "feat: the cherry title");
      await git(["checkout", "main"], repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: main change");
      await gitExpectFail(["cherry-pick", pickHash], repoDir);

      const result = await makeDetector().detect();
      if (result.state.status !== "cherry-pick") throw new Error("Expected cherry-pick state");

      expect(result.state.originalTitle).toBe("feat: the cherry title");
    });

    test("short-circuits: completedSteps ends at 'cherry-pick-detect'", async () => {
      await initRepo(repoDir);
      await setupCherryPickConflict();

      const result = await makeDetector().detect();

      expect(result.completedSteps).toEqual([
        "bisect-detect", "rebase-detect", "merge-detect", "squash-detect", "cherry-pick-detect",
      ]);
      expect(result.finalStep).toBe("cherry-pick-detect");
    });
  });

  // ── Revert ─

  describe("detect() — revert in progress", () => {
    test("detects revert state from REVERT_HEAD", async () => {
      await initRepo(repoDir);
      const hash = await makeCommit(repoDir, "file.txt", "content", "feat: to revert");
      await git(["revert", "--no-commit", hash], repoDir);

      const result = await makeDetector().detect();

      expect(result.state.status).toBe("revert");
    });

    test("revertHead matches the reverted commit hash", async () => {
      await initRepo(repoDir);
      const hash = await makeCommit(repoDir, "file.txt", "content", "feat: to revert");
      await git(["revert", "--no-commit", hash], repoDir);

      const result = await makeDetector().detect();
      if (result.state.status !== "revert") throw new Error("Expected revert state");

      expect(result.state.revertHead).toBe(hash);
    });

    test("originalTitle is resolved from the real commit via git log", async () => {
      await initRepo(repoDir);
      const hash = await makeCommit(repoDir, "file.txt", "content", "feat: the revert title");
      await git(["revert", "--no-commit", hash], repoDir);

      const result = await makeDetector().detect();
      if (result.state.status !== "revert") throw new Error("Expected revert state");

      expect(result.state.originalTitle).toBe("feat: the revert title");
    });

    test("completedSteps contains all 6 steps when revert is the final detection", async () => {
      await initRepo(repoDir);
      const hash = await makeCommit(repoDir, "file.txt", "content", "feat: to revert");
      await git(["revert", "--no-commit", hash], repoDir);

      const result = await makeDetector().detect();

      expect(result.completedSteps).toEqual([
        "bisect-detect", "rebase-detect", "merge-detect",
        "squash-detect", "cherry-pick-detect", "revert-detect",
      ]);
      expect(result.finalStep).toBe("revert-detect");
    });
  });

  // ── Rebase ─

  describe("detect() — rebase in progress (merge backend)", () => {
    test("detects rebase state when rebase-merge/ dir is created by conflict", async () => {
      await initRepo(repoDir);
      await setupRebaseConflict(repoDir);

      const result = await makeDetector().detect();

      expect(result.state.status).toBe("rebase");
    });

    test("rebaseType is 'merge' for the merge backend", async () => {
      await initRepo(repoDir);
      await setupRebaseConflict(repoDir);

      const result = await makeDetector().detect();
      if (result.state.status !== "rebase") throw new Error("Expected rebase state");

      expect(result.state.rebaseType).toBe("merge");
    });

    test("originalMessage is non-null and contains the conflicting commit's message", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "file.txt", "initial", "chore: initial");
      await git(["checkout", "-b", "feature"], repoDir);
      await makeCommit(repoDir, "file.txt", "feature content", "feat: rebase original message");
      await git(["checkout", "main"], repoDir);
      await makeCommit(repoDir, "file.txt", "main content", "chore: main change");
      await git(["checkout", "feature"], repoDir);
      await gitExpectFail(["rebase", "--merge", "main"], repoDir);

      const result = await makeDetector().detect();
      if (result.state.status !== "rebase") throw new Error("Expected rebase state");

      expect(result.state.originalMessage).not.toBeNull();
      expect(result.state.originalMessage).toContain("feat: rebase original message");
    });

    test("short-circuits: completedSteps ends at 'rebase-detect'", async () => {
      await initRepo(repoDir);
      await setupRebaseConflict(repoDir);

      const result = await makeDetector().detect();

      expect(result.completedSteps).toEqual(["bisect-detect", "rebase-detect"]);
      expect(result.finalStep).toBe("rebase-detect");
    });

    test("detect() returns clean state after rebase is aborted", async () => {
      await initRepo(repoDir);
      await setupRebaseConflict(repoDir);
      await expect(makeDetector().detect()).resolves.toMatchObject({ state: { status: "rebase" } });

      await git(["rebase", "--abort"], repoDir);

      await expect(makeDetector().detect()).resolves.toMatchObject({ state: { status: "clean" } });
    });
  });

  // ── Detection priority ──

  describe("detect() — detection priority", () => {
    test("rebase takes priority over merge: rebase-merge/ wins over MERGE_HEAD", async () => {
      // During rebase with conflict, git's merge machinery may create MERGE_HEAD in .git/;
      // StateDetector must report "rebase" not "merge"
      await initRepo(repoDir);
      await setupRebaseConflict(repoDir);

      const result = await makeDetector().detect();

      // Must be "rebase", never "merge"
      expect(result.state.status).toBe("rebase");
    });

    test("bisect blocks detection even when other state files exist alongside BISECT_LOG", async () => {
      await initRepo(repoDir);
      await setupBisect(repoDir);
      // Manually plant MERGE_HEAD to simulate a parallel state — bisect must still win
      await Bun.write(join(gitDir, "MERGE_HEAD"), "abc123def456abc123def456abc123def456abc1\n");

      await expect(makeDetector().detect()).rejects.toMatchObject({
        code: GitCode.BISECT_IN_PROGRESS,
      });
    });
  });
});