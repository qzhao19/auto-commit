/**
 * E2E tests for RepoChecker — exercises the full precheck pipeline
 * against real git repositories on disk, using the real GitRunner.
 *
 * No git commands are mocked. Every assertion reflects actual git behavior.
 *
 * Requires: git >= 2.28 (for `git init -b <branch>`)
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GitCode, GitError } from "../../src/shared/exceptions/index";
import { RepoChecker } from "../../src/core/git/context/repo-checker";
import { GitRunner } from "../../src/core/git/runner/git-runner";

// ── Helpers ───

/**
 * Spawns a raw git command. Used for test setup only — never via the module
 * under test, so setup failures are clearly distinguishable from SUT failures.
 */
async function git(args: string[], cwd: string): Promise<string> {
  const proc = Bun.spawn(["git", ...args], {
    cwd,
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = await new Response(proc.stdout).text();
  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    throw new Error(`git ${args.join(" ")} failed (exit ${exitCode}): ${stderr.trim()}`);
  }
  return stdout.trim();
}

/**
 * Initialises a real git repo with a deterministic branch name and a
 * test identity so commits can be created without global git config.
 */
async function initRepo(dir: string, branch = "main"): Promise<void> {
  await git(["init", "-b", branch], dir);
  await git(["config", "user.email", "test@example.com"], dir);
  await git(["config", "user.name", "Test User"], dir);
}

/**
 * Writes a file under the repo root and stages it with `git add`.
 */
async function stageFile(
  gitRoot: string,
  relPath: string,
  content = "content",
): Promise<void> {
  const fullPath = join(gitRoot, relPath);
  await mkdir(join(fullPath, ".."), { recursive: true });
  await writeFile(fullPath, content);
  await git(["add", relPath], gitRoot);
}

/**
 * Stages a file and commits it.
 */
async function makeCommit(
  gitRoot: string,
  relPath: string,
  message = "chore: test commit",
): Promise<void> {
  await stageFile(gitRoot, relPath, "content");
  await git(["commit", "-m", message], gitRoot);
}

/**
 * Captures a thrown error from an async call; rethrows if nothing was thrown.
 */
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

// ── Fixtures ──

let repoDir: string;

beforeEach(async () => {
  const tmp = await mkdtemp(join(tmpdir(), "repo-checker-e2e-"));
  repoDir = await realpath(tmp);  // resolve macOS /var → /private/var symlink
});

afterEach(async () => {
  await rm(repoDir, { recursive: true, force: true });
});

// ── Tests ──────

describe("RepoChecker e2e", () => {

  // ── Success path ───

  describe("check() — success path", () => {
    test("returns GitRepoPrecheckResult with success=true", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.success).toBe(true);
    });

    test("gitDir resolves to the absolute .git directory path", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.context.gitDir).toBe(join(repoDir, ".git"));
    });

    test("gitDir is correctly resolved when running from a subdirectory", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "src/index.ts");
      // Runner uses subdirectory as cwd; git rev-parse --git-dir returns a relative path
      const checker = new RepoChecker(new GitRunner({ cwd: join(repoDir, "src") }));

      const result = await checker.check();

      // Must be absolute, not relative ("../.git")
      expect(result.context.gitDir).toBe(join(repoDir, ".git"));
    });

    test("workTree resolves to the repository root", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.context.workTree).toBe(repoDir);
    });

    test("isInitialCommit is true before the first commit", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.context.isInitialCommit).toBe(true);
    });

    test("isInitialCommit is false after at least one commit exists", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      await stageFile(repoDir, "second.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.context.isInitialCommit).toBe(false);
    });

    test("currentBranch reflects the actual branch name", async () => {
      await initRepo(repoDir, "main");
      await stageFile(repoDir, "file.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.context.currentBranch).toBe("main");
    });

    test("currentBranch is updated when the active branch changes", async () => {
      await initRepo(repoDir, "main");
      await makeCommit(repoDir, "initial.txt");
      await git(["checkout", "-b", "feature/my-branch"], repoDir);
      await stageFile(repoDir, "feature.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.context.currentBranch).toBe("feature/my-branch");
    });

    test("completedSteps contains all 6 steps in execution order", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.completedSteps).toEqual([
        "is-repo",
        "lock-check",
        "staging-check",
        "resolve-worktree",
        "initial-commit-check",
        "detached-head-check",
      ]);
    });

    test("finalStep is 'detached-head-check' when all steps complete", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.finalStep).toBe("detached-head-check");
    });
  });

  // ── Detached HEAD ──

  describe("check() — detached HEAD", () => {
    test("isDetachedHead is true when HEAD is detached", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      await git(["checkout", "--detach", "HEAD"], repoDir);
      await stageFile(repoDir, "new.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.context.isDetachedHead).toBe(true);
    });

    test("currentBranch is null when HEAD is detached", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      await git(["checkout", "--detach", "HEAD"], repoDir);
      await stageFile(repoDir, "new.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const result = await checker.check();

      expect(result.context.currentBranch).toBeNull();
    });

    test("check() resolves successfully in detached HEAD state (does not throw)", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      await git(["checkout", "--detach", "HEAD"], repoDir);
      await stageFile(repoDir, "new.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      await expect(checker.check()).resolves.toMatchObject({ success: true });
    });
  });

  // ── NOT_A_REPO ─────

  describe("check() — NOT_A_REPO", () => {
    test("throws GitError(NOT_A_REPO) for a plain directory", async () => {
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      await expect(checker.check()).rejects.toMatchObject({
        code: GitCode.NOT_A_REPO,
      });
    });

    test("thrown error is a GitError instance", async () => {
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      await expect(checker.check()).rejects.toBeInstanceOf(GitError);
    });

    test("error details include cwd", async () => {
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const error = asGitError(await captureError(() => checker.check()));

      expect(error.details?.cwd).toBe(repoDir);
    });

    test("error details include step='is-repo' and completedSteps=[]", async () => {
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const error = asGitError(await captureError(() => checker.check()));

      expect(error.details?.step).toBe("is-repo");
      expect(error.details?.completedSteps).toEqual([]);
    });
  });

  // ── LOCK_FILE_EXISTS ───

  describe("check() — LOCK_FILE_EXISTS", () => {
    test("throws GitError(LOCK_FILE_EXISTS) when index.lock is present", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      await Bun.write(join(repoDir, ".git", "index.lock"), "");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      await expect(checker.check()).rejects.toMatchObject({
        code: GitCode.LOCK_FILE_EXISTS,
      });
    });

    test("error details include the absolute lockPath", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      const lockPath = join(repoDir, ".git", "index.lock");
      await Bun.write(lockPath, "");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const error = asGitError(await captureError(() => checker.check()));

      expect(error.details?.lockPath).toBe(lockPath);
    });

    test("check() succeeds after the lock file is removed", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      const lockPath = join(repoDir, ".git", "index.lock");
      await Bun.write(lockPath, "");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      await expect(checker.check()).rejects.toMatchObject({ code: GitCode.LOCK_FILE_EXISTS });

      // Remove lock and retry
      await rm(lockPath);
      await expect(checker.check()).resolves.toMatchObject({ success: true });
    });

    test("error details include step='lock-check' and completedSteps=['is-repo']", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.txt");
      await Bun.write(join(repoDir, ".git", "index.lock"), "");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const error = asGitError(await captureError(() => checker.check()));

      expect(error.details?.step).toBe("lock-check");
      expect(error.details?.completedSteps).toEqual(["is-repo"]);
    });
  });

  // ── STAGING_EMPTY ──

  describe("check() — STAGING_EMPTY", () => {
    test("throws GitError(STAGING_EMPTY) when tracked files are modified but not staged", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      // Modify without staging
      await writeFile(join(repoDir, "initial.txt"), "modified content");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      await expect(checker.check()).rejects.toMatchObject({
        code: GitCode.STAGING_EMPTY,
      });
    });

    test("error details include the list of unstaged files", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      await writeFile(join(repoDir, "initial.txt"), "modified content");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const error = asGitError(await captureError(() => checker.check()));

      expect(Array.isArray(error.details?.unstagedFiles)).toBe(true);
      expect(error.details?.unstagedFiles as string[]).toContain("initial.txt");
    });

    test("multiple unstaged files all appear in the error details", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "a.txt");
      await makeCommit(repoDir, "b.txt");
      await writeFile(join(repoDir, "a.txt"), "modified a");
      await writeFile(join(repoDir, "b.txt"), "modified b");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const error = asGitError(await captureError(() => checker.check()));
      const files = error.details?.unstagedFiles as string[];

      expect(files).toContain("a.txt");
      expect(files).toContain("b.txt");
    });

    test("error details include step='staging-check' and completedSteps=['is-repo','lock-check']", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      await writeFile(join(repoDir, "initial.txt"), "modified");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const error = asGitError(await captureError(() => checker.check()));

      expect(error.details?.step).toBe("staging-check");
      expect(error.details?.completedSteps).toEqual(["is-repo", "lock-check"]);
    });
  });

  // ── NOTHING_TO_COMMIT ──

  describe("check() — NOTHING_TO_COMMIT", () => {
    test("throws GitError(NOTHING_TO_COMMIT) when the working tree is clean after commits", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      // Working tree is now clean
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      await expect(checker.check()).rejects.toMatchObject({
        code: GitCode.NOTHING_TO_COMMIT,
      });
    });

    test("throws GitError(NOTHING_TO_COMMIT) in a fresh repo with no staged files", async () => {
      await initRepo(repoDir);
      // No files staged, no commits — untracked files are invisible to `git diff`
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      await expect(checker.check()).rejects.toMatchObject({
        code: GitCode.NOTHING_TO_COMMIT,
      });
    });

    test("error details include step='staging-check'", async () => {
      await initRepo(repoDir);
      await makeCommit(repoDir, "initial.txt");
      const checker = new RepoChecker(new GitRunner({ cwd: repoDir }));

      const error = asGitError(await captureError(() => checker.check()));

      expect(error.details?.step).toBe("staging-check");
      expect(error.details?.completedSteps).toEqual(["is-repo", "lock-check"]);
    });
  });
});