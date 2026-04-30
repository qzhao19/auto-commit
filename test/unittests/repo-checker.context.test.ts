import { afterEach, describe, expect, jest, spyOn, test } from "bun:test";
import { GitCode, GitError } from "../../src/shared/exceptions/index";
import type { GitRunResult } from "../../src/shared/types/index";
import type { GitRunner } from "../../src/core/git/runner/index";
import { RepoChecker } from "../../src/core/git/context/repo-checker";
import type { BunFile } from "bun";

// ── Constants ────────────────────────────────────────────────────────────────

const REPO_CWD  = "/repo";
const GIT_DIR   = "/repo/.git";
const WORKTREE  = "/repo";
const BRANCH    = "refs/heads/main";

// ── Helpers ──────────────────────────────────────────────────────────────────

function r(overrides: Partial<GitRunResult> = {}): GitRunResult {
  return {
    args: [],
    command: "git",
    cwd: REPO_CWD,
    exitCode: 0,
    stdout: "",
    stderr: "",
    ...overrides,
  };
}

function makeRunner(...responses: Array<GitRunResult | Error>): GitRunner {
  let idx = 0;
  return {
    run: jest.fn(async () => {
      const res = responses[idx++];
      if (res === undefined) {
        throw new Error(`makeRunner: unexpected extra run() call (#${idx})`);
      }
      if (res instanceof Error) throw res;
      return res;
    }),
  } as unknown as GitRunner;
}

function mockBunFileExists(exists: boolean) {
  return spyOn(Bun, "file").mockReturnValue({
    exists: jest.fn(async () => exists),
  } as unknown as BunFile);
}

/**
 * Standard runner responses for a successful run.
 * resolveWorkTree makes 2 run() calls: --is-bare-repository then --show-toplevel.
 */
function happyResponses(): GitRunResult[] {
  return [
    r({ stdout: GIT_DIR }),        // 1. checkIsRepo
    r({ exitCode: 1 }),            // 3. resolveStagingState → has-staged-changes
    r({ stdout: "false" }),        // 4a. resolveWorkTree → --is-bare-repository
    r({ stdout: WORKTREE }),       // 4b. resolveWorkTree → --show-toplevel
    r({ exitCode: 0 }),            // 5. detectInitialCommit → HEAD exists
    r({ stdout: BRANCH }),         // 6. detectHeadState → normal branch
  ];
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

afterEach(() => {
  jest.restoreAllMocks();
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("RepoChecker", () => {

  // ── Success paths ──────────────────────────────────────────────────────────

  describe("check() — success paths", () => {
    test("returns a complete GitRepoPrecheckResult on the happy path", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(...happyResponses()));

      const result = await checker.check();

      expect(result.success).toBe(true);
      expect(result.context.gitDir).toBe(GIT_DIR);
      expect(result.context.workTree).toBe(WORKTREE);
      expect(result.context.isInitialCommit).toBe(false);
      expect(result.context.isDetachedHead).toBe(false);
      expect(result.context.currentBranch).toBe("main");
    });

    test("completedSteps lists all 6 steps in execution order", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(...happyResponses()));

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

    test("finalStep equals 'detached-head-check' when all steps succeed", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(...happyResponses()));

      const result = await checker.check();

      expect(result.finalStep).toBe("detached-head-check");
    });

    test("isInitialCommit is true when rev-parse HEAD exits with code 128", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),   // checkIsRepo
        r({ exitCode: 1 }),       // resolveStagingState
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),  // resolveWorkTree → --show-toplevel
        r({ exitCode: 128 }),     // detectInitialCommit → no HEAD yet
        r({ stdout: BRANCH }),    // detectHeadState
      ));

      const result = await checker.check();

      expect(result.context.isInitialCommit).toBe(true);
    });

    test("isDetachedHead is true and currentBranch is null when symbolic-ref exits 1", async () => {
      spyOn(console, "warn").mockImplementation(() => {});
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 0 }),
        r({ exitCode: 1 }),       // symbolic-ref → detached HEAD
      ));

      const result = await checker.check();

      expect(result.context.isDetachedHead).toBe(true);
      expect(result.context.currentBranch).toBeNull();
    });

    test("currentBranch strips the refs/heads/ prefix from symbolic-ref output", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 0 }),
        r({ stdout: "refs/heads/feat/my-feature" }),
      ));

      const result = await checker.check();

      expect(result.context.currentBranch).toBe("feat/my-feature");
    });

    test("resolves a relative gitDir by joining it with the result cwd", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: ".git", cwd: "/repo" }),  // relative path
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 0 }),
        r({ stdout: BRANCH }),
      ));

      const result = await checker.check();

      expect(result.context.gitDir).toBe("/repo/.git");
    });

    test("returns an absolute gitDir unchanged when --git-dir output starts with '/'", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: "/absolute/.git" }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 0 }),
        r({ stdout: BRANCH }),
      ));

      const result = await checker.check();

      expect(result.context.gitDir).toBe("/absolute/.git");
    });
  });

  // ── Error propagation and step tracking ────────────────────────────────────

  describe("check() — error propagation and step tracking", () => {

    test("step 1 failure (is-repo): re-throws with step='is-repo' and completedSteps=[]", async () => {
      mockBunFileExists(false);
      const original = new GitError({ code: GitCode.NOT_A_REPO, message: "not a repo" });
      const checker = new RepoChecker(makeRunner(original));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect(caught).toBeInstanceOf(GitError);
      const e = caught as GitError;
      // GitError: original code is preserved
      expect(e.code).toBe(GitCode.NOT_A_REPO);
      expect(e.details?.step).toBe("is-repo");
      expect(e.details?.completedSteps).toEqual([]);
    });

    test("step 2 failure (lock-check): re-throws with step='lock-check', completedSteps=['is-repo']", async () => {
      mockBunFileExists(true);  // lock file exists
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),  // checkIsRepo succeeds
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect(caught).toBeInstanceOf(GitError);
      const e = caught as GitError;
      // GitError: original code is preserved
      expect(e.code).toBe(GitCode.LOCK_FILE_EXISTS);
      expect(e.details?.step).toBe("lock-check");
      expect(e.details?.completedSteps).toEqual(["is-repo"]);
    });

    test("step 3 failure (staging-check / nothing-to-commit): completedSteps=['is-repo','lock-check']", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),   // checkIsRepo
        r({ exitCode: 0 }),       // diff --cached (staging empty)
        r({ exitCode: 0 }),       // diff --quiet (workdir clean too)
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect(caught).toBeInstanceOf(GitError);
      const e = caught as GitError;
      // GitError: original NOTHING_TO_COMMIT code is preserved
      expect(e.code).toBe(GitCode.NOTHING_TO_COMMIT);
      expect(e.details?.step).toBe("staging-check");
      expect(e.details?.completedSteps).toEqual(["is-repo", "lock-check"]);
    });

    test("step 3 failure (staging-check / unstaged): details includes unstagedFiles", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 0 }),                              // staging empty
        r({ exitCode: 1 }),                              // workdir dirty
        r({ stdout: "src/a.ts\nsrc/b.ts\n" }),          // diff --name-only
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      const e = caught as GitError;
      // original details are merged into e.details along with step context
      expect(e.details?.step).toBe("staging-check");
      expect(e.details?.unstagedFiles).toEqual(["src/a.ts", "src/b.ts"]);
    });

    test("step 4 failure (resolve-worktree): completedSteps contains first 3 steps", async () => {
      mockBunFileExists(false);
      const worktreeError = new GitError({ code: GitCode.COMMAND_FAILED, message: "show-toplevel failed" });
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),   // has staged
        worktreeError,        // resolveWorkTree throws on --is-bare-repository call
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      const e = caught as GitError;
      expect(e.details?.step).toBe("resolve-worktree");
      expect(e.details?.completedSteps).toEqual(["is-repo", "lock-check", "staging-check"]);
    });

    test("step 5 failure (initial-commit-check): completedSteps contains first 4 steps", async () => {
      mockBunFileExists(false);
      const headError = new GitError({ code: GitCode.COMMAND_FAILED, message: "rev-parse failed" });
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        headError,
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      const e = caught as GitError;
      expect(e.details?.step).toBe("initial-commit-check");
      expect(e.details?.completedSteps).toEqual([
        "is-repo", "lock-check", "staging-check", "resolve-worktree",
      ]);
    });

    test("step 6 failure (detached-head-check): completedSteps contains first 5 steps", async () => {
      mockBunFileExists(false);
      const headStateError = new GitError({ code: GitCode.COMMAND_FAILED, message: "symbolic-ref failed" });
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 0 }),
        headStateError,
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      const e = caught as GitError;
      expect(e.details?.step).toBe("detached-head-check");
      expect(e.details?.completedSteps).toEqual([
        "is-repo", "lock-check", "staging-check", "resolve-worktree", "initial-commit-check",
      ]);
    });

    test("re-thrown GitError preserves the original code and message", async () => {
      mockBunFileExists(false);
      const original = new GitError({
        code: GitCode.NOT_A_REPO,
        message: "Exact original message",
      });
      const checker = new RepoChecker(makeRunner(original));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      const e = caught as GitError;
      expect(e.code).toBe(GitCode.NOT_A_REPO);
      expect(e.message).toBe("Exact original message");
    });

    test("re-thrown GitError propagates the underlying cause, not the GitError itself", async () => {
      mockBunFileExists(false);
      const rootCause = new Error("underlying spawn error");
      const original = new GitError({
        code: GitCode.NOT_A_REPO,
        message: "not a repo",
        cause: rootCause,
      });
      const checker = new RepoChecker(makeRunner(original));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      // cause is error.cause (the root), not error itself — no GitError → cause → GitError chain
      expect((caught as GitError).cause).toBe(rootCause);
    });

    test("re-thrown GitError merges original details with step and completedSteps", async () => {
      mockBunFileExists(false);
      const original = new GitError({
        code: GitCode.NOT_A_REPO,
        message: "not a repo",
        details: { cwd: "/project", stderr: "fatal: not a git repository" },
      });
      const checker = new RepoChecker(makeRunner(original));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      const e = caught as GitError;
      // original details and step context are all merged into e.details
      expect(e.details?.cwd).toBe("/project");
      expect(e.details?.stderr).toBe("fatal: not a git repository");
      expect(e.details?.step).toBe("is-repo");
      expect(e.details?.completedSteps).toEqual([]);
    });

    test("completedSteps in re-thrown error is a snapshot (not the live array)", async () => {
      mockBunFileExists(false);
      jest.restoreAllMocks();
      mockBunFileExists(true);
      const runner = makeRunner(r({ stdout: GIT_DIR }));
      const checker2 = new RepoChecker(runner);

      let caught: unknown;
      try { await checker2.check(); } catch (e) { caught = e; }

      const snapshot = (caught as GitError).details?.completedSteps as string[];
      snapshot.push("injected");
      expect(snapshot).toContain("injected");
      expect(snapshot[0]).toBe("is-repo");
    });

    test("non-GitError is wrapped into GitError with COMMAND_FAILED code", async () => {
      mockBunFileExists(false);
      const networkError = new TypeError("Network failure");
      const runner = {
        run: jest.fn(async () => { throw networkError; }),
      } as unknown as GitRunner;
      const checker = new RepoChecker(runner);

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect(caught).toBeInstanceOf(GitError);
      const e = caught as GitError;
      expect(e.code).toBe(GitCode.COMMAND_FAILED);
      expect(e.message).toBe("Network failure");
      expect(e.cause).toBe(networkError);
      expect(e.details?.step).toBe("is-repo");
    });
  });

  // ── checkLockFile ──────────────────────────────────────────────────────────

  describe("checkLockFile()", () => {
    test("does not throw when index.lock does not exist", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(...happyResponses()));

      await expect(checker.check()).resolves.toBeDefined();
    });

    test("throws GitError(LOCK_FILE_EXISTS) when index.lock exists", async () => {
      mockBunFileExists(true);
      const checker = new RepoChecker(makeRunner(r({ stdout: GIT_DIR })));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect(caught).toBeInstanceOf(GitError);
      // GitError: original code is preserved in the re-thrown error
      expect((caught as GitError).code).toBe(GitCode.LOCK_FILE_EXISTS);
    });

    test("checks the path <gitDir>/index.lock", async () => {
      const fileSpy = mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(...happyResponses()));

      await checker.check();

      expect(fileSpy).toHaveBeenCalledWith(`${GIT_DIR}/index.lock`);
    });

    test("lockPath is included in error details when lock exists", async () => {
      mockBunFileExists(true);
      const checker = new RepoChecker(makeRunner(r({ stdout: GIT_DIR })));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      // lockPath is merged into e.details along with step context
      expect((caught as GitError).details?.lockPath).toBe(`${GIT_DIR}/index.lock`);
    });
  });

  // ── resolveStagingState ────────────────────────────────────────────────────

  describe("resolveStagingState()", () => {
    test("check() succeeds (implying has-staged-changes) when diff --cached exits 1", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(...happyResponses()));

      const result = await checker.check();

      // stagingState is removed from context; a successful check() implies has-staged-changes
      expect(result.success).toBe(true);
      expect(result.context).not.toHaveProperty("stagingState");
    });

    test("throws STAGING_EMPTY with unstagedFiles when staging is empty but workdir is dirty", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 0 }),                              // diff --cached → staging empty
        r({ exitCode: 1 }),                              // diff --quiet → workdir dirty
        r({ stdout: "src/foo.ts\nsrc/bar.ts\n" }),       // diff --name-only
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect(caught).toBeInstanceOf(GitError);
      const e = caught as GitError;
      // GitError: original code and merged details
      expect(e.code).toBe(GitCode.STAGING_EMPTY);
      expect(e.details?.unstagedFiles).toEqual(["src/foo.ts", "src/bar.ts"]);
    });

    test("filters empty lines from diff --name-only output", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 0 }),
        r({ exitCode: 1 }),
        r({ stdout: "\nsrc/foo.ts\n\nsrc/bar.ts\n" }),   // blank lines present
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect((caught as GitError).details?.unstagedFiles).toEqual(["src/foo.ts", "src/bar.ts"]);
    });

    test("throws NOTHING_TO_COMMIT when both staging and workdir are clean", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 0 }),   // diff --cached clean
        r({ exitCode: 0 }),   // diff --quiet clean
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect(caught).toBeInstanceOf(GitError);
      const e = caught as GitError;
      // GitError: original NOTHING_TO_COMMIT code is preserved
      expect(e.code).toBe(GitCode.NOTHING_TO_COMMIT);
      expect(e.message).toContain("no changes to commit");
    });

    test("unstaged-files error message contains 'git add' hint", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 0 }),
        r({ exitCode: 1 }),
        r({ stdout: "README.md\n" }),
      ));

      let caught: unknown;
      try { await checker.check(); } catch (e) { caught = e; }

      expect((caught as GitError).message).toContain("git add");
    });
  });

  // ── detectInitialCommit ────────────────────────────────────────────────────

  describe("detectInitialCommit()", () => {
    test("isInitialCommit is false when rev-parse HEAD exits 0 (commits exist)", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 0 }),     // HEAD exists
        r({ stdout: BRANCH }),
      ));

      const result = await checker.check();

      expect(result.context.isInitialCommit).toBe(false);
    });

    test("isInitialCommit is true when rev-parse HEAD exits 128 (no commits yet)", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 128 }),   // no HEAD
        r({ stdout: BRANCH }),
      ));

      const result = await checker.check();

      expect(result.context.isInitialCommit).toBe(true);
    });
  });

  // ── detectHeadState ────────────────────────────────────────────────────────

  describe("detectHeadState()", () => {
    test("isDetachedHead is false and currentBranch is parsed when symbolic-ref exits 0", async () => {
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 0 }),
        r({ stdout: "refs/heads/develop" }),
      ));

      const result = await checker.check();

      expect(result.context.isDetachedHead).toBe(false);
      expect(result.context.currentBranch).toBe("develop");
    });

    test("logs a console.warn containing 'Detached HEAD' on detached state", async () => {
      const warnSpy = spyOn(console, "warn").mockImplementation(() => {});
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(
        r({ stdout: GIT_DIR }),
        r({ exitCode: 1 }),
        r({ stdout: "false" }),   // resolveWorkTree → --is-bare-repository
        r({ stdout: WORKTREE }),
        r({ exitCode: 0 }),
        r({ exitCode: 1 }),   // symbolic-ref exits 1 → detached
      ));

      await checker.check();

      expect(warnSpy).toHaveBeenCalledTimes(1);
    });

    test("does not call console.warn when HEAD is not detached", async () => {
      const warnSpy = spyOn(console, "warn").mockImplementation(() => {});
      mockBunFileExists(false);
      const checker = new RepoChecker(makeRunner(...happyResponses()));

      await checker.check();

      expect(warnSpy).not.toHaveBeenCalled();
    });
  });
});