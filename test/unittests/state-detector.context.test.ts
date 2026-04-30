import { afterEach, beforeEach, describe, expect, jest, test } from "bun:test";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GitCode, GitError } from "../../src/shared/exceptions/index";
import { StateDetector } from "../../src/core/git/context/state-detector";
import type { GitRunner } from "../../src/core/git/runner/index";
import type { GitInternalOpState, GitRunResult } from "../../src/shared/types/index";

// ── Constants ────────────────────────────────────────────────────────────────

const COMMIT_HASH = "abc123def456abc123def456abc123def456abc1";
const COMMIT_TITLE = "feat: add some feature";
const MERGE_MSG = "Merge branch 'feature' into main";
const SQUASH_MSG = "squashed commit message";
const REBASE_MSG = "chore: update dependencies";

// ── Helpers ──────────────────────────────────────────────────────────────────

function r(overrides: Partial<GitRunResult> = {}): GitRunResult {
  return {
    args: [],
    command: "git",
    cwd: "/repo",
    exitCode: 0,
    stdout: "",
    stderr: "",
    ...overrides,
  };
}

function makeRunner(...responses: Array<GitRunResult | Error>): {
  runner: GitRunner;
  runMock: ReturnType<typeof jest.fn>;
} {
  let idx = 0;

  const runMock = jest.fn(async () => {
    const res = responses[idx++];
    if (res === undefined) {
      throw new Error(`makeRunner: unexpected extra run() call (#${idx})`);
    }
    if (res instanceof Error) {
      throw res;
    }
    return res;
  });

  return {
    runner: { run: runMock } as unknown as GitRunner,
    runMock,
  };
}

function titleRunner(title = COMMIT_TITLE): {
  runner: GitRunner;
  runMock: ReturnType<typeof jest.fn>;
} {
  return makeRunner(r({ exitCode: 0, stdout: title }));
}

function noTitleRunner(): {
  runner: GitRunner;
  runMock: ReturnType<typeof jest.fn>;
} {
  return makeRunner(r({ exitCode: 0, stdout: "" }));
}

function unusedRunner(): {
  runner: GitRunner;
  runMock: ReturnType<typeof jest.fn>;
} {
  const runMock = jest.fn(async () => {
    throw new Error("runner.run() should not have been called in this scenario");
  });

  return {
    runner: { run: runMock } as unknown as GitRunner,
    runMock,
  };
}

async function writeGitFile(gitDir: string, filename: string, content: string): Promise<void> {
  await Bun.write(join(gitDir, filename), content + "\n");
}

async function makeRebaseDir(gitDir: string, type: "merge" | "apply"): Promise<string> {
  const dir = join(gitDir, type === "merge" ? "rebase-merge" : "rebase-apply");
  await mkdir(dir, { recursive: true });
  return dir;
}

async function captureError(fn: () => Promise<unknown>): Promise<unknown> {
  try {
    await fn();
  } catch (error) {
    return error;
  }

  throw new Error("Expected function to throw");
}

function expectGitError(error: unknown): GitError {
  expect(error).toBeInstanceOf(GitError);
  return error as GitError;
}

function expectRebaseState(
  state: GitInternalOpState,
): Extract<GitInternalOpState, { status: "rebase" }> {
  expect(state.status).toBe("rebase");
  if (state.status !== "rebase") {
    throw new Error(`Expected rebase state, received ${state.status}`);
  }
  return state;
}

function expectMergeState(
  state: GitInternalOpState,
): Extract<GitInternalOpState, { status: "merge" }> {
  expect(state.status).toBe("merge");
  if (state.status !== "merge") {
    throw new Error(`Expected merge state, received ${state.status}`);
  }
  return state;
}

function expectSquashState(
  state: GitInternalOpState,
): Extract<GitInternalOpState, { status: "squash-merge" }> {
  expect(state.status).toBe("squash-merge");
  if (state.status !== "squash-merge") {
    throw new Error(`Expected squash-merge state, received ${state.status}`);
  }
  return state;
}

function expectCherryPickState(
  state: GitInternalOpState,
): Extract<GitInternalOpState, { status: "cherry-pick" }> {
  expect(state.status).toBe("cherry-pick");
  if (state.status !== "cherry-pick") {
    throw new Error(`Expected cherry-pick state, received ${state.status}`);
  }
  return state;
}

function expectRevertState(
  state: GitInternalOpState,
): Extract<GitInternalOpState, { status: "revert" }> {
  expect(state.status).toBe("revert");
  if (state.status !== "revert") {
    throw new Error(`Expected revert state, received ${state.status}`);
  }
  return state;
}

// ── Setup / Teardown ─────────────────────────────────────────────────────────

let gitDir: string;

beforeEach(async () => {
  gitDir = await mkdtemp(join(tmpdir(), "state-detector-test-"));
});

afterEach(async () => {
  await rm(gitDir, { recursive: true, force: true });
});

// ── Tests ────────────────────────────────────────────────────────────────────

describe("StateDetector", () => {
  describe("detect() — clean state", () => {
    test("returns { status: 'clean' } when no special files exist", async () => {
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.state).toEqual({ status: "clean" });
    });

    test("success is true", async () => {
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      expect((await detector.detect()).success).toBe(true);
    });

    test("completedSteps contains all 6 steps in execution order", async () => {
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.completedSteps).toEqual([
        "bisect-detect",
        "rebase-detect",
        "merge-detect",
        "squash-detect",
        "cherry-pick-detect",
        "revert-detect",
      ]);
    });

    test("finalStep is 'revert-detect' when all steps run without finding anything", async () => {
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      expect((await detector.detect()).finalStep).toBe("revert-detect");
    });
  });

  describe("detect() — bisect (hard exit)", () => {
    test("throws GitError(BISECT_IN_PROGRESS) when BISECT_LOG exists", async () => {
      await writeGitFile(gitDir, "BISECT_LOG", "git bisect start\n# bad: abc123");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      await expect(detector.detect()).rejects.toMatchObject({
        code: GitCode.BISECT_IN_PROGRESS,
      });
    });

    test("thrown error is a GitError instance", async () => {
      await writeGitFile(gitDir, "BISECT_LOG", "content");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      await expect(detector.detect()).rejects.toBeInstanceOf(GitError);
    });

    test("error details include bisectLogPath", async () => {
      await writeGitFile(gitDir, "BISECT_LOG", "content");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.details?.bisectLogPath).toBe(join(gitDir, "BISECT_LOG"));
    });

    test("error is wrapped with step='bisect-detect' and completedSteps=[]", async () => {
      await writeGitFile(gitDir, "BISECT_LOG", "content");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.details?.step).toBe("bisect-detect");
      expect(error.details?.completedSteps).toEqual([]);
    });

    test("absent BISECT_LOG → no throw, detection continues normally", async () => {
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      await expect(detector.detect()).resolves.toMatchObject({
        state: { status: "clean" },
      });
    });
  });

  describe("detect() — rebase (merge backend)", () => {
    test("detects rebase-merge state and reads 'message'", async () => {
      const rebaseDir = await makeRebaseDir(gitDir, "merge");
      await Bun.write(join(rebaseDir, "message"), REBASE_MSG);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.state).toEqual({
        status: "rebase",
        rebaseType: "merge",
        originalMessage: REBASE_MSG,
      });
    });

    test("originalMessage is null when 'message' file is absent", async () => {
      await makeRebaseDir(gitDir, "merge");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.state).toMatchObject({
        status: "rebase",
        rebaseType: "merge",
        originalMessage: null,
      });
    });

    test("'message' file content is trimmed", async () => {
      const rebaseDir = await makeRebaseDir(gitDir, "merge");
      await Bun.write(join(rebaseDir, "message"), "  spaced message  \n");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRebaseState(result.state);

      expect(state.originalMessage).toBe("spaced message");
    });

    test("short-circuits: completedSteps = ['bisect-detect', 'rebase-detect']", async () => {
      await makeRebaseDir(gitDir, "merge");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.completedSteps).toEqual(["bisect-detect", "rebase-detect"]);
      expect(result.finalStep).toBe("rebase-detect");
    });
  });

  describe("detect() — rebase (apply backend)", () => {
    test("uses 'final-commit' when all three files are present", async () => {
      const rebaseDir = await makeRebaseDir(gitDir, "apply");
      await Bun.write(join(rebaseDir, "final-commit"), "final message");
      await Bun.write(join(rebaseDir, "msg-clean"), "msg-clean message");
      await Bun.write(join(rebaseDir, "msg"), "raw message");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRebaseState(result.state);

      expect(state.originalMessage).toBe("final message");
    });

    test("falls back to 'msg-clean' when 'final-commit' is absent", async () => {
      const rebaseDir = await makeRebaseDir(gitDir, "apply");
      await Bun.write(join(rebaseDir, "msg-clean"), "msg-clean message");
      await Bun.write(join(rebaseDir, "msg"), "raw message");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRebaseState(result.state);

      expect(state.originalMessage).toBe("msg-clean message");
    });

    test("falls back to 'msg' when 'final-commit' and 'msg-clean' are absent", async () => {
      const rebaseDir = await makeRebaseDir(gitDir, "apply");
      await Bun.write(join(rebaseDir, "msg"), "raw message");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRebaseState(result.state);

      expect(state.originalMessage).toBe("raw message");
    });

    test("originalMessage is null when all apply message files are absent", async () => {
      await makeRebaseDir(gitDir, "apply");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRebaseState(result.state);

      expect(state.originalMessage).toBeNull();
    });

    test("rebaseType is 'apply'", async () => {
      await makeRebaseDir(gitDir, "apply");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRebaseState(result.state);

      expect(state.rebaseType).toBe("apply");
    });

    test("short-circuits: completedSteps = ['bisect-detect', 'rebase-detect']", async () => {
      await makeRebaseDir(gitDir, "apply");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.completedSteps).toEqual(["bisect-detect", "rebase-detect"]);
    });
  });

  describe("detect() — invalid rebase state (both dirs exist)", () => {
    test("throws GitError(COMMAND_FAILED) when both rebase-merge and rebase-apply exist", async () => {
      await makeRebaseDir(gitDir, "merge");
      await makeRebaseDir(gitDir, "apply");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      await expect(detector.detect()).rejects.toMatchObject({
        code: GitCode.COMMAND_FAILED,
      });
    });

    test("error message mentions both directory names", async () => {
      await makeRebaseDir(gitDir, "merge");
      await makeRebaseDir(gitDir, "apply");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.message).toContain("rebase-merge");
      expect(error.message).toContain("rebase-apply");
    });

    test("error is wrapped with step='rebase-detect' and completedSteps=['bisect-detect']", async () => {
      await makeRebaseDir(gitDir, "merge");
      await makeRebaseDir(gitDir, "apply");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.details?.step).toBe("rebase-detect");
      expect(error.details?.completedSteps).toEqual(["bisect-detect"]);
    });

    test("error details preserve both rebase directory paths", async () => {
      const rebaseMergeDir = await makeRebaseDir(gitDir, "merge");
      const rebaseApplyDir = await makeRebaseDir(gitDir, "apply");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.details?.rebaseMergeDir).toBe(rebaseMergeDir);
      expect(error.details?.rebaseApplyDir).toBe(rebaseApplyDir);
    });
  });

  describe("detect() — merge", () => {
    test("detects merge state from MERGE_HEAD", async () => {
      await writeGitFile(gitDir, "MERGE_HEAD", COMMIT_HASH);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.state).toMatchObject({
        status: "merge",
        mergeHead: COMMIT_HASH,
      });
    });

    test("includes mergeMessage when MERGE_MSG is present", async () => {
      await writeGitFile(gitDir, "MERGE_HEAD", COMMIT_HASH);
      await writeGitFile(gitDir, "MERGE_MSG", MERGE_MSG);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectMergeState(result.state);

      expect(state.mergeMessage).toBe(MERGE_MSG);
    });

    test("mergeMessage is null when MERGE_MSG is absent", async () => {
      await writeGitFile(gitDir, "MERGE_HEAD", COMMIT_HASH);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectMergeState(result.state);

      expect(state.mergeMessage).toBeNull();
    });

    test("mergeHead is trimmed", async () => {
      await Bun.write(join(gitDir, "MERGE_HEAD"), "  " + COMMIT_HASH + "  \n");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectMergeState(result.state);

      expect(state.mergeHead).toBe(COMMIT_HASH);
    });

    test("short-circuits: completedSteps = ['bisect-detect','rebase-detect','merge-detect']", async () => {
      await writeGitFile(gitDir, "MERGE_HEAD", COMMIT_HASH);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.completedSteps).toEqual([
        "bisect-detect",
        "rebase-detect",
        "merge-detect",
      ]);
      expect(result.finalStep).toBe("merge-detect");
    });
  });

  describe("detect() — squash-merge", () => {
    test("detects squash-merge state from SQUASH_MSG", async () => {
      await writeGitFile(gitDir, "SQUASH_MSG", SQUASH_MSG);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.state).toMatchObject({
        status: "squash-merge",
        squashMessage: SQUASH_MSG,
      });
    });

    test("squashMessage content is trimmed", async () => {
      await Bun.write(join(gitDir, "SQUASH_MSG"), "  " + SQUASH_MSG + "  \n");
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectSquashState(result.state);

      expect(state.squashMessage).toBe(SQUASH_MSG);
    });

    test("short-circuits: completedSteps ends at 'squash-detect'", async () => {
      await writeGitFile(gitDir, "SQUASH_MSG", SQUASH_MSG);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.completedSteps).toEqual([
        "bisect-detect",
        "rebase-detect",
        "merge-detect",
        "squash-detect",
      ]);
      expect(result.finalStep).toBe("squash-detect");
    });
  });

  describe("detect() — cherry-pick", () => {
    test("detects cherry-pick state from CHERRY_PICK_HEAD", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const { runner } = noTitleRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.state).toMatchObject({
        status: "cherry-pick",
        cherryPickHead: COMMIT_HASH,
      });
    });

    test("originalTitle is populated when runner returns a title", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const { runner } = titleRunner(COMMIT_TITLE);
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectCherryPickState(result.state);

      expect(state.originalTitle).toBe(COMMIT_TITLE);
    });

    test("originalTitle is null when runner returns empty stdout", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const { runner } = noTitleRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectCherryPickState(result.state);

      expect(state.originalTitle).toBeNull();
    });

    test("originalTitle is null when runner exits with code 128 (unreachable hash)", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const { runner } = makeRunner(r({ exitCode: 128, stdout: "" }));
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectCherryPickState(result.state);

      expect(state.originalTitle).toBeNull();
    });

    test("runner is called with [log, -1, --format=%s, <hash>] and allowedExitCodes [0,128]", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const { runner, runMock } = titleRunner();
      const detector = new StateDetector(gitDir, runner);

      await detector.detect();

      expect(runMock).toHaveBeenCalledWith(
        ["log", "-1", "--format=%s", COMMIT_HASH],
        { allowedExitCodes: [0, 128] },
      );
    });

    test("short-circuits: completedSteps ends at 'cherry-pick-detect'", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const { runner } = noTitleRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.completedSteps).toEqual([
        "bisect-detect",
        "rebase-detect",
        "merge-detect",
        "squash-detect",
        "cherry-pick-detect",
      ]);
    });
  });

  describe("detect() — revert", () => {
    test("detects revert state from REVERT_HEAD", async () => {
      await writeGitFile(gitDir, "REVERT_HEAD", COMMIT_HASH);
      const { runner } = noTitleRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.state).toMatchObject({
        status: "revert",
        revertHead: COMMIT_HASH,
      });
    });

    test("originalTitle is populated when runner returns a title", async () => {
      await writeGitFile(gitDir, "REVERT_HEAD", COMMIT_HASH);
      const { runner } = titleRunner(COMMIT_TITLE);
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRevertState(result.state);

      expect(state.originalTitle).toBe(COMMIT_TITLE);
    });

    test("originalTitle is null when runner returns empty stdout", async () => {
      await writeGitFile(gitDir, "REVERT_HEAD", COMMIT_HASH);
      const { runner } = noTitleRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRevertState(result.state);

      expect(state.originalTitle).toBeNull();
    });

    test("originalTitle is null when runner exits with code 128", async () => {
      await writeGitFile(gitDir, "REVERT_HEAD", COMMIT_HASH);
      const { runner } = makeRunner(r({ exitCode: 128, stdout: "" }));
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();
      const state = expectRevertState(result.state);

      expect(state.originalTitle).toBeNull();
    });

    test("runner is called with the commit hash from REVERT_HEAD", async () => {
      await writeGitFile(gitDir, "REVERT_HEAD", COMMIT_HASH);
      const { runner, runMock } = titleRunner();
      const detector = new StateDetector(gitDir, runner);

      await detector.detect();

      expect(runMock).toHaveBeenCalledWith(
        ["log", "-1", "--format=%s", COMMIT_HASH],
        { allowedExitCodes: [0, 128] },
      );
    });

    test("completedSteps contains all 6 steps when revert is the final detection", async () => {
      await writeGitFile(gitDir, "REVERT_HEAD", COMMIT_HASH);
      const { runner } = noTitleRunner();
      const detector = new StateDetector(gitDir, runner);

      const result = await detector.detect();

      expect(result.finalStep).toBe("revert-detect");
      expect(result.completedSteps).toEqual([
        "bisect-detect",
        "rebase-detect",
        "merge-detect",
        "squash-detect",
        "cherry-pick-detect",
        "revert-detect",
      ]);
    });
  });

  describe("detect() — detection priority", () => {
    test("rebase-merge takes priority over MERGE_HEAD", async () => {
      await makeRebaseDir(gitDir, "merge");
      await writeGitFile(gitDir, "MERGE_HEAD", COMMIT_HASH);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      expect((await detector.detect()).state.status).toBe("rebase");
    });

    test("rebase-apply takes priority over CHERRY_PICK_HEAD", async () => {
      await makeRebaseDir(gitDir, "apply");
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      expect((await detector.detect()).state.status).toBe("rebase");
    });

    test("MERGE_HEAD takes priority over SQUASH_MSG", async () => {
      await writeGitFile(gitDir, "MERGE_HEAD", COMMIT_HASH);
      await writeGitFile(gitDir, "SQUASH_MSG", SQUASH_MSG);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      expect((await detector.detect()).state.status).toBe("merge");
    });

    test("SQUASH_MSG takes priority over CHERRY_PICK_HEAD", async () => {
      await writeGitFile(gitDir, "SQUASH_MSG", SQUASH_MSG);
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      expect((await detector.detect()).state.status).toBe("squash-merge");
    });

    test("CHERRY_PICK_HEAD takes priority over REVERT_HEAD", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      await writeGitFile(gitDir, "REVERT_HEAD", COMMIT_HASH);
      const { runner } = noTitleRunner();
      const detector = new StateDetector(gitDir, runner);

      expect((await detector.detect()).state.status).toBe("cherry-pick");
    });

    test("bisect blocks all other detections even when MERGE_HEAD is present", async () => {
      await writeGitFile(gitDir, "BISECT_LOG", "bisect content");
      await writeGitFile(gitDir, "MERGE_HEAD", COMMIT_HASH);
      const { runner } = unusedRunner();
      const detector = new StateDetector(gitDir, runner);

      await expect(detector.detect()).rejects.toMatchObject({
        code: GitCode.BISECT_IN_PROGRESS,
      });
    });
  });

  describe("detect() — error wrapping", () => {
    test("non-GitError is wrapped as GitError(COMMAND_FAILED)", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const plainError = new Error("unexpected plain error");
      const { runner } = makeRunner(plainError);
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.code).toBe(GitCode.COMMAND_FAILED);
      expect(error.message).toBe("unexpected plain error");
      expect(error.cause).toBe(plainError);
    });

    test("GitError from runner is wrapped with step and completedSteps added to details", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const original = new GitError({
        code: GitCode.COMMAND_FAILED,
        message: "git log failed",
      });
      const { runner } = makeRunner(original);
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.details?.step).toBe("cherry-pick-detect");
      expect(error.details?.completedSteps).toEqual([
        "bisect-detect",
        "rebase-detect",
        "merge-detect",
        "squash-detect",
      ]);
    });

    test("re-thrown GitError preserves original code and message", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const original = new GitError({
        code: GitCode.COMMAND_FAILED,
        message: "original message",
      });
      const { runner } = makeRunner(original);
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.code).toBe(GitCode.COMMAND_FAILED);
      expect(error.message).toBe("original message");
    });

    test("re-thrown GitError preserves original error's cause", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const rootCause = new Error("root cause");
      const original = new GitError({
        code: GitCode.COMMAND_FAILED,
        message: "original",
        cause: rootCause,
      });
      const { runner } = makeRunner(original);
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.cause).toBe(rootCause);
    });

    test("existing details on original GitError are preserved after wrapping", async () => {
      await writeGitFile(gitDir, "CHERRY_PICK_HEAD", COMMIT_HASH);
      const original = new GitError({
        code: GitCode.COMMAND_FAILED,
        message: "failed",
        details: { exitCode: 128, extra: "payload" },
      });
      const { runner } = makeRunner(original);
      const detector = new StateDetector(gitDir, runner);

      const error = expectGitError(await captureError(() => detector.detect()));

      expect(error.details?.exitCode).toBe(128);
      expect(error.details?.extra).toBe("payload");
      expect(error.details?.step).toBe("cherry-pick-detect");
    });
  });
});
