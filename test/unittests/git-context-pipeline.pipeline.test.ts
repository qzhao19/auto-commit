// test/unittests/git-context-pipeline.pipeline.test.ts
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  jest,
  spyOn,
  test,
} from "bun:test";
import { GitCode, GitError } from "../../src/shared/exceptions/index";
import type { BudgetThresholds } from "../../src/shared/types/index";
import type { GitRunner } from "../../src/core/git/runner/index";
import { GitContextPipeline } from "../../src/core/git/pipeline/git-context-pipeline";
import { RepoChecker } from "../../src/core/git/context/repo-checker";
import { StateDetector } from "../../src/core/git/context/state-detector";
import { DiffCollector } from "../../src/core/git/diff/diff-collector";
import { FileClassifier } from "../../src/core/git/diff/file-classifier";
import { BudgetPlanner } from "../../src/core/git/diff/budget-planner";

// ── Constants ─────────────────────────────────────────────────────────────────

const GIT_DIR = "/repo/.git";
const WORKTREE = "/repo";
const BRANCH = "main";
const HASH_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

// ── Fixtures ──────────────────────────────────────────────────────────────────

const REPO_CONTEXT = {
  gitDir: GIT_DIR,
  workTree: WORKTREE,
  isInitialCommit: false,
  isDetachedHead: false,
  currentBranch: BRANCH,
};

const PRECHECK_RESULT = {
  finalStep: "complete" as const,
  completedSteps: [] as any[],
  context: REPO_CONTEXT,
};

const CLEAN_DETECT = {
  finalStep: "complete" as const,
  completedSteps: [] as any[],
  state: { status: "clean" as const },
};

const MOCK_FILE = {
  path: "src/index.ts",
  oldPath: null,
  changeType: "modified" as const,
  similarityScore: null,
  isBinary: false,
  isSubmodule: false,
  insertions: 10,
  deletions: 3,
  diff: null,
};

const MOCK_DIFF_SUMMARY = {
  totalFiles: 1,
  totalInsertions: 10,
  totalDeletions: 3,
  hasBinaryFiles: false,
  hasSubmodules: false,
  files: [MOCK_FILE],
};

const MOCK_CLASSIFIED_FILE = {
  file: MOCK_FILE,
  isNoise: false as const,
  nonNoiseCategory: "source" as const,
};

const MOCK_CLASSIFIED = {
  noiseCount: 0,
  nonNoiseCount: 1,
  files: [MOCK_CLASSIFIED_FILE],
};

const MOCK_DIFF_PLAN = {
  estimate: {
    totalFiles: 1,
    nonNoiseFiles: 1,
    noiseFiles: 0,
    renamedNoContentChangeCount: 0,
    maxSingleFileLines: 13,
    totalChangedLines: 13,
    estimatedTokensIfFull: 180,
    tokenBudget: 16_000,
    isWithinBudget: true,
  },
  plans: [
    { file: MOCK_CLASSIFIED_FILE, mode: "full" as const, estimatedTokens: 180 },
  ],
  fullDiffCount: 1,
  degradedCount: 0,
};

const MOCK_DIFF_TEXTS = new Map([
  [
    "src/index.ts",
    "diff --git a/src/index.ts b/src/index.ts\n+export const x = 1;",
  ],
]);

// ── Helpers ───

const MOCK_RUNNER = { run: jest.fn() } as unknown as GitRunner;

function makePipeline(thresholds?: BudgetThresholds) {
  return new GitContextPipeline(MOCK_RUNNER, thresholds);
}

type SpyOf<T> = T extends (...args: infer A) => infer R
  ? jest.Mock<(...args: A) => R>
  : never;

/** Wire all spies for the happy-path full route. */
function setupFullPath(
  ck: SpyOf<RepoChecker["check"]>,
  dt: SpyOf<StateDetector["detect"]>,
  cl: SpyOf<DiffCollector["collect"]>,
  cf: SpyOf<FileClassifier["classify"]>,
  pl: SpyOf<BudgetPlanner["plan"]>,
  cd: SpyOf<DiffCollector["collectDiff"]>,
) {
  ck.mockResolvedValue(PRECHECK_RESULT);
  dt.mockResolvedValue(CLEAN_DETECT);
  cl.mockResolvedValue(MOCK_DIFF_SUMMARY);
  cf.mockResolvedValue(MOCK_CLASSIFIED);
  pl.mockReturnValue(MOCK_DIFF_PLAN);
  cd.mockResolvedValue(MOCK_DIFF_TEXTS);
}

/** Wire spies so the state detector returns a non-clean internal-op state. */
function setupInterruptedPath(
  ck: SpyOf<RepoChecker["check"]>,
  dt: SpyOf<StateDetector["detect"]>,
  state: Record<string, unknown>,
) {
  ck.mockResolvedValue(PRECHECK_RESULT);
  dt.mockResolvedValue({
    finalStep: "complete",
    completedSteps: [],
    state,
  } as unknown as Awaited<ReturnType<StateDetector["detect"]>>);
}

// ── Spy management ────

let checkSpy: SpyOf<RepoChecker["check"]>;
let detectSpy: SpyOf<StateDetector["detect"]>;
let collectSpy: SpyOf<DiffCollector["collect"]>;
let collectDiffSpy: SpyOf<DiffCollector["collectDiff"]>;
let classifySpy: SpyOf<FileClassifier["classify"]>;
let planSpy: SpyOf<BudgetPlanner["plan"]>;

beforeEach(() => {
  checkSpy = spyOn(RepoChecker.prototype, "check") as typeof checkSpy;
  detectSpy = spyOn(StateDetector.prototype, "detect") as typeof detectSpy;
  collectSpy = spyOn(DiffCollector.prototype, "collect") as typeof collectSpy;
  collectDiffSpy = spyOn(
    DiffCollector.prototype,
    "collectDiff",
  ) as typeof collectDiffSpy;
  classifySpy = spyOn(
    FileClassifier.prototype,
    "classify",
  ) as typeof classifySpy;
  planSpy = spyOn(BudgetPlanner.prototype, "plan") as typeof planSpy;
});

afterEach(() => {
  checkSpy.mockRestore();
  detectSpy.mockRestore();
  collectSpy.mockRestore();
  collectDiffSpy.mockRestore();
  classifySpy.mockRestore();
  planSpy.mockRestore();
});

// ═════════════════════════════════════════════════════════════════════════════
// execute() — interrupted route
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline.execute() — interrupted route: merge", () => {
  test("merge with mergeMessage → commitMessage uses mergeMessage", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "merge",
      mergeHead: HASH_A,
      mergeMessage: "Merge branch 'feature' into main",
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe("Merge branch 'feature' into main");
    }
  });

  test("merge with null mergeMessage → falls back to 'Merge {hash}'", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "merge",
      mergeHead: HASH_A,
      mergeMessage: null,
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe(`Merge ${HASH_A}`);
    }
  });

  test("merge: gitState carries the full merge state object", async () => {
    const mergeState = {
      status: "merge" as const,
      mergeHead: HASH_A,
      mergeMessage: "Merge branch 'dev'",
    };
    setupInterruptedPath(checkSpy, detectSpy, mergeState);
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.gitState).toEqual(mergeState);
    }
  });
});

describe("GitContextPipeline.execute() — interrupted route: squash-merge", () => {
  test("squash-merge → commitMessage uses squashMessage", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "squash-merge",
      squashMessage: "squash changes from feature/payments",
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe("squash changes from feature/payments");
    }
  });
});

describe("GitContextPipeline.execute() — interrupted route: cherry-pick", () => {
  test("cherry-pick with originalTitle → 'cherry-pick: {title}'", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "cherry-pick",
      cherryPickHead: HASH_A,
      originalTitle: "fix: resolve memory leak",
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe(
        "cherry-pick: fix: resolve memory leak",
      );
    }
  });

  test("cherry-pick with null originalTitle → 'cherry-pick {hash}'", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "cherry-pick",
      cherryPickHead: HASH_B,
      originalTitle: null,
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe(`cherry-pick ${HASH_B}`);
    }
  });
});

describe("GitContextPipeline.execute() — interrupted route: revert", () => {
  test("revert with originalTitle → 'revert: {title}'", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "revert",
      revertHead: HASH_A,
      originalTitle: "feat: add dangerous migration",
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe(
        "revert: feat: add dangerous migration",
      );
    }
  });

  test("revert with null originalTitle → 'revert {hash}'", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "revert",
      revertHead: HASH_B,
      originalTitle: null,
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe(`revert ${HASH_B}`);
    }
  });
});

describe("GitContextPipeline.execute() — interrupted route: rebase", () => {
  test("rebase (merge) with originalMessage → uses originalMessage", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "rebase",
      rebaseType: "merge",
      originalMessage: "feat: implement streaming pipeline",
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe("feat: implement streaming pipeline");
    }
  });

  test("rebase (apply) with null originalMessage → 'rebase (apply)'", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "rebase",
      rebaseType: "apply",
      originalMessage: null,
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe("rebase (apply)");
    }
  });

  test("rebase (merge) with null originalMessage → 'rebase (merge)'", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "rebase",
      rebaseType: "merge",
      originalMessage: null,
    });
    const result = await makePipeline().execute();
    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.commitMessage).toBe("rebase (merge)");
    }
  });
});

describe("GitContextPipeline.execute() — interrupted route: shared assertions", () => {
  test("completedSteps contains exactly ['repo-precheck', 'state-detect']", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "merge",
      mergeHead: HASH_A,
      mergeMessage: null,
    });
    const result = await makePipeline().execute();
    expect(result.completedSteps).toEqual(["repo-precheck", "state-detect"]);
  });

  test("diff pipeline steps are NOT executed on interrupted route", async () => {
    setupInterruptedPath(checkSpy, detectSpy, {
      status: "merge",
      mergeHead: HASH_A,
      mergeMessage: null,
    });
    await makePipeline().execute();
    expect(collectSpy).not.toHaveBeenCalled();
    expect(classifySpy).not.toHaveBeenCalled();
    expect(planSpy).not.toHaveBeenCalled();
    expect(collectDiffSpy).not.toHaveBeenCalled();
  });

  test("route is 'interrupted' for every non-clean state", async () => {
    const states = [
      { status: "merge", mergeHead: HASH_A, mergeMessage: null },
      { status: "squash-merge", squashMessage: "squash" },
      { status: "cherry-pick", cherryPickHead: HASH_A, originalTitle: null },
      { status: "revert", revertHead: HASH_A, originalTitle: null },
      { status: "rebase", rebaseType: "merge", originalMessage: null },
    ];
    for (const state of states) {
      setupInterruptedPath(checkSpy, detectSpy, state);
      const result = await makePipeline().execute();
      expect(result.route).toBe("interrupted");

      // reset spies between iterations so mockResolvedValue is clean each time
      checkSpy.mockReset();
      detectSpy.mockReset();
    }
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// execute() — full route
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline.execute() — full route: return shape", () => {
  test("returns route: 'full'", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const result = await makePipeline().execute();
    expect(result.route).toBe("full");
  });

  test("repoContext matches the context from RepoChecker", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const result = await makePipeline().execute();
    expect(result.route).toBe("full");
    if (result.route === "full") {
      expect(result.repoContext).toEqual(REPO_CONTEXT);
    }
  });

  test("diffSummary is the value returned by DiffCollector.collect()", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const result = await makePipeline().execute();
    expect(result.route).toBe("full");
    if (result.route === "full") {
      expect(result.diffSummary).toBe(MOCK_DIFF_SUMMARY);
    }
  });

  test("diffPlan is the value returned by BudgetPlanner.plan()", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const result = await makePipeline().execute();
    expect(result.route).toBe("full");
    if (result.route === "full") {
      expect(result.diffPlan).toBe(MOCK_DIFF_PLAN);
    }
  });

  test("diffTexts is the map returned by DiffCollector.collectDiff()", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const result = await makePipeline().execute();
    expect(result.route).toBe("full");
    if (result.route === "full") {
      expect(result.diffTexts).toBe(MOCK_DIFF_TEXTS);
    }
  });

  test("full route completedSteps contains all 6 steps in order", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const result = await makePipeline().execute();
    expect(result.completedSteps).toEqual([
      "repo-precheck",
      "state-detect",
      "diff-collect",
      "file-classify",
      "budget-plan",
      "diff-fetch",
    ]);
  });
});

describe("GitContextPipeline.execute() — full route: data flow between steps", () => {
  test("FileClassifier.classify() is called with the diffSummary from collect()", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    await makePipeline().execute();
    expect(classifySpy).toHaveBeenCalledWith(MOCK_DIFF_SUMMARY);
  });

  test("BudgetPlanner.plan() is called with the classified result from classify()", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    await makePipeline().execute();
    expect(planSpy).toHaveBeenCalledWith(MOCK_CLASSIFIED);
  });

  test("collectDiff is called only with paths whose plan mode is 'full'", async () => {
    const degradedFile = { ...MOCK_FILE, path: "package-lock.json" };
    const degradedClassified = {
      file: degradedFile,
      isNoise: false as const,
      nonNoiseCategory: "lockfile" as const,
    };
    const mixedPlan = {
      ...MOCK_DIFF_PLAN,
      plans: [
        {
          file: MOCK_CLASSIFIED_FILE,
          mode: "full" as const,
          estimatedTokens: 180,
        },
        {
          file: degradedClassified,
          mode: "degraded" as const,
          degradationReason: "budget-exceeded" as const,
          estimatedTokens: null,
        },
      ],
      fullDiffCount: 1,
      degradedCount: 1,
    };
    checkSpy.mockResolvedValue(PRECHECK_RESULT);
    detectSpy.mockResolvedValue(CLEAN_DETECT);
    collectSpy.mockResolvedValue(MOCK_DIFF_SUMMARY);
    classifySpy.mockResolvedValue(MOCK_CLASSIFIED);
    planSpy.mockReturnValue(mixedPlan);
    collectDiffSpy.mockResolvedValue(MOCK_DIFF_TEXTS);

    await makePipeline().execute();

    const pathsArg: readonly string[] = collectDiffSpy.mock.calls[0]![0];
    expect(pathsArg).toContain("src/index.ts");
    expect(pathsArg).not.toContain("package-lock.json");
    expect(pathsArg).toHaveLength(1);
  });

  test("collectDiff is called with [] when every plan is degraded", async () => {
    const allDegradedPlan = {
      ...MOCK_DIFF_PLAN,
      plans: [
        {
          file: MOCK_CLASSIFIED_FILE,
          mode: "degraded" as const,
          degradationReason: "oversized" as const,
          estimatedTokens: null,
        },
      ],
      fullDiffCount: 0,
      degradedCount: 1,
    };
    checkSpy.mockResolvedValue(PRECHECK_RESULT);
    detectSpy.mockResolvedValue(CLEAN_DETECT);
    collectSpy.mockResolvedValue(MOCK_DIFF_SUMMARY);
    classifySpy.mockResolvedValue(MOCK_CLASSIFIED);
    planSpy.mockReturnValue(allDegradedPlan);
    collectDiffSpy.mockResolvedValue(new Map());

    await makePipeline().execute();

    expect(collectDiffSpy.mock.calls[0]![0]).toEqual([]);
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// execute() — error propagation
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline.execute() — error propagation", () => {
  test("GitError from RepoChecker.check() propagates as-is", async () => {
    const err = new GitError({
      code: GitCode.NOT_A_REPO,
      message: "not a repo",
    });
    checkSpy.mockRejectedValue(err);

    let caught: unknown;
    try {
      await makePipeline().execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.NOT_A_REPO);
    expect((caught as GitError).message).toBe("not a repo");
  });

  test("GitError from StateDetector.detect() propagates as-is", async () => {
    const err = new GitError({
      code: GitCode.BISECT_IN_PROGRESS,
      message: "bisect active",
    });
    checkSpy.mockResolvedValue(PRECHECK_RESULT);
    detectSpy.mockRejectedValue(err);

    let caught: unknown;
    try {
      await makePipeline().execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.BISECT_IN_PROGRESS);
  });

  test("GitError from DiffCollector.collect() propagates as-is", async () => {
    const err = new GitError({
      code: GitCode.COMMAND_FAILED,
      message: "git diff failed",
    });
    checkSpy.mockResolvedValue(PRECHECK_RESULT);
    detectSpy.mockResolvedValue(CLEAN_DETECT);
    collectSpy.mockRejectedValue(err);

    let caught: unknown;
    try {
      await makePipeline().execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.COMMAND_FAILED);
    expect((caught as GitError).message).toBe("git diff failed");
  });

  test("GitError from FileClassifier.classify() propagates as-is", async () => {
    const err = new GitError({
      code: GitCode.COMMAND_FAILED,
      message: "cat-file failed",
    });
    checkSpy.mockResolvedValue(PRECHECK_RESULT);
    detectSpy.mockResolvedValue(CLEAN_DETECT);
    collectSpy.mockResolvedValue(MOCK_DIFF_SUMMARY);
    classifySpy.mockRejectedValue(err);

    let caught: unknown;
    try {
      await makePipeline().execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.COMMAND_FAILED);
  });

  test("GitError from DiffCollector.collectDiff() propagates as-is", async () => {
    const err = new GitError({
      code: GitCode.COMMAND_FAILED,
      message: "diff fetch failed",
    });
    checkSpy.mockResolvedValue(PRECHECK_RESULT);
    detectSpy.mockResolvedValue(CLEAN_DETECT);
    collectSpy.mockResolvedValue(MOCK_DIFF_SUMMARY);
    classifySpy.mockResolvedValue(MOCK_CLASSIFIED);
    planSpy.mockReturnValue(MOCK_DIFF_PLAN);
    collectDiffSpy.mockRejectedValue(err);

    let caught: unknown;
    try {
      await makePipeline().execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).message).toBe("diff fetch failed");
  });

  test("non-GitError from RepoChecker propagates without being swallowed", async () => {
    const plain = new TypeError("unexpected crash");
    checkSpy.mockRejectedValue(plain);

    let caught: unknown;
    try {
      await makePipeline().execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(TypeError);
    expect((caught as TypeError).message).toBe("unexpected crash");
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// constructor — thresholds injection
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline constructor — thresholds", () => {
  test("pipeline executes successfully without explicit thresholds (defaults applied)", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const result = await new GitContextPipeline(MOCK_RUNNER).execute();
    expect(result.route).toBe("full");
  });

  test("pipeline executes successfully with custom thresholds", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const custom: BudgetThresholds = {
      maxTotalTokens: 500,
      maxLinesPerFile: 20,
      tokensPerLine: 5,
      tokensPerFileOverhead: 25,
    };
    const result = await makePipeline(custom).execute();
    expect(result.route).toBe("full");
    expect(planSpy).toHaveBeenCalledTimes(1);
  });

  test("BudgetPlanner.plan() is called exactly once per execute() call", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    await makePipeline().execute();
    expect(planSpy).toHaveBeenCalledTimes(1);
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// execute() — edge cases
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline.execute() — edge cases", () => {
  test("isInitialCommit=true in repoContext is passed through to result", async () => {
    const initialPrecheck = {
      ...PRECHECK_RESULT,
      context: { ...REPO_CONTEXT, isInitialCommit: true },
    };
    checkSpy.mockResolvedValue(initialPrecheck);
    detectSpy.mockResolvedValue(CLEAN_DETECT);
    collectSpy.mockResolvedValue(MOCK_DIFF_SUMMARY);
    classifySpy.mockResolvedValue(MOCK_CLASSIFIED);
    planSpy.mockReturnValue(MOCK_DIFF_PLAN);
    collectDiffSpy.mockResolvedValue(MOCK_DIFF_TEXTS);

    const result = await makePipeline().execute();
    expect(result.route).toBe("full");
    if (result.route === "full") {
      expect(result.repoContext.isInitialCommit).toBe(true);
    }
  });

  test("isDetachedHead=true / currentBranch=null is passed through to result", async () => {
    const detachedPrecheck = {
      ...PRECHECK_RESULT,
      context: { ...REPO_CONTEXT, isDetachedHead: true, currentBranch: null },
    };
    checkSpy.mockResolvedValue(detachedPrecheck);
    detectSpy.mockResolvedValue(CLEAN_DETECT);
    collectSpy.mockResolvedValue(MOCK_DIFF_SUMMARY);
    classifySpy.mockResolvedValue(MOCK_CLASSIFIED);
    planSpy.mockReturnValue(MOCK_DIFF_PLAN);
    collectDiffSpy.mockResolvedValue(MOCK_DIFF_TEXTS);

    const result = await makePipeline().execute();
    expect(result.route).toBe("full");
    if (result.route === "full") {
      expect(result.repoContext.isDetachedHead).toBe(true);
      expect(result.repoContext.currentBranch).toBeNull();
    }
  });

  test("all-degraded plan → diffTexts is empty, route is still 'full'", async () => {
    const allDegradedPlan = {
      ...MOCK_DIFF_PLAN,
      plans: [
        {
          file: MOCK_CLASSIFIED_FILE,
          mode: "degraded" as const,
          degradationReason: "oversized" as const,
          estimatedTokens: null,
        },
      ],
      fullDiffCount: 0,
      degradedCount: 1,
    };
    checkSpy.mockResolvedValue(PRECHECK_RESULT);
    detectSpy.mockResolvedValue(CLEAN_DETECT);
    collectSpy.mockResolvedValue(MOCK_DIFF_SUMMARY);
    classifySpy.mockResolvedValue(MOCK_CLASSIFIED);
    planSpy.mockReturnValue(allDegradedPlan);
    collectDiffSpy.mockResolvedValue(new Map());

    const result = await makePipeline().execute();
    expect(result.route).toBe("full");
    if (result.route === "full") {
      expect(result.diffTexts.size).toBe(0);
    }
  });

  test("completedSteps array is independent across multiple execute() calls", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const pipeline = makePipeline();

    const r1 = await pipeline.execute();
    const r2 = await pipeline.execute();

    expect(r1.completedSteps).not.toBe(r2.completedSteps);
    expect(r1.completedSteps).toHaveLength(6);
    expect(r2.completedSteps).toHaveLength(6);
  });

  test("each step is appended only once even when pipeline is reused", async () => {
    setupFullPath(
      checkSpy,
      detectSpy,
      collectSpy,
      classifySpy,
      planSpy,
      collectDiffSpy,
    );
    const result = await makePipeline().execute();
    const unique = new Set(result.completedSteps);
    expect(unique.size).toBe(result.completedSteps.length);
  });
});
