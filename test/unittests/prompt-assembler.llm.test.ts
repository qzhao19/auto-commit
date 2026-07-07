import { describe, expect, test } from "bun:test";
import type {
  GitInternalOpState,
  GitPipelineResult,
  GitRepoPrecheckContext,
  StagedFileChange,
  ClassifiedFile,
  StagedDiffSummary,
  DiffPlanResult,
  BudgetEstimate,
} from "../../src/shared/types/index";
import { PromptAssembler } from "../../src/core/llm/prompt/prompt-assembler";

// ── Constants ──

const HASH_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

// ── Fixtures ──

const REPO_CONTEXT: GitRepoPrecheckContext = {
  gitDir: "/repo/.git",
  workTree: "/repo",
  isInitialCommit: false,
  isDetachedHead: false,
  currentBranch: "main",
};

const MOCK_FILE: StagedFileChange = {
  path: "src/app.ts",
  oldPath: null,
  changeType: "modified",
  similarityScore: null,
  isBinary: false,
  isSubmodule: false,
  insertions: 10,
  deletions: 5,
  diff: null,
};

const MOCK_RENAMED_FILE: StagedFileChange = {
  path: "src/new-name.ts",
  oldPath: "src/old-name.ts",
  changeType: "renamed",
  similarityScore: 95,
  isBinary: false,
  isSubmodule: false,
  insertions: 2,
  deletions: 1,
  diff: null,
};

const MOCK_BINARY_FILE: StagedFileChange = {
  path: "assets/image.png",
  oldPath: null,
  changeType: "added",
  similarityScore: null,
  isBinary: true,
  isSubmodule: false,
  insertions: null,
  deletions: null,
  diff: null,
};

const MOCK_CLASSIFIED_SOURCE: ClassifiedFile = {
  file: MOCK_FILE,
  isNoise: false,
  nonNoiseCategory: "source",
};

const MOCK_CLASSIFIED_RENAMED: ClassifiedFile = {
  file: MOCK_RENAMED_FILE,
  isNoise: false,
  nonNoiseCategory: "source",
};

const MOCK_CLASSIFIED_BINARY: ClassifiedFile = {
  file: MOCK_BINARY_FILE,
  isNoise: true,
  noiseCategory: "binary",
};

const MOCK_DIFF_SUMMARY: StagedDiffSummary = {
  totalFiles: 1,
  totalInsertions: 10,
  totalDeletions: 5,
  hasBinaryFiles: false,
  hasSubmodules: false,
  files: [MOCK_FILE],
};

const MOCK_BUDGET_ESTIMATE: BudgetEstimate = {
  totalFiles: 1,
  nonNoiseFiles: 1,
  noiseFiles: 0,
  renamedNoContentChangeCount: 0,
  maxSingleFileLines: 15,
  totalChangedLines: 15,
  estimatedTokensIfFull: 200,
  tokenBudget: 16_000,
  isWithinBudget: true,
};

const MOCK_DIFF_PLAN: DiffPlanResult = {
  estimate: MOCK_BUDGET_ESTIMATE,
  plans: [
    {
      file: MOCK_CLASSIFIED_SOURCE,
      mode: "full",
      estimatedTokens: 200,
    },
  ],
  fullDiffCount: 1,
  degradedCount: 0,
};

// ── Helper functions ──────────────────────────────────────────────────────────

function makeFullInput(overrides: Partial<Extract<GitPipelineResult, { route: "full" }>> = {}): Extract<GitPipelineResult, { route: "full" }> {
  return {
    route: "full",
    repoContext: REPO_CONTEXT,
    diffSummary: MOCK_DIFF_SUMMARY,
    diffPlan: MOCK_DIFF_PLAN,
    diffTexts: new Map(),
    completedSteps: ["repo-precheck", "state-detect", "diff-collect", "file-classify", "budget-plan", "diff-fetch"],
    ...overrides,
  };
}

type NonCleanState = Exclude<GitInternalOpState, { status: "clean" }>;

function makeInterruptedInput(
  status: "merge" | "squash-merge" | "cherry-pick" | "revert" | "rebase",
  stateOverrides: Record<string, unknown> = {},
): Extract<GitPipelineResult, { route: "interrupted" }> {
   const baseStates: Record<typeof status, NonCleanState> = {
    merge: { status: "merge" as const, mergeHead: HASH_A, mergeMessage: null, ...stateOverrides },
    "squash-merge": { status: "squash-merge" as const, squashMessage: "Squash feature branch", ...stateOverrides },
    "cherry-pick": { status: "cherry-pick" as const, cherryPickHead: HASH_A, originalTitle: null, ...stateOverrides },
    revert: { status: "revert" as const, revertHead: HASH_A, originalTitle: null, ...stateOverrides },
    rebase: { status: "rebase" as const, rebaseType: "merge" as const, originalMessage: null, ...stateOverrides },
  };

  return {
    route: "interrupted",
    gitState: baseStates[status],
    commitMessage: "Suggested commit message",
    completedSteps: ["repo-precheck", "state-detect"],
  };
}


// ── Tests ──────

describe("PromptAssembler", () => {
  const assembler = new PromptAssembler();

  // ── assemble() top-level structure ──────────────────────────────────────────

  test("assemble() returns system + user messages with token estimate", () => {
    const input = makeFullInput();
    const result = assembler.assemble(input);

    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]!.role).toBe("system");
    expect(result.messages[1]!.role).toBe("user");
    expect(result.tokenEstimate).toBeGreaterThan(0);
  });

  test("assemble() token estimate scales with content length", () => {
    const shortInput = makeFullInput();
    const shortResult = assembler.assemble(shortInput);

    const longInput = makeFullInput({
      diffTexts: new Map([["src/app.ts", "a".repeat(10000)]]),
    });
    const longResult = assembler.assemble(longInput);

    expect(longResult.tokenEstimate).toBeGreaterThan(shortResult.tokenEstimate);
  });

  // ── Full route: buildRepoSection ───────────────────────────────────────────

  test("full route: repo section shows branch name", () => {
    const input = makeFullInput();
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Repository");
    expect(userMsg).toContain("Branch: main");
  });

  test("full route: repo section shows detached HEAD", () => {
    const input = makeFullInput({
      repoContext: { ...REPO_CONTEXT, isDetachedHead: true, currentBranch: null },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("Branch: (detached HEAD)");
    expect(userMsg).toContain("HEAD: detached");
  });

  test("full route: repo section shows initial commit flag", () => {
    const input = makeFullInput({
      repoContext: { ...REPO_CONTEXT, isInitialCommit: true },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("Initial commit: yes (repository has no prior commits)");
  });

  // ── Full route: buildSummarySection ─────────────────────────────────────────

  test("full route: summary shows file count and line stats", () => {
    const input = makeFullInput();
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Summary");
    expect(userMsg).toContain("1 file(s) changed — +10 insertions, -5 deletions");
  });

  test("full route: summary shows noise files separately", () => {
    const input = makeFullInput({
      diffSummary: {
        ...MOCK_DIFF_SUMMARY,
        totalFiles: 2,
        hasBinaryFiles: true,
      },
      diffPlan: {
        ...MOCK_DIFF_PLAN,
        estimate: {
          ...MOCK_BUDGET_ESTIMATE,
          totalFiles: 2,
          nonNoiseFiles: 1,
          noiseFiles: 1,
        },
      },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("1 content file(s), 1 noise file(s)");
    expect(userMsg).toContain("Contains binary files");
  });

  test("full route: summary shows renamed-no-change count", () => {
    const input = makeFullInput({
      diffPlan: {
        ...MOCK_DIFF_PLAN,
        estimate: {
          ...MOCK_BUDGET_ESTIMATE,
          renamedNoContentChangeCount: 3,
        },
      },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("3 rename(s) with no content change");
  });

  test("full route: summary shows budget exceeded warning", () => {
    const input = makeFullInput({
      diffPlan: {
        ...MOCK_DIFF_PLAN,
        estimate: {
          ...MOCK_BUDGET_ESTIMATE,
          isWithinBudget: false,
        },
        fullDiffCount: 2,
        degradedCount: 3,
      },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("Token budget exceeded: 2 file(s) with full diff, 3 file(s) omitted");
  });

  test("full route: summary shows submodule flag", () => {
    const input = makeFullInput({
      diffSummary: {
        ...MOCK_DIFF_SUMMARY,
        hasSubmodules: true,
      },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("Contains submodule changes");
  });

  // ── Full route: buildFileManifest ───────────────────────────────────────────

  test("full route: file manifest shows full-mode file", () => {
    const input = makeFullInput();
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Files");
    expect(userMsg).toContain("[full diff below] src/app.ts");
    expect(userMsg).toContain("(modified, +10 -5)");
    expect(userMsg).toContain("[source]");
  });

  test("full route: file manifest shows degraded file with reason", () => {
    const input = makeFullInput({
      diffPlan: {
        ...MOCK_DIFF_PLAN,
        plans: [
          {
            file: MOCK_CLASSIFIED_SOURCE,
            mode: "degraded",
            degradationReason: "budget-exceeded",
            estimatedTokens: null,
          },
        ],
        fullDiffCount: 0,
        degradedCount: 1,
      },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("[omitted: budget-exceeded]");
  });

  test("full route: file manifest shows renamed file with arrow", () => {
    const input = makeFullInput({
      diffSummary: {
        ...MOCK_DIFF_SUMMARY,
        files: [MOCK_RENAMED_FILE],
      },
      diffPlan: {
        ...MOCK_DIFF_PLAN,
        plans: [
          {
            file: MOCK_CLASSIFIED_RENAMED,
            mode: "full",
            estimatedTokens: 50,
          },
        ],
      },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("src/old-name.ts → src/new-name.ts");
    expect(userMsg).toContain("95% similar");
  });

  test("full route: file manifest shows binary file as noise", () => {
    const input = makeFullInput({
      diffSummary: {
        ...MOCK_DIFF_SUMMARY,
        files: [MOCK_BINARY_FILE],
        hasBinaryFiles: true,
      },
      diffPlan: {
        ...MOCK_DIFF_PLAN,
        plans: [
          {
            file: MOCK_CLASSIFIED_BINARY,
            mode: "degraded",
            degradationReason: "noise",
            estimatedTokens: null,
          },
        ],
        estimate: {
          ...MOCK_BUDGET_ESTIMATE,
          nonNoiseFiles: 0,
          noiseFiles: 1,
        },
      },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("[omitted: noise]");
    expect(userMsg).toContain("[binary]");
  });

  test("full route: file manifest handles null insertions/deletions", () => {
    const fileWithNullStats: StagedFileChange = {
      ...MOCK_FILE,
      insertions: null,
      deletions: null,
    };
    const input = makeFullInput({
      diffSummary: {
        ...MOCK_DIFF_SUMMARY,
        files: [fileWithNullStats],
      },
      diffPlan: {
        ...MOCK_DIFF_PLAN,
        plans: [
          {
            file: { ...MOCK_CLASSIFIED_SOURCE, file: fileWithNullStats },
            mode: "full",
            estimatedTokens: 50,
          },
        ],
      },
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("+0 -0");
  });

  // ── Full route: buildDiffSection ────────────────────────────────────────────

  test("full route: diff section shows diff text when available", () => {
    const diffText = "--- a/src/app.ts\n+++ b/src/app.ts\n@@ -1,3 +1,4 @@";
    const input = makeFullInput({
      diffTexts: new Map([["src/app.ts", diffText]]),
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Diffs");
    expect(userMsg).toContain("### src/app.ts");
    expect(userMsg).toContain("```diff");
    expect(userMsg).toContain(diffText);
  });

  test("full route: diff section omitted when no diffs available", () => {
    const input = makeFullInput({
      diffTexts: new Map(),
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).not.toContain("## Diffs");
  });

  test("full route: diff section skips degraded files", () => {
    const input = makeFullInput({
      diffPlan: {
        ...MOCK_DIFF_PLAN,
        plans: [
          {
            file: MOCK_CLASSIFIED_SOURCE,
            mode: "degraded",
            degradationReason: "oversized",
            estimatedTokens: null,
          },
        ],
        fullDiffCount: 0,
        degradedCount: 1,
      },
      diffTexts: new Map([["src/app.ts", "some diff content"]]),
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).not.toContain("## Diffs");
  });

  test("full route: diff section skips empty diff text", () => {
    const input = makeFullInput({
      diffTexts: new Map([["src/app.ts", ""]]),
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).not.toContain("## Diffs");
  });

  test("full route: diff section skips files with undefined diff text", () => {
    const input = makeFullInput({
      diffTexts: new Map(), // src/app.ts not in map → undefined
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).not.toContain("## Diffs");
  });

  // ── Interrupted route: merge ────────────────────────────────────────────────

  test("interrupted route: merge without message", () => {
    const input = makeInterruptedInput("merge");
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Git operation: merge");
    expect(userMsg).not.toContain("Merge message:");
    expect(userMsg).toContain("## Suggested message");
    expect(userMsg).toContain("Suggested commit message");
    expect(userMsg).toContain("Refine the suggested message if needed");
  });

  test("interrupted route: merge with message", () => {
    const input = makeInterruptedInput("merge", {
      mergeMessage: "Merge feature branch into main\n\nAdds new features",
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("Merge message:");
    expect(userMsg).toContain("  Merge feature branch into main");
    expect(userMsg).toContain("  Adds new features");
  });

  // ── Interrupted route: squash-merge ─────────────────────────────────────────

  test("interrupted route: squash-merge always has message", () => {
    const input = makeInterruptedInput("squash-merge");
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Git operation: squash merge");
    expect(userMsg).toContain("Squash message:");
    expect(userMsg).toContain("  Squash feature branch");
  });

  // ── Interrupted route: cherry-pick ──────────────────────────────────────────

  test("interrupted route: cherry-pick without original title", () => {
    const input = makeInterruptedInput("cherry-pick");
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Git operation: cherry-pick");
    expect(userMsg).not.toContain("Original commit:");
  });

  test("interrupted route: cherry-pick with original title", () => {
    const input = makeInterruptedInput("cherry-pick", {
      originalTitle: "feat: add user authentication",
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("Original commit: feat: add user authentication");
  });

  // ── Interrupted route: revert ───────────────────────────────────────────────

  test("interrupted route: revert without original title", () => {
    const input = makeInterruptedInput("revert");
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Git operation: revert");
    expect(userMsg).not.toContain("Reverted commit:");
  });

  test("interrupted route: revert with original title", () => {
    const input = makeInterruptedInput("revert", {
      originalTitle: "fix: broken API endpoint",
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("Reverted commit: fix: broken API endpoint");
  });

  // ── Interrupted route: rebase ───────────────────────────────────────────────

  test("interrupted route: rebase without original message", () => {
    const input = makeInterruptedInput("rebase");
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Git operation: rebase (merge)");
    expect(userMsg).not.toContain("Original message:");
  });

  test("interrupted route: rebase with original message", () => {
    const input = makeInterruptedInput("rebase", {
      rebaseType: "apply",
      originalMessage: "WIP: refactor database layer",
    });
    const result = assembler.assemble(input);
    const userMsg = result.messages[1]!.content;

    expect(userMsg).toContain("## Git operation: rebase (apply)");
    expect(userMsg).toContain("Original message:");
    expect(userMsg).toContain("  WIP: refactor database layer");
  });

  // ── System message ──────────────────────────────────────────────────────────

  test("system message contains Conventional Commits instruction", () => {
    const input = makeFullInput();
    const result = assembler.assemble(input);
    const systemMsg = result.messages[0]!.content;

    expect(systemMsg).toContain("Conventional Commits 1.0.0");
    expect(systemMsg).toContain("feat | fix | docs | style | refactor | perf | test | chore | build | ci");
    expect(systemMsg).toContain("≤72 characters");
    expect(systemMsg).toContain("Output ONLY the commit message");
  });

  test("system message does NOT contain merge/cherry-pick rules", () => {
    const input = makeFullInput();
    const result = assembler.assemble(input);
    const systemMsg = result.messages[0]!.content;

    expect(systemMsg).not.toContain("merge/squash");
    expect(systemMsg).not.toContain("cherry-pick/revert");
  });
});
