// End-to-end tests for GitContextPipeline.
// Each test spins up a real temporary git repository, sets up git state via
// real git commands, and drives the full pipeline stack (RepoChecker →
// StateDetector → DiffCollector → FileClassifier → BudgetPlanner).
// No mocks are used: every git command is executed by a real GitRunner.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GitCode, GitError } from "../../src/shared/exceptions/index";
import { GitRunner } from "../../src/core/git/runner/index";
import { GitContextPipeline } from "../../src/core/git/pipeline/git-pipeline";
import type { BudgetThresholds } from "../../src/shared/types/index";

// ── Git process helpers ───────────────────────────────────────────────────────

/**
 * Author / committer env injected into every git invocation.
 * Prevents tests from requiring global git identity configuration.
 */
const GIT_ENV: Record<string, string> = {
  GIT_TERMINAL_PROMPT: "0",
  GIT_ASKPASS: "true",
  GIT_AUTHOR_NAME: "E2E Test",
  GIT_AUTHOR_EMAIL: "e2e@test.local",
  GIT_COMMITTER_NAME: "E2E Test",
  GIT_COMMITTER_EMAIL: "e2e@test.local",
};

/** Run a git command; throws if the process exits non-zero. */
async function git(cwd: string, ...args: string[]): Promise<string> {
  const proc = Bun.spawn(["git", ...args], {
    cwd,
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, ...GIT_ENV },
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  if (code !== 0) {
    throw new Error(`git ${args.join(" ")} failed (${code}): ${stderr.trim()}`);
  }
  return stdout.trim();
}

/**
 * Run a git command that may legitimately exit non-zero (conflict, bisect, …).
 * Returns the exit code; never throws.
 */
async function gitMayFail(cwd: string, ...args: string[]): Promise<number> {
  const proc = Bun.spawn(["git", ...args], {
    cwd,
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, ...GIT_ENV },
  });
  const [, , code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return code;
}

// ── Repo bootstrap helpers ────────────────────────────────────────────────────

/**
 * Create a temp directory with an empty, properly configured git repository.
 */
async function initRepo(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "ac-e2e-"));
  await git(dir, "init", "-b", "main");
  await git(dir, "config", "user.email", "e2e@test.local");
  await git(dir, "config", "user.name", "E2E Test");
  return dir;
}

/**
 * Create a repo and land an initial commit so HEAD exists.
 * Returns the directory and the first commit's hash.
 */
async function initRepoWithCommit(
  file = "README.md",
  content = "# Test Repo\n",
): Promise<{ dir: string; firstHash: string }> {
  const dir = await initRepo();
  await mkdir(join(dir, file, ".."), { recursive: true });
  await writeFile(join(dir, file), content);
  await git(dir, "add", ".");
  await git(dir, "commit", "-m", "init: initial commit");
  const firstHash = await git(dir, "rev-parse", "HEAD");
  return { dir, firstHash };
}

/**
 * Build a GitContextPipeline backed by a real GitRunner for the given directory.
 */
function makePipeline(dir: string, thresholds?: BudgetThresholds) {
  return new GitContextPipeline(new GitRunner({ cwd: dir }), thresholds);
}

// ── Per-test cleanup ──────────────────────────────────────────────────────────

let testDir: string | null = null;

beforeEach(() => {
  testDir = null;
});

afterEach(async () => {
  if (testDir) {
    await rm(testDir, { recursive: true, force: true });
    testDir = null;
  }
});

// ═════════════════════════════════════════════════════════════════════════════
// Full route — basic file types
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline E2E — full route: basic staged file types", () => {
  test("added TypeScript file: route=full, changeType=added, diff content present", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await mkdir(join(dir, "src"), { recursive: true });
    await writeFile(
      join(dir, "src/index.ts"),
      "export const hello = () => 'world';\n",
    );
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    expect(result.completedSteps).toEqual([
      "repo-precheck",
      "state-detect",
      "diff-collect",
      "file-classify",
      "budget-plan",
      "diff-fetch",
    ]);

    expect(result.diffSummary.totalFiles).toBe(1);
    const file = result.diffSummary.files[0]!;
    expect(file.path).toBe("src/index.ts");
    expect(file.changeType).toBe("added");
    expect(file.isBinary).toBe(false);
    expect(file.isSubmodule).toBe(false);
    expect(file.insertions).toBeGreaterThan(0);

    expect(result.diffPlan.fullDiffCount).toBe(1);
    expect(result.diffPlan.degradedCount).toBe(0);

    const diffText = result.diffTexts.get("src/index.ts");
    expect(diffText).toBeDefined();
    expect(diffText).toContain("diff --git");
    expect(diffText).toContain("+export const hello");
  });

  test("modified file: changeType=modified, insertions and deletions counted", async () => {
    const { dir } = await initRepoWithCommit("src/app.ts", "const a = 1;\n");
    testDir = dir;

    await writeFile(join(dir, "src/app.ts"), "const a = 1;\nconst b = 2;\n");
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    const file = result.diffSummary.files[0]!;
    expect(file.changeType).toBe("modified");
    expect(file.insertions).toBeGreaterThan(0);
    expect(file.deletions).toBe(0);
  });

  test("deleted file: changeType=deleted, deletions>0, diff text present", async () => {
    const { dir } = await initRepoWithCommit(
      "src/old.ts",
      "export const x = 1;\n",
    );
    testDir = dir;

    await rm(join(dir, "src/old.ts"));
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    const file = result.diffSummary.files[0]!;
    expect(file.changeType).toBe("deleted");
    expect(file.insertions).toBe(0);
    expect(file.deletions).toBeGreaterThan(0);
  });

  test("renamed file: changeType=renamed, oldPath and path set, similarityScore=100", async () => {
    const { dir } = await initRepoWithCommit(
      "src/old-name.ts",
      "export const x = 1;\n",
    );
    testDir = dir;

    await git(dir, "mv", "src/old-name.ts", "src/new-name.ts");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    const file = result.diffSummary.files[0]!;
    expect(file.changeType).toBe("renamed");
    expect(file.path).toBe("src/new-name.ts");
    expect(file.oldPath).toBe("src/old-name.ts");
    expect(file.similarityScore).toBe(100);
  });

  test("multiple files staged: totalFiles and diffTexts both include all files", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await mkdir(join(dir, "src"), { recursive: true });
    await writeFile(join(dir, "src/a.ts"), "const a = 1;\n");
    await writeFile(join(dir, "src/b.ts"), "const b = 2;\n");
    await writeFile(join(dir, "src/c.ts"), "const c = 3;\n");
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    expect(result.diffSummary.totalFiles).toBe(3);
    expect(result.diffTexts.size).toBe(3);
    expect(result.diffTexts.has("src/a.ts")).toBe(true);
    expect(result.diffTexts.has("src/b.ts")).toBe(true);
    expect(result.diffTexts.has("src/c.ts")).toBe(true);
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// Full route — file classification
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline E2E — full route: file classification", () => {
  test("source file: classified as non-noise/source, plan mode=full", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await writeFile(join(dir, "main.ts"), "console.log('hi');\n");
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    const plan = result.diffPlan.plans[0]!;
    expect(plan.mode).toBe("full");
    expect(plan.file.isNoise).toBe(false);
    if (!plan.file.isNoise) {
      expect(plan.file.nonNoiseCategory).toBe("source");
    }
    expect(result.diffPlan.estimate.nonNoiseFiles).toBe(1);
    expect(result.diffPlan.estimate.noiseFiles).toBe(0);
  });

  test("package-lock.json: classified as non-noise/lockfile", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await writeFile(
      join(dir, "package-lock.json"),
      JSON.stringify({ lockfileVersion: 3, packages: {} }, null, 2),
    );
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    const plan = result.diffPlan.plans[0]!;
    expect(plan.file.isNoise).toBe(false);
    if (!plan.file.isNoise) {
      expect(plan.file.nonNoiseCategory).toBe("lockfile");
    }
  });

  test("binary file: isBinary=true, classified as noise, no diff text emitted", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    // PNG magic bytes → git detects as binary
    const png = Buffer.from([
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00,
    ]);
    await Bun.write(join(dir, "icon.png"), png);
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    const file = result.diffSummary.files[0]!;
    expect(file.isBinary).toBe(true);
    expect(file.insertions).toBeNull();
    expect(file.deletions).toBeNull();

    const plan = result.diffPlan.plans[0]!;
    expect(plan.mode).toBe("degraded");
    expect(plan.file.isNoise).toBe(true);
    if (plan.file.isNoise) {
      expect(plan.file.noiseCategory).toBe("binary");
    }

    expect(result.diffTexts.has("icon.png")).toBe(false);
  });

  test("mixed source + lockfile + binary: counts and plans are split correctly", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await mkdir(join(dir, "src"), { recursive: true });
    await writeFile(join(dir, "src/app.ts"), "const x = 1;\n");
    await writeFile(join(dir, "package-lock.json"), "{'lockfileVersion':3}");
    const pngHeader = Buffer.from([
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    ]);
    await Bun.write(
      join(dir, "logo.png"),
      Buffer.concat([pngHeader, Buffer.alloc(256)]),
    );
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    expect(result.diffSummary.totalFiles).toBe(3);
    expect(result.diffPlan.estimate.nonNoiseFiles).toBe(2); // source + lockfile
    expect(result.diffPlan.estimate.noiseFiles).toBe(1); // binary

    // Source and lockfile receive full diffs; binary does not
    expect(result.diffTexts.has("src/app.ts")).toBe(true);
    expect(result.diffTexts.has("package-lock.json")).toBe(true);
    expect(result.diffTexts.has("logo.png")).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// Full route — repo context edge cases
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline E2E — full route: repo context", () => {
  test("normal branch: isInitialCommit=false, isDetachedHead=false, currentBranch=main", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await writeFile(join(dir, "check.ts"), "// check\n");
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    expect(result.repoContext.currentBranch).toBe("main");
    expect(result.repoContext.isInitialCommit).toBe(false);
    expect(result.repoContext.isDetachedHead).toBe(false);
    expect(result.repoContext.gitDir).toEndWith(".git");
    // expect(result.repoContext.workTree).toBe(dir);
  });

  test("initial commit (no prior commits): isInitialCommit=true, pipeline still completes", async () => {
    const dir = await initRepo();
    testDir = dir;

    await writeFile(join(dir, "first.ts"), "// first file\n");
    await git(dir, "add", ".");
    // Intentionally no commit — this is the very first staged set

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    expect(result.repoContext.isInitialCommit).toBe(true);
    expect(result.diffSummary.totalFiles).toBe(1);
    expect(result.diffTexts.has("first.ts")).toBe(true);
  });

  test("detached HEAD: isDetachedHead=true, currentBranch=null", async () => {
    const { dir, firstHash } = await initRepoWithCommit(
      "a.ts",
      "const a = 1;\n",
    );
    testDir = dir;

    // Create a second commit, then detach HEAD to the first
    await writeFile(join(dir, "b.ts"), "const b = 2;\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "chore: second commit");
    await git(dir, "checkout", "--detach", firstHash);

    // Stage a file while in detached HEAD
    await writeFile(join(dir, "detached.ts"), "// detached\n");
    await git(dir, "add", ".");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    expect(result.repoContext.isDetachedHead).toBe(true);
    expect(result.repoContext.currentBranch).toBeNull();
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// Full route — budget control
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline E2E — full route: budget control", () => {
  test("file exceeding maxLinesPerFile is degraded: no diff text for that file", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    // 100-line file — well above our test maxLinesPerFile of 10
    const largeContent =
      Array.from({ length: 100 }, (_, i) => `const line${i} = ${i};`).join(
        "\n",
      ) + "\n";
    await writeFile(join(dir, "large.ts"), largeContent);
    await git(dir, "add", ".");

    const tightThresholds: BudgetThresholds = {
      maxTotalTokens: 500,
      maxLinesPerFile: 10, // 100 lines > 10 → oversized → degraded
      tokensPerLine: 10,
      tokensPerFileOverhead: 50,
    };

    const result = await makePipeline(dir, tightThresholds).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    expect(result.diffPlan.degradedCount).toBe(1);
    expect(result.diffPlan.fullDiffCount).toBe(0);
    const plan = result.diffPlan.plans[0]!;
    expect(plan.mode).toBe("degraded");
    expect(plan.degradationReason).toBe("oversized");
    expect(result.diffTexts.has("large.ts")).toBe(false);
  });

  test("small file within budget: mode=full, diff text populated", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await writeFile(join(dir, "tiny.ts"), "const x = 1;\n");
    await git(dir, "add", ".");

    const generousThresholds: BudgetThresholds = {
      maxTotalTokens: 100_000,
      maxLinesPerFile: 1_000,
      tokensPerLine: 10,
      tokensPerFileOverhead: 50,
    };

    const result = await makePipeline(dir, generousThresholds).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    expect(result.diffPlan.fullDiffCount).toBe(1);
    expect(result.diffPlan.degradedCount).toBe(0);
    expect(result.diffTexts.has("tiny.ts")).toBe(true);
  });

  test("budget exceeded by total tokens: source files prioritised over lockfiles in greedy fill", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await mkdir(join(dir, "src"), { recursive: true });
    // Source file (small, higher priority in greedy sort)
    await writeFile(join(dir, "src/app.ts"), "const a = 1;\n");
    // Lockfile (also small, lower priority than source)
    await writeFile(
      join(dir, "package-lock.json"),
      Array.from({ length: 40 }, (_, i) => `  "dep${i}": "1.0.${i}"`).join(
        ",\n",
      ) + "\n",
    );
    await git(dir, "add", ".");

    // Budget tight enough that only one file fits after overhead
    // 2 files × 50 overhead = 100 reserved; availableDiff = 50
    // source file: ~1 line → 10 tokens (fits)
    // lockfile: ~40 lines → 400 tokens (doesn't fit)
    const tightThresholds: BudgetThresholds = {
      maxTotalTokens: 150,
      maxLinesPerFile: 500,
      tokensPerLine: 10,
      tokensPerFileOverhead: 50,
    };

    const result = await makePipeline(dir, tightThresholds).execute();

    expect(result.route).toBe("full");
    if (result.route !== "full") return;

    // Source file fits; lockfile is budget-exceeded
    expect(result.diffPlan.fullDiffCount).toBe(1);
    expect(result.diffPlan.degradedCount).toBe(1);

    expect(result.diffTexts.has("src/app.ts")).toBe(true);
    expect(result.diffTexts.has("package-lock.json")).toBe(false);

    const lockPlan = result.diffPlan.plans.find(
      (p) => p.file.file.path === "package-lock.json",
    );
    expect(lockPlan?.degradationReason).toBe("budget-exceeded");
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// Interrupted route — all git internal operations
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline E2E — interrupted route: merge", () => {
  test("merge --no-commit: route=interrupted, gitState.status=merge", async () => {
    const { dir } = await initRepoWithCommit("base.txt", "base\n");
    testDir = dir;

    await git(dir, "checkout", "-b", "feature");
    await writeFile(join(dir, "feature.txt"), "from feature\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "feat: add feature file");

    await git(dir, "checkout", "main");
    // --no-commit leaves MERGE_HEAD intact even without a conflict
    await gitMayFail(dir, "merge", "--no-commit", "--no-ff", "feature");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("interrupted");
    if (result.route !== "interrupted") return;

    expect(result.gitState.status).toBe("merge");
    expect(result.completedSteps).toEqual(["repo-precheck", "state-detect"]);
    expect(result.commitMessage).toMatch(/merge/i);
  });

  test("merge: gitState.mergeHead matches the feature branch tip hash", async () => {
    const { dir } = await initRepoWithCommit("base.txt", "base\n");
    testDir = dir;

    await git(dir, "checkout", "-b", "feature");
    await writeFile(join(dir, "f.txt"), "f\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "feat: feature");
    const featureTip = await git(dir, "rev-parse", "HEAD");

    await git(dir, "checkout", "main");
    await gitMayFail(dir, "merge", "--no-commit", "--no-ff", "feature");

    const result = await makePipeline(dir).execute();

    if (result.route !== "interrupted") return;
    if (result.gitState.status !== "merge") return;

    expect(result.gitState.mergeHead).toBe(featureTip);
  });

  test("merge conflict: route=interrupted after resolving conflict and staging", async () => {
    const { dir } = await initRepoWithCommit("file.txt", "original\n");
    testDir = dir;

    await git(dir, "checkout", "-b", "feature");
    await writeFile(join(dir, "file.txt"), "from-feature\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "feat: branch change");

    await git(dir, "checkout", "main");
    await writeFile(join(dir, "file.txt"), "from-main\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "chore: main change");

    // Conflict: both branches changed the same line
    await gitMayFail(dir, "merge", "--no-edit", "feature");

    // Resolve and stage
    await writeFile(join(dir, "file.txt"), "resolved\n");
    await git(dir, "add", "file.txt");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("interrupted");
    if (result.route === "interrupted") {
      expect(result.gitState.status).toBe("merge");
    }
  });
});

describe("GitContextPipeline E2E — interrupted route: squash-merge", () => {
  test("squash-merge: route=interrupted, gitState.status=squash-merge, commitMessage from SQUASH_MSG", async () => {
    const { dir } = await initRepoWithCommit("base.txt", "base\n");
    testDir = dir;

    await git(dir, "checkout", "-b", "feature");
    await writeFile(join(dir, "squashed.ts"), "const x = 1;\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "feat: add squashed file");

    await git(dir, "checkout", "main");
    // git merge --squash stages files and writes SQUASH_MSG without creating MERGE_HEAD
    await git(dir, "merge", "--squash", "feature");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("interrupted");
    if (result.route !== "interrupted") return;

    expect(result.gitState.status).toBe("squash-merge");
    if (result.gitState.status === "squash-merge") {
      expect(result.gitState.squashMessage).toBeTruthy();
      expect(result.commitMessage).toBe(result.gitState.squashMessage);
    }
    expect(result.completedSteps).toEqual(["repo-precheck", "state-detect"]);
  });
});

describe("GitContextPipeline E2E — interrupted route: cherry-pick", () => {
  test("cherry-pick conflict: route=interrupted, gitState.status=cherry-pick", async () => {
    const { dir } = await initRepoWithCommit("file.txt", "original\n");
    testDir = dir;

    // Feature: change "original" → "from-feature"
    await git(dir, "checkout", "-b", "feature");
    await writeFile(join(dir, "file.txt"), "from-feature\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "feat: change from feature");
    const featureHash = await git(dir, "rev-parse", "HEAD");

    // Main diverges: change "original" → "from-main"
    await git(dir, "checkout", "main");
    await writeFile(join(dir, "file.txt"), "from-main\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "chore: change from main");

    // Cherry-pick the feature commit → conflict (both changed same line)
    await gitMayFail(dir, "cherry-pick", featureHash);
    // CHERRY_PICK_HEAD now exists; resolve and stage
    await writeFile(join(dir, "file.txt"), "resolved\n");
    await git(dir, "add", "file.txt");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("interrupted");
    if (result.route !== "interrupted") return;

    expect(result.gitState.status).toBe("cherry-pick");
    if (result.gitState.status === "cherry-pick") {
      expect(result.gitState.cherryPickHead).toBe(featureHash);
    }
    expect(result.completedSteps).toEqual(["repo-precheck", "state-detect"]);
  });

  test("cherry-pick: commitMessage includes original commit title", async () => {
    const { dir } = await initRepoWithCommit("file.txt", "original\n");
    testDir = dir;

    await git(dir, "checkout", "-b", "feature");
    await writeFile(join(dir, "file.txt"), "from-feature\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "fix: resolve the critical bug");
    const featureHash = await git(dir, "rev-parse", "HEAD");

    await git(dir, "checkout", "main");
    await writeFile(join(dir, "file.txt"), "from-main\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "chore: diverge main");

    await gitMayFail(dir, "cherry-pick", featureHash);
    await writeFile(join(dir, "file.txt"), "resolved\n");
    await git(dir, "add", "file.txt");

    const result = await makePipeline(dir).execute();

    if (result.route !== "interrupted") return;
    expect(result.commitMessage).toBe(
      "cherry-pick: fix: resolve the critical bug",
    );
  });
});

describe("GitContextPipeline E2E — interrupted route: revert", () => {
  test("revert conflict: route=interrupted, gitState.status=revert", async () => {
    const { dir } = await initRepoWithCommit("file.txt", "original\n");
    testDir = dir;

    // Commit B: "original" → "by-B"
    await writeFile(join(dir, "file.txt"), "by-B\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "chore: commit B");
    const commitBHash = await git(dir, "rev-parse", "HEAD");

    // Commit C: "by-B" → "by-C" (same line — reverting B will now conflict)
    await writeFile(join(dir, "file.txt"), "by-C\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "chore: commit C");

    // Revert B: base="by-B", ours="by-C", theirs="original" → conflict
    await gitMayFail(dir, "revert", "--no-edit", commitBHash);
    // REVERT_HEAD now exists; resolve and stage
    await writeFile(join(dir, "file.txt"), "resolved\n");
    await git(dir, "add", "file.txt");

    const result = await makePipeline(dir).execute();

    expect(result.route).toBe("interrupted");
    if (result.route !== "interrupted") return;

    expect(result.gitState.status).toBe("revert");
    if (result.gitState.status === "revert") {
      expect(result.gitState.revertHead).toBe(commitBHash);
      expect(result.commitMessage).toMatch(/revert/i);
    }
    expect(result.completedSteps).toEqual(["repo-precheck", "state-detect"]);
  });

  test("revert: commitMessage includes original commit title when available", async () => {
    const { dir } = await initRepoWithCommit("file.txt", "original\n");
    testDir = dir;

    await writeFile(join(dir, "file.txt"), "by-B\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "fix: the bad change");
    const commitBHash = await git(dir, "rev-parse", "HEAD");

    await writeFile(join(dir, "file.txt"), "by-C\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "chore: further change");

    await gitMayFail(dir, "revert", "--no-edit", commitBHash);
    await writeFile(join(dir, "file.txt"), "resolved\n");
    await git(dir, "add", "file.txt");

    const result = await makePipeline(dir).execute();

    if (result.route !== "interrupted") return;
    expect(result.commitMessage).toBe("revert: fix: the bad change");
  });
});

// ═════════════════════════════════════════════════════════════════════════════
// Error cases
// ═════════════════════════════════════════════════════════════════════════════

describe("GitContextPipeline E2E — error cases", () => {
  test("not a git repository: throws GitError(NOT_A_REPO)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ac-e2e-noRepo-"));
    testDir = dir;

    let caught: unknown;
    try {
      await makePipeline(dir).execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.NOT_A_REPO);
  });

  test("nothing to commit (clean working tree): throws GitError(NOTHING_TO_COMMIT)", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;
    // No changes — working tree is clean

    let caught: unknown;
    try {
      await makePipeline(dir).execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.NOTHING_TO_COMMIT);
  });

  test("changes only in working tree (not staged): throws GitError(STAGING_EMPTY)", async () => {
    const { dir } = await initRepoWithCommit("src/app.ts", "const a = 1;\n");
    testDir = dir;

    // Modify but deliberately do NOT stage
    await writeFile(join(dir, "src/app.ts"), "const a = 2;\n");

    let caught: unknown;
    try {
      await makePipeline(dir).execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.STAGING_EMPTY);
  });

  test("another git process holding index.lock: throws GitError(LOCK_FILE_EXISTS)", async () => {
    const { dir } = await initRepoWithCommit();
    testDir = dir;

    await writeFile(join(dir, "staged.ts"), "const s = 1;\n");
    await git(dir, "add", ".");

    // Simulate a concurrent git process by writing index.lock directly
    const lockPath = join(dir, ".git", "index.lock");
    await writeFile(lockPath, "");

    let caught: unknown;
    try {
      await makePipeline(dir).execute();
    } catch (e) {
      caught = e;
    }

    // Clean up lock before afterEach rm
    await rm(lockPath, { force: true });

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.LOCK_FILE_EXISTS);
  });

  test("bisect in progress: throws GitError(BISECT_IN_PROGRESS) before diff steps", async () => {
    const { dir } = await initRepoWithCommit("a.ts", "const a = 1;\n");
    testDir = dir;

    await writeFile(join(dir, "b.ts"), "const b = 2;\n");
    await git(dir, "add", ".");
    await git(dir, "commit", "-m", "chore: second commit");

    // Stage something so staging-check passes (bisect-detect runs after staging-check)
    await writeFile(join(dir, "staged.ts"), "const s = 1;\n");
    await git(dir, "add", "staged.ts");

    // Start a bisect session — creates .git/BISECT_LOG after marking bad
    await git(dir, "bisect", "start");
    await gitMayFail(dir, "bisect", "bad", "HEAD");

    let caught: unknown;
    try {
      await makePipeline(dir).execute();
    } catch (e) {
      caught = e;
    }

    // Always reset bisect so afterEach rm can succeed
    await gitMayFail(dir, "bisect", "reset");

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.BISECT_IN_PROGRESS);
  });

  test("bare repository: throws GitError(BARE_REPO_UNSUPPORTED)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ac-e2e-bare-"));
    testDir = dir;

    await git(dir, "init", "--bare");

    let caught: unknown;
    try {
      await makePipeline(dir).execute();
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
    expect((caught as GitError).code).toBe(GitCode.BARE_REPO_UNSUPPORTED);
  });
});
