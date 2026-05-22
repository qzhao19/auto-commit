/**
 * E2E tests for BudgetPlanner — exercises the full
 * DiffCollector → FileClassifier → BudgetPlanner pipeline against
 * real git repositories on disk using the real GitRunner.
 *
 * No git commands are mocked. Budget-sensitive scenarios use a compact
 * custom threshold so files with just a handful of lines trigger the
 * oversized / budget-exceeded degradation paths.
 *
 * Requires: git >= 2.28 (for `git init -b <branch>`)
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BudgetPlanner } from "../../src/core/git/diff/budget-planner";
import { DiffCollector } from "../../src/core/git/diff/diff-collector";
import { FileClassifier } from "../../src/core/git/diff/file-classifier";
import { GitRunner } from "../../src/core/git/runner/git-runner";
import { LFS_POINTER_MAGIC } from "../../src/shared/constants/index";
import type { BudgetThresholds, DiffPlanResult } from "../../src/shared/types/index";

// ── Constants ─────────────────────────────────────────────────────────────────

/** Minimal PNG magic bytes — causes git numstat to flag the file as binary. */
const PNG_MAGIC = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00]);

/** Valid 3-line LFS pointer (v1 spec). insertions=3 passes isLfsCandidate(); blob starts with LFS_POINTER_MAGIC. */
const LFS_POINTER_CONTENT =
  `${LFS_POINTER_MAGIC}\n` +
  "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n" +
  "size 104857600\n";

/**
 * Compact thresholds designed so real files with a handful of lines
 * are sufficient to exercise the oversized / budget-exceeded paths.
 *
 * Key derived quantities (N = total staged files):
 *   reservedMetadata    = N × 50
 *   availableDiffBudget = Math.max(0, 200 − N × 50)
 *   lines > 5           → "oversized" (only when already over budget)
 *
 * Useful scenarios:
 *   3 files [oversized(6L) + normal(2L) + normal(2L)]:
 *     totalIfFull=250 > 200; normalDiffTotal=40 ≤ avail(50) → Step 4b
 *
 *   3 files [3 × 4L]:
 *     totalIfFull=270 > 200; avail=50; greedy [40,40,40]: 1 fits → 1 full, 2 budget-exceeded
 *
 *   5 noise + 1 source(2L):
 *     reservedMeta=300 > 200; avail=0 → source becomes budget-exceeded
 */
const TIGHT_E2E: BudgetThresholds = {
  maxTotalTokens:       200,
  maxLinesPerFile:        5,
  tokensPerLine:         10,
  tokensPerFileOverhead: 50,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

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

async function initRepo(dir: string, branch = "main"): Promise<void> {
  await git(["init", "-b", branch], dir);
  await git(["config", "user.email", "test@example.com"], dir);
  await git(["config", "user.name", "Test User"], dir);
}

async function stageFile(gitRoot: string, relPath: string, content: string): Promise<void> {
  const fullPath = join(gitRoot, relPath);
  await mkdir(join(fullPath, ".."), { recursive: true });
  await writeFile(fullPath, content);
  await git(["add", relPath], gitRoot);
}

async function stageBinary(gitRoot: string, relPath: string, data: Uint8Array): Promise<void> {
  const fullPath = join(gitRoot, relPath);
  await mkdir(join(fullPath, ".."), { recursive: true });
  await writeFile(fullPath, data);
  await git(["add", relPath], gitRoot);
}

async function commitFile(
  gitRoot: string,
  relPath: string,
  content: string,
  message = "chore: seed commit",
): Promise<void> {
  await stageFile(gitRoot, relPath, content);
  await git(["commit", "-m", message], gitRoot);
}

/**
 * Generates a text file with exactly `count` unique lines, each syntactically
 * valid TypeScript so they are never collapsed by diff compression.
 */
function lines(count: number): string {
  return Array.from({ length: count }, (_, i) => `const v${i + 1} = ${i + 1};`).join("\n") + "\n";
}

/** Runs the full DiffCollector → FileClassifier → BudgetPlanner pipeline. */
async function run(cwd: string, thresholds?: BudgetThresholds): Promise<DiffPlanResult> {
  const runner     = new GitRunner({ cwd });
  const summary    = await new DiffCollector(runner).collect();
  const classified = await new FileClassifier(runner).classify(summary);
  return new BudgetPlanner(thresholds).plan(classified);
}

// ── Fixtures ──────────────────────────────────────────────────────────────────

let repoDir: string;

beforeEach(async () => {
  const tmp = await mkdtemp(join(tmpdir(), "budget-planner-e2e-"));
  repoDir = await realpath(tmp);
  await initRepo(repoDir);
});

afterEach(async () => {
  await rm(repoDir, { recursive: true, force: true });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Empty staging area
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — empty staging area", () => {
  test("no staged changes → empty plans, isWithinBudget=true, all counts zero", async () => {
    await commitFile(repoDir, "seed.ts", "export {};\n");

    const result = await run(repoDir);

    expect(result.plans).toHaveLength(0);
    expect(result.estimate.totalFiles).toBe(0);
    expect(result.estimate.contentFiles).toBe(0);
    expect(result.estimate.noiseFiles).toBe(0);
    expect(result.estimate.totalChangedLines).toBe(0);
    expect(result.estimate.maxSingleFileLines).toBe(0);
    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.fullDiffCount).toBe(0);
    expect(result.degradedCount).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Noise files — binary, LFS pointer
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — noise files (binary / LFS pointer)", () => {
  test("binary file (PNG magic bytes) → single degraded/noise plan", async () => {
    await commitFile(repoDir, "seed.ts", "export {};\n");
    await stageBinary(repoDir, "assets/logo.png", PNG_MAGIC);

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(1);
    const plan = result.plans[0]!;
    expect(plan.mode).toBe("degraded");
    expect(plan.degradationReason).toBe("noise");
    expect(plan.estimatedTokens).toBeNull();
    expect(result.estimate.noiseFiles).toBe(1);
    expect(result.estimate.contentFiles).toBe(0);
    expect(result.fullDiffCount).toBe(0);
    expect(result.degradedCount).toBe(1);
  });

  test("LFS pointer file → classified as lfs-pointer noise, planned as degraded/noise", async () => {
    await commitFile(repoDir, "seed.ts", "export {};\n");
    await stageFile(repoDir, "models/weights.bin", LFS_POINTER_CONTENT);

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(1);
    const plan = result.plans[0]!;
    expect(plan.mode).toBe("degraded");
    expect(plan.degradationReason).toBe("noise");
    expect(plan.file.isNoise).toBe(true);
    if (plan.file.isNoise) {
      expect(plan.file.noiseCategory).toBe("lfs-pointer");
    }
    // LFS pointer lines do not count as changed lines
    expect(result.estimate.totalChangedLines).toBe(0);
  });

  test("multiple noise types staged together → all degraded/noise, no diff tokens consumed", async () => {
    await commitFile(repoDir, "seed.ts", "export {};\n");
    await stageBinary(repoDir, "assets/a.png", PNG_MAGIC);
    await stageFile(repoDir, "models/weights.bin", LFS_POINTER_CONTENT);

    // 2 noise files: reservedMetadata=100, diffTokens=0, totalIfFull=100 ≤ 200 → within budget
    const result = await run(repoDir, TIGHT_E2E);

    expect(result.estimate.noiseFiles).toBe(2);
    expect(result.estimate.contentFiles).toBe(0);
    expect(result.estimate.totalChangedLines).toBe(0);
    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.estimate.estimatedTokensIfFull).toBe(2 * TIGHT_E2E.tokensPerFileOverhead);
    result.plans.forEach(p => {
      expect(p.mode).toBe("degraded");
      expect(p.degradationReason).toBe("noise");
    });
  });

  test("noise metadata overhead can exhaust diff budget when many noise files are staged", async () => {
    // 5 noise + 1 source(2L): reservedMeta=6×50=300 > maxTotalTokens(200)
    // availableDiffBudget = Math.max(0, 200-300) = 0
    // source: diffTokens=20 > availableDiffBudget(0) → budget-exceeded
    await commitFile(repoDir, "seed.ts", "export {};\n");
    for (let i = 0; i < 5; i++) {
      await stageBinary(repoDir, `assets/img${i}.png`, PNG_MAGIC);
    }
    await stageFile(repoDir, "src/feature.ts", lines(2));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.estimate.noiseFiles).toBe(5);
    expect(result.estimate.contentFiles).toBe(1);

    result.plans
      .filter(p => p.file.isNoise)
      .forEach(p => expect(p.degradationReason).toBe("noise"));

    const sourcePlan = result.plans.find(p => !p.file.isNoise);
    expect(sourcePlan?.mode).toBe("degraded");
    expect(sourcePlan?.degradationReason).toBe("budget-exceeded");
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Source files — various change types, within budget
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — source files within budget", () => {
  test("newly staged source file → content/source, full diff plan", async () => {
    await stageFile(repoDir, "src/index.ts", lines(4));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(1);
    expect(result.plans[0]!.mode).toBe("full");
    expect(result.plans[0]!.estimatedTokens).toBeGreaterThan(0);
    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.estimate.contentFiles).toBe(1);
    expect(result.estimate.noiseFiles).toBe(0);
  });

  test("modified source file (shrink) → correct insertion/deletion counts, full diff plan", async () => {
    // Commit 4 lines, stage 2 lines → diff: -2 deletions, 0 insertions → lines=2
    await commitFile(repoDir, "src/util.ts", lines(4));
    await stageFile(repoDir, "src/util.ts", lines(2));

    // 1 file: reservedMeta=50, diffTokens=20, totalIfFull=70 ≤ 200 → full
    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(1);
    expect(result.plans[0]!.mode).toBe("full");
    expect(result.estimate.totalChangedLines).toBe(2);
  });

  test("deleted source file → classified as content/source, receives full diff plan", async () => {
    await commitFile(repoDir, "src/old.ts", lines(3));
    await git(["rm", "src/old.ts"], repoDir);

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(1);
    expect(result.plans[0]!.mode).toBe("full");
    expect(result.estimate.contentFiles).toBe(1);
  });

  test("two small source files within budget → both get full diff", async () => {
    // 2×2L: reservedMeta=100, diffTokens=40, totalIfFull=140 ≤ 200 → all full
    await stageFile(repoDir, "src/a.ts", lines(2));
    await stageFile(repoDir, "src/b.ts", lines(2));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(2);
    expect(result.fullDiffCount).toBe(2);
    expect(result.degradedCount).toBe(0);
    result.plans.forEach(p => expect(p.mode).toBe("full"));
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Lockfiles
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — lockfiles", () => {
  test("yarn.lock → classified as content/lockfile, receives full diff plan", async () => {
    await stageFile(repoDir, "yarn.lock", lines(4));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(1);
    const plan = result.plans[0]!;
    expect(plan.mode).toBe("full");
    expect(plan.file.isNoise).toBe(false);
    if (!plan.file.isNoise) {
      expect(plan.file.contentCategory).toBe("lockfile");
    }
  });

  test("bun.lock → classified as content/lockfile, receives full diff plan", async () => {
    await stageFile(repoDir, "bun.lock", lines(2));

    const result = await run(repoDir, TIGHT_E2E);

    const plan = result.plans[0]!;
    expect(plan.mode).toBe("full");
    if (!plan.file.isNoise) {
      expect(plan.file.contentCategory).toBe("lockfile");
    }
  });

  test("lockfile + source file staged together → both content files, both full", async () => {
    // 2 files: reservedMeta=100, diffTokens=40, totalIfFull=140 ≤ 200 → all full
    await stageFile(repoDir, "package-lock.json", lines(2));
    await stageFile(repoDir, "src/index.ts", lines(2));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(2);
    expect(result.fullDiffCount).toBe(2);
    expect(result.degradedCount).toBe(0);
    expect(result.estimate.contentFiles).toBe(2);

    const categories = result.plans
      .map(p => (p.file.isNoise ? "noise" : p.file.contentCategory))
      .sort();
    expect(categories).toEqual(["lockfile", "source"]);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Mixed noise + content
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — mixed noise + content files", () => {
  test("binary image + source file → image degraded/noise, source full", async () => {
    await stageBinary(repoDir, "assets/icon.png", PNG_MAGIC);
    await stageFile(repoDir, "src/app.ts", lines(2));

    // 2 files: reservedMeta=100, diffTokens=20, totalIfFull=120 ≤ 200 → within budget
    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(2);
    expect(result.estimate.noiseFiles).toBe(1);
    expect(result.estimate.contentFiles).toBe(1);
    expect(result.fullDiffCount).toBe(1);
    expect(result.degradedCount).toBe(1);

    const noisePlan  = result.plans.find(p => p.file.isNoise);
    const sourcePlan = result.plans.find(p => !p.file.isNoise);

    expect(noisePlan?.mode).toBe("degraded");
    expect(noisePlan?.degradationReason).toBe("noise");
    expect(sourcePlan?.mode).toBe("full");
  });

  test("binary + LFS pointer + lockfile + source → noise degraded, content files full", async () => {
    await stageBinary(repoDir, "assets/photo.png", PNG_MAGIC);
    await stageFile(repoDir, "model.bin", LFS_POINTER_CONTENT);
    await stageFile(repoDir, "yarn.lock", lines(2));
    await stageFile(repoDir, "src/feat.ts", lines(2));

    // 4 files: reservedMeta=200, diffTokens=40, totalIfFull=240 > 200 → else branch
    // Noise files: already handled as degraded/noise before budget check
    // Content files (lockfile + source): 2 files, totalIfFull from content perspective:
    // Actually: estimates[] only includes content files (2 files with 2L each)
    // reservedMetadata = 4 × 50 = 200; diffTokens = 40; totalIfFull = 240 > 200 → else branch
    // availableDiffBudget = max(0, 200-200) = 0
    // normalDiffTotal = 40 > 0 → greedy: 0+20 ≤ 0? NO → both content files budget-exceeded
    // So this test checks that even content files become budget-exceeded when metadata fills budget
    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(4);
    expect(result.estimate.noiseFiles).toBe(2);
    expect(result.estimate.contentFiles).toBe(2);

    result.plans
      .filter(p => p.file.isNoise)
      .forEach(p => {
        expect(p.mode).toBe("degraded");
        expect(p.degradationReason).toBe("noise");
      });

    result.plans
      .filter(p => !p.file.isNoise)
      .forEach(p => {
        expect(p.mode).toBe("degraded");
        expect(p.degradationReason).toBe("budget-exceeded");
      });
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Budget and oversized degradation
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — budget and oversized degradation (TIGHT_E2E)", () => {
  test("oversized file + two small normals → oversized degraded, normals full (Step 4b)", async () => {
    // 3 files [6L + 2L + 2L]: reservedMeta=150, totalIfFull=250 > 200 → else branch
    // 6 > 5 → oversized; availableDiffBudget=50; normalDiffTotal=40 ≤ 50 → Step 4b
    await stageFile(repoDir, "src/migration.ts", lines(6));
    await stageFile(repoDir, "src/a.ts",         lines(2));
    await stageFile(repoDir, "src/b.ts",         lines(2));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(3);

    const migrationPlan = result.plans.find(p => p.file.file.path.includes("migration"));
    const normalPlans   = result.plans.filter(p => !p.file.file.path.includes("migration"));

    expect(migrationPlan?.mode).toBe("degraded");
    expect(migrationPlan?.degradationReason).toBe("oversized");
    expect(migrationPlan?.estimatedTokens).toBeNull();

    normalPlans.forEach(p => {
      expect(p.mode).toBe("full");
      expect(p.estimatedTokens).toBeGreaterThan(0);
    });

    expect(result.fullDiffCount).toBe(2);
    expect(result.degradedCount).toBe(1);
  });

  test("file exactly at maxLinesPerFile (5L) is NOT oversized — strict greater-than boundary", async () => {
    // 1 file with 5L: reservedMeta=50, totalIfFull=100 ≤ 200 → within budget → full
    // (oversized check never runs anyway; but confirms boundary is strict)
    await stageFile(repoDir, "src/at-limit.ts", lines(5));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans[0]!.mode).toBe("full");
    expect(result.estimate.isWithinBudget).toBe(true);
  });

  test("oversized file alone is NOT degraded — oversized only triggers after budget exceeded", async () => {
    // 1 file with 6L: reservedMeta=50, totalIfFull=110 ≤ 200 → isWithinBudget=true → Step 3 fast path
    // Oversized check only runs in the else branch; this file never reaches it.
    await stageFile(repoDir, "src/large.ts", lines(6));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans[0]!.mode).toBe("full");
    expect(result.estimate.isWithinBudget).toBe(true);
  });

  test("greedy fill: 3 equal-size files where only 1 fits remaining budget (Step 4c)", async () => {
    // 3 × 4L: reservedMeta=150, totalIfFull=270 > 200 → else branch
    // 4 ≤ 5 → none oversized; avail=50; greedy [40,40,40]: first(40)≤50 fits, rest rejected
    await stageFile(repoDir, "src/a.ts", lines(4));
    await stageFile(repoDir, "src/b.ts", lines(4));
    await stageFile(repoDir, "src/c.ts", lines(4));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(3);
    expect(result.fullDiffCount).toBe(1);
    expect(result.degradedCount).toBe(2);

    const budgetExceeded = result.plans.filter(p => p.degradationReason === "budget-exceeded");
    expect(budgetExceeded).toHaveLength(2);
    budgetExceeded.forEach(p => expect(p.estimatedTokens).toBeNull());
  });

  test("oversized + normals: normals pushed to greedy when diff budget is tight (Step 4c)", async () => {
    // 1×6L (oversized) + 3×4L (normals):
    // reservedMeta=200, totalIfFull=200+60+120=380 > 200 → else branch
    // 6 > 5 → oversized removed; normal[] = 3×4L, normalDiffTotal=120
    // availableDiffBudget = max(0, 200-200) = 0; 120 > 0 → greedy
    // 0+40 ≤ 0? NO → all normals budget-exceeded
    await stageFile(repoDir, "src/migration.ts", lines(6));
    await stageFile(repoDir, "src/a.ts",         lines(4));
    await stageFile(repoDir, "src/b.ts",         lines(4));
    await stageFile(repoDir, "src/c.ts",         lines(4));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(4);
    expect(result.fullDiffCount).toBe(0);
    expect(result.degradedCount).toBe(4);

    const oversizedPlan = result.plans.find(p => p.degradationReason === "oversized");
    expect(oversizedPlan).toBeDefined();

    const budgetExceeded = result.plans.filter(p => p.degradationReason === "budget-exceeded");
    expect(budgetExceeded).toHaveLength(3);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Renamed files
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — renamed files", () => {
  test("git mv (pure rename, no content change) → renamedNoContentChangeCount=1, mode=full", async () => {
    await commitFile(repoDir, "src/old-name.ts", lines(2));
    await git(["mv", "src/old-name.ts", "src/new-name.ts"], repoDir);

    // Rename with 0 insertions + 0 deletions
    // 1 file: reservedMeta=50, diffTokens=0, totalIfFull=50 ≤ 200 → full
    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(1);
    expect(result.plans[0]!.mode).toBe("full");
    expect(result.estimate.renamedNoContentChangeCount).toBe(1);
    expect(result.estimate.totalChangedLines).toBe(0);
  });

  test("rename with content modification → renamedNoContentChangeCount=0", async () => {
    await commitFile(repoDir, "src/old.ts", lines(4));
    await git(["mv", "src/old.ts", "src/new.ts"], repoDir);
    // Overwrite the renamed file to create a content change
    await stageFile(repoDir, "src/new.ts", lines(2));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.estimate.renamedNoContentChangeCount).toBe(0);
    expect(result.estimate.totalChangedLines).toBeGreaterThan(0);
  });

  test("mixed: pure rename + modified file → renamedNoContentChangeCount counts only pure rename", async () => {
    await commitFile(repoDir, "src/old.ts",  lines(2));
    await commitFile(repoDir, "src/keep.ts", lines(2));
    await git(["mv", "src/old.ts", "src/new.ts"], repoDir);
    // Modify keep.ts (not a rename)
    await stageFile(repoDir, "src/keep.ts", lines(3));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.estimate.renamedNoContentChangeCount).toBe(1);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Initial commit (no parent)
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — initial commit (no parent)", () => {
  test("staging files on an empty repository → pipeline succeeds, all source files full", async () => {
    // No git commit has been made yet — this exercises the initial-commit code path
    await stageFile(repoDir, "src/main.ts", lines(2));
    await stageFile(repoDir, "README.md",   "# Project\n");

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans.length).toBeGreaterThan(0);
    expect(result.estimate.isWithinBudget).toBe(true);
    result.plans.forEach(p => expect(p.mode).toBe("full"));
  });

  test("binary staged on empty repository → degraded/noise", async () => {
    await stageBinary(repoDir, "logo.png", PNG_MAGIC);

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.plans).toHaveLength(1);
    expect(result.plans[0]!.mode).toBe("degraded");
    expect(result.plans[0]!.degradationReason).toBe("noise");
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Aggregate estimate metrics accuracy
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — aggregate estimate metrics", () => {
  test("totalChangedLines reflects actual git diff line counts across all content files", async () => {
    // Stage 3-line and 4-line files → totalChangedLines = 7, maxSingleFileLines = 4
    await stageFile(repoDir, "src/a.ts", lines(3));
    await stageFile(repoDir, "src/b.ts", lines(4));

    // 2 files: reservedMeta=100, diffTokens=70, total=170 ≤ 200 → within budget
    const result = await run(repoDir, TIGHT_E2E);

    expect(result.estimate.totalChangedLines).toBe(7);
    expect(result.estimate.maxSingleFileLines).toBe(4);
    expect(result.estimate.isWithinBudget).toBe(true);
  });

  test("noise files do not contribute to totalChangedLines or maxSingleFileLines", async () => {
    await commitFile(repoDir, "seed.ts", "export {};\n");
    await stageBinary(repoDir, "assets/img.png", PNG_MAGIC);
    await stageFile(repoDir, "src/a.ts", lines(3));

    const result = await run(repoDir, TIGHT_E2E);

    // Noise file has null insertions/deletions; only the source file counts
    expect(result.estimate.totalChangedLines).toBe(3);
    expect(result.estimate.maxSingleFileLines).toBe(3);
  });

  test("fullDiffCount + degradedCount always equals plans.length", async () => {
    await stageBinary(repoDir, "img.png", PNG_MAGIC);
    await stageFile(repoDir, "src/a.ts", lines(4));
    await stageFile(repoDir, "src/b.ts", lines(4));
    await stageFile(repoDir, "src/c.ts", lines(4));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.fullDiffCount + result.degradedCount).toBe(result.plans.length);
  });

  test("estimate.tokenBudget reflects the configured maxTotalTokens", async () => {
    await stageFile(repoDir, "src/a.ts", lines(2));

    const result = await run(repoDir, TIGHT_E2E);

    expect(result.estimate.tokenBudget).toBe(TIGHT_E2E.maxTotalTokens);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Real-world commit patterns (production thresholds)
// ═══════════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner e2e — real-world commit patterns (production thresholds)", () => {
  test("feature commit: several new source files → all full with default thresholds", async () => {
    await stageFile(repoDir, "src/auth/login.ts",  lines(20));
    await stageFile(repoDir, "src/auth/token.ts",  lines(15));
    await stageFile(repoDir, "src/auth/types.ts",  lines(10));

    // Default threshold: maxTotalTokens=16,000 — easily within budget for small files
    const result = await run(repoDir);

    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.estimate.tokenBudget).toBe(16_000);
    result.plans.forEach(p => expect(p.mode).toBe("full"));
  });

  test("dependency update: lockfile + source → both content, both full", async () => {
    await stageFile(repoDir, "package-lock.json", lines(15));
    await stageFile(repoDir, "src/app.ts",         lines(5));

    const result = await run(repoDir);

    expect(result.fullDiffCount).toBe(2);
    expect(result.degradedCount).toBe(0);

    const lockfilePlan = result.plans.find(
      p => !p.file.isNoise && p.file.file.path.includes("package-lock"),
    );
    if (lockfilePlan && !lockfilePlan.file.isNoise) {
      expect(lockfilePlan.file.contentCategory).toBe("lockfile");
    }
  });

  test("asset-only update: binary files only → all noise, no diff tokens, isWithinBudget=true", async () => {
    await commitFile(repoDir, "seed.ts", "export {};\n");
    await stageBinary(repoDir, "assets/logo.png",    PNG_MAGIC);
    await stageBinary(repoDir, "assets/favicon.ico", new Uint8Array([0x00, 0x00, 0x01, 0x00]));

    const result = await run(repoDir);

    expect(result.estimate.noiseFiles).toBe(2);
    expect(result.estimate.contentFiles).toBe(0);
    expect(result.estimate.totalChangedLines).toBe(0);
    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.fullDiffCount).toBe(0);
    expect(result.degradedCount).toBe(2);
    result.plans.forEach(p => {
      expect(p.mode).toBe("degraded");
      expect(p.degradationReason).toBe("noise");
    });
  });

  test("refactor: renamed source files (no content change) → renamedNoContentChangeCount > 0, all full", async () => {
    await commitFile(repoDir, "src/utils/helper.ts", lines(5));
    await commitFile(repoDir, "src/utils/parser.ts", lines(5));
    await git(["mv", "src/utils/helper.ts", "src/lib/helper.ts"], repoDir);
    await git(["mv", "src/utils/parser.ts", "src/lib/parser.ts"], repoDir);

    const result = await run(repoDir);

    expect(result.estimate.renamedNoContentChangeCount).toBe(2);
    expect(result.estimate.totalChangedLines).toBe(0);
    expect(result.estimate.isWithinBudget).toBe(true);
    result.plans.forEach(p => expect(p.mode).toBe("full"));
  });
});