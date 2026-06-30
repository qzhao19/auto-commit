// test/unittests/budget-planner.diff.test.ts
import { describe, expect, test } from "bun:test";
import type {
  BudgetThresholds,
  ClassifiedFile,
  NonNoiseFile,
  FileClassificationResult,
  NoiseFile,
  StagedFileChange,
} from "../../src/shared/types/index";
import { BudgetPlanner } from "../../src/core/git/diff/budget-planner";

// ── Test thresholds (small, deterministic numbers) ────────────────────────────
//
// maxTotalTokens:       1000
// maxLinesPerFile:        50  (lines > 50 → "oversized")
// tokensPerLine:           10  (lines × 10 = diffTokens)
// tokensPerFileOverhead:   50  (fixed metadata cost, charged for EVERY file)
//
// Key derived quantities:
//   reservedMetadataTokens = files.length × 50
//   estimatedTokensIfFull  = reservedMetadataTokens + Σ(lines × 10)  [content only]
//   availableDiffBudget    = Math.max(0, 1000 − reservedMetadataTokens)
//   isWithinBudget         = estimatedTokensIfFull ≤ 1000
const TIGHT: BudgetThresholds = {
  maxTotalTokens: 1_000,
  maxLinesPerFile: 50,
  tokensPerLine: 10,
  tokensPerFileOverhead: 50,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

let _pathId = 0;
function uid(prefix = "src/file"): string {
  return `${prefix}${++_pathId}.ts`;
}

/** Minimal StagedFileChange – overrides applied last. */
function rawFile(overrides: Partial<StagedFileChange> = {}): StagedFileChange {
  return {
    path: uid(),
    oldPath: null,
    changeType: "modified",
    similarityScore: null,
    isBinary: false,
    isSubmodule: false,
    insertions: 10,
    deletions: 5,
    diff: null,
    ...overrides,
  };
}

function contentFile(overrides: Partial<StagedFileChange> = {}): NonNoiseFile {
  return {
    file: rawFile(overrides),
    isNoise: false,
    nonNoiseCategory: "source",
  };
}

function noiseFile(
  overrides: Partial<StagedFileChange> = {},
  category: NoiseFile["noiseCategory"] = "binary",
): NoiseFile {
  return {
    file: rawFile({
      isBinary: true,
      insertions: null,
      deletions: null,
      ...overrides,
    }),
    isNoise: true,
    noiseCategory: category,
  };
}

/** Build a FileClassificationResult from an ordered list of ClassifiedFile objects. */
function classify(...files: ClassifiedFile[]): FileClassificationResult {
  return {
    noiseCount: files.filter((f) => f.isNoise).length,
    nonNoiseCount: files.filter((f) => !f.isNoise).length,
    files,
  };
}

/** Build N noise-file stubs quickly. */
function noiseFiles(count: number): NoiseFile[] {
  return Array.from({ length: count }, () => noiseFile());
}

// ═══════════════════════════════════════════════════════════════════════════
// plan() — empty staging area
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — empty staging area", () => {
  test("returns zero counts, empty plans, and isWithinBudget=true", () => {
    const result = new BudgetPlanner(TIGHT).plan(classify());

    expect(result.estimate.totalFiles).toBe(0);
    expect(result.estimate.nonNoiseFiles).toBe(0);
    expect(result.estimate.noiseFiles).toBe(0);
    expect(result.estimate.totalChangedLines).toBe(0);
    expect(result.estimate.maxSingleFileLines).toBe(0);
    expect(result.estimate.estimatedTokensIfFull).toBe(0);
    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.plans).toHaveLength(0);
    expect(result.fullDiffCount).toBe(0);
    expect(result.degradedCount).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// plan() — noise files only (Step 1 unconditional degradation)
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — noise files only", () => {
  test("single binary noise file → degraded/noise, estimatedTokens null", () => {
    const noise = noiseFile({ path: "assets/logo.png" });
    const result = new BudgetPlanner(TIGHT).plan(classify(noise));

    expect(result.plans).toHaveLength(1);
    const plan = result.plans[0]!;
    expect(plan.mode).toBe("degraded");
    expect(plan.degradationReason).toBe("noise");
    expect(plan.estimatedTokens).toBeNull();

    expect(result.fullDiffCount).toBe(0);
    expect(result.degradedCount).toBe(1);
    expect(result.estimate.noiseFiles).toBe(1);
    expect(result.estimate.nonNoiseFiles).toBe(0);
  });

  test("binary / submodule / lfs-pointer noise files → all degraded/noise", () => {
    const files = [
      noiseFile({ path: "assets/image.png" }, "binary"),
      noiseFile({ path: "vendor/sub" }, "submodule"),
      noiseFile({ path: "models/weight.bin" }, "lfs-pointer"),
    ];
    const result = new BudgetPlanner(TIGHT).plan(classify(...files));

    expect(result.plans).toHaveLength(3);
    for (const plan of result.plans) {
      expect(plan.mode).toBe("degraded");
      expect(plan.degradationReason).toBe("noise");
      expect(plan.estimatedTokens).toBeNull();
    }
    expect(result.estimate.noiseFiles).toBe(3);
    expect(result.estimate.totalChangedLines).toBe(0);
    expect(result.estimate.maxSingleFileLines).toBe(0);
    expect(result.fullDiffCount).toBe(0);
    expect(result.degradedCount).toBe(3);
  });

  test("noise files do not contribute to changed-line metrics", () => {
    // Even though the raw StagedFileChange could theoretically have insertions,
    // noise files are excluded from the estimates[] array.
    const noise = noiseFile({ insertions: 100, deletions: 50 });
    const result = new BudgetPlanner(TIGHT).plan(classify(noise));

    expect(result.estimate.totalChangedLines).toBe(0);
    expect(result.estimate.maxSingleFileLines).toBe(0);
    expect(result.estimate.estimatedTokensIfFull).toBe(
      1 /* file */ * 50 /* overhead */ + 0 /* no diff tokens */,
    );
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// plan() — content files within budget (Step 3: all-full fast path)
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — content files within budget (Step 3)", () => {
  test("single file: correct diffTokens, fullTokens, and metrics", () => {
    // lines=15, diffTokens=150, fullTokens=200
    // reservedMetadataTokens=50, estimatedTokensIfFull=200 ≤ 1000
    const file = contentFile({ insertions: 10, deletions: 5 });
    const result = new BudgetPlanner(TIGHT).plan(classify(file));

    const plan = result.plans[0]!;
    expect(plan.mode).toBe("full");
    expect(plan.estimatedTokens).toBe(200); // 15×10 + 50

    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.estimate.estimatedTokensIfFull).toBe(200);
    expect(result.estimate.totalChangedLines).toBe(15);
    expect(result.estimate.maxSingleFileLines).toBe(15);
    expect(result.estimate.tokenBudget).toBe(1000);
    expect(result.fullDiffCount).toBe(1);
    expect(result.degradedCount).toBe(0);
  });

  test("two files both within budget → both get 'full'", () => {
    // each: 20 ins + 0 del = 20 lines → diffTokens=200, fullTokens=250
    // reservedMetadataTokens=100, estimatedTokensIfFull=500 ≤ 1000
    const f1 = contentFile({ insertions: 20, deletions: 0 });
    const f2 = contentFile({ insertions: 20, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(classify(f1, f2));

    expect(result.fullDiffCount).toBe(2);
    expect(result.degradedCount).toBe(0);
    for (const plan of result.plans) {
      expect(plan.mode).toBe("full");
      expect(plan.estimatedTokens).toBe(250); // 200+50
    }
    expect(result.estimate.totalChangedLines).toBe(40);
    expect(result.estimate.maxSingleFileLines).toBe(20);
  });

  test("null insertions/deletions default to 0 — no NaN, no RangeError", () => {
    // lines = (null??0)+(null??0) = 0 → diffTokens=0, fullTokens=50
    const file = contentFile({ insertions: null, deletions: null });
    const result = new BudgetPlanner(TIGHT).plan(classify(file));

    const plan = result.plans[0]!;
    expect(plan.mode).toBe("full");
    expect(plan.estimatedTokens).toBe(50); // 0+50
    expect(result.estimate.totalChangedLines).toBe(0);
    expect(result.estimate.maxSingleFileLines).toBe(0);
  });

  test("budget boundary: estimatedTokensIfFull === maxTotalTokens → isWithinBudget=true", () => {
    // reservedMetadataTokens = 50; need diffTokens = 950 → lines = 95
    const file = contentFile({ insertions: 95, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(classify(file));

    expect(result.estimate.estimatedTokensIfFull).toBe(1_000);
    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.plans[0]!.mode).toBe("full");
  });

  test("budget boundary + 1: estimatedTokensIfFull > maxTotalTokens → isWithinBudget=false", () => {
    // reservedMetadataTokens=50, diffTokens=960, total=1010 > 1000
    const file = contentFile({ insertions: 96, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(classify(file));

    expect(result.estimate.estimatedTokensIfFull).toBe(1_010);
    expect(result.estimate.isWithinBudget).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// plan() — oversized file degradation (Step 4a)
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — oversized file degradation (Step 4a)", () => {
  // Helper: 3 files where one is oversized, producing an over-budget scenario.
  // oversized(60 lines) + normal1(30 lines) + normal2(30 lines)
  // reservedMetadata=150, estimatedIfFull=1350 > 1000 → enters else branch
  // availableDiffBudget = 850, normalDiffTotal = 600 ≤ 850 → Step 4b applies after removing oversized

  test("file with lines > maxLinesPerFile → degraded/oversized", () => {
    // reservedMetadata=150, estimatedIfFull=150+600+300+300=1350 > 1000 → enters else branch
    const oversized = contentFile({ insertions: 60, deletions: 0 }); // 60 > 50
    const normal1 = contentFile({ insertions: 30, deletions: 0 });
    const normal2 = contentFile({ insertions: 30, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(
      classify(oversized, normal1, normal2),
    );

    expect(result.plans[0]!.mode).toBe("degraded");
    expect(result.plans[0]!.degradationReason).toBe("oversized");
    expect(result.plans[0]!.estimatedTokens).toBeNull();
  });

  test("file with lines === maxLinesPerFile (50) → NOT oversized (boundary is strict >)", () => {
    const atLimit = contentFile({ insertions: 50, deletions: 0 }); // exactly 50, NOT > 50
    // Need to be over budget: add enough overhead via extra files
    // 15 more normal files + atLimit → reservedMetadata=16*50=800, atLimit diffTokens=500
    // total normal files diff = 500, estimatedIfFull = 800 + 500 = 1300 > 1000 (over budget)
    // but wait: normal files (not oversized), and we don't want them to trigger greedy
    // Let me use: atLimit(50 lines) + 1 noise file (for overhead)
    // 2 total files: reservedMetadata=100, atLimit diffTokens=500, total=600 ≤ 1000 → within budget
    // That would be within budget, so 4a never runs. Need a different setup.
    //
    // Use: atLimit + 8 noise files = 9 total files
    // reservedMetadata = 9*50 = 450, atLimit diffTokens=500, total=950 ≤ 1000 → still within!
    //
    // Use: atLimit + 9 noise files = 10 total files
    // reservedMetadata = 10*50 = 500, atLimit diffTokens=500, total=1000 ≤ 1000 → within boundary
    //
    // Use: atLimit + 10 noise files = 11 total files
    // reservedMetadata=550, diffTokens=500, total=1050 > 1000 → OVER BUDGET → enters else
    // atLimit: 50 lines, NOT > 50 → stays in normal[]
    // normalDiffTotal=500, availableDiffBudget=1000-550=450
    // 500 > 450 → greedy fill
    // atLimit: 500 > 450 → BUDGET-EXCEEDED (not oversized, just doesn't fit)
    // So: mode should be "budget-exceeded", degradationReason "budget-exceeded" (not "oversized")
    const noises = noiseFiles(10);
    const result = new BudgetPlanner(TIGHT).plan(classify(...noises, atLimit));
    const lastPlan = result.plans[result.plans.length - 1]!;

    // Key assertion: it must NOT be labelled "oversized" — the line limit is strict greater-than
    expect(lastPlan.degradationReason).not.toBe("oversized");
  });

  test("oversized file removed; remaining normal files within diff budget → all normal get 'full'", () => {
    // oversized(60 lines) + normalA(30 lines) + normalB(30 lines)
    // reservedMetadata=150, estimatedIfFull=1350 > 1000 → over budget
    // availableDiffBudget=850, normalDiffTotal=600 ≤ 850 → Step 4b
    const oversized = contentFile({ insertions: 60, deletions: 0 });
    const normalA = contentFile({ insertions: 30, deletions: 0 });
    const normalB = contentFile({ insertions: 30, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(
      classify(oversized, normalA, normalB),
    );

    expect(result.plans[0]!.degradationReason).toBe("oversized");
    expect(result.plans[1]!.mode).toBe("full");
    expect(result.plans[1]!.estimatedTokens).toBe(350); // 300+50
    expect(result.plans[2]!.mode).toBe("full");
    expect(result.fullDiffCount).toBe(2);
    expect(result.degradedCount).toBe(1);
  });

  test("all content files oversized → all degraded/oversized", () => {
    const a = contentFile({ insertions: 60, deletions: 0 });
    const b = contentFile({ insertions: 80, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(classify(a, b));

    for (const plan of result.plans) {
      expect(plan.mode).toBe("degraded");
      expect(plan.degradationReason).toBe("oversized");
    }
    expect(result.fullDiffCount).toBe(0);
    expect(result.degradedCount).toBe(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// plan() — greedy fill (Step 4c)
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — greedy fill (Step 4c)", () => {
  test("4 equal-sized files over budget: 3 accepted, 1 budget-exceeded", () => {
    // 4 × 25-line files → diffTokens=250 each
    // reservedMetadata=200, estimatedIfFull=1200 > 1000 → over budget
    // availableDiffBudget=800, normalDiffTotal=1000 > 800 → greedy
    // greedy (sorted asc, all equal): accept A(250), B(500), C(750), reject D(1000>800)
    const [a, b, c, d] = [
      contentFile({ insertions: 25, deletions: 0 }),
      contentFile({ insertions: 25, deletions: 0 }),
      contentFile({ insertions: 25, deletions: 0 }),
      contentFile({ insertions: 25, deletions: 0 }),
    ] as const;
    const result = new BudgetPlanner(TIGHT).plan(classify(a, b, c, d));

    expect(result.fullDiffCount).toBe(3);
    expect(result.degradedCount).toBe(1);
    expect(result.plans[3]!.mode).toBe("degraded");
    expect(result.plans[3]!.degradationReason).toBe("budget-exceeded");
  });

  test("greedy sorts ascending by diffTokens — smaller files accepted first", () => {
    // Input order: large(40 lines/400 tokens), small(20/200), medium(30/300)
    // reservedMetadata=150, total=1050 > 1000 → over budget
    // availableDiffBudget=850, normalDiffTotal=900 > 850 → greedy
    // sorted: small(200), medium(300), large(400)
    //   accept small: 200 ≤ 850 ✓
    //   accept medium: 500 ≤ 850 ✓
    //   reject large: 900 > 850 ✗
    const large = contentFile({ insertions: 40, deletions: 0 });
    const small = contentFile({ insertions: 20, deletions: 0 });
    const medium = contentFile({ insertions: 30, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(
      classify(large, small, medium),
    );

    expect(result.fullDiffCount).toBe(2);
    expect(result.degradedCount).toBe(1);
    // Output follows INPUT order, not sort order
    expect(result.plans[0]!.mode).toBe("degraded");
    expect(result.plans[0]!.degradationReason).toBe("budget-exceeded"); // large
    expect(result.plans[1]!.mode).toBe("full"); // small
    expect(result.plans[2]!.mode).toBe("full"); // medium
  });

  test("noise files shrink availableDiffBudget, causing content file to be budget-exceeded", () => {
    // 16 noise files + 1 content file (20 lines)
    // reservedMetadata = 17 × 50 = 850
    // diffTokens = 200
    // estimatedIfFull = 850 + 200 = 1050 > 1000 → over budget
    // availableDiffBudget = 1000 − 850 = 150
    // normalDiffTotal = 200 > 150 → greedy
    // 200 > 150 → budget-exceeded
    const noises = noiseFiles(16);
    const content = contentFile({ insertions: 20, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(classify(...noises, content));

    const contentPlan = result.plans[result.plans.length - 1]!;
    expect(contentPlan.mode).toBe("degraded");
    expect(contentPlan.degradationReason).toBe("budget-exceeded");
  });

  test("availableDiffBudget clamped to 0 when metadata overhead exceeds maxTotalTokens", () => {
    // 21 noise files + 1 content file (10 lines)
    // reservedMetadata = 22 × 50 = 1100 > 1000
    // availableDiffBudget = Math.max(0, 1000 − 1100) = 0
    // diffTokens = 100 > 0 → budget-exceeded
    const noises = noiseFiles(21);
    const content = contentFile({ insertions: 10, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(classify(...noises, content));

    const contentPlan = result.plans[result.plans.length - 1]!;
    expect(contentPlan.mode).toBe("degraded");
    expect(contentPlan.degradationReason).toBe("budget-exceeded");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// plan() — mixed noise + content
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — mixed noise + content", () => {
  test("1 noise + 2 content within budget: noise=degraded, content=full", () => {
    // reservedMetadata = 3 × 50 = 150; 2 × diffTokens(200) = 400; total=550 ≤ 1000
    const noise = noiseFile();
    const c1 = contentFile({ insertions: 20, deletions: 0 });
    const c2 = contentFile({ insertions: 20, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(classify(noise, c1, c2));

    expect(result.estimate.totalFiles).toBe(3);
    expect(result.estimate.noiseFiles).toBe(1);
    expect(result.estimate.nonNoiseFiles).toBe(2);
    expect(result.estimate.isWithinBudget).toBe(true);

    expect(result.plans[0]!.mode).toBe("degraded");
    expect(result.plans[0]!.degradationReason).toBe("noise");
    expect(result.plans[1]!.mode).toBe("full");
    expect(result.plans[2]!.mode).toBe("full");
    expect(result.fullDiffCount).toBe(2);
    expect(result.degradedCount).toBe(1);
  });

  test("plans array preserves the original input order across noise and content files", () => {
    const n = noiseFile();
    const c1 = contentFile({ insertions: 10, deletions: 0 });
    const c2 = contentFile({ insertions: 10, deletions: 0 });
    const n2 = noiseFile();
    const c3 = contentFile({ insertions: 10, deletions: 0 });

    const classified = classify(n, c1, c2, n2, c3);
    const result = new BudgetPlanner(TIGHT).plan(classified);

    // Each plan.file must point back to the corresponding input ClassifiedFile
    for (let i = 0; i < classified.files.length; i++) {
      expect(result.plans[i]!.file.file.path).toBe(
        classified.files[i]!.file.path,
      );
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// plan() — renamedNoContentChangeCount
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — renamedNoContentChangeCount", () => {
  test("renamed file with 0 ins + 0 del → counted", () => {
    const renamed = contentFile({
      changeType: "renamed",
      insertions: 0,
      deletions: 0,
      similarityScore: 100,
    });
    const result = new BudgetPlanner(TIGHT).plan(classify(renamed));

    expect(result.estimate.renamedNoContentChangeCount).toBe(1);
  });

  test("renamed file with null ins + null del → counted via ?? 0 fallback", () => {
    const renamed = contentFile({
      changeType: "renamed",
      insertions: null,
      deletions: null,
      similarityScore: 100,
    });
    const result = new BudgetPlanner(TIGHT).plan(classify(renamed));

    expect(result.estimate.renamedNoContentChangeCount).toBe(1);
  });

  test("renamed file WITH content changes → not counted", () => {
    const renamed = contentFile({
      changeType: "renamed",
      insertions: 5,
      deletions: 3,
      similarityScore: 85,
    });
    const result = new BudgetPlanner(TIGHT).plan(classify(renamed));

    expect(result.estimate.renamedNoContentChangeCount).toBe(0);
  });

  test("non-renamed file with 0 lines → not counted", () => {
    const modified = contentFile({
      changeType: "modified",
      insertions: 0,
      deletions: 0,
    });
    const result = new BudgetPlanner(TIGHT).plan(classify(modified));

    expect(result.estimate.renamedNoContentChangeCount).toBe(0);
  });

  test("mixed: only pure-rename files (0 content change) are counted", () => {
    const pureRename = contentFile({
      changeType: "renamed",
      insertions: 0,
      deletions: 0,
      similarityScore: 100,
    });
    const nullRename = contentFile({
      changeType: "renamed",
      insertions: null,
      deletions: null,
      similarityScore: 100,
    });
    const contentRename = contentFile({
      changeType: "renamed",
      insertions: 5,
      deletions: 3,
      similarityScore: 85,
    });
    const modified = contentFile({
      changeType: "modified",
      insertions: 0,
      deletions: 0,
    });
    const result = new BudgetPlanner(TIGHT).plan(
      classify(pureRename, nullRename, contentRename, modified),
    );

    expect(result.estimate.renamedNoContentChangeCount).toBe(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// plan() — aggregate metric correctness
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — aggregate metrics", () => {
  test("totalChangedLines is sum of insertions+deletions across all content files", () => {
    // f1: 10+5=15, f2: 20+0=20 → total=35
    const f1 = contentFile({ insertions: 10, deletions: 5 });
    const f2 = contentFile({ insertions: 20, deletions: 0 });
    const result = new BudgetPlanner(TIGHT).plan(classify(f1, f2));

    expect(result.estimate.totalChangedLines).toBe(35);
  });

  test("maxSingleFileLines is the max across all content files", () => {
    const f1 = contentFile({ insertions: 10, deletions: 5 }); // 15
    const f2 = contentFile({ insertions: 30, deletions: 10 }); // 40
    const f3 = contentFile({ insertions: 5, deletions: 3 }); //  8
    const result = new BudgetPlanner(TIGHT).plan(classify(f1, f2, f3));

    expect(result.estimate.maxSingleFileLines).toBe(40);
  });

  test("maxSingleFileLines is 0 when there are no content files", () => {
    const result = new BudgetPlanner(TIGHT).plan(classify(noiseFile()));

    expect(result.estimate.maxSingleFileLines).toBe(0);
  });

  test("estimate.tokenBudget mirrors the configured maxTotalTokens", () => {
    const result = new BudgetPlanner(TIGHT).plan(classify());
    expect(result.estimate.tokenBudget).toBe(TIGHT.maxTotalTokens);
  });

  test("estimatedTokensIfFull includes reserved metadata tokens for noise files", () => {
    // 1 noise + 1 content (20 lines)
    // reservedMetadata = 2×50 = 100, diffTokens = 200, total = 300
    const result = new BudgetPlanner(TIGHT).plan(
      classify(noiseFile(), contentFile({ insertions: 20, deletions: 0 })),
    );
    expect(result.estimate.estimatedTokensIfFull).toBe(300);
  });

  test("fullDiffCount + degradedCount always equals plans.length", () => {
    const scenarios: FileClassificationResult[] = [
      classify(),
      classify(noiseFile()),
      classify(contentFile()),
      classify(noiseFile(), contentFile()),
      // over-budget mixed scenario
      classify(
        ...noiseFiles(5),
        contentFile({ insertions: 25, deletions: 0 }),
        contentFile({ insertions: 25, deletions: 0 }),
        contentFile({ insertions: 25, deletions: 0 }),
      ),
    ];
    for (const scenario of scenarios) {
      const result = new BudgetPlanner(TIGHT).plan(scenario);
      expect(result.fullDiffCount + result.degradedCount).toBe(
        result.plans.length,
      );
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// plan() — default thresholds (no-arg constructor)
// ═══════════════════════════════════════════════════════════════════════════

describe("BudgetPlanner.plan() — default constructor", () => {
  test("uses DEFAULT_BUDGET_THRESHOLDS (maxTotalTokens=16,000) when no argument provided", () => {
    // With default thresholds: 1 file × 20 lines → total ≪ 16 000
    const file = contentFile({ insertions: 20, deletions: 0 });
    const result = new BudgetPlanner().plan(classify(file));

    expect(result.estimate.tokenBudget).toBe(16_000);
    expect(result.estimate.isWithinBudget).toBe(true);
    expect(result.plans[0]!.mode).toBe("full");
  });
});
