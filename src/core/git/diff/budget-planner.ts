import {
  type BudgetEstimate,
  type BudgetThresholds,
  type ClassifiedFile,
  type DiffPlanResult,
  type FileDiffPlan,
  type FileClassificationResult,
} from "../../../shared/types/index";
import { DEFAULT_BUDGET_THRESHOLDS } from "../../../shared/constants/index";

/** Internal per-file estimate used only during plan() computation. */
interface FileEstimate {
  readonly file: ClassifiedFile;
  readonly path: string;
  readonly lines: number;
  readonly diffTokens: number;
  readonly fullTokens: number;
}

export class BudgetPlanner {
  private readonly thresholds: BudgetThresholds;

  constructor(thresholds: BudgetThresholds = DEFAULT_BUDGET_THRESHOLDS) {
    this.thresholds = thresholds;
  }

  public plan(classified: FileClassificationResult): DiffPlanResult {
    const {
      maxTotalTokens,
      maxLinesPerFile,
      tokensPerLine,
      tokensPerFileOverhead,
    } = this.thresholds;

    // ── Step 1: separate noise from content ──
    const noiseFiles = classified.files.filter((file) => file.isNoise);
    const nonNoiseFiles = classified.files.filter((file) => !file.isNoise);

    // ── Step 2: per-file token estimates ──
    const estimates: FileEstimate[] = nonNoiseFiles.map((file) => {
      const lines = (file.file.insertions ?? 0) + (file.file.deletions ?? 0);
      const diffTokens = lines * tokensPerLine;
      const fullTokens = diffTokens + tokensPerFileOverhead;
      return {
        file,
        path: file.file.path,
        lines,
        diffTokens,
        fullTokens,
      };
    });

    // Every file (noise or not) contributes at least one overhead slot in the prompt.
    // Any filenames, modification types, etc., that are ultimately added to the prompt
    // and seen by the LLM will be fixed as the cost of `tokensPerFileOverhead`.
    const reservedMetadataTokens =
      classified.files.length * tokensPerFileOverhead;
    const estimatedTokensIfFull =
      reservedMetadataTokens +
      estimates.reduce((sum, estimate) => sum + estimate.diffTokens, 0);
    const isWithinBudget = estimatedTokensIfFull <= maxTotalTokens;

    // Excluding this absolutely fixed overhead (reserved tokens),
    // how many tokens can we allocate during the content processing
    const availableDiffBudget = Math.max(
      0,
      maxTotalTokens - reservedMetadataTokens,
    );

    // ── Step 3 / 4: determine mode for every file, keyed by path ──
    const diffPlanByPath: Map<string, FileDiffPlan> = new Map();

    // Noise files: always degraded, no diff to fetch.
    for (const file of noiseFiles) {
      diffPlanByPath.set(file.file.path, {
        file,
        mode: "degraded",
        degradationReason: "noise",
        estimatedTokens: null,
      });
    }

    if (isWithinBudget) {
      for (const estimate of estimates) {
        diffPlanByPath.set(estimate.path, {
          file: estimate.file,
          mode: "full",
          estimatedTokens: estimate.fullTokens,
        });
      }
    } else {
      // 4a: oversized files
      const normal: FileEstimate[] = [];
      for (const estimate of estimates) {
        // If greater than a specific number of lines
        if (estimate.lines > maxLinesPerFile) {
          diffPlanByPath.set(estimate.path, {
            file: estimate.file,
            mode: "degraded",
            degradationReason: "oversized",
            estimatedTokens: null,
          });
        } else {
          normal.push(estimate);
        }
      }

      // 4b: check remaining diff budget after oversized removal
      // After excluding the largest files, recalculate the remaining costs.
      const normalDiffTotal = normal.reduce(
        (sum, estimate) => sum + estimate.diffTokens,
        0,
      );
      if (normalDiffTotal <= availableDiffBudget) {
        for (const estimate of normal) {
          diffPlanByPath.set(estimate.path, {
            file: estimate.file,
            mode: "full",
            estimatedTokens: estimate.fullTokens,
          });
        }
      } else {
        // 4c: greedy fill — sort asc to maximise file count within budget
        const sorted = [...normal].sort((a, b) => {
          // Priority tier 1: source files come before lockfiles
          const catA = a.file.isNoise ? null : a.file.nonNoiseCategory;
          const catB = b.file.isNoise ? null : b.file.nonNoiseCategory;

          if (catA !== catB) {
            // "source" < "lockfile" alphabetically, so direct string compare works
            // But for explicitness and maintainability:
            if (catA === "source" && catB === "lockfile") return -1;
            if (catA === "lockfile" && catB === "source") return 1;
          }

          // Priority tier 2: within same category, smaller files first (maximize count)
          return a.diffTokens - b.diffTokens;
        });

        const accepted: Set<string> = new Set();
        let accumulated = 0;
        for (const estimate of sorted) {
          if (accumulated + estimate.diffTokens <= availableDiffBudget) {
            accepted.add(estimate.path);
            accumulated += estimate.diffTokens;
          }
        }
        for (const estimate of normal) {
          if (accepted.has(estimate.path)) {
            diffPlanByPath.set(estimate.path, {
              file: estimate.file,
              mode: "full",
              estimatedTokens: estimate.fullTokens,
            });
          } else {
            diffPlanByPath.set(estimate.path, {
              file: estimate.file,
              mode: "degraded",
              degradationReason: "budget-exceeded",
              estimatedTokens: null,
            });
          }
        }
      }
    }

    // ── Assemble plans in original file order ──
    // diffPlanByPath now covers ALL files (noise + content), so no branch needed here.
    const plans: FileDiffPlan[] = classified.files.map(
      (file) => diffPlanByPath.get(file.file.path)!,
    );

    // ── Aggregate estimate metrics ──
    const changedLines = estimates.map((estimate) => estimate.lines);
    const totalChangedLines = changedLines.reduce(
      (sum, lines) => sum + lines,
      0,
    );
    const maxSingleFileLines = changedLines.reduce(
      (max, lines) => Math.max(max, lines),
      0,
    );

    const renamedNoContentChangeCount = nonNoiseFiles.filter(
      (file) =>
        file.file.changeType === "renamed" &&
        (file.file.insertions ?? 0) === 0 &&
        (file.file.deletions ?? 0) === 0,
    ).length;

    const estimate: BudgetEstimate = {
      totalFiles: classified.files.length,
      nonNoiseFiles: classified.nonNoiseCount,
      noiseFiles: classified.noiseCount,
      renamedNoContentChangeCount,
      maxSingleFileLines,
      totalChangedLines,
      estimatedTokensIfFull,
      tokenBudget: maxTotalTokens,
      isWithinBudget,
    };

    const fullDiffCount = plans.filter((plan) => plan.mode === "full").length;
    const degradedCount = plans.length - fullDiffCount;

    return { estimate, plans, fullDiffCount, degradedCount };
  }
}
