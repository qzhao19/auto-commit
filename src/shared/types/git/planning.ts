import { type ClassifiedFile } from "./classify";

/**
 * - maxTotalTokens: token ceiling for the diff section
 * - maxLinesPerFile: files with more total changed lines than 
 *                    this are "oversized" and always degraded. 
 * - tokensPerLine: tokens-per-changed-line multiplier used for estimation
 * - tokensPerFileOverhead: Fixed per-file overhead 
 */
export interface BudgetThresholds {
  readonly maxTotalTokens: number;
  readonly maxLinesPerFile: number;
  readonly tokensPerLine: number;
  readonly tokensPerFileOverhead: number;
}

// Whether a file will receive a full diff fetch or be handled in degraded mode.
export type FileDiffMode = "full" | "degraded";

/**
 * Why a file was placed in degraded mode:
 *   - "noise"           → noise file (binary / submodule / lfs-pointer); no diff to pull
 *   - "oversized"       → single-file changed lines exceed maxLinesPerFile
 *   - "budget-exceeded" → file would push the cumulative token estimate over maxTotalTokens
 */
export type DegradationReason = "noise" | "oversized" | "budget-exceeded";

/**
 * - estimatedTokens: Coarse estimated tokens for this file's diff contribution.
 *                    Null for degraded files where no diff will be fetched.
 */
export interface FileDiffPlan {
  readonly file: ClassifiedFile;
  readonly mode: FileDiffMode;
  readonly degradationReason?: DegradationReason;
  readonly estimatedTokens: number | null;
}

/**
 * Summary of the coarse estimation pass, computed before any git diff calls.
 * Exposes all four estimation inputs from the spec:
 *   - file count / content file count
 *   - total changed lines
 *   - renamed-with-no-content-change file count
 *   - max changed lines in a single file
 */
export interface BudgetEstimate {
  readonly totalFiles: number;
  readonly contentFiles: number;
  readonly noiseFiles: number;
  /** Renamed files whose insertions and deletions are both zero. */
  readonly renamedNoContentChangeCount: number;
  /** Largest changed-line count (insertions + deletions) across all content files. */
  readonly maxSingleFileLines: number;
  /** Sum of insertions + deletions across all content files. */
  readonly totalChangedLines: number;
  /** Estimated tokens if every content file received a full diff. */
  readonly estimatedTokensIfFull: number;
  /** Configured token budget ceiling (from BudgetThresholds.maxTotalTokens). */
  readonly tokenBudget: number;
  /** True when estimatedTokensIfFull ≤ tokenBudget. */
  readonly isWithinBudget: boolean;
}

/**
 * Planning result
 * - estimate: aggregate metrics from the coarse estimation pass
 * - plans: per-file plan, in the same order as FileClassificationResult.files
 * - fullDiffCount: number of files that will receive a full diff fetch
 * - degradedCount: number of files that will be handled in degraded mode
 */
export interface DiffPlanResult {
  readonly estimate: BudgetEstimate;
  readonly plans: readonly FileDiffPlan[];
  readonly fullDiffCount: number;
  readonly degradedCount: number;
}