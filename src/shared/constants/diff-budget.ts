import { type BudgetThresholds } from "../types/index";

/**
 * Default budget thresholds.
 *
 * Calibration rationale:
 *   - maxTotalTokens: leaves ~112k of a 128k context window for system prompt,
 *     history, and the generated commit message.
 *   - maxLinesPerFile: 500 changed lines ≈ a complete file rewrite; unlikely to
 *     add incremental value once the LLM sees its neighbours.
 *   - tokensPerLine: empirical average for mixed code/config content (~40 chars/line
 *     × ~0.25 tokens/char ≈ 10; conservative to avoid under-estimation).
 *   - tokensPerFileOverhead: filename + diff hunk header + separators.
 */
export const DEFAULT_BUDGET_THRESHOLDS: BudgetThresholds = {
  maxTotalTokens:       16_000,
  maxLinesPerFile:        500,
  tokensPerLine:           10,
  tokensPerFileOverhead:   50,
};