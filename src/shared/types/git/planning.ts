import { type FileClassificationResult } from "./classify";
import { type StagedChangeType } from "./diff";

export interface DiffBudgetConfig {
  readonly maxPromptTokens: number;
  readonly softTokenBudget: number;
}

export const DEFAULT_DIFF_BUDGET: DiffBudgetConfig = {
  maxPromptTokens: 20_000,
  softTokenBudget: 16_000,
};

export interface DiffBudgetEstimate {
  readonly estimatedTokens: number;
  readonly estimatedChars: number;
  readonly oversized: boolean;
  readonly textFileCount: number;
  readonly totalChangeLines: number;
}

export type DiffSelectionMode = "full" | "trimmed" | "summary-only";

export interface DiffSelectionPlan {
  readonly mode: DiffSelectionMode;
  readonly estimate: DiffBudgetEstimate;
  readonly classification: FileClassificationResult;
  readonly fullDiffPaths: readonly string[];
  readonly summarizedTextPaths: readonly string[];
}

export type FileLLMPayload =
  | {
      kind: "diff";
      path: string;
      oldPath: string | null;
      changeType: StagedChangeType;
      insertions: number | null;
      deletions: number | null;
      diff: string;
    }
  | {
      kind: "stats";
      path: string;
      oldPath: string | null;
      changeType: StagedChangeType;
      insertions: number | null;
      deletions: number | null;
      annotation: string | null;
    }
  | {
      kind: "binary";
      path: string;
      changeType: StagedChangeType;
    }
  | {
      kind: "submodule";
      path: string;
      changeType: StagedChangeType;
    };

export interface DiffBuildResult {
  readonly plan: DiffSelectionPlan;
  readonly payloads: readonly FileLLMPayload[];
  readonly totalDiffChars: number;
}