import type { GitRepoPrecheckContext, GitInternalOpState } from "../git/context";
import type { StagedDiffSummary } from "../git/diff";
import type { DiffPlanResult } from "../git/planning";

export type LLMMessageRole = "system" | "user" | "assistant";

export interface LLMMessage {
  readonly role: LLMMessageRole;
  readonly content: string;
}

export interface PromptAssemblyInput {
  readonly repoContext: GitRepoPrecheckContext;             // Phase 0: branch, isInitialCommit
  readonly gitState: GitInternalOpState;                    // Phase 1: merge/rebase/cherry-pick context
  readonly diffSummary: StagedDiffSummary;                  // Phase 2: +ins/-del aggregate stats
  readonly diffPlan: DiffPlanResult;                        // Phase 3b: per-file mode + budget estimate
  readonly diffTexts: ReadonlyMap<string, string>;          // Resolved full-diff content (full-mode files only)
}

export interface AssembledPrompt {
  readonly messages: readonly LLMMessage[];
  readonly tokenEstimate: number; // Includes system message overhead not in diffPlan.estimate
}
