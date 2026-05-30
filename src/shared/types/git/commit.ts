import type { GitRepoPrecheckContext } from "./context";
import type { StagedDiffSummary } from "./diff";
import type { DiffPlanResult } from "./planning";

// Only Git data-collection steps; orchestration steps (prompt/llm/commit) are not here
export type GitPipelineStep =
  | "repo-precheck"
  | "state-detect"
  | "diff-collect"
  | "file-classify"
  | "budget-plan"
  | "diff-fetch";

export type GitPipelineResult =
  | {
      readonly route: "internal-op";
      readonly commitMessage: string;
      readonly completedSteps: GitPipelineStep[];
    }
  | {
      readonly route: "clean";
      readonly repoContext: GitRepoPrecheckContext;
      readonly diffSummary: StagedDiffSummary;
      readonly diffPlan: DiffPlanResult;
      readonly diffTexts: ReadonlyMap<string, string>;
      readonly completedSteps: GitPipelineStep[];
    };