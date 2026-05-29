export type CommitExecutionStep =
  | "repo-precheck"
  | "state-detect"
  | "diff-collect"
  | "file-classify"
  | "budget-plan"
  | "diff-fetch"
  | "prompt-assemble"
  | "llm-generate"
  | "git-commit";

export interface CommitExecutionResult {
  readonly commitHash: string;
  readonly commitMessage: string;
  readonly completedSteps: CommitExecutionStep[];
}