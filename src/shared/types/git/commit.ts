import { type GitInternalOpState, type GitRepoPrecheckContext } from "./context";
import { type DiffBuildResult } from "./planning";

export interface CommitHintSet {
  readonly suggestedTypes: readonly string[];
  readonly confidence: "high" | "low" | "none";
  readonly reasons: readonly string[];
  readonly heuristicFlags: readonly string[];
}

export interface CommitGenerationInput {
  readonly repoContext: GitRepoPrecheckContext;
  readonly gitState: GitInternalOpState;
  readonly diffBuild: DiffBuildResult | null;
  readonly hints: CommitHintSet;
  readonly prompt: string;
}

export interface CommitMessageDraft {
  readonly type: string;
  readonly scope: string | null;
  readonly subject: string;
  readonly body: string | null;
  readonly breaking: boolean;
  readonly source: "llm" | "template" | "reused" | "fallback";
}

export interface ValidatedCommitMessage {
  readonly title: string;
  readonly body: string | null;
  readonly type: string;
  readonly scope: string | null;
  readonly subject: string;
  readonly breaking: boolean;
  readonly source: CommitMessageDraft["source"];
  readonly warnings: readonly string[];
}