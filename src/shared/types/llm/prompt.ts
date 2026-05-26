import type { GitRepoPrecheckContext, GitInternalOpState } from "../git/context";
import type { FileContentCategory, FileNoiseCategory } from "../git/classify";
import type { FileDiffPlan, DiffPlanResult } from "../git/planning";

export type LLMMessageRole = "system" | "user" | "assistant";

export interface LLMMessage {
  readonly role: LLMMessageRole;
  readonly content: string;
}

export type PromptFileCategory = FileContentCategory | FileNoiseCategory;

export interface PromptFileEntry {
  readonly plan: FileDiffPlan;
  readonly category: PromptFileCategory;
  readonly diffText: string | null;
}

export interface PromptAssemblyInput {
  readonly repoContext: GitRepoPrecheckContext;
  readonly gitState: GitInternalOpState;
  readonly diffPlan: DiffPlanResult;
  readonly diffTexts: ReadonlyMap<string, string>;
}

export interface AssembledPrompt {
  readonly messages: readonly LLMMessage[];
  readonly fileEntries: readonly PromptFileEntry[];
  readonly gitState: GitInternalOpState;
  readonly tokenEstimate: number;
  readonly isComplete: boolean;
  readonly fullPaths: readonly string[];
  readonly degradedPaths: readonly string[];
}
