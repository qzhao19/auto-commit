import type { GitPipelineResult } from "../git/pipeline";

export type LLMMessageRole = "system" | "user" | "assistant";

export interface LLMMessage {
  readonly role: LLMMessageRole;
  readonly content: string;
}

export type PromptAssemblyInput = GitPipelineResult;

export interface AssembledPrompt {
  readonly messages: readonly LLMMessage[];
  readonly tokenEstimate: number; // Includes system message overhead not in diffPlan.estimate
}
