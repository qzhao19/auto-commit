import { type RequestGuardsConfig, type InternalRequestGuardsConfig } from "./request-guards";
import { type LLMGenerationConfig } from "./model-settings";

/**
 * Provider-level configuration (user-facing / factory-facing).
 * apiKey/baseUrl are optional because local/ollama style providers may not require them.
 */
interface ProviderConfig {
  apiKey?: string;
  baseUrl?: string;
  model: string;
  provider: string;
  generationConfig?: LLMGenerationConfig;
  requestGuardsConfig?: RequestGuardsConfig;
}

/**
 * Fully resolved provider config returned by ConfigLoader.
 */
export interface ResolvedProviderConfig extends Omit<ProviderConfig, "requestGuardsConfig" | "generationConfig"> {
  generationConfig: LLMGenerationConfig;
  requestGuardsConfig: InternalRequestGuardsConfig;
  verbose: boolean;
}

/**
 * Unified request payload passed from BaseProvider to adapter implementations.
 */
export interface ProviderInvokeOptions {
  prompt: string;
  model: string;
  generationConfig: LLMGenerationConfig;
  signal?: AbortSignal;
}