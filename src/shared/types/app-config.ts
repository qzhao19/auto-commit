import { type RequestGuardsConfig, type InternalRequestGuardsConfig } from "./request-guards";
import { type LLMGenerationConfig } from "./model-settings";

/**
 * Provider-level configuration (per adapter instance).
 * 
 * Properties:
 * - apiKey: API key for authentication (not required for local/Ollama)
 * - baseUrl: Custom API base URL (e.g., DeepSeek endpoint, Ollama localhost)
 * - model: Default model identifier or name
 * - modelParams: Default generation parameters, can be overridden per invoke() call 
 * - requestConfig: LLM request config (retry / timeout / rate-limiter)
 */
interface ProviderConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
  provider: string;
  generationConfig?: LLMGenerationConfig;
  requestGuardsConfig?: RequestGuardsConfig;
}

/**
 * Internally parsed configuration (ready for direct use by retry guards)
 */
export interface ResolvedProviderConfig extends Omit<ProviderConfig, "requestGuardsConfig"> {
  requestGuardsConfig: InternalRequestGuardsConfig;
}