import { type RequestSafetyConfig } from "./request-safety";
import { type LLModelParams } from "./model-settings";

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
export interface ProviderConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
  provider: string;
  modelParams?: LLModelParams;
  requestConfig?: RequestSafetyConfig;
}

