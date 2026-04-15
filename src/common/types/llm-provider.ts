import { type LLMRequestConfig } from "./llm-request-config";
import { type LLModelParams } from "./llm-model-params";

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
  apiKey?: string;
  baseUrl?: string;
  model: string;
  modelParams?: LLModelParams;
  requestConfig?: LLMRequestConfig;
}

