import { type LLMRequestConfig } from "./llm-request-config";
import { type LLModelParams } from "./llm-model-params";

/**
 * Provider-level configuration (per adapter instance).
 */
export interface ProviderConfig {
  /** API key for authentication (not required for local/Ollama) */
  apiKey?: string;
  /** Custom API base URL (e.g., DeepSeek endpoint, Ollama localhost) */
  baseUrl?: string;
  /** Default model identifier or name */
  model: string;
  /** Default generation parameters, can be overridden per invoke() call */
  modelParams?: LLModelParams;
  /** LLM request config (retry / timeout / rate-limiter) */
  requestConfig?: LLMRequestConfig;
}

