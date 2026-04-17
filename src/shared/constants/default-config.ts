import { type RequestGuardsConfig, type LLMConfig } from "../types/index"

export const DEFAULT_REQUEST_GUARDS_CONFIG: RequestGuardsConfig = {
  retry: {
    maxRetries: 3,
    initialDelayMs: 1000,
    maxDelayMs: 10_000,
    factor: 2,
    jitter: true,
  },
  timeout: {
    timeoutMs: 30_000,
  },
  rateLimiter: {
    maxRequestsPerMinute: 20,
    maxQueueSize: 1000,
    requestTimeout: 30_000,
  },
}

export const DEFAULT_LLM_CONFIG: LLMConfig = {
  provider: "",
  model: "",
  baseUrl: "",
  temperature: 0.8,
  maxTokens: 4096,
  topP: 0.9,
  frequencyPenalty: 0,
  presencePenalty: 0,
  
}