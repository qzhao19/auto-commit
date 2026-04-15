/**
 * Configuration for rate limiter
 */
export interface RateLimiterConfig {
  maxRequestsPerMinute: number;
  maxQueueSize: number;
  requestTimeout: number;
}

/**
 * Configuration for retry
 */
export interface RetryConfig {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
  factor: number;
  retryableErrors: RegExp[];
  jitter: boolean;
}


/**
 * Configuration for timeout
 */
export interface TimeoutConfig {
  timeoutMs: number;
}

/**
 * Request policies for LLM calls..
 * All fields optional — only instantiate the guards you need.
 */
export interface LLMRequestConfig {
  retry?: RetryConfig;
  timeout?: TimeoutConfig;
  rateLimiter?: RateLimiterConfig;
}
