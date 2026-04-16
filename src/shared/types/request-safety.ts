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
 * Options for executing a timeout-guarded operation.
 * Reuses the same structure as `TimeoutConfig`.
 */
export type TimeoutExecuteOptions = TimeoutConfig;

/**
 * Options for acquiring a token from a rate limiter with a timeout.
 * Reuses the same structure as `TimeoutConfig`.
 */
export type AcquireOptions = TimeoutConfig;

/**
 * Request policies for LLM calls.
 * All fields optional — only instantiate the guards you need.
 */
export interface RequestSafetyConfig {
  retry: Omit<RetryConfig, "retryableErrors">;
  timeout: TimeoutConfig;
  rateLimiter: RateLimiterConfig;
}
