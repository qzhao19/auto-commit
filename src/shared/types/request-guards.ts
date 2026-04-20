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
 * User-configurable retry behavior (retryableErrors are not allowed)
 */
export type PublicRetryConfig = Omit<RetryConfig, "retryableErrors">;

/**
 * Request policies for LLM calls.
 * All fields optional — only instantiate the guards you need.
 */
export interface RequestGuardsConfig {
  retry: PublicRetryConfig;
  timeout: TimeoutConfig;
  rateLimiter: RateLimiterConfig;
}

/**
 * Complete guards types used by the internal runtime including retryableErrors
 */
export interface InternalRequestGuardsConfig extends Omit<RequestGuardsConfig, "retry"> {
  retry: RetryConfig;
}

/**
 * Partial configuration for request guards
 * All fields are optional, and each guard's config can be partially provided
 */
export type PartialRequestGuardsConfig = {
  retry?: Partial<PublicRetryConfig>;
  timeout?: Partial<TimeoutConfig>;
  rateLimiter?: Partial<RateLimiterConfig>;
};
