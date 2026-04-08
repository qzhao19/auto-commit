/**
 * Configuration for rate limiter
 */
export interface RateLimiterConfig {
  maxRequestsPerMinute: number;
  maxQueueSize: number;
  requestTimeout: number;
}