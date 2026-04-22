import { type RetryConfig } from "../../../shared/types/index";
import { GuardError, GuardErrorCode } from "../../../shared/exceptions/index";

// Helper function 
const sleep = (delayMs: number) => 
  new Promise((resolve) => setTimeout(resolve, delayMs));

export class Retry {
  private readonly maxRetries: number;
  private readonly maxDelayMs: number;
  private readonly initialDelayMs: number;
  private readonly factor: number;
  private readonly retryableErrors: RegExp[];
  private readonly jitter: boolean;

  constructor(config: RetryConfig) {
    this.maxRetries = config.maxRetries;
    this.maxDelayMs = config.maxDelayMs;
    this.initialDelayMs = config.initialDelayMs;
    this.factor = config.factor;
    this.retryableErrors = config.retryableErrors;
    this.jitter = config.jitter;
  }

  /**
   * Executes `func` with exponential backoff retry.
   */
  public async execute<T>(func: () => Promise<T>): Promise<T> {
    let lastError: Error | undefined;

    // attempt 0 = first try, attempts 1..maxRetries = retries
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const startTimestamp: number = Date.now();
        const result: T = await func();
        const duration: number = Date.now() - startTimestamp;

        // Record only when a retry succeeded, not the first attemp
        if (attempt > 0) {
          console.log(
            JSON.stringify({
              event: "retry_succeeded",
              attempt,
              maxRetries: this.maxRetries,
              durationMs: duration,
            }),
          );
        }

        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Last attempt
        if (attempt === this.maxRetries) {
          throw new GuardError({
            code: GuardErrorCode.RETRY_EXHAUSTED,
            message: `[Retry] All ${this.maxRetries} retries exhausted: ${lastError.message}`,
            cause: lastError,
          });
          break;
        }

        // Check if error is retryable
        if (!this.isRetryable(lastError)) {
          break;
        }

        // Calculate delay: initialDelayMs * factor^attempt
        const delayMs: number = this.calculateDelay(attempt);
        console.warn(
          JSON.stringify({
            event: "retry_attempt_failed",
            attempt,
            maxRetries: this.maxRetries,
            retryInMs: Math.round(delayMs),
            error: lastError.message,
          }),
        );

        await sleep(delayMs);
      }
    }
    // All retries exhausted or non-retryable error
    throw lastError!;
  }

  private isRetryable(error: Error): boolean {
    // If no patterns configured, retry all errors
    if (!this.retryableErrors || this.retryableErrors.length === 0) {
      return true;
    }
    return this.retryableErrors.some((pattern) => pattern.test(error.message));
  }

  /**
   * Calculate delay with exponential backoff and optional jitter
   */
  private calculateDelay(attempt: number): number {
    let delayMs: number = this.initialDelayMs * Math.pow(this.factor, attempt);

    if (this.jitter) {
      // Add ±20% random jitter
      const jitterValue: number = delayMs * 0.4 * (Math.random() - 0.5);
      delayMs += jitterValue;
    }

    // Clamp to [0, maxDelayMs]
    return Math.min(Math.max(0, delayMs), this.maxDelayMs);
  }
}
