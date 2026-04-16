import { type RateLimiterConfig, type AcquireOptions } from "../../../shared/types/index";

// A type for items in internal queue
type WaitingResolver = {
  resolve: () => void;
  reject: (error: Error) => void;
  enqueuedAt: number;
  timeoutId: ReturnType<typeof setTimeout>;
};

export class RateLimiter {
  private readonly maxTokens: number;
  private readonly refillRatePerSecond: number;
  private readonly maxQueueSize: number;
  private readonly requestTimeout: number;

  // Current number of tokens in the bucket
  private tokenBucket: number;
  private lastRefillTimestamp: number;
  private waitingQueue: WaitingResolver[] = [];

  // Drain scheduling
  private isDraining: boolean = false;
  private scheduledDrainId: ReturnType<typeof setTimeout> | null = null;

  constructor(config: RateLimiterConfig) {
    this.maxTokens = config.maxRequestsPerMinute;
    this.refillRatePerSecond = config.maxRequestsPerMinute / 60;
    this.maxQueueSize = config.maxQueueSize;
    this.requestTimeout = config.requestTimeout;

    // Initialize the bucket as full
    this.tokenBucket = this.maxTokens;
    this.lastRefillTimestamp = Date.now();
  }

  /**
   * Acquires a token. Resolves when a token is available, 
   * rejects if the queue is full or the request times out.
   */
  public async acquire(options?: AcquireOptions): Promise<void> {
    if (this.waitingQueue.length >= this.maxQueueSize) {
      throw new Error(
        `[RateLimiter] Rate limiter queue is full (${this.maxQueueSize} requests)`,
      );
    }

    const effectiveTimeoutMs = options?.timeoutMs ?? this.requestTimeout;
    if (!Number.isFinite(effectiveTimeoutMs) || effectiveTimeoutMs <= 0) {
      throw new Error("[RateLimiter] Acquire timeout must be a finite number > 0");
    }

    return new Promise<void>((resolve, reject) => {
      // Set a separate timeout timer for each request
      const timeoutId = setTimeout(() => {
        const index = this.waitingQueue.indexOf(item);
        if (index >= 0) {
          this.waitingQueue.splice(index, 1);
        }
        reject(new Error(`[RateLimiter] Request timeout after ${effectiveTimeoutMs}ms`));
      }, effectiveTimeoutMs);

      // Create object of waiting queue
      const item: WaitingResolver = {
        enqueuedAt: Date.now(),
        timeoutId,
        resolve: () => {
          clearTimeout(timeoutId);
          resolve();
        },
        reject: (error: Error) => {
          clearTimeout(timeoutId);
          reject(error);
        },
      };

      // Enqueue item
      this.waitingQueue.push(item);
      this.scheduleDrain();
    });
  }

  /**
   * Current number of requests waiting in the queue
   */
  public get queueLength(): number {
    return this.waitingQueue.length;
  }

  /**
   * Current available tokens
   */
  public get availableTokens(): number {
    return this.tokenBucket;
  }

  /**
   * Add the token based on time
   */
  private refillTokens(): void {
    const currentTimestamp: number = Date.now();
    // Calculate the time difference in seconds
    const elapsedSeconds: number = (currentTimestamp - this.lastRefillTimestamp) / 1000;
    if (elapsedSeconds <= 0) return;

    // Calculate the number of tokens that should theoretically be added
    const tokensToAdd: number = elapsedSeconds * this.refillRatePerSecond;
    this.tokenBucket = Math.min(this.maxTokens, this.tokenBucket + tokensToAdd); 
    this.lastRefillTimestamp = currentTimestamp;
  }

  /**
   * Schedules a drain pass
   */
  private scheduleDrain(): void {
    // If a drain is already scheduled or actively running
    if (this.scheduledDrainId !== null || this.isDraining) return;

    this.scheduledDrainId = setTimeout(() => {
      this.scheduledDrainId = null;
      this.drain()
    }, 0);
  }

  /**
   * Drains the queue
   */
  private drain(): void {
    if (this.isDraining) return;
    this.isDraining = true;

    try {
      // Firstly add token
      this.refillTokens();

      // Release all requests, queue has item and token exists
      while (this.waitingQueue.length > 0 && this.tokenBucket >= 1) {
        this.tokenBucket -= 1;
        const item: WaitingResolver | undefined = this.waitingQueue.shift();
        item?.resolve();
      }

      // If items remain, schedule the next drain at the moment 
      // that the next token becomes available
      if (this.waitingQueue.length > 0) {
        const deficit: number = 1 - this.tokenBucket;
        const waitMs: number = Math.max(
          1, 
          Math.ceil((deficit / this.refillRatePerSecond) * 1000),
        );
        this.scheduledDrainId = setTimeout(() => {
          this.scheduledDrainId = null;
          this.drain();
        }, waitMs);
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error : new Error("Rate limiter drain failed");
      // Reject all pending to avoid hangs
      while (this.waitingQueue.length > 0) {
        const item: WaitingResolver | undefined = this.waitingQueue.shift();
        item?.reject(errorMsg);
      }
    } finally {
      this.isDraining = false;
    }
  }
}