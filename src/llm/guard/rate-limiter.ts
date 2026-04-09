import { type RateLimiterConfig } from "../../common/types/index";

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

  private tokenBucket: number;
  private lastRefillTimestamp: number;
  private waitingQueue: WaitingResolver[] = [];

  // Drain scheduling — replaces the polling loop
  private isDraining = false;
  private scheduledDrainId: ReturnType<typeof setTimeout> | null = null;

  constructor(config: RateLimiterConfig) {
    this.maxTokens = config.maxRequestsPerMinute;
    this.refillRatePerSecond = config.maxRequestsPerMinute / 60;
    this.maxQueueSize = config.maxQueueSize;
    this.requestTimeout = config.requestTimeout;

    this.tokenBucket = this.maxTokens;
    this.lastRefillTimestamp = Date.now();
  }

  /**
   * Acquires a token. Resolves when a token is available,
   * rejects if the queue is full or the request times out.
   */
  public async acquire(): Promise<void> {
    if (this.waitingQueue.length >= this.maxQueueSize) {
      throw new Error(
        `Rate limiter queue is full (${this.maxQueueSize} requests)`,
      );
    }

    return new Promise<void>((resolve, reject) => {
      // Per-request timeout — fires precisely, regardless of drain state
      const timeoutId = setTimeout(() => {
        const idx = this.waitingQueue.indexOf(item);
        if (idx >= 0) {
          this.waitingQueue.splice(idx, 1);
        }
        reject(new Error(`Request timed out after ${this.requestTimeout}ms`));
      }, this.requestTimeout);

      const item: WaitingResolver = {
        enqueuedAt: Date.now(),
        timeoutId,
        resolve: () => {
          clearTimeout(timeoutId);
          resolve();
        },
        reject: (err: Error) => {
          clearTimeout(timeoutId);
          reject(err);
        },
      };

      this.waitingQueue.push(item);
      this.scheduleDrain();
    });
  }


  private refillTokens(): void {
    const now = Date.now();
    const elapsedSeconds = (now - this.lastRefillTimestamp) / 1000;
    if (elapsedSeconds <= 0) return;

    const tokensToAdd = elapsedSeconds * this.refillRatePerSecond;
    this.tokenBucket = Math.min(this.maxTokens, this.tokenBucket + tokensToAdd);
    this.lastRefillTimestamp = now;
  }

  /**
   * Schedules a drain pass. If a drain is already scheduled or
   * actively running, this is a no-op - the current drain will
   * pick up the newly enqueued item.
   */
  private scheduleDrain(): void {
    if (this.scheduledDrainId !== null || this.isDraining) return;
    // Use queueMicrotask so the drain runs after the current
    // synchronous block (including the push into waitingQueue).
    this.scheduledDrainId = setTimeout(() => {
      this.scheduledDrainId = null;
      this.drain();
    }, 0);
  }

  /**
   * Drains the queue: grants tokens to as many waiting requests as
   * currently possible, then — if items remain — computes the exact
   * time until the next token is available and schedules itself once.
   */
  private drain(): void {
    if (this.isDraining) return;
    this.isDraining = true;

    try {
      this.refillTokens();

      // Release all requests we can satisfy right now
      while (this.waitingQueue.length > 0 && this.tokenBucket >= 1) {
        this.tokenBucket -= 1;
        const next = this.waitingQueue.shift();
        next?.resolve();
      }

      // If items remain, schedule the next drain at the precise
      // moment the next token becomes available.
      if (this.waitingQueue.length > 0) {
        const deficit = 1 - this.tokenBucket;
        const waitMs = Math.max(
          1,
          Math.ceil((deficit / this.refillRatePerSecond) * 1000),
        );

        this.scheduledDrainId = setTimeout(() => {
          this.scheduledDrainId = null;
          this.drain();
        }, waitMs);
      }
    } catch (error) {
      // On unexpected error, reject all pending to avoid hangs
      const err =
        error instanceof Error
          ? error
          : new Error("Rate limiter drain failed");

      while (this.waitingQueue.length > 0) {
        const item = this.waitingQueue.shift();
        item?.reject(err);
      }
    } finally {
      this.isDraining = false;
    }
  }
}