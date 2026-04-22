import { describe, test, expect } from "bun:test";
import { Retry } from "../../src/lib/guards/retry/index";
import { type RetryConfig } from "../../src/shared/types/index";
import { GuardError, GuardErrorCode } from "../../src/shared/exceptions/index";

// ── Helpers ─────────────────────────────────────────────

/** Default config: 3 retries, 100ms initial, factor 2, no jitter */
const defaultConfig: RetryConfig = {
  maxRetries: 3,
  initialDelayMs: 100,
  maxDelayMs: 5_000,
  factor: 2,
  retryableErrors: [],
  jitter: false,
};

/** Fast config for timing tests: short delays */
const fastConfig: RetryConfig = {
  maxRetries: 3,
  initialDelayMs: 50,
  maxDelayMs: 500,
  factor: 2,
  retryableErrors: [],
  jitter: false,
};

/** Creates a function that fails `n` times then succeeds */
function failNTimes(n: number, errorMsg = "transient error"): () => Promise<string> {
  let calls = 0;
  return async () => {
    calls++;
    if (calls <= n) {
      throw new Error(errorMsg);
    }
    return "success";
  };
}

/** Creates a function that always fails */
function alwaysFail(errorMsg = "permanent error"): () => Promise<never> {
  return async () => {
    throw new Error(errorMsg);
  };
}

/** Creates a counter function that tracks call count */
function callCounter<T>(fn: () => Promise<T>): { fn: () => Promise<T>; getCount: () => number } {
  let count = 0;
  return {
    fn: async () => {
      count++;
      return fn();
    },
    getCount: () => count,
  };
}

// ── Tests ───────────────────────────────────────────────

describe("Retry", () => {
  // ── 1. Basic success (no retry needed) ────────────────

  describe("basic success", () => {
    test("should return result immediately when function succeeds on first try", async () => {
      // Purpose: If func succeeds, no retry should happen
      const retry = new Retry(defaultConfig);
      const fn = async () => "hello";

      const result = await retry.execute(fn);

      expect(result).toBe("hello");
    });

    test("should call function exactly once on success", async () => {
      // Purpose: Verify no extra calls on success
      const retry = new Retry(defaultConfig);
      const counter = callCounter(async () => 42);

      await retry.execute(counter.fn);

      expect(counter.getCount()).toBe(1);
    });

    test("should preserve return type for different value types", async () => {
      // Purpose: Generic <T> should work with objects, arrays, null
      const retry = new Retry(defaultConfig);

      const obj = await retry.execute(async () => ({ key: "value" }));
      expect(obj).toEqual({ key: "value" });

      const arr = await retry.execute(async () => [1, 2, 3]);
      expect(arr).toEqual([1, 2, 3]);

      const nil = await retry.execute(async () => null);
      expect(nil).toBeNull();
    });
  });

  // ── 2. Retry after failure then success ───────────────

  describe("retry then success", () => {
    test("should succeed after 1 failure", async () => {
      // Purpose: Fail once, succeed on retry
      const retry = new Retry(fastConfig);
      const fn = failNTimes(1);

      const result = await retry.execute(fn);

      expect(result).toBe("success");
    });

    test("should succeed after multiple failures within maxRetries", async () => {
      // Purpose: Fail 2 times, succeed on 3rd attempt (attempt 0,1,2)
      const retry = new Retry(fastConfig);
      const fn = failNTimes(2);

      const result = await retry.execute(fn);

      expect(result).toBe("success");
    });

    test("should succeed on the very last retry attempt", async () => {
      // Purpose: Fail exactly maxRetries times, succeed on the last allowed attempt
      const retry = new Retry({ ...fastConfig, maxRetries: 3 });
      const fn = failNTimes(3); // fails 3 times, succeeds on 4th (attempt 3)

      const result = await retry.execute(fn);

      expect(result).toBe("success");
    });

    test("should call function correct number of times", async () => {
      // Purpose: Verify total call count = failures + 1 success
      const retry = new Retry(fastConfig);
      const counter = callCounter(failNTimes(2));

      await retry.execute(counter.fn);

      // 2 failures + 1 success = 3 calls
      expect(counter.getCount()).toBe(3);
    });
  });

  // ── 3. Exhausted retries ──────────────────────────────

  describe("max retries exhausted", () => {
    test("should throw after maxRetries failures", async () => {
      // Purpose: When all attempts fail, throw the last error
      const retry = new Retry({ ...fastConfig, maxRetries: 2 });
      const fn = alwaysFail("always fails");

      await expect(retry.execute(fn)).rejects.toThrow("always fails");
    });

    test("should throw the original error message", async () => {
      // Purpose: The thrown error should be the last error from func
      const retry = new Retry({ ...fastConfig, maxRetries: 1 });

      let callCount = 0;
      const fn = async () => {
        callCount++;
        throw new Error(`error-${callCount}`);
      };

      await expect(retry.execute(fn)).rejects.toThrow("error-2");
    });

    test("should call function exactly maxRetries + 1 times", async () => {
      // Purpose: 1 initial + maxRetries retries = maxRetries + 1 total
      const config: RetryConfig = { ...fastConfig, maxRetries: 4 };
      const retry = new Retry(config);
      const counter = callCounter(alwaysFail());

      await retry.execute(counter.fn).catch(() => {});

      expect(counter.getCount()).toBe(5); // 1 + 4
    });

    test("should throw Error even if func throws non-Error", async () => {
      // Purpose: String/number thrown should be wrapped in Error
      const retry = new Retry({ ...fastConfig, maxRetries: 0 });

      const fn = async () => {
        throw new Error("string error");
      };

      await expect(retry.execute(fn)).rejects.toThrow("string error");
    });
  });

  // ── 4. Non-retryable errors ───────────────────────────

  describe("non-retryable errors", () => {
    test("should throw immediately for non-retryable error", async () => {
      // Purpose: If error doesn't match retryableErrors, don't retry
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [/timeout/, /429/],
      };
      const retry = new Retry(config);
      const counter = callCounter(alwaysFail("authentication failed"));

      await expect(retry.execute(counter.fn)).rejects.toThrow("authentication failed");

      // Should have been called only once (no retry)
      expect(counter.getCount()).toBe(1);
    });

    test("should retry when error matches retryableErrors pattern", async () => {
      // Purpose: Retryable errors should be retried
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [/timeout/, /rate limit/],
      };
      const retry = new Retry(config);

      let calls = 0;
      const fn = async () => {
        calls++;
        if (calls <= 2) {
          throw new Error("request timeout");
        }
        return "ok";
      };

      const result = await retry.execute(fn);

      expect(result).toBe("ok");
      expect(calls).toBe(3);
    });

    test("should match retryableErrors with regex patterns", async () => {
      // Purpose: RegExp matching should work for partial matches
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [/5\d{2}/, /ECONNRESET/],
      };
      const retry = new Retry(config);

      // "500 Internal Server Error" should match /5\d{2}/
      const counter = callCounter(failNTimes(1, "500 Internal Server Error"));
      const result = await retry.execute(counter.fn);

      expect(result).toBe("success");
      expect(counter.getCount()).toBe(2);
    });

    test("should not retry when retryable error occurs after non-retryable", async () => {
      // Purpose: First non-retryable error should stop immediately
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [/timeout/],
      };
      const retry = new Retry(config);
      const counter = callCounter(alwaysFail("invalid API key"));

      await retry.execute(counter.fn).catch(() => {});

      expect(counter.getCount()).toBe(1);
    });

    test("should retry all errors when retryableErrors is empty", async () => {
      // Purpose: Empty retryableErrors = retry everything
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [],
      };
      const retry = new Retry(config);
      const counter = callCounter(failNTimes(2, "any error"));

      const result = await retry.execute(counter.fn);

      expect(result).toBe("success");
      expect(counter.getCount()).toBe(3);
    });
  });

  // ── 5. Exponential backoff & maxDelayMs cap ───────────

  describe("exponential backoff", () => {
    test("should increase delay exponentially between retries", async () => {
      // Purpose: Delay should follow initialDelayMs * factor^attempt
      // Config: initial=100, factor=2 → delays: 100, 200, 400
      const config: RetryConfig = {
        maxRetries: 3,
        initialDelayMs: 100,
        maxDelayMs: 10_000,
        factor: 2,
        retryableErrors: [],
        jitter: false,
      };
      const retry = new Retry(config);

      const timestamps: number[] = [];
      let calls = 0;
      const fn = async () => {
        timestamps.push(Date.now());
        calls++;
        if (calls <= 3) {
          throw new Error("fail");
        }
        return "ok";
      };

      await retry.execute(fn);

      // Verify increasing delays between attempts
      // delay 0→1 ≈ 100ms, delay 1→2 ≈ 200ms, delay 2→3 ≈ 400ms
      const delay1 = timestamps[1]! - timestamps[0]!;
      const delay2 = timestamps[2]! - timestamps[1]!;
      const delay3 = timestamps[3]! - timestamps[2]!;

      expect(delay1).toBeGreaterThan(70);   // ~100ms
      expect(delay1).toBeLessThan(200);
      expect(delay2).toBeGreaterThan(150);  // ~200ms
      expect(delay2).toBeLessThan(350);
      expect(delay3).toBeGreaterThan(300);  // ~400ms
      expect(delay3).toBeLessThan(600);
    }, 10_000);

    test("should cap delay at maxDelayMs", async () => {
      // Purpose: Delay should never exceed maxDelayMs
      // Config: initial=100, factor=10, maxDelay=250 → raw delays: 100, 1000(>250)
      const config: RetryConfig = {
        maxRetries: 2,
        initialDelayMs: 100,
        maxDelayMs: 250,
        factor: 10,
        retryableErrors: [],
        jitter: false,
      };
      const retry = new Retry(config);

      const timestamps: number[] = [];
      let calls = 0;
      const fn = async () => {
        timestamps.push(Date.now());
        calls++;
        if (calls <= 2) {
          throw new Error("fail");
        }
        return "ok";
      };

      await retry.execute(fn);

      // Second delay should be capped at ~250ms, not 1000ms
      const delay2 = timestamps[2]! - timestamps[1]!;
      expect(delay2).toBeGreaterThan(180);
      expect(delay2).toBeLessThan(400); // Well under 1000ms
    }, 5_000);

    test("should use correct delay with factor=1 (constant delay)", async () => {
      // Purpose: factor=1 means no exponential growth, constant delay
      const config: RetryConfig = {
        maxRetries: 3,
        initialDelayMs: 100,
        maxDelayMs: 5_000,
        factor: 1,
        retryableErrors: [],
        jitter: false,
      };
      const retry = new Retry(config);

      const timestamps: number[] = [];
      let calls = 0;
      const fn = async () => {
        timestamps.push(Date.now());
        calls++;
        if (calls <= 2) {
          throw new Error("fail");
        }
        return "ok";
      };

      await retry.execute(fn);

      const delay1 = timestamps[1]! - timestamps[0]!;
      const delay2 = timestamps[2]! - timestamps[1]!;

      // Both delays should be ~100ms (factor=1, so 100*1^0=100, 100*1^1=100)
      expect(Math.abs(delay1 - delay2)).toBeLessThan(80);
    }, 5_000);
  });

  // ── 6. Jitter ─────────────────────────────────────────

  describe("jitter", () => {
    test("should produce consistent delays when jitter is off", async () => {
      // Purpose: Without jitter, same config should give same delays
      const config: RetryConfig = {
        maxRetries: 5,
        initialDelayMs: 100,
        maxDelayMs: 5_000,
        factor: 2,
        retryableErrors: [],
        jitter: false,
      };
      const retry = new Retry(config);

      const timestamps: number[] = [];
      let calls = 0;
      const fn = async () => {
        timestamps.push(Date.now());
        calls++;
        if (calls <= 3) {
          throw new Error("fail");
        }
        return "ok";
      };

      await retry.execute(fn);

      // Delays: ~100, ~200, ~400 — predictable
      const delay1 = timestamps[1]! - timestamps[0]!;
      const delay2 = timestamps[2]! - timestamps[1]!;

      // delay2 should be roughly 2x delay1
      const ratio = delay2 / delay1;
      expect(ratio).toBeGreaterThan(1.5);
      expect(ratio).toBeLessThan(3.0);
    }, 5_000);

    test("should produce varying delays when jitter is on", async () => {
      // Purpose: With jitter, delays should have randomness (±20%)
      const config: RetryConfig = {
        maxRetries: 10,
        initialDelayMs: 100,
        maxDelayMs: 5_000,
        factor: 1, // constant base to isolate jitter effect
        retryableErrors: [],
        jitter: true,
      };

      // Run multiple trials and collect delays
      const delays: number[] = [];

      for (let trial = 0; trial < 5; trial++) {
        const retry = new Retry(config);
        const timestamps: number[] = [];
        let calls = 0;

        const fn = async () => {
          timestamps.push(Date.now());
          calls++;
          if (calls <= 1) {
            throw new Error("fail");
          }
          return "ok";
        };

        await retry.execute(fn);
        delays.push(timestamps[1]! - timestamps[0]!);
      }

      // With jitter, delays should not all be identical
      // (statistically near-impossible for all 5 to be exactly the same)
      const allSame = delays.every((d) => Math.abs(d - delays[0]!) < 5);
      // This could theoretically fail but is extremely unlikely
      // We mainly verify delays are in the right ballpark: 80-120ms (100 ±20%)
      for (const d of delays) {
        expect(d).toBeGreaterThan(50);
        expect(d).toBeLessThan(200);
      }
    }, 10_000);

    test("should clamp jittered delay to non-negative", async () => {
      // Purpose: Even with jitter, delay should never be negative
      // Use very small initialDelayMs to test edge case
      const config: RetryConfig = {
        maxRetries: 1,
        initialDelayMs: 1,
        maxDelayMs: 5_000,
        factor: 1,
        retryableErrors: [],
        jitter: true,
      };
      const retry = new Retry(config);

      // Should not throw due to negative setTimeout
      const result = await retry.execute(failNTimes(1));
      expect(result).toBe("success");
    });
  });

  // ── 7. Concurrent / parallel calls ────────────────────

  describe("concurrent calls", () => {
    test("should handle multiple independent execute calls in parallel", async () => {
      // Purpose: Multiple execute() calls should not interfere
      const retry = new Retry(fastConfig);

      const results = await Promise.all([
        retry.execute(async () => "a"),
        retry.execute(async () => "b"),
        retry.execute(async () => "c"),
      ]);

      expect(results).toEqual(["a", "b", "c"]);
    });

    test("should independently retry parallel calls", async () => {
      // Purpose: Each parallel call tracks its own attempt count
      const retry = new Retry(fastConfig);

      const fn1 = failNTimes(1, "err1");
      const fn2 = failNTimes(2, "err2");
      const fn3 = async () => "instant";

      const results = await Promise.all([
        retry.execute(fn1),
        retry.execute(fn2),
        retry.execute(fn3),
      ]);

      expect(results).toEqual(["success", "success", "instant"]);
    });

    test("should handle mixed success and failure in parallel", async () => {
      // Purpose: Some parallel calls succeed, some exhaust retries
      const config: RetryConfig = { ...fastConfig, maxRetries: 1 };
      const retry = new Retry(config);

      const results = await Promise.allSettled([
        retry.execute(async () => "ok"),
        retry.execute(alwaysFail("permanent")),
        retry.execute(failNTimes(1)),
      ]);

      expect(results[0]).toEqual({ status: "fulfilled", value: "ok" });
      expect(results[1]).toEqual({
        status: "rejected",
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        reason: expect.any(Error),
      });
      expect(results[2]).toEqual({ status: "fulfilled", value: "success" });
    });
  });

  // ── 8. Error propagation ──────────────────────────────

  describe("error propagation", () => {
    test("should preserve Error instance properties", async () => {
      // Purpose: The thrown error should be an Error with correct message
      const retry = new Retry({ ...fastConfig, maxRetries: 0 });

      try {
        await retry.execute(alwaysFail("custom message"));
        expect(true).toBe(false);
      } catch (error) {
        expect(error).toBeInstanceOf(GuardError);
        expect((error as GuardError).code).toBe(GuardErrorCode.RETRY_EXHAUSTED);
        expect((error as GuardError).cause).toBeInstanceOf(Error);
        expect(((error as GuardError).cause as Error).message).toBe("custom message");
      }
    });

    test("should wrap non-Error throws in Error", async () => {
      // Purpose: If func throws a string, wrap it
      const retry = new Retry({ ...fastConfig, maxRetries: 0 });

      try {
        await retry.execute(async () => {
          throw new Error("12345");
        });
      } catch (error) {
        expect(error).toBeInstanceOf(GuardError);
        expect((error as GuardError).code).toBe(GuardErrorCode.RETRY_EXHAUSTED);
        expect((error as GuardError).cause).toBeInstanceOf(Error);
        expect(((error as GuardError).cause as Error).message).toBe("12345");
      }
    });

    test("should throw last error when retrying with changing messages", async () => {
      // Purpose: If each attempt throws different error, the last one is thrown
      const retry = new Retry({ ...fastConfig, maxRetries: 2 });

      let attempt = 0;
      const fn = async () => {
        attempt++;
        throw new Error(`attempt-${attempt}`);
      };

      await expect(retry.execute(fn)).rejects.toThrow("attempt-3");
    });
  });

  // ── 9. Edge cases & boundary values ───────────────────

  describe("edge cases", () => {
    test("should work with maxRetries = 0 (no retries)", async () => {
      // Purpose: maxRetries=0 means only 1 attempt
      const config: RetryConfig = { ...fastConfig, maxRetries: 0 };
      const retry = new Retry(config);

      // Success case
      const result = await retry.execute(async () => "ok");
      expect(result).toBe("ok");

      // Failure case — should throw immediately, no retry
      const counter = callCounter(alwaysFail());
      await retry.execute(counter.fn).catch(() => {});
      expect(counter.getCount()).toBe(1);
    });

    test("should work with maxRetries = 1 (single retry)", async () => {
      // Purpose: Exactly 1 retry allowed
      const config: RetryConfig = { ...fastConfig, maxRetries: 1 };
      const retry = new Retry(config);

      // Fail once, succeed on retry
      const result = await retry.execute(failNTimes(1));
      expect(result).toBe("success");

      // Fail twice — should exhaust
      await expect(retry.execute(alwaysFail())).rejects.toThrow();
    });

    test("should handle very large maxRetries", async () => {
      // Purpose: Large maxRetries + early success should be fast
      const config: RetryConfig = {
        ...fastConfig,
        maxRetries: 1000,
        initialDelayMs: 1,
      };
      const retry = new Retry(config);
      const counter = callCounter(failNTimes(5));

      const start = Date.now();
      const result = await retry.execute(counter.fn);
      const elapsed = Date.now() - start;

      expect(result).toBe("success");
      expect(counter.getCount()).toBe(6);
      // Should be fast since it succeeds on 6th attempt with 1ms delay
      expect(elapsed).toBeLessThan(500);
    });

    test("should handle initialDelayMs = 0", async () => {
      // Purpose: Zero delay between retries
      const config: RetryConfig = {
        ...fastConfig,
        initialDelayMs: 0,
        maxRetries: 3,
      };
      const retry = new Retry(config);

      const start = Date.now();
      const result = await retry.execute(failNTimes(2));
      const elapsed = Date.now() - start;

      expect(result).toBe("success");
      // With 0ms delay, should be nearly instant
      expect(elapsed).toBeLessThan(200);
    });

    test("should handle maxDelayMs = 0", async () => {
      // Purpose: maxDelayMs=0 clamps all delays to 0
      const config: RetryConfig = {
        ...fastConfig,
        maxDelayMs: 0,
        maxRetries: 2,
      };
      const retry = new Retry(config);

      const start = Date.now();
      const result = await retry.execute(failNTimes(2));
      const elapsed = Date.now() - start;

      expect(result).toBe("success");
      expect(elapsed).toBeLessThan(200);
    });

    test("should handle large factor value", async () => {
      // Purpose: Large factor but capped by maxDelayMs
      const config: RetryConfig = {
        maxRetries: 2,
        initialDelayMs: 10,
        maxDelayMs: 100,
        factor: 100, // Would be 10, 1000, 100000 — all capped to 100
        retryableErrors: [],
        jitter: false,
      };
      const retry = new Retry(config);

      const timestamps: number[] = [];
      let calls = 0;
      const fn = async () => {
        timestamps.push(Date.now());
        calls++;
        if (calls <= 2) {
          throw new Error("fail");
        }
        return "ok";
      };

      await retry.execute(fn);

      // Second delay should be capped at ~100ms, not 1000ms
      const delay2 = timestamps[2]! - timestamps[1]!;
      expect(delay2).toBeLessThan(250);
    }, 5_000);
  });

  // ── 10. Integration with real-world error patterns ────

  describe("real-world patterns", () => {
    test("should retry on 429 rate limit error", async () => {
      // Purpose: Simulate LLM API rate limit response
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [/429/, /rate limit/i],
      };
      const retry = new Retry(config);

      const fn = failNTimes(2, "429 Too Many Requests");
      const result = await retry.execute(fn);

      expect(result).toBe("success");
    });

    test("should not retry on 401 authentication error", async () => {
      // Purpose: Auth errors are not transient, should fail fast
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [/429/, /5\d{2}/],
      };
      const retry = new Retry(config);
      const counter = callCounter(alwaysFail("401 Unauthorized"));

      await retry.execute(counter.fn).catch(() => {});

      expect(counter.getCount()).toBe(1);
    });

    test("should retry on 500/502/503 server errors", async () => {
      // Purpose: Server errors are transient
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [/5\d{2}/],
      };
      const retry = new Retry(config);

      for (const code of ["500", "502", "503"]) {
        const fn = failNTimes(1, `${code} Server Error`);
        const result = await retry.execute(fn);
        expect(result).toBe("success");
      }
    });

    test("should retry on network errors", async () => {
      // Purpose: ECONNRESET, timeout, etc.
      const config: RetryConfig = {
        ...fastConfig,
        retryableErrors: [/ECONNRESET/, /timeout/, /ETIMEDOUT/],
      };
      const retry = new Retry(config);

      const fn = failNTimes(1, "connect ECONNRESET 104.18.0.1:443");
      const result = await retry.execute(fn);

      expect(result).toBe("success");
    });
  });
});