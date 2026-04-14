import { describe, test, expect } from "bun:test";
import { Timeout } from "../../src/common/guards/llm/timeout";
import { type TimeoutConfig } from "../../src/common/types/index";

// ── Helpers ─────────────────────────────────────────────

/** Default config: 1 second timeout */
const defaultConfig: TimeoutConfig = {
  timeoutMs: 1_000,
};

/** Short config: 100ms timeout for fast tests */
const shortConfig: TimeoutConfig = {
  timeoutMs: 100,
};

/** Creates a function that resolves with `value` immediately */
function immediate<T>(value: T): (signal: AbortSignal) => Promise<T> {
  return async (_signal) => value;
}

/** Creates a function that resolves with `value` after `ms` milliseconds */
function delayedSuccess<T>(value: T, ms: number): (signal: AbortSignal) => Promise<T> {
  return (_signal) =>
    new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

/** Creates a function that rejects with `error` after `ms` milliseconds */
function delayedThrow(error: Error, ms: number): (signal: AbortSignal) => Promise<never> {
  return (_signal) =>
    new Promise((_, reject) => setTimeout(() => reject(error), ms));
}

/** Creates a function that throws `error` immediately */
function immediateThrow(error: Error): (signal: AbortSignal) => Promise<never> {
  return async (_signal) => {
    throw error;
  };
}

// ── Tests ───────────────────────────────────────────────

describe("Timeout", () => {
  // ── 1. Basic success ─────────────────────────────────

  describe("basic success", () => {
    test("should return result when function completes within timeout", async () => {
      // Purpose: Function that finishes before timeout should succeed normally
      const timeout = new Timeout(shortConfig);

      const result = await timeout.execute(immediate("hello"));

      expect(result).toBe("hello");
    });

    test("should preserve return type for different value types", async () => {
      // Purpose: Generic <T> should work with string, number, object, array, null
      const timeout = new Timeout(shortConfig);

      expect(await timeout.execute(immediate("string"))).toBe("string");
      expect(await timeout.execute(immediate(42))).toBe(42);
      expect(await timeout.execute(immediate({ key: "value" }))).toEqual({ key: "value" });
      expect(await timeout.execute(immediate([1, 2, 3]))).toEqual([1, 2, 3]);
      expect(await timeout.execute(immediate(null))).toBeNull();
    });

    test("should succeed when function completes just before deadline", async () => {
      // Purpose: 50ms function with 100ms timeout should succeed
      const timeout = new Timeout(shortConfig); // 100ms

      const result = await timeout.execute(delayedSuccess("done", 50));

      expect(result).toBe("done");
    });
  });

  // ── 2. Timeout behavior ──────────────────────────────

  describe("timeout behavior", () => {
    test("should throw when function exceeds timeout", async () => {
      // Purpose: 300ms function with 100ms timeout must be rejected
      const timeout = new Timeout(shortConfig);

      await expect(timeout.execute(delayedSuccess("too late", 300))).rejects.toThrow();
    });

    test("should throw error message containing the configured timeoutMs value", async () => {
      // Purpose: Error message should embed the timeout duration for debugging
      const timeout = new Timeout({ timeoutMs: 50 });

      await expect(timeout.execute(delayedSuccess("too late", 300))).rejects.toThrow("50ms");
    });

    test("should throw an Error instance on timeout", async () => {
      // Purpose: Thrown value should be instanceof Error, not a plain string/object
      const timeout = new Timeout(shortConfig);

      let caughtError: unknown;
      try {
        await timeout.execute(delayedSuccess("too late", 300));
      } catch (e) {
        caughtError = e;
      }

      expect(caughtError).toBeInstanceOf(Error);
    });

    test("should reject within approximately the configured timeout duration", async () => {
      // Purpose: Timeout must fire close to timeoutMs, not significantly later
      const timeoutMs = 100;
      const timeout = new Timeout({ timeoutMs });

      const start = Date.now();
      await timeout.execute(delayedSuccess("too late", 2_000)).catch(() => {});
      const elapsed = Date.now() - start;

      // Should reject around 100ms, allow ±80ms tolerance for event loop variance
      expect(elapsed).toBeGreaterThanOrEqual(80);
      expect(elapsed).toBeLessThan(300);
    });
  });

  // ── 3. AbortSignal ───────────────────────────────────

  describe("AbortSignal", () => {
    test("should pass a valid AbortSignal instance to the function", async () => {
      // Purpose: The signal parameter must be a real AbortSignal
      const timeout = new Timeout(defaultConfig);
      let capturedSignal: AbortSignal | null = null;

      await timeout.execute(async (signal) => {
        capturedSignal = signal;
        return "done";
      });

      expect(capturedSignal).toBeInstanceOf(AbortSignal);
    });

    test("should pass a non-aborted signal when the function starts", async () => {
      // Purpose: Signal must be active (not pre-aborted) when func begins executing
      const timeout = new Timeout(defaultConfig);
      let signalAbortedAtStart = true;

      await timeout.execute(async (signal) => {
        signalAbortedAtStart = signal.aborted;
        return "done";
      });

      expect(signalAbortedAtStart).toBe(false);
    });

    test("should abort the signal when timeout fires", async () => {
      // Purpose: After timeout, signal.aborted must be true so callers can react
      const timeout = new Timeout({ timeoutMs: 50 });
      let capturedSignal: AbortSignal | null = null;

      await timeout.execute(async (signal) => {
        capturedSignal = signal;
        await Bun.sleep(300);
        return "done";
      }).catch(() => {});

      expect(capturedSignal!.aborted).toBe(true);
    });

    test("should allow func to detect abort via signal.aborted", async () => {
      // Purpose: Function can observe AbortSignal when timeout fires
      const timeout = new Timeout({ timeoutMs: 100 });
      let detectedAbort = false;

      let resolveAbortObserved: (() => void) | null = null;
      const abortObserved = new Promise<void>((resolve) => {
        resolveAbortObserved = resolve;
      });

      await timeout.execute(async (signal) => {
        signal.addEventListener(
          "abort",
          () => {
            detectedAbort = true;
            resolveAbortObserved?.();
          },
          { once: true },
        );

        // Keep the task alive long enough so timeout can fire
        await Bun.sleep(300);
        return "done";
      })
      .catch(() => {});

      // Wait until abort side-effect is definitely observed
      await abortObserved;

      expect(detectedAbort).toBe(true);
    });
  });

  // ── 4. Error propagation ─────────────────────────────

  describe("error propagation", () => {
    test("should propagate error when function throws before timeout", async () => {
      // Purpose: Early function error should be thrown as-is, not wrapped as timeout error
      const timeout = new Timeout(defaultConfig); // 1000ms

      await expect(timeout.execute(immediateThrow(new Error("user error")))).rejects.toThrow(
        "user error",
      );
    });

    test("should propagate error thrown with a short delay before timeout", async () => {
      // Purpose: Error at 50ms with 300ms timeout should preserve the original error
      const timeout = new Timeout({ timeoutMs: 300 });

      await expect(
        timeout.execute(delayedThrow(new Error("early failure"), 50)),
      ).rejects.toThrow("early failure");
    });

    test("should propagate the exact error instance thrown by the function", async () => {
      // Purpose: Error identity (===) should be preserved, not cloned or wrapped
      const timeout = new Timeout(defaultConfig);
      const originalError = new Error("original");

      let caughtError: unknown;
      try {
        await timeout.execute(immediateThrow(originalError));
      } catch (e) {
        caughtError = e;
      }

      expect(caughtError).toBe(originalError);
    });

    test("should not report a timeout error when func fails before the deadline", async () => {
      // Purpose: A fast failure must not be mistaken for a timeout
      const timeout = new Timeout({ timeoutMs: 300 });

      let errorMessage = "";
      try {
        await timeout.execute(delayedThrow(new Error("fast failure"), 30));
      } catch (e) {
        if (e instanceof Error) errorMessage = e.message;
      }

      expect(errorMessage).toBe("fast failure");
      expect(errorMessage).not.toContain("timed out");
    });
  });

  // ── 5. Resource cleanup ──────────────────────────────

  describe("resource cleanup", () => {
    test("should not abort the signal after successful completion", async () => {
      // Purpose: clearTimeout must run in finally so the timer never fires after success
      const timeout = new Timeout(defaultConfig);
      let signalAbortedAfterReturn = false;

      await timeout.execute(async (signal) => {
        await Bun.sleep(10);
        // Check signal state just before returning
        signalAbortedAfterReturn = signal.aborted;
        return "done";
      });

      expect(signalAbortedAfterReturn).toBe(false);
    });

    test("should not abort the signal when function throws early", async () => {
      // Purpose: clearTimeout in finally must cancel the timer even on early rejection
      const timeout = new Timeout(defaultConfig);
      let capturedSignal: AbortSignal | null = null;

      await timeout
        .execute(async (signal) => {
          capturedSignal = signal;
          throw new Error("early");
        })
        .catch(() => {});

      // Give a small window to verify the timer did NOT fire afterward
      await Bun.sleep(20);

      expect(capturedSignal!.aborted).toBe(false);
    });
  });

  // ── 6. Edge cases ────────────────────────────────────

  describe("edge cases", () => {
    test("should time out with very short timeoutMs (1ms)", async () => {
      // Purpose: Even at minimum granularity, timeout must still fire
      const timeout = new Timeout({ timeoutMs: 1 });

      await expect(timeout.execute(delayedSuccess("late", 500))).rejects.toThrow();
    });

    test("should handle multiple independent executions concurrently", async () => {
      // Purpose: Each execute() creates its own controller+timer, no shared state
      const timeout = new Timeout(shortConfig);

      const results = await Promise.all([
        timeout.execute(immediate("a")),
        timeout.execute(immediate("b")),
        timeout.execute(immediate("c")),
      ]);

      expect(results).toEqual(["a", "b", "c"]);
    });

    test("should correctly settle concurrent mixed fast and slow executions", async () => {
      // Purpose: Fast calls succeed, slow calls time out — entirely independently
      const timeout = new Timeout(shortConfig); // 100ms

      const results = await Promise.allSettled([
        timeout.execute(immediate("fast")),             // succeeds
        timeout.execute(delayedSuccess("slow", 300)),   // times out
      ]);

      expect(results[0].status).toBe("fulfilled");
      expect(results[1].status).toBe("rejected");
    });

    test("should work correctly with a very large timeoutMs", async () => {
      // Purpose: Large timeout value should not cause issues; fast func still succeeds
      const timeout = new Timeout({ timeoutMs: 60_000 });

      const result = await timeout.execute(immediate("fast"));

      expect(result).toBe("fast");
    });
  });
});