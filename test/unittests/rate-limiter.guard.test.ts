import { describe, test, expect} from "bun:test";
import { RateLimiter } from "../../src/lib/guards/rate-limit/index";
import { type RateLimiterConfig } from "../../src/shared/types/index";

// ── Helpers ──────────────

/** Default config: 60 RPM, queue 10, timeout 5s */
const defaultConfig: RateLimiterConfig = {
  maxRequestsPerMinute: 60,
  maxQueueSize: 10,
  requestTimeout: 5_000,
};

/** Fast config for time-sensitive tests: 6 RPM = 1 token/10s */
const slowRefillConfig: RateLimiterConfig = {
  maxRequestsPerMinute: 6,
  maxQueueSize: 10,
  requestTimeout: 5_000,
};


describe("RateLimiter", () => {
  // ── 1. Initialization ──

  describe("initialization", () => {
    test("should initialize with full token bucket", () => {
      // Purpose: Verify that the bucket starts full (maxTokens = maxRequestsPerMinute)
      const limiter = new RateLimiter(defaultConfig);

      expect(limiter.availableTokens).toBe(60);
      expect(limiter.queueLength).toBe(0);
    });

    test("should initialize with correct values for different configs", () => {
      // Purpose: Verify config values are correctly assigned
      const limiter = new RateLimiter(slowRefillConfig);

      expect(limiter.availableTokens).toBe(6);
      expect(limiter.queueLength).toBe(0);
    });

    test("should initialize with minimal config (1 RPM)", () => {
      // Purpose: Edge case — minimum meaningful config
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 1,
        maxQueueSize: 1,
        requestTimeout: 1_000,
      };
      const limiter = new RateLimiter(config);

      expect(limiter.availableTokens).toBe(1);
    });
  });

  // ── 2. Basic acquire ─-

  describe("basic acquire", () => {
    test("should resolve immediately when tokens are available", async () => {
      // Purpose: First call with full bucket should not block
      const limiter = new RateLimiter(defaultConfig);

      const start = Date.now();
      await limiter.acquire();
      const elapsed = Date.now() - start;

      // Should complete nearly instantly (< 50ms with setTimeout(0) overhead)
      expect(elapsed).toBeLessThan(100);
      expect(limiter.availableTokens).toBeLessThanOrEqual(59);
    });

    test("should consume one token per acquire", async () => {
      // Purpose: Each acquire should decrement the bucket by 1
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 5,
        maxQueueSize: 10,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      await limiter.acquire();
      await limiter.acquire();
      await limiter.acquire();

      // 5 - 3 = 2 (approximately, may have tiny refill)
      expect(limiter.availableTokens).toBeLessThanOrEqual(2.1);
      expect(limiter.availableTokens).toBeGreaterThanOrEqual(1.9);
    });

    test("should drain all tokens when burst requests equal bucket size", async () => {
      // Purpose: Exhaust entire bucket with burst
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 3,
        maxQueueSize: 10,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      await Promise.all([
        limiter.acquire(),
        limiter.acquire(),
        limiter.acquire(),
      ]);

      // All 3 tokens consumed, bucket should be near 0
      expect(limiter.availableTokens).toBeLessThan(1);
    });
  });

  // ── 3. Token refill ────

  describe("token refill", () => {
    test("should refill tokens over time", async () => {
      // Purpose: After consuming tokens, waiting should refill them
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 60, // 1 token/second
        maxQueueSize: 10,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      // Consume all tokens
      for (let i = 0; i < 60; i++) {
        await limiter.acquire();
      }

      // Wait 1 second — should refill ~1 token
      await Bun.sleep(1_100);

      // availableTokens getter doesn't trigger refill (it's passive),
      // but acquire does via drain → refillTokens
      // We test by successfully acquiring after waiting
      await limiter.acquire();

      // If we got here without timeout, refill worked
      expect(true).toBe(true);
    }, 10_000);

    test("should not refill beyond maxTokens", async () => {
      // Purpose: Bucket should cap at maxTokens even after long idle
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 5,
        maxQueueSize: 10,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      // Wait some time (bucket is already full)
      await Bun.sleep(500);

      // Trigger a drain to force refill calculation
      await limiter.acquire();

      // Even after idle, previous availableTokens should not exceed 5
      // (consumed 1, so ~4 + tiny refill ≈ 4.x, capped at 5)
      expect(limiter.availableTokens).toBeLessThanOrEqual(5);
    });
  });

  // ── 4. Blocking / waiting ─────────────────────────────

  describe("blocking wait", () => {
    test("should block and wait when no tokens are available", async () => {
      // Purpose: When bucket is empty, acquire should wait for refill
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 60, // 1 token/second
        maxQueueSize: 65,
        requestTimeout: 10_000,
      };
      const limiter = new RateLimiter(config);

      // Exhaust all tokens
      const promises: Promise<void>[] = [];
      for (let i = 0; i < 60; i++) {
        promises.push(limiter.acquire());
      }
      await Promise.all(promises);

      // Next acquire should be delayed (~1 second for 1 token)
      const start = Date.now();
      await limiter.acquire();
      const elapsed = Date.now() - start;

      // Should wait approximately 1 second (tolerance: 500ms ~ 2500ms)
      expect(elapsed).toBeGreaterThan(500);
      expect(elapsed).toBeLessThan(3_000);
    }, 15_000);

    test("should queue requests in FIFO order", async () => {
      // Purpose: Earlier requests should resolve before later ones
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 60,
        maxQueueSize: 10,
        requestTimeout: 10_000,
      };
      const limiter = new RateLimiter(config);

      // Exhaust tokens
      for (let i = 0; i < 60; i++) {
        await limiter.acquire();
      }

      // Queue 3 requests — they should resolve in order
      const order: number[] = [];

      const p1 = limiter.acquire().then(() => order.push(1));
      const p2 = limiter.acquire().then(() => order.push(2));
      const p3 = limiter.acquire().then(() => order.push(3));

      await Promise.all([p1, p2, p3]);

      expect(order).toEqual([1, 2, 3]);
    }, 15_000);
  });

  // ── 5. Concurrent requests ────────────────────────────

  describe("concurrent requests", () => {
    test("should handle multiple concurrent acquires correctly", async () => {
      // Purpose: Many parallel acquires should all eventually resolve
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 60,
        maxQueueSize: 20,
        requestTimeout: 10_000,
      };
      const limiter = new RateLimiter(config);

      // Fire 10 concurrent requests
      const results = await Promise.all(
        Array.from({ length: 10 }, () => limiter.acquire().then(() => "ok")),
      );

      expect(results).toHaveLength(10);
      expect(results.every((r) => r === "ok")).toBe(true);
    });

    test("should not exceed rate over time with sustained requests", async () => {
      // Purpose: Verify rate limiting actually throttles sustained load
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 30, // 0.5 token/second
        maxQueueSize: 35,
        requestTimeout: 15_000,
      };
      const limiter = new RateLimiter(config);

      const start = Date.now();
      // Request 32 tokens (30 burst + 2 must wait for refill)
      const promises = Array.from({ length: 32 }, () => limiter.acquire());
      await Promise.all(promises);
      const elapsed = Date.now() - start;

      // 30 tokens burst instantly; 2 additional need refill
      // At 0.5 token/sec, ~4 seconds for 2 tokens
      expect(elapsed).toBeGreaterThan(2_000);
    }, 20_000);
  });

  // ── 6. Queue full (backpressure) ──────────────────────

  describe("queue full", () => {
    test("should throw when queue is full", async () => {
      // Purpose: Backpressure — reject immediately when queue overflows
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 1,
        maxQueueSize: 2,
        requestTimeout: 10_000,
      };
      const limiter = new RateLimiter(config);

      // Consume the only token
      await limiter.acquire();

      // Queue 2 requests (fills the queue)
      limiter.acquire().catch(() => {}); // queued #1
      limiter.acquire().catch(() => {}); // queued #2

      // Wait for queue to fill
      await Bun.sleep(10);

      // 3rd queued request should throw
      expect(() => limiter.acquire()).toThrow(
        "Rate limiter queue is full (2 requests)",
      );
    });

    test("should report correct queueLength when full", async () => {
      // Purpose: Verify queueLength getter reflects actual state
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 1,
        maxQueueSize: 3,
        requestTimeout: 10_000,
      };
      const limiter = new RateLimiter(config);

      await limiter.acquire();

      limiter.acquire().catch(() => {});
      limiter.acquire().catch(() => {});

      await Bun.sleep(10);

      expect(limiter.queueLength).toBe(2);
    });
  });

  // ── 7. Timeout ─────────

  describe("timeout", () => {
    test("should reject with timeout error when request waits too long", async () => {
      // Purpose: requests that exceed requestTimeout should be rejected
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 1, // very slow refill
        maxQueueSize: 5,
        requestTimeout: 300, // 300ms timeout
      };
      const limiter = new RateLimiter(config);

      // Consume the only token
      await limiter.acquire();

      // Next request should timeout (need ~60s for next token, but timeout is 300ms)
      expect(limiter.acquire()).rejects.toThrow(
        "Request timeout after 300ms",
      );
    }, 5_000);

    test("should remove timed-out request from queue", async () => {
      // Purpose: After timeout, the request should be cleaned from the queue
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 1,
        maxQueueSize: 5,
        requestTimeout: 200,
      };
      const limiter = new RateLimiter(config);

      await limiter.acquire();

      // Queue a request that will timeout
      const promise = limiter.acquire().catch(() => {});
      await Bun.sleep(10);
      expect(limiter.queueLength).toBe(1);

      // Wait for timeout
      await promise;
      await Bun.sleep(50);

      expect(limiter.queueLength).toBe(0);
    }, 5_000);

    test("should reject multiple timed-out requests independently", async () => {
      // Purpose: Each request has its own timeout timer
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 1,
        maxQueueSize: 5,
        requestTimeout: 300,
      };
      const limiter = new RateLimiter(config);

      await limiter.acquire();

      const errors: string[] = [];

      const p1 = limiter.acquire().catch((e: Error) => errors.push(e.message));
      const p2 = limiter.acquire().catch((e: Error) => errors.push(e.message));

      await Promise.all([p1, p2]);

      expect(errors).toHaveLength(2);
      expect(errors[0]).toContain("timeout");
      expect(errors[1]).toContain("timeout");
    }, 5_000);
  });

  // ── 8. Edge cases & boundary values ───────────────────

  describe("edge cases", () => {
    test("should handle maxRequestsPerMinute = 1", async () => {
      // Purpose: Smallest meaningful rate — only 1 burst token
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 1,
        maxQueueSize: 5,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      // First acquire should succeed
      await limiter.acquire();
      expect(limiter.availableTokens).toBeLessThan(1);
    });

    test("should handle maxQueueSize = 1", async () => {
      // Purpose: Minimal queue — only 1 request can wait
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 1,
        maxQueueSize: 1,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      await limiter.acquire();

      // Queue 1 request (fills the queue)
      limiter.acquire().catch(() => {});
      await Bun.sleep(10);

      // Next should throw immediately
      expect(() => limiter.acquire()).toThrow("Rate limiter queue is full");
    });

    test("should handle very high RPM", async () => {
      // Purpose: Large burst capacity should work without issues
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 10_000,
        maxQueueSize: 100,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      // Acquire 100 tokens rapidly
      const promises = Array.from({ length: 100 }, () => limiter.acquire());
      await Promise.all(promises);

      expect(limiter.availableTokens).toBeLessThanOrEqual(9_900);
    });

    test("should handle rapid sequential acquires", async () => {
      // Purpose: Sequential (not parallel) rapid acquires
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 60,
        maxQueueSize: 10,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      for (let i = 0; i < 20; i++) {
        await limiter.acquire();
      }

      // All 20 should have succeeded (60 RPM = plenty of burst)
      expect(limiter.availableTokens).toBeLessThanOrEqual(40.1);
    });
  });

  // ── 9. Drain scheduling behavior ─────────────────────

  describe("drain scheduling", () => {
    test("should schedule drain asynchronously", async () => {
      // Purpose: drain runs via setTimeout(0), not synchronously
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 60,
        maxQueueSize: 10,
        requestTimeout: 5_000,
      };
      const limiter = new RateLimiter(config);

      // Queue a request — it should be pending (not yet resolved)
      // because drain is scheduled asynchronously
      let resolved = false;
      const promise = limiter.acquire().then(() => {
        resolved = true;
      });

      // At this point (synchronously), the drain hasn't fired yet
      // but the queueLength should be 1
      expect(limiter.queueLength).toBe(1);

      // After awaiting, drain has run
      await promise;
      expect(resolved).toBe(true);
      expect(limiter.queueLength).toBe(0);
    });

    test("should re-schedule drain when queue has remaining items", async () => {
      // Purpose: After partial drain, next drain should be scheduled
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 60,
        maxQueueSize: 100,
        requestTimeout: 15_000,
      };
      const limiter = new RateLimiter(config);

      // Exhaust bucket + queue extras
      const allPromises: Promise<void>[] = [];
      for (let i = 0; i < 63; i++) {
        allPromises.push(limiter.acquire());
      }

      // All should eventually resolve (60 burst + 3 via refill)
      await Promise.all(allPromises);

      // If we reach here, the drain successfully re-scheduled itself
      expect(true).toBe(true);
    }, 15_000);
  });

  // ── 10. Mixed scenarios (chain/integration) ───────────

  describe("mixed scenarios", () => {
    test("should handle acquire-timeout-acquire cycle", async () => {
      // Purpose: After a timeout, subsequent acquires should still work
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 60,
        maxQueueSize: 5,
        requestTimeout: 200,
      };
      const limiter = new RateLimiter(config);

      // Exhaust tokens
      for (let i = 0; i < 60; i++) {
        await limiter.acquire();
      }

      // This should timeout
      await limiter.acquire().catch(() => {});

      // Wait for refill
      await Bun.sleep(1_200);

      // Should succeed again after refill
      await limiter.acquire();
      expect(true).toBe(true);
    }, 10_000);

    test("should handle interleaved success and timeout", async () => {
      // Purpose: Some requests succeed, others timeout, limiter stays healthy
      const config: RateLimiterConfig = {
        maxRequestsPerMinute: 2,
        maxQueueSize: 10,
        requestTimeout: 500,
      };
      const limiter = new RateLimiter(config);

      const results: string[] = [];

      // 2 should succeed (burst), 2 should timeout (slow refill)
      const p1 = limiter.acquire().then(() => results.push("ok"));
      const p2 = limiter.acquire().then(() => results.push("ok"));
      const p3 = limiter
        .acquire()
        .then(() => results.push("ok"))
        .catch(() => results.push("timeout"));
      const p4 = limiter
        .acquire()
        .then(() => results.push("ok"))
        .catch(() => results.push("timeout"));

      await Promise.all([p1, p2, p3, p4]);

      const okCount = results.filter((r) => r === "ok").length;
      const timeoutCount = results.filter((r) => r === "timeout").length;

      // At least 2 should succeed (burst tokens)
      expect(okCount).toBeGreaterThanOrEqual(2);
      // At least some might timeout (depends on timing)
      expect(okCount + timeoutCount).toBe(4);
    }, 5_000);

    test("should work correctly with multiple limiter instances", async () => {
      // Purpose: Instances should be independent (no shared state)
      const limiterA = new RateLimiter({
        maxRequestsPerMinute: 2,
        maxQueueSize: 5,
        requestTimeout: 5_000,
      });
      const limiterB = new RateLimiter({
        maxRequestsPerMinute: 100,
        maxQueueSize: 5,
        requestTimeout: 5_000,
      });

      // Exhaust A
      await limiterA.acquire();
      await limiterA.acquire();

      // B should still have plenty of tokens
      await limiterB.acquire();
      expect(limiterB.availableTokens).toBeGreaterThan(90);
      expect(limiterA.availableTokens).toBeLessThan(1);
    });
  });
});