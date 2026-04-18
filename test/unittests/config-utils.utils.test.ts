import { describe, test, expect } from "bun:test";
import { deepMerge } from "../../src/lib/utils/config-utils";
import {
  type RuntimeConfig,
  type PartialRuntimeConfig,
} from "../../src/shared/types/index";

// ── Helpers ──

function createBaseConfig(): RuntimeConfig {
  return {
    llm: {
      provider: "openai",
      model: "gpt-4o-mini",
      baseUrl: "",
      apiKey: "sk-test-key",
      temperature: 0.8,
      maxTokens: 4096,
      topP: 0.9,
      frequencyPenalty: 0,
      presencePenalty: 0,
    },
    requestGuards: {
      retry: {
        maxRetries: 3,
        initialDelayMs: 1000,
        maxDelayMs: 10_000,
        factor: 2,
        jitter: true,
      },
      timeout: {
        timeoutMs: 30_000,
      },
      rateLimiter: {
        maxRequestsPerMinute: 20,
        maxQueueSize: 1000,
        requestTimeout: 30_000,
      },
    },
  };
}

// ── Tests ──

describe("deepMerge", () => {

  // ── Identity / no-op merges ──

  test("empty partial returns base config unchanged", () => {
    const base = createBaseConfig();
    const result = deepMerge(base, {});
    expect(result).toEqual(base);
  });

  test("partial with only undefined fields returns base unchanged", () => {
    const base = createBaseConfig();
    const partial: PartialRuntimeConfig = {
      llm: undefined,
      requestGuards: undefined,
    };
    const result = deepMerge(base, partial);
    expect(result).toEqual(base);
  });

  // ── LLM: single field overrides ──

  describe("LLM single-field overrides", () => {
    test("overrides temperature only", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, { llm: { temperature: 0.2 } });

      expect(result.llm.temperature).toBe(0.2);
      expect(result.llm.maxTokens).toBe(4096);
      expect(result.llm.provider).toBe("openai");
      expect(result.llm.model).toBe("gpt-4o-mini");
    });

    test("overrides apiKey only", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, { llm: { apiKey: "sk-new" } });

      expect(result.llm.apiKey).toBe("sk-new");
      expect(result.llm.provider).toBe("openai");
      expect(result.llm.temperature).toBe(0.8);
    });

    test("overrides model only", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, { llm: { model: "deepseek-chat" } });

      expect(result.llm.model).toBe("deepseek-chat");
      expect(result.llm.provider).toBe("openai");
    });
  });

  // ── LLM: multi-field overrides ──

  describe("LLM multi-field overrides", () => {
    test("overrides all generation fields at once", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        llm: {
          temperature: 1.5,
          maxTokens: 2048,
          topP: 0.5,
          frequencyPenalty: 0.3,
          presencePenalty: 0.1,
        },
      });

      expect(result.llm.temperature).toBe(1.5);
      expect(result.llm.maxTokens).toBe(2048);
      expect(result.llm.topP).toBe(0.5);
      expect(result.llm.frequencyPenalty).toBe(0.3);
      expect(result.llm.presencePenalty).toBe(0.1);
      expect(result.llm.provider).toBe("openai");
    });

    test("overrides provider + model + apiKey together", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        llm: {
          provider: "anthropic",
          model: "claude-3-5-sonnet",
          apiKey: "sk-ant",
        },
      });

      expect(result.llm.provider).toBe("anthropic");
      expect(result.llm.model).toBe("claude-3-5-sonnet");
      expect(result.llm.apiKey).toBe("sk-ant");
      expect(result.llm.temperature).toBe(0.8);
    });
  });

  // ── RequestGuards overrides ──

  describe("RequestGuards overrides", () => {
    test("overrides a single retry field", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        requestGuards: { retry: { maxRetries: 5 } },
      });

      expect(result.requestGuards.retry.maxRetries).toBe(5);
      expect(result.requestGuards.retry.initialDelayMs).toBe(1000);
      expect(result.requestGuards.retry.factor).toBe(2);
      expect(result.requestGuards.retry.jitter).toBe(true);
    });

    test("overrides all retry fields", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        requestGuards: {
          retry: {
            maxRetries: 5,
            initialDelayMs: 2000,
            maxDelayMs: 30_000,
            factor: 3,
            jitter: false,
          },
        },
      });

      expect(result.requestGuards.retry).toEqual({
        maxRetries: 5,
        initialDelayMs: 2000,
        maxDelayMs: 30_000,
        factor: 3,
        jitter: false,
      });
      expect(result.requestGuards.timeout.timeoutMs).toBe(30_000);
      expect(result.requestGuards.rateLimiter.maxRequestsPerMinute).toBe(20);
    });

    test("overrides timeout only", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        requestGuards: { timeout: { timeoutMs: 60_000 } },
      });

      expect(result.requestGuards.timeout.timeoutMs).toBe(60_000);
      expect(result.requestGuards.retry.maxRetries).toBe(3);
    });

    test("overrides partial rateLimiter fields", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        requestGuards: {
          rateLimiter: { maxRequestsPerMinute: 60, maxQueueSize: 500 },
        },
      });

      expect(result.requestGuards.rateLimiter.maxRequestsPerMinute).toBe(60);
      expect(result.requestGuards.rateLimiter.maxQueueSize).toBe(500);
      expect(result.requestGuards.rateLimiter.requestTimeout).toBe(30_000);
    });

    test("overrides across all three guard sections simultaneously", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        requestGuards: {
          retry: { maxRetries: 10 },
          timeout: { timeoutMs: 5_000 },
          rateLimiter: { maxRequestsPerMinute: 100 },
        },
      });

      expect(result.requestGuards.retry.maxRetries).toBe(10);
      expect(result.requestGuards.retry.initialDelayMs).toBe(1000);
      expect(result.requestGuards.timeout.timeoutMs).toBe(5_000);
      expect(result.requestGuards.rateLimiter.maxRequestsPerMinute).toBe(100);
      expect(result.requestGuards.rateLimiter.maxQueueSize).toBe(1000);
    });
  });

  // ── Cross-section: LLM + Guards ──

  test("overrides LLM and guards in a single partial", () => {
    const base = createBaseConfig();
    const result = deepMerge(base, {
      llm: { provider: "deepseek", temperature: 0.5 },
      requestGuards: {
        retry: { maxRetries: 1 },
        timeout: { timeoutMs: 10_000 },
      },
    });

    expect(result.llm.provider).toBe("deepseek");
    expect(result.llm.temperature).toBe(0.5);
    expect(result.llm.maxTokens).toBe(4096);
    expect(result.requestGuards.retry.maxRetries).toBe(1);
    expect(result.requestGuards.retry.factor).toBe(2);
    expect(result.requestGuards.timeout.timeoutMs).toBe(10_000);
    expect(result.requestGuards.rateLimiter.maxRequestsPerMinute).toBe(20);
  });

  // ── Chained merges (full pipeline simulation) ──

  describe("chained merges (Layer 0 → 1 → 2 → 3)", () => {
    test("later layers override earlier layers", () => {
      const base = createBaseConfig();

      // Layer 1: TOML
      const afterToml = deepMerge(base, {
        llm: { model: "gpt-4o-mini", temperature: 0.8, maxTokens: 4096 },
        requestGuards: {
          retry: { maxRetries: 3 },
          timeout: { timeoutMs: 30_000 },
        },
      });

      // Layer 2: ENV
      const afterEnv = deepMerge(afterToml, {
        llm: { provider: "openai", apiKey: "sk-prod" },
        requestGuards: { rateLimiter: { maxRequestsPerMinute: 50 } },
      });

      // Layer 3: CLI
      const final = deepMerge(afterEnv, {
        llm: { temperature: 0.2 },
      });

      expect(final.llm.temperature).toBe(0.2);
      expect(final.llm.provider).toBe("openai");
      expect(final.llm.apiKey).toBe("sk-prod");
      expect(final.llm.model).toBe("gpt-4o-mini");
      expect(final.llm.maxTokens).toBe(4096);
      expect(final.requestGuards.retry.maxRetries).toBe(3);
      expect(final.requestGuards.timeout.timeoutMs).toBe(30_000);
      expect(final.requestGuards.rateLimiter.maxRequestsPerMinute).toBe(50);
      expect(final.requestGuards.rateLimiter.maxQueueSize).toBe(1000);
    });

    test("each layer only affects its own fields", () => {
      const base = createBaseConfig();
      const l1 = deepMerge(base, { llm: { model: "claude-3-5-sonnet" } });
      const l2 = deepMerge(l1, { llm: { apiKey: "sk-ant" } });
      const l3 = deepMerge(l2, { llm: { temperature: 1.0 } });

      expect(l3.llm.model).toBe("claude-3-5-sonnet");
      expect(l3.llm.apiKey).toBe("sk-ant");
      expect(l3.llm.temperature).toBe(1.0);
      expect(l3.llm.topP).toBe(0.9);
      expect(l3.llm.frequencyPenalty).toBe(0);
    });
  });

  // ── Edge cases: falsy but valid values ──

  describe("falsy but valid values", () => {
    test("0 is preserved (not treated as undefined)", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        llm: { temperature: 0, frequencyPenalty: 0, presencePenalty: 0 },
        requestGuards: { retry: { maxRetries: 0 } },
      });

      expect(result.llm.temperature).toBe(0);
      expect(result.llm.frequencyPenalty).toBe(0);
      expect(result.llm.presencePenalty).toBe(0);
      expect(result.requestGuards.retry.maxRetries).toBe(0);
    });

    test("empty string is preserved", () => {
      const base = createBaseConfig();
      base.llm.baseUrl = "https://old-proxy.example.com";

      const result = deepMerge(base, { llm: { baseUrl: "" } });
      expect(result.llm.baseUrl).toBe("");
    });

    test("false is preserved (for jitter)", () => {
      const base = createBaseConfig();
      expect(base.requestGuards.retry.jitter).toBe(true);

      const result = deepMerge(base, {
        requestGuards: { retry: { jitter: false } },
      });
      expect(result.requestGuards.retry.jitter).toBe(false);
    });
  });

  // ── Immutability ──

  describe("immutability", () => {
    test("does not mutate the base config", () => {
      const base = createBaseConfig();
      const origTemp = base.llm.temperature;
      const origRetries = base.requestGuards.retry.maxRetries;

      deepMerge(base, {
        llm: { temperature: 999 },
        requestGuards: { retry: { maxRetries: 999 } },
      });

      expect(base.llm.temperature).toBe(origTemp);
      expect(base.requestGuards.retry.maxRetries).toBe(origRetries);
    });

    test("does not mutate the partial config", () => {
      const base = createBaseConfig();
      const partial: PartialRuntimeConfig = { llm: { temperature: 0.5 } };

      deepMerge(base, partial);

      expect(partial.llm?.temperature).toBe(0.5);
      expect(partial.requestGuards).toBeUndefined();
    });

    test("returned config is a new object reference", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {});

      expect(result).not.toBe(base);
      expect(result).toEqual(base);
    });

    test("nested objects in result are new references", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, { llm: { temperature: 0.5 } });

      expect(result.llm).not.toBe(base.llm);
      expect(result.requestGuards).not.toBe(base.requestGuards);
    });
  });

  // ── Structural completeness ──

  describe("structural completeness", () => {
    test("result always has both top-level keys", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, { llm: { temperature: 0.1 } });

      expect(result).toHaveProperty("llm");
      expect(result).toHaveProperty("requestGuards");
    });

    test("result.requestGuards always has all three guard sections", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, {
        requestGuards: { retry: { maxRetries: 1 } },
      });

      expect(result.requestGuards).toHaveProperty("retry");
      expect(result.requestGuards).toHaveProperty("timeout");
      expect(result.requestGuards).toHaveProperty("rateLimiter");
    });

    test("result.llm retains all fields after partial override", () => {
      const base = createBaseConfig();
      const result = deepMerge(base, { llm: { temperature: 0.1 } });

      const keys = Object.keys(result.llm);
      for (const k of [
        "provider", "model", "baseUrl", "apiKey",
        "temperature", "maxTokens", "topP",
        "frequencyPenalty", "presencePenalty",
      ]) {
        expect(keys).toContain(k);
      }
    });
  });
});