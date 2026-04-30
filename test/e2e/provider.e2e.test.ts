// Usage instructions (requires setting corresponding env vars to execute):
//   E2E_OPENAI_API_KEY=sk-...   bun test test/e2e/
//   E2E_DEEPSEEK_API_KEY=sk-... bun test test/e2e/
//   E2E_OLLAMA_BASE_URL=http://localhost:11434 E2E_OLLAMA_MODEL=llama3 bun test test/e2e/

import { describe, test, expect } from "bun:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createProvider } from "../../src/core/llm/provider/registry";
import { OpenAIProvider } from "../../src/core/llm/provider/adapters/openai";

// Non-existent TOML path to make ConfigLoader skip the file layer
const NO_TOML = join(tmpdir(), "__e2e_no_config__.toml");

// Minimal token-consuming prompt
const PING_PROMPT = "Reply with exactly the single word 'pong' and nothing else.";

// ── Helpers ───

/**
 * Construct minimal env injected into ConfigLoader, disable retries for fast e2e failure.
 * In real API calls, no need for high retry counts—failure indicates config or network issues, 
 * should be exposed immediately.
 */
function makeEnv(overrides: Record<string, string>): NodeJS.ProcessEnv {
  return {
    AUTOCOMMIT_RETRY_MAX_RETRIES: "0",        // fail fast
    AUTOCOMMIT_TIMEOUT_MS: "30000",           // 30s per call
    ...overrides,
  };
}

/**
 * Detect if local service is reachable。
 */
async function isReachable(url: string, timeoutMs = 3000): Promise<boolean> {
  try {
    await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
    return true;
  } catch {
    return false;
  }
}

// ── OpenAI ────

const OPENAI_API_KEY = process.env.E2E_OPENAI_API_KEY ?? "";
const OPENAI_MODEL   = process.env.E2E_OPENAI_MODEL ?? "gpt-4o-mini";

(OPENAI_API_KEY ? describe : describe.skip)(
  "E2E: OpenAI provider",
  () => {
    test(
      "createProvider returns OpenAIProvider",
      async () => {
        const provider = await createProvider({
          argv: [],
          env: makeEnv({
            AUTOCOMMIT_PROVIDER: "openai",
            AUTOCOMMIT_MODEL: OPENAI_MODEL,
            AUTOCOMMIT_API_KEY: OPENAI_API_KEY,
          }),
          configFilePath: NO_TOML,
        });

        expect(provider).toBeInstanceOf(OpenAIProvider);
      },
      30_000,
    );

    test(
      "invoke returns non-empty string from real API",
      async () => {
        const provider = await createProvider({
          argv: [],
          env: makeEnv({
            AUTOCOMMIT_PROVIDER: "openai",
            AUTOCOMMIT_MODEL: OPENAI_MODEL,
            AUTOCOMMIT_API_KEY: OPENAI_API_KEY,
          }),
          configFilePath: NO_TOML,
        });

        const result = await provider.invoke(
          PING_PROMPT,
          { maxTokens: 20 },
        );

        expect(typeof result).toBe("string");
        expect(result.trim().length).toBeGreaterThan(0);
      },
      60_000,
    );

    test(
      "AbortSignal cancels in-flight request",
      async () => {
        const provider = await createProvider({
          argv: [],
          env: makeEnv({
            AUTOCOMMIT_PROVIDER: "openai",
            AUTOCOMMIT_MODEL: OPENAI_MODEL,
            AUTOCOMMIT_API_KEY: OPENAI_API_KEY,
            AUTOCOMMIT_TIMEOUT_MS: "60000",
          }),
          configFilePath: NO_TOML,
        });

        const controller = new AbortController();
        controller.abort(new Error("cancelled by test"));

        await expect(
          provider.invoke(PING_PROMPT, { maxTokens: 20 }, controller.signal),
        ).rejects.toMatchObject({ name: "ProviderError" });
      },
      15_000,
    );
  },
);

// ── DeepSeek ──

const DEEPSEEK_API_KEY = process.env.E2E_DEEPSEEK_API_KEY ?? "";
const DEEPSEEK_MODEL   = process.env.E2E_DEEPSEEK_MODEL ?? "deepseek-chat";

(DEEPSEEK_API_KEY ? describe : describe.skip)(
  "E2E: DeepSeek provider",
  () => {
    test(
      "createProvider returns OpenAIProvider",
      async () => {
        const provider = await createProvider({
          argv: [],
          env: makeEnv({
            AUTOCOMMIT_PROVIDER: "deepseek",
            AUTOCOMMIT_MODEL: DEEPSEEK_MODEL,
            AUTOCOMMIT_API_KEY: DEEPSEEK_API_KEY,
          }),
          configFilePath: NO_TOML,
        });

        expect(provider).toBeInstanceOf(OpenAIProvider);
      },
      30_000,
    );

    test(
      "invoke returns non-empty string from real API",
      async () => {
        const provider = await createProvider({
          argv: [],
          env: makeEnv({
            AUTOCOMMIT_PROVIDER: "deepseek",
            AUTOCOMMIT_MODEL: DEEPSEEK_MODEL,
            AUTOCOMMIT_API_KEY: DEEPSEEK_API_KEY,
          }),
          configFilePath: NO_TOML,
        });

        const result = await provider.invoke(
          PING_PROMPT,
          { maxTokens: 20 },
        );

        expect(typeof result).toBe("string");
        expect(result.trim().length).toBeGreaterThan(0);
      },
      60_000,
    );

    test(
      "uses correct DeepSeek baseUrl (api.deepseek.com)",
      async () => {
        const provider = await createProvider({
          argv: [],
          env: makeEnv({
            AUTOCOMMIT_PROVIDER: "deepseek",
            AUTOCOMMIT_MODEL: DEEPSEEK_MODEL,
            AUTOCOMMIT_API_KEY: DEEPSEEK_API_KEY,
          }),
          configFilePath: NO_TOML,
        });

        const result = await provider.invoke(PING_PROMPT, { maxTokens: 20 });
        expect(result.trim().length).toBeGreaterThan(0);
      },
      60_000,
    );
  },
);

// ── Ollama ────

const OLLAMA_BASE_URL = process.env.E2E_OLLAMA_BASE_URL ?? "http://localhost:11434";
const OLLAMA_MODEL    = process.env.E2E_OLLAMA_MODEL ?? "llama3";

describe("E2E: Ollama provider", () => {
  test(
    "createProvider returns OpenAIProvider and invoke returns non-empty string",
    async () => {
      const reachable = await isReachable(`${OLLAMA_BASE_URL}/api/tags`);
      if (!reachable) {
        console.log(`[e2e] skipped: Ollama not reachable at ${OLLAMA_BASE_URL}`);
        return;
      }

      const provider = await createProvider({
        argv: [],
        env: makeEnv({
          AUTOCOMMIT_PROVIDER: "ollama",
          AUTOCOMMIT_MODEL: OLLAMA_MODEL,
          AUTOCOMMIT_BASE_URL: `${OLLAMA_BASE_URL}/v1`,
          // Ollama do not need apiKey
        }),
        configFilePath: NO_TOML,
      });

      expect(provider).toBeInstanceOf(OpenAIProvider);

      const result = await provider.invoke(PING_PROMPT, { maxTokens: 20 });

      expect(typeof result).toBe("string");
      expect(result.trim().length).toBeGreaterThan(0);
    },
    120_000, // Local model first inference may be slower
  );

  test(
    "ollama with explicit apiKey override still works",
    async () => {
      const reachable = await isReachable(`${OLLAMA_BASE_URL}/api/tags`);
      if (!reachable) {
        console.log(`[e2e] skipped: Ollama not reachable at ${OLLAMA_BASE_URL}`);
        return;
      }

      const provider = await createProvider({
        argv: [],
        env: makeEnv({
          AUTOCOMMIT_PROVIDER: "ollama",
          AUTOCOMMIT_MODEL: OLLAMA_MODEL,
          AUTOCOMMIT_BASE_URL: `${OLLAMA_BASE_URL}/v1`,
          AUTOCOMMIT_API_KEY: "any-placeholder-key",
        }),
        configFilePath: NO_TOML,
      });

      const result = await provider.invoke(PING_PROMPT, { maxTokens: 20 });
      expect(result.trim().length).toBeGreaterThan(0);
    },
    120_000,
  );
});