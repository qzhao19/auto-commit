import { beforeEach, afterEach, describe, expect, test } from "bun:test";
import { BaseProvider } from "../../src/core/llm/provider/base-provider";
import { OpenAIProvider } from "../../src/core/llm/provider/adapters/openai";
import { createProvider } from "../../src/core/llm/provider/registry";
import {
  ProviderError,
  ProviderErrorCode,
} from "../../src/shared/exceptions/index";
import {
  type ProviderInvokeOptions,
  type ResolvedProviderConfig,
} from "../../src/shared/types/index";

function makeConfig(
  provider: string,
  overrides: Partial<ResolvedProviderConfig> = {},
): ResolvedProviderConfig {
  return {
    provider,
    model: "test-model",
    apiKey: "sk-test",
    baseUrl: "http://localhost:11434/v1",
    generationConfig: {
      temperature: 0.7,
      topP: 0.9,
      maxTokens: 128,
      frequencyPenalty: 0,
      presencePenalty: 0,
    },
    requestGuardsConfig: {
      retry: {
        maxRetries: 0,
        initialDelayMs: 1,
        maxDelayMs: 1,
        factor: 2,
        jitter: false,
        retryableErrors: [/.*/],
      },
      timeout: { timeoutMs: 5000 },
      rateLimiter: {
        maxRequestsPerMinute: 600,
        maxQueueSize: 100,
        requestTimeout: 5000,
      },
    },
    verbose: false,
    ...overrides,
  };
}

class TestableOpenAIProvider extends OpenAIProvider {
  public invokeRaw(options: ProviderInvokeOptions): Promise<string> {
    return this.doInvoke(options);
  }
}

function setFakeClient(
  provider: OpenAIProvider,
  createImpl: (
    req: Record<string, unknown>,
    opts?: { signal?: AbortSignal },
  ) => Promise<unknown>,
): void {
  (
    provider as unknown as {
      client: {
        chat: {
          completions: {
            create: (
              req: Record<string, unknown>,
              opts?: { signal?: AbortSignal },
            ) => Promise<unknown>;
          };
        };
      };
    }
  ).client = {
    chat: {
      completions: {
        create: createImpl,
      },
    },
  };
}

// const originalLoadConfig = BaseProvider.loadConfig;
const originalLoadConfig: typeof BaseProvider.loadConfig =
BaseProvider.loadConfig.bind(BaseProvider);

function mockLoadConfig(config: ResolvedProviderConfig): void {
  (BaseProvider as unknown as { loadConfig: typeof BaseProvider.loadConfig }).loadConfig =
    async () => config;
}

describe("provider pipeline", () => {
  beforeEach(() => {
    (BaseProvider as unknown as { loadConfig: typeof BaseProvider.loadConfig }).loadConfig =
      originalLoadConfig;
  });

  afterEach(() => {
    (BaseProvider as unknown as { loadConfig: typeof BaseProvider.loadConfig }).loadConfig =
      originalLoadConfig;
  });

  describe("createProvider dispatch", () => {
    test("dispatch openai to OpenAIProvider", async () => {
      mockLoadConfig(makeConfig("openai"));
      const provider = await createProvider();
      expect(provider).toBeInstanceOf(OpenAIProvider);
    });

    test("dispatch deepseek to OpenAIProvider", async () => {
      mockLoadConfig(makeConfig("deepseek"));
      const provider = await createProvider();
      expect(provider).toBeInstanceOf(OpenAIProvider);
    });

    test("dispatch ollama to OpenAIProvider", async () => {
      mockLoadConfig(makeConfig("ollama", { apiKey: undefined }));
      const provider = await createProvider();
      expect(provider).toBeInstanceOf(OpenAIProvider);
    });

    test("anthropic should return standardized not implemented error", async () => {
      mockLoadConfig(makeConfig("anthropic"));

      await expect(createProvider()).rejects.toMatchObject({
        name: "ProviderError",
        code: ProviderErrorCode.ADAPTER_NOT_IMPLEMENTED,
      });
    });

    test("unsupported provider should return standardized unsupported error", async () => {
      mockLoadConfig(makeConfig("foo-provider"));

      await expect(createProvider()).rejects.toMatchObject({
        name: "ProviderError",
        code: ProviderErrorCode.UNSUPPORTED_PROVIDER,
      });
    });
  });

  describe("OpenAIProvider constructor validation", () => {
    test("throws CONFIG_INVALID for unsupported provider", () => {
      expect(() => new OpenAIProvider(makeConfig("foo-provider"))).toThrow();

      try {
        new OpenAIProvider(makeConfig("foo-provider"));
      } catch (error) {
        expect(error).toMatchObject({
          name: "ProviderError",
          code: ProviderErrorCode.CONFIG_INVALID,
          provider: "foo-provider",
        });
      }
    });

    test("throws CONFIG_INVALID when openai apiKey is missing", () => {
      expect(() =>
        new OpenAIProvider(
          makeConfig("openai", { apiKey: undefined }),
        ),
      ).toThrow();

      try {
        new OpenAIProvider(makeConfig("openai", { apiKey: undefined }));
      } catch (error) {
        expect(error).toMatchObject({
          name: "ProviderError",
          code: ProviderErrorCode.CONFIG_INVALID,
          provider: "openai",
        });
      }
    });

    test("throws CONFIG_INVALID when deepseek apiKey is blank", () => {
      expect(() =>
        new OpenAIProvider(
          makeConfig("deepseek", { apiKey: "   " }),
        ),
      ).toThrow();

      try {
        new OpenAIProvider(makeConfig("deepseek", { apiKey: "   " }));
      } catch (error) {
        expect(error).toMatchObject({
          name: "ProviderError",
          code: ProviderErrorCode.CONFIG_INVALID,
          provider: "deepseek",
        });
      }
    });

    test("does not throw when ollama apiKey is missing", () => {
      expect(() =>
        new OpenAIProvider(
          makeConfig("ollama", {
            apiKey: undefined,
            baseUrl: undefined,
          }),
        ),
      ).not.toThrow();
    });

    test("provider check is case-insensitive", () => {
      expect(() => new OpenAIProvider(makeConfig("OpenAI"))).not.toThrow();
    });
  });

  describe("OpenAIProvider factory methods", () => {
    test("createFromConfig returns instance", () => {
      const provider = OpenAIProvider.createFromConfig(makeConfig("openai"));
      expect(provider).toBeInstanceOf(OpenAIProvider);
    });

    test("create uses BaseProvider.loadConfig", async () => {
      (BaseProvider as unknown as { loadConfig: typeof BaseProvider.loadConfig }).loadConfig =
        async () => makeConfig("deepseek");

      const provider = await OpenAIProvider.create();
      expect(provider).toBeInstanceOf(OpenAIProvider);
    });
  });

  describe("OpenAIProvider doInvoke", () => {
    test("maps request fields and returns content", async () => {
      const provider = new TestableOpenAIProvider(makeConfig("openai"));

      let capturedReq: Record<string, unknown> | undefined;
      let capturedSignal: AbortSignal | undefined;

      setFakeClient(provider, async (req, opts) => {
        capturedReq = req;
        capturedSignal = opts?.signal;
        return {
          choices: [{ message: { content: "feat: add commit message generator" } }],
        };
      });

      const controller = new AbortController();

      const result = await provider.invokeRaw({
        prompt: "summarize staged changes",
        model: "gpt-4o-mini",
        generationConfig: {
          temperature: 0.2,
          topP: 0.8,
          maxTokens: 64,
          frequencyPenalty: 0.1,
          presencePenalty: 0.3,
        },
        signal: controller.signal,
      });

      expect(result).toBe("feat: add commit message generator");
      expect(capturedReq).toMatchObject({
        model: "gpt-4o-mini",
        temperature: 0.2,
        top_p: 0.8,
        max_completion_tokens: 64,
        frequency_penalty: 0.1,
        presence_penalty: 0.3,
      });
      expect(capturedReq?.messages).toEqual([
        { role: "user", content: "summarize staged changes" },
      ]);
      expect(capturedSignal).toBe(controller.signal);
    });

    test("throws EMPTY_RESPONSE when content is empty", async () => {
      const provider = new TestableOpenAIProvider(makeConfig("openai"));

      setFakeClient(provider, async () => ({
        choices: [{ message: { content: "   " } }],
      }));

      await expect(
        provider.invokeRaw({
          prompt: "x",
          model: "gpt-4o-mini",
          generationConfig: {},
        }),
      ).rejects.toMatchObject({
        name: "ProviderError",
        code: ProviderErrorCode.EMPTY_RESPONSE,
        provider: "openai",
      });
    });

    test("throws EMPTY_RESPONSE when content missing", async () => {
      const provider = new TestableOpenAIProvider(makeConfig("openai"));

      setFakeClient(provider, async () => ({
        choices: [{ message: {} }],
      }));

      await expect(
        provider.invokeRaw({
          prompt: "x",
          model: "gpt-4o-mini",
          generationConfig: {},
        }),
      ).rejects.toMatchObject({
        name: "ProviderError",
        code: ProviderErrorCode.EMPTY_RESPONSE,
        provider: "openai",
      });
    });

    test("wraps unknown error as REQUEST_FAILED", async () => {
      const provider = new TestableOpenAIProvider(makeConfig("deepseek"));

      setFakeClient(provider, async () => {
        throw new Error("network down");
      });

      await expect(
        provider.invokeRaw({
          prompt: "x",
          model: "deepseek-chat",
          generationConfig: {},
        }),
      ).rejects.toMatchObject({
        name: "ProviderError",
        code: ProviderErrorCode.REQUEST_FAILED,
        provider: "deepseek",
      });
    });

    test("keeps ProviderError unchanged", async () => {
      const provider = new TestableOpenAIProvider(makeConfig("openai"));

      setFakeClient(provider, async () => {
        throw new ProviderError({
          code: ProviderErrorCode.RATE_LIMITED,
          provider: "openai",
          message: "rate limited",
        });
      });

      await expect(
        provider.invokeRaw({
          prompt: "x",
          model: "gpt-4o-mini",
          generationConfig: {},
        }),
      ).rejects.toMatchObject({
        name: "ProviderError",
        code: ProviderErrorCode.RATE_LIMITED,
        provider: "openai",
      });
    });

    test("ollama without apiKey can still invoke", async () => {
      const provider = new TestableOpenAIProvider(
        makeConfig("ollama", {
          apiKey: undefined,
          baseUrl: undefined,
          model: "llama3",
        }),
      );

      setFakeClient(provider, async () => ({
        choices: [{ message: { content: "ok" } }],
      }));

      await expect(
        provider.invokeRaw({
          prompt: "x",
          model: "llama3",
          generationConfig: {},
        }),
      ).resolves.toBe("ok");
    });
  });

  describe("BaseProvider.invoke guard chain order", () => {
    class FakeProvider extends BaseProvider {
      public sequence: string[] = [];
      public lastInvoke?: ProviderInvokeOptions;

      constructor(config: ResolvedProviderConfig) {
        super(config);
      }

      protected override async doInvoke(options: ProviderInvokeOptions): Promise<string> {
        this.sequence.push("doInvoke");
        this.lastInvoke = options;
        return "ok";
      }
    }

    test("order should be retry -> rateLimiter -> timeout -> doInvoke", async () => {
      const provider = new FakeProvider(makeConfig("deepseek"));

      (
        provider as unknown as {
          retry: { execute: (fn: () => Promise<string>) => Promise<string> };
        }
      ).retry = {
        execute: async (fn) => {
          provider.sequence.push("retry");
          return fn();
        },
      };

      (
        provider as unknown as {
          rateLimiter: { acquire: () => Promise<void> };
        }
      ).rateLimiter = {
        acquire: async () => {
          provider.sequence.push("rateLimiter");
        },
      };

      (
        provider as unknown as {
          timeout: {
            execute: (
              fn: (signal: AbortSignal) => Promise<string>,
            ) => Promise<string>;
          };
        }
      ).timeout = {
        execute: async (fn) => {
          provider.sequence.push("timeout");
          return fn(new AbortController().signal);
        },
      };

      const result = await provider.invoke("hello");

      expect(result).toBe("ok");
      expect(provider.sequence).toEqual([
        "retry",
        "rateLimiter",
        "timeout",
        "doInvoke",
      ]);
    });

    test("generation overrides should merge with config defaults", async () => {
      const provider = new FakeProvider(makeConfig("deepseek"));

      (
        provider as unknown as {
          retry: { execute: (fn: () => Promise<string>) => Promise<string> };
        }
      ).retry = {
        execute: async (fn) => fn(),
      };
      (
        provider as unknown as {
          rateLimiter: { acquire: () => Promise<void> };
        }
      ).rateLimiter = {
        acquire: async () => {},
      };
      (
        provider as unknown as {
          timeout: {
            execute: (
              fn: (signal: AbortSignal) => Promise<string>,
            ) => Promise<string>;
          };
        }
      ).timeout = {
        execute: async (fn) => fn(new AbortController().signal),
      };

      await provider.invoke("hello", { temperature: 0.2, maxTokens: 32 });

      expect(provider.lastInvoke?.generationConfig.temperature).toBe(0.2);
      expect(provider.lastInvoke?.generationConfig.maxTokens).toBe(32);
      expect(provider.lastInvoke?.generationConfig.topP).toBe(0.9);
    });
  });
});