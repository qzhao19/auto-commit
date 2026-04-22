import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ConfigLoader } from "../../src/lib/config/config-loader";
import {
  DEFAULT_LLM_CONFIG,
  DEFAULT_REQUEST_GUARDS_CONFIG,
} from "../../src/shared/constants/index";

// ── Helpers ──

/** Minimal env that satisfies validateConfig (provider + model + apiKey required). */
function validEnv(extra: Record<string, string> = {}): NodeJS.ProcessEnv {
  return {
    AUTOCOMMIT_PROVIDER: "openai",
    AUTOCOMMIT_MODEL: "gpt-4o-mini",
    AUTOCOMMIT_API_KEY: "sk-test-key",
    ...extra,
  };
}

/** Argv with no recognized CLI flags (commander sees nothing). */
const EMPTY_ARGV: string[] = [];

let tempDir: string;

async function writeToml(content: string): Promise<string> {
  const path = join(tempDir, "autocommit.toml");
  await Bun.write(path, content);
  return path;
}

/** Path that definitely doesn't exist. */
function nonExistentPath(): string {
  return join(tempDir, "does-not-exist.toml");
}

// ── Setup / Teardown ──

beforeEach(async () => {
  tempDir = await mkdtemp(join(tmpdir(), "config-loader-test-"));
});

afterEach(async () => {
  await rm(tempDir, { recursive: true, force: true });
});

// ═════════════════════════════════════════════════════════
// 1. Defaults Only (no TOML, no env overrides, no CLI)
// ═════════════════════════════════════════════════════════

describe("Layer 0 – defaults only", () => {
  test("returns default config when no TOML file exists", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(DEFAULT_LLM_CONFIG.temperature);
    expect(cfg.generationConfig.maxTokens).toBe(DEFAULT_LLM_CONFIG.maxTokens);
    expect(cfg.generationConfig.topP).toBe(DEFAULT_LLM_CONFIG.topP);
    expect(cfg.generationConfig.frequencyPenalty).toBe(DEFAULT_LLM_CONFIG.frequencyPenalty);
    expect(cfg.generationConfig.presencePenalty).toBe(DEFAULT_LLM_CONFIG.presencePenalty);
    const { retryableErrors: _, ...retryPublic } = cfg.requestGuardsConfig.retry;
    expect({ ...cfg.requestGuardsConfig, retry: retryPublic }).toEqual(DEFAULT_REQUEST_GUARDS_CONFIG);
  });
});

// ═════════════════════════════════════════════════════════
// 2. Layer 1 – TOML File
// ═════════════════════════════════════════════════════════

describe("Layer 1 – TOML file", () => {
  test("loads a complete TOML config and overrides defaults", async () => {
    const path = await writeToml(`
[llm]
provider = "deepseek"
model = "deepseek-chat"
baseUrl = "https://api.deepseek.com/v1"
apiKey = "sk-ds-key"
temperature = 0.5
maxTokens = 2048
topP = 0.8
frequencyPenalty = 0.1
presencePenalty = 0.2

[requestGuards.retry]
maxRetries = 5
initialDelayMs = 500
maxDelayMs = 5000
factor = 3
jitter = false

[requestGuards.timeout]
timeoutMs = 15000

[requestGuards.rateLimiter]
maxRequestsPerMinute = 10
maxQueueSize = 500
requestTimeout = 15000
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.provider).toBe("deepseek");
    expect(cfg.model).toBe("deepseek-chat");
    expect(cfg.baseUrl).toBe("https://api.deepseek.com/v1");
    expect(cfg.apiKey).toBe("sk-ds-key");
    expect(cfg.generationConfig.temperature).toBe(0.5);
    expect(cfg.generationConfig.maxTokens).toBe(2048);
    expect(cfg.generationConfig.topP).toBe(0.8);
    expect(cfg.generationConfig.frequencyPenalty).toBe(0.1);
    expect(cfg.generationConfig.presencePenalty).toBe(0.2);

    expect(cfg.requestGuardsConfig.retry.maxRetries).toBe(5);
    expect(cfg.requestGuardsConfig.retry.initialDelayMs).toBe(500);
    expect(cfg.requestGuardsConfig.retry.maxDelayMs).toBe(5000);
    expect(cfg.requestGuardsConfig.retry.factor).toBe(3);
    expect(cfg.requestGuardsConfig.retry.jitter).toBe(false);
    expect(cfg.requestGuardsConfig.timeout.timeoutMs).toBe(15000);
    expect(cfg.requestGuardsConfig.rateLimiter.maxRequestsPerMinute).toBe(10);
    expect(cfg.requestGuardsConfig.rateLimiter.maxQueueSize).toBe(500);
    expect(cfg.requestGuardsConfig.rateLimiter.requestTimeout).toBe(15000);
  });

  test("partial TOML only overrides specified fields, rest stay default", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"
temperature = 1.2
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(1.2);
    expect(cfg.generationConfig.maxTokens).toBe(DEFAULT_LLM_CONFIG.maxTokens);
    expect(cfg.generationConfig.topP).toBe(DEFAULT_LLM_CONFIG.topP);
    const { retryableErrors: _, ...retryPublic } = cfg.requestGuardsConfig.retry;
    expect({ ...cfg.requestGuardsConfig, retry: retryPublic }).toEqual(DEFAULT_REQUEST_GUARDS_CONFIG);
  });

  test("empty baseUrl in TOML is omitted (falls back to default)", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"
baseUrl = ""
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.baseUrl).toBeUndefined();
  });

  test("backward compatibility: RequestGuards (PascalCase) → requestGuards", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[RequestGuards.retry]
maxRetries = 10
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.retry.maxRetries).toBe(10);
  });

  test("throws on unknown top-level key in TOML", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[unknown]
foo = "bar"
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("Unknown key root.unknown");
  });

  test("throws on unknown key inside [llm]", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"
unknownField = 123
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("Unknown key llm.unknownField");
  });

  test("throws on unknown key inside [requestGuards]", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[requestGuards.unknownSection]
foo = 1
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("Unknown key requestGuards.unknownSection");
  });

  test("throws on unknown key inside [requestGuards.retry]", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[requestGuards.retry]
maxRetries = 3
badKey = 999
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("Unknown key requestGuards.retry.badKey");
  });

  test("throws when llm.temperature is not a number", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"
temperature = "hot"
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("llm.temperature must be a finite number");
  });

  test("throws when llm.maxTokens is a float", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"
maxTokens = 1.5
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("llm.maxTokens must be an integer");
  });

  test("throws when llm.provider is not a string (number)", async () => {
    const path = await writeToml(`
[llm]
provider = 123
model = "gpt-4o"
apiKey = "sk-key"
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("llm.provider must be a string");
  });

  test("throws when requestGuards.retry is not a table", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[requestGuards]
retry = "not-a-table"
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("requestGuards.retry must be a table");
  });

  test("throws on invalid TOML syntax", async () => {
    const path = await writeToml("this is not valid toml ][");

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });

    await expect(loader.load()).rejects.toThrow("Failed to parse config file");
  });

  test("TOML with only requestGuards, no llm section", async () => {
    const path = await writeToml(`
[requestGuards.timeout]
timeoutMs = 5000
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv(),
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.timeout.timeoutMs).toBe(5000);
    expect(cfg.provider).toBe("openai");
  });
});

// ═════════════════════════════════════════════════════════
// 3. Layer 2 – Environment Variables
// ═════════════════════════════════════════════════════════

describe("Layer 2 – environment variables", () => {
  test("overrides LLM provider config from env", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_BASE_URL: "https://custom.api/v1",
      }),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.provider).toBe("openai");
    expect(cfg.model).toBe("gpt-4o-mini");
    expect(cfg.apiKey).toBe("sk-test-key");
    expect(cfg.baseUrl).toBe("https://custom.api/v1");
  });

  test("overrides retry config from env", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_RETRY_MAX_RETRIES: "5",
        AUTOCOMMIT_RETRY_INITIAL_DELAY_MS: "2000",
        AUTOCOMMIT_RETRY_MAX_DELAY_MS: "20000",
        AUTOCOMMIT_RETRY_FACTOR: "3.5",
        AUTOCOMMIT_RETRY_JITTER: "false",
      }),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.retry.maxRetries).toBe(5);
    expect(cfg.requestGuardsConfig.retry.initialDelayMs).toBe(2000);
    expect(cfg.requestGuardsConfig.retry.maxDelayMs).toBe(20000);
    expect(cfg.requestGuardsConfig.retry.factor).toBe(3.5);
    expect(cfg.requestGuardsConfig.retry.jitter).toBe(false);
  });

  test("overrides timeout from env", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_TIMEOUT_MS: "60000",
      }),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.timeout.timeoutMs).toBe(60000);
  });

  test("overrides rateLimiter from env", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_RATE_LIMITER_MAX_RPM: "100",
        AUTOCOMMIT_RATE_LIMITER_MAX_QUEUE_SIZE: "5000",
        AUTOCOMMIT_RATE_LIMITER_REQUEST_TIMEOUT: "60000",
      }),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.rateLimiter.maxRequestsPerMinute).toBe(100);
    expect(cfg.requestGuardsConfig.rateLimiter.maxQueueSize).toBe(5000);
    expect(cfg.requestGuardsConfig.rateLimiter.requestTimeout).toBe(60000);
  });

  test("empty env vars are ignored", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_RETRY_MAX_RETRIES: "",
        AUTOCOMMIT_TIMEOUT_MS: "  ",
      }),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.retry.maxRetries).toBe(DEFAULT_REQUEST_GUARDS_CONFIG.retry.maxRetries);
    expect(cfg.requestGuardsConfig.timeout.timeoutMs).toBe(DEFAULT_REQUEST_GUARDS_CONFIG.timeout.timeoutMs);
  });

  test("throws on non-integer env var for integer field", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_RETRY_MAX_RETRIES: "abc",
      }),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow(
      "Invalid env var AUTOCOMMIT_RETRY_MAX_RETRIES=\"abc\"",
    );
  });

  test("throws on non-numeric env var for float field", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_RETRY_FACTOR: "not-a-number",
      }),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow(
      "Invalid env var AUTOCOMMIT_RETRY_FACTOR=\"not-a-number\"",
    );
  });

  test("throws on invalid boolean env var", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_RETRY_JITTER: "maybe",
      }),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow(
      "Invalid env var AUTOCOMMIT_RETRY_JITTER=\"maybe\"",
    );
  });

  test("trims whitespace from env values", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({
        AUTOCOMMIT_PROVIDER: "  deepseek  ",
        AUTOCOMMIT_MODEL: "  deepseek-chat  ",
        AUTOCOMMIT_API_KEY: "  sk-trimmed  ",
      }),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.provider).toBe("deepseek");
    expect(cfg.model).toBe("deepseek-chat");
    expect(cfg.apiKey).toBe("sk-trimmed");
  });

  test("env jitter accepts '1' and '0'", async () => {
    const loader1 = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({ AUTOCOMMIT_RETRY_JITTER: "1" }),
      configFilePath: nonExistentPath(),
    });
    expect((await loader1.load()).requestGuardsConfig.retry.jitter).toBe(true);

    const loader0 = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({ AUTOCOMMIT_RETRY_JITTER: "0" }),
      configFilePath: nonExistentPath(),
    });
    expect((await loader0.load()).requestGuardsConfig.retry.jitter).toBe(false);
  });
});

// ═════════════════════════════════════════════════════════
// 4. Layer 3 – CLI Arguments
// ═════════════════════════════════════════════════════════

describe("Layer 3 – CLI arguments", () => {
  test("overrides temperature from CLI", async () => {
    const loader = new ConfigLoader({
      argv: ["--temperature", "0.3"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(0.3);
  });

  test("overrides maxTokens from CLI", async () => {
    const loader = new ConfigLoader({
      argv: ["--max-tokens", "1024"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.maxTokens).toBe(1024);
  });

  test("overrides topP from CLI", async () => {
    const loader = new ConfigLoader({
      argv: ["--top-p", "0.5"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.topP).toBe(0.5);
  });

  test("overrides frequencyPenalty from CLI", async () => {
    const loader = new ConfigLoader({
      argv: ["--frequency-penalty", "0.7"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.frequencyPenalty).toBe(0.7);
  });

  test("overrides presencePenalty from CLI", async () => {
    const loader = new ConfigLoader({
      argv: ["--presence-penalty", "0.4"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.presencePenalty).toBe(0.4);
  });

  test("multiple CLI flags at once", async () => {
    const loader = new ConfigLoader({
      argv: ["--temperature", "1.0", "--max-tokens", "512", "--top-p", "0.7"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(1.0);
    expect(cfg.generationConfig.maxTokens).toBe(512);
    expect(cfg.generationConfig.topP).toBe(0.7);
    expect(cfg.generationConfig.frequencyPenalty).toBe(DEFAULT_LLM_CONFIG.frequencyPenalty);
    expect(cfg.generationConfig.presencePenalty).toBe(DEFAULT_LLM_CONFIG.presencePenalty);
  });

  test("no CLI flags → defaults remain", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(DEFAULT_LLM_CONFIG.temperature);
    expect(cfg.generationConfig.maxTokens).toBe(DEFAULT_LLM_CONFIG.maxTokens);
  });

  test("throws on non-numeric --temperature", async () => {
    const loader = new ConfigLoader({
      argv: ["--temperature", "hot"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("--temperature must be a finite number");
  });

  test("throws on non-integer --max-tokens", async () => {
    const loader = new ConfigLoader({
      argv: ["--max-tokens", "abc"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("--max-tokens must be an integer");
  });

  test("throws on float --max-tokens", async () => {
    const loader = new ConfigLoader({
      argv: ["--max-tokens", "10.5"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("--max-tokens must be an integer");
  });

  test("unknown CLI flags are allowed (allowUnknownOption)", async () => {
    const loader = new ConfigLoader({
      argv: ["--unknown-flag", "value", "--temperature", "0.5"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(0.5);
  });
});

// ═════════════════════════════════════════════════════════
// 5. Merge Priority (defaults < TOML < env < CLI)
// ═════════════════════════════════════════════════════════

describe("merge priority", () => {
  test("TOML overrides defaults", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-toml"
temperature = 1.5
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(1.5);
    expect(cfg.generationConfig.temperature).not.toBe(DEFAULT_LLM_CONFIG.temperature);
  });

  test("env overrides TOML", async () => {
    const path = await writeToml(`
[llm]
provider = "deepseek"
model = "deepseek-chat"
apiKey = "sk-toml"

[requestGuards.retry]
maxRetries = 10
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {
        AUTOCOMMIT_PROVIDER: "openai",
        AUTOCOMMIT_MODEL: "gpt-4o",
        AUTOCOMMIT_API_KEY: "sk-env",
        AUTOCOMMIT_RETRY_MAX_RETRIES: "2",
      },
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.provider).toBe("openai");
    expect(cfg.model).toBe("gpt-4o");
    expect(cfg.apiKey).toBe("sk-env");
    expect(cfg.requestGuardsConfig.retry.maxRetries).toBe(2);
  });

  test("CLI overrides TOML and env", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-toml"
temperature = 0.5
`);

    const loader = new ConfigLoader({
      argv: ["--temperature", "1.8"],
      env: validEnv(),
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(1.8);
  });

  test("full three-layer override chain", async () => {
    const path = await writeToml(`
[llm]
provider = "toml-provider"
model = "toml-model"
apiKey = "sk-toml"
temperature = 0.3
maxTokens = 1000

[requestGuards.retry]
maxRetries = 10
`);

    const loader = new ConfigLoader({
      argv: ["--temperature", "1.9"],
      env: {
        AUTOCOMMIT_PROVIDER: "env-provider",
        AUTOCOMMIT_MODEL: "env-model",
        AUTOCOMMIT_API_KEY: "sk-env",
        AUTOCOMMIT_RETRY_MAX_RETRIES: "1",
      },
      configFilePath: path,
    });
    const cfg = await loader.load();

    // Provider/model/apiKey: env wins over TOML
    expect(cfg.provider).toBe("env-provider");
    expect(cfg.model).toBe("env-model");
    expect(cfg.apiKey).toBe("sk-env");
    // temperature: CLI wins over TOML
    expect(cfg.generationConfig.temperature).toBe(1.9);
    // maxTokens: TOML wins over default (no CLI/env for it)
    expect(cfg.generationConfig.maxTokens).toBe(1000);
    // retry.maxRetries: env wins over TOML
    expect(cfg.requestGuardsConfig.retry.maxRetries).toBe(1);
  });
});

// ═════════════════════════════════════════════════════════
// 6. validateConfig – Final Validation
// ═════════════════════════════════════════════════════════

describe("validateConfig errors", () => {
  test("throws when provider is missing", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {
        AUTOCOMMIT_MODEL: "gpt-4o",
        AUTOCOMMIT_API_KEY: "sk-key",
      },
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("llm.provider is required");
  });

  test("throws when model is missing", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {
        AUTOCOMMIT_PROVIDER: "openai",
        AUTOCOMMIT_API_KEY: "sk-key",
      },
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("llm.model is required");
  });

  test("throws when apiKey is missing for non-ollama provider", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {
        AUTOCOMMIT_PROVIDER: "openai",
        AUTOCOMMIT_MODEL: "gpt-4o",
      },
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("llm.apiKey is required");
  });

  test("does not throw when apiKey is missing for ollama provider", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {
        AUTOCOMMIT_PROVIDER: "ollama",
        AUTOCOMMIT_MODEL: "llama3",
      },
      configFilePath: nonExistentPath(),
    });

    const cfg = await loader.load();
    expect(cfg.provider).toBe("ollama");
  });

  test("does not throw when apiKey is missing for local provider", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {
        AUTOCOMMIT_PROVIDER: "local",
        AUTOCOMMIT_MODEL: "local-model",
      },
      configFilePath: nonExistentPath(),
    });

    const cfg = await loader.load();
    expect(cfg.provider).toBe("local");
  });

  test("throws when temperature > 2", async () => {
    const loader = new ConfigLoader({
      argv: ["--temperature", "2.5"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("temperature must be between 0 and 2");
  });

  test("throws when temperature < 0", async () => {
    const loader = new ConfigLoader({
      argv: ["--temperature", "-0.1"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("temperature must be between 0 and 2");
  });

  test("throws when topP out of range", async () => {
    const loader = new ConfigLoader({
      argv: ["--top-p", "1.5"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("topP must be between 0 and 1");
  });

  test("throws when maxTokens <= 0", async () => {
    const loader = new ConfigLoader({
      argv: ["--max-tokens", "0"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("maxTokens must be > 0");
  });

  test("throws when retry.maxRetries < 0", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({ AUTOCOMMIT_RETRY_MAX_RETRIES: "-1" }),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("retry.maxRetries must be >= 0");
  });

  test("throws when retry.factor <= 0", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({ AUTOCOMMIT_RETRY_FACTOR: "0" }),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("retry.factor must be > 0");
  });

  test("throws when timeout.timeoutMs <= 0", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({ AUTOCOMMIT_TIMEOUT_MS: "0" }),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("timeout.timeoutMs must be > 0");
  });

  test("throws when rateLimiter.maxRequestsPerMinute <= 0", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({ AUTOCOMMIT_RATE_LIMITER_MAX_RPM: "0" }),
      configFilePath: nonExistentPath(),
    });

    await expect(loader.load()).rejects.toThrow("rateLimiter.maxRequestsPerMinute must be > 0");
  });
});

// ═════════════════════════════════════════════════════════
// 7. Edge Cases
// ═════════════════════════════════════════════════════════

describe("edge cases", () => {
  test("TOML with all numeric fields as integers (native TOML types)", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"
temperature = 1
maxTokens = 512
topP = 0
frequencyPenalty = 0
presencePenalty = 0
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(1);
    expect(cfg.generationConfig.maxTokens).toBe(512);
    expect(cfg.generationConfig.topP).toBe(0);
  });

  test("TOML jitter = true (native bool)", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[requestGuards.retry]
jitter = true
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.retry.jitter).toBe(true);
  });

  test("TOML jitter = false (native bool)", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[requestGuards.retry]
jitter = false
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.retry.jitter).toBe(false);
  });

  test("argv starting with non-flag gets sliced (simulates bun run)", async () => {
    const loader = new ConfigLoader({
      argv: ["index.ts", "--temperature", "0.6"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(0.6);
  });

  test("TOML with empty requestGuards sections falls back to defaults", async () => {
    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[requestGuards.retry]

[requestGuards.timeout]

[requestGuards.rateLimiter]
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    const cfg = await loader.load();

    const { retryableErrors: _, ...retryPublic } = cfg.requestGuardsConfig.retry;
    expect({ ...cfg.requestGuardsConfig, retry: retryPublic }).toEqual(DEFAULT_REQUEST_GUARDS_CONFIG);
  });

  test("env var with only whitespace is ignored", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv({ AUTOCOMMIT_BASE_URL: "   " }),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.baseUrl).toBe(DEFAULT_LLM_CONFIG.baseUrl);
  });

  test("CLI --temperature 0 is valid (boundary)", async () => {
    const loader = new ConfigLoader({
      argv: ["--temperature", "0"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(0);
  });

  test("CLI --temperature 2 is valid (boundary)", async () => {
    const loader = new ConfigLoader({
      argv: ["--temperature", "2"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.generationConfig.temperature).toBe(2);
  });

  test("CLI --top-p 0 and 1 are valid (boundaries)", async () => {
    const loader0 = new ConfigLoader({
      argv: ["--top-p", "0"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    expect((await loader0.load()).generationConfig.topP).toBe(0);

    const loader1 = new ConfigLoader({
      argv: ["--top-p", "1"],
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    expect((await loader1.load()).generationConfig.topP).toBe(1);
  });

  test("TOML does not mutate DEFAULT constants", async () => {
    const originalRetry = { ...DEFAULT_REQUEST_GUARDS_CONFIG.retry };

    const path = await writeToml(`
[llm]
provider = "openai"
model = "gpt-4o"
apiKey = "sk-key"

[requestGuards.retry]
maxRetries = 99
`);

    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: {},
      configFilePath: path,
    });
    await loader.load();

    expect(DEFAULT_REQUEST_GUARDS_CONFIG.retry.maxRetries).toBe(originalRetry.maxRetries);
  });
});

// ═════════════════════════════════════════════════════════
// 8. retryableErrors Injection
// ═════════════════════════════════════════════════════════

describe("retryableErrors injection", () => {
  test("retryableErrors are injected into requestGuardsConfig.retry", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    expect(cfg.requestGuardsConfig.retry.retryableErrors).toBeInstanceOf(Array);
    expect(cfg.requestGuardsConfig.retry.retryableErrors.length).toBeGreaterThan(0);
  });

  test("retryableErrors includes default network/HTTP error patterns", async () => {
    const loader = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const cfg = await loader.load();

    const patterns = cfg.requestGuardsConfig.retry.retryableErrors;
    expect(patterns.some((r) => r.test("ECONNRESET"))).toBe(true);
    expect(patterns.some((r) => r.test("ETIMEDOUT"))).toBe(true);
    expect(patterns.some((r) => r.test("ECONNREFUSED"))).toBe(true);
    expect(patterns.some((r) => r.test("socket hang up"))).toBe(true);
    expect(patterns.some((r) => r.test("429"))).toBe(true);
    expect(patterns.some((r) => r.test("503"))).toBe(true);
    expect(patterns.some((r) => r.test("502"))).toBe(true);
  });

  test("retryableErrors are independent copies (not shared references)", async () => {
    const loader1 = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });
    const loader2 = new ConfigLoader({
      argv: EMPTY_ARGV,
      env: validEnv(),
      configFilePath: nonExistentPath(),
    });

    const cfg1 = await loader1.load();
    const cfg2 = await loader2.load();

    expect(cfg1.requestGuardsConfig.retry.retryableErrors).not.toBe(
      cfg2.requestGuardsConfig.retry.retryableErrors,
    );
  });
});