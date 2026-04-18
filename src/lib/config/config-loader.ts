import { Command, Option } from "commander";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  type RuntimeConfig,
  type PartialRuntimeConfig,
} from "../../shared/types/index";
import { deepMerge } from "../utils/index";
import { DEFAULT_LLM_CONFIG, DEFAULT_REQUEST_GUARDS_CONFIG } from "../../shared/constants/index"

export class ConfigLoader {
  private readonly argv: string[];
  private readonly env: NodeJS.ProcessEnv;
  private readonly configFilePath: string;

  constructor(options?: {
    argv?: string[];
    env?: NodeJS.ProcessEnv;
    configFilePath?: string;
  }) {
    this.argv = options?.argv ?? Bun.argv;
    this.env = options?.env ?? process.env;
    this.configFilePath = options?.configFilePath ?? join(homedir(), "autocommit.toml");
  }

  // ── Layer 0: Defaults ──

  private createDefaultConfig(): RuntimeConfig {
    return {
      llm: DEFAULT_LLM_CONFIG,
      requestGuards: DEFAULT_REQUEST_GUARDS_CONFIG,
    }
  }

  // // ── Layer 1: TOML File ──

  private async loadTomlFile(): Promise<PartialRuntimeConfig> {
    const file = Bun.file(this.configFilePath);
    if (!(await file.exists())) {
      throw new Error(
        `Failed to open file ${this.configFilePath}.`
      );
    }

    let content: string;
    try {
      content = await file.text();
    } catch (error) {
      throw new Error(
        `Failed to read config file ${this.configFilePath}: ${this.errorMsg(error)}`,
        { cause: error }
      );
    }

    try {
      return Bun.TOML.parse(content) as PartialRuntimeConfig;
    } catch (error) {
      throw new Error(
        `Failed to parse config file ${this.configFilePath}: ${this.errorMsg(error)}`,
        { cause: error }
      );
    }
  }

  // ── Layer 2: Environment Variables ───

  // Supports:
  // 1) RequestGuards overrides
  // 2) LLMProviderConfig overrides (provider/model/baseUrl/apiKey)

  private static readonly ENV_OVERRIDE_MAP: ReadonlyArray<{
    key: string;
    apply: (overrides: PartialRuntimeConfig, value: string) => void;
  }> = [
    // ── LLM provider config ──
    {
      key: "AUTOCOMMIT_PROVIDER",
      apply: (overrides, value) => {
        const llm = ConfigLoader.ensureLLMConfig(overrides);
        llm.provider = value.trim();
      },
    },
    {
      key: "AUTOCOMMIT_MODEL",
      apply: (overrides, value) => {
        const llm = ConfigLoader.ensureLLMConfig(overrides);
        llm.model = value.trim();
      },
    },
    {
      key: "AUTOCOMMIT_BASE_URL",
      apply: (overrides, value) => {
        const llm = ConfigLoader.ensureLLMConfig(overrides);
        llm.baseUrl = value.trim();
      },
    },
    {
      key: "AUTOCOMMIT_API_KEY",
      apply: (overrides, value) => {
        const llm = ConfigLoader.ensureLLMConfig(overrides);
        llm.apiKey = value.trim();
      },
    },

    // ── RequestGuards.retry ──
    {
      key: "AUTOCOMMIT_RETRY_MAX_RETRIES",
      apply: (overrides, value) => {
        const retry = ConfigLoader.ensureRetryConfig(overrides);
        retry.maxRetries = ConfigLoader.mustParseInt(value, "AUTOCOMMIT_RETRY_MAX_RETRIES");
      },
    },
    {
      key: "AUTOCOMMIT_RETRY_INITIAL_DELAY_MS",
      apply: (overrides, value) => {
        const retry = ConfigLoader.ensureRetryConfig(overrides);
        retry.initialDelayMs = ConfigLoader.mustParseInt(value, "AUTOCOMMIT_RETRY_INITIAL_DELAY_MS");
      },
    },
    {
      key: "AUTOCOMMIT_RETRY_MAX_DELAY_MS",
      apply: (overrides, value) => {
        const retry = ConfigLoader.ensureRetryConfig(overrides);
        retry.maxDelayMs = ConfigLoader.mustParseInt(value, "AUTOCOMMIT_RETRY_MAX_DELAY_MS");
      },
    },
    {
      key: "AUTOCOMMIT_RETRY_FACTOR",
      apply: (overrides, value) => {
        const retry = ConfigLoader.ensureRetryConfig(overrides);
        retry.factor = ConfigLoader.mustParseFloat(value, "AUTOCOMMIT_RETRY_FACTOR");
      },
    },
    {
      key: "AUTOCOMMIT_RETRY_JITTER",
      apply: (overrides, value) => {
        const retry = ConfigLoader.ensureRetryConfig(overrides);
        retry.jitter = ConfigLoader.mustParseBool(value, "AUTOCOMMIT_RETRY_JITTER");
      },
    },

    // ── RequestGuards.timeout ──
    {
      key: "AUTOCOMMIT_TIMEOUT_MS",
      apply: (overrides, value) => {
        const timeout = ConfigLoader.ensureTimeoutConfig(overrides);
        timeout.timeoutMs = ConfigLoader.mustParseInt(value, "AUTOCOMMIT_TIMEOUT_MS");
      },
    },

    // ── RequestGuards.rateLimiter ──
    {
      key: "AUTOCOMMIT_RATE_LIMITER_MAX_RPM",
      apply: (overrides, value) => {
        const rl = ConfigLoader.ensureRateLimiterConfig(overrides);
        rl.maxRequestsPerMinute = ConfigLoader.mustParseInt(value, "AUTOCOMMIT_RATE_LIMITER_MAX_RPM");
      },
    },
    {
      key: "AUTOCOMMIT_RATE_LIMITER_MAX_QUEUE_SIZE",
      apply: (overrides, value) => {
        const rl = ConfigLoader.ensureRateLimiterConfig(overrides);
        rl.maxQueueSize = ConfigLoader.mustParseInt(value, "AUTOCOMMIT_RATE_LIMITER_MAX_QUEUE_SIZE");
      },
    },
    {
      key: "AUTOCOMMIT_RATE_LIMITER_REQUEST_TIMEOUT",
      apply: (overrides, value) => {
        const rl = ConfigLoader.ensureRateLimiterConfig(overrides);
        rl.requestTimeout = ConfigLoader.mustParseInt(value, "AUTOCOMMIT_RATE_LIMITER_REQUEST_TIMEOUT");
      },
    },
  ];

  /**
   * Read AUTOCOMMIT_* env vars into an overrides object.
   * Caller merges this overrides with defaults/file config.
   */
  private loadEnvGuardOverrides(): PartialRuntimeConfig {
    const overrides: PartialRuntimeConfig = {};

    for (const entry of ConfigLoader.ENV_OVERRIDE_MAP) {
      const value = this.env[entry.key];
      if (value === undefined || value.trim() === "") continue;

      try {
        entry.apply(overrides, value);
      } catch (error) {
        throw new Error(
          `Invalid env var ${entry.key}="${value}": ${this.errorMsg(error)}`,
          { cause: error },
        );
      }
    }

    return overrides;
  }

  // ── Layer 3: CLI Arguments (LLM only) ──

  private loadCliLLMOverrides(): PartialRuntimeConfig {
    const program = new Command();

    program
      .exitOverride()
      .allowUnknownOption(true)
      .allowExcessArguments(true)
      // CLI layer is intentionally limited to LLMGenerationConfig only.
      .addOption(new Option("--temperature <n>", "Sampling temperature"))
      .addOption(new Option("--max-tokens <n>", "Max output tokens"))
      .addOption(new Option("--top-p <n>", "Nucleus sampling"))
      .addOption(new Option("--frequency-penalty <n>", "Frequency penalty"))
      .addOption(new Option("--presence-penalty <n>", "Presence penalty"));

    try {
      program.parse(this.argv, { from: "user" });
    } catch (error) {
      throw new Error(
        `Invalid CLI arguments: ${this.errorMsg(error)}`,
        { cause: error}
      );
    }

    const opts = program.opts<{
      temperature?: string;
      maxTokens?: string;
      topP?: string;
      frequencyPenalty?: string;
      presencePenalty?: string;
    }>();

    const overrides: PartialRuntimeConfig = {};
    let hasLLMGenerationOverride = false;

    const setFloatOverride = (
      raw: string | undefined,
      argName: string,
      setter: (value: number) => void,
    ): void => {
      if (raw === undefined) return;
      const value = ConfigLoader.mustParseFloat(raw, argName);
      setter(value);
      hasLLMGenerationOverride = true;
    };

    const setIntOverride = (
      raw: string | undefined,
      argName: string,
      setter: (value: number) => void,
    ): void => {
      if (raw === undefined) return;
      const value = ConfigLoader.mustParseInt(raw, argName);
      setter(value);
      hasLLMGenerationOverride = true;
    };

    setFloatOverride(opts.temperature, "--temperature", (value) => {
      ConfigLoader.ensureLLMConfig(overrides).temperature = value;
    });
    setIntOverride(opts.maxTokens, "--max-tokens", (value) => {
      ConfigLoader.ensureLLMConfig(overrides).maxTokens = value;
    });
    setFloatOverride(opts.topP, "--top-p", (value) => {
      ConfigLoader.ensureLLMConfig(overrides).topP = value;
    });
    setFloatOverride(opts.frequencyPenalty, "--frequency-penalty", (value) => {
      ConfigLoader.ensureLLMConfig(overrides).frequencyPenalty = value;
    });
    setFloatOverride(opts.presencePenalty, "--presence-penalty", (value) => {
      ConfigLoader.ensureLLMConfig(overrides).presencePenalty = value;
    });

    if (!hasLLMGenerationOverride) return {};
    return overrides;
  }


  // 
  private mergeConfigs(configs: PartialRuntimeConfig[]): RuntimeConfig {
    const defaultConfig: RuntimeConfig = this.createDefaultConfig();

    let mergedConfig: RuntimeConfig = { ...defaultConfig };
    for (const partial of configs) {
      mergedConfig = deepMerge(mergedConfig, partial);
    }

    return mergedConfig;
  }


  // ── Safe config initializers (avoid undefined nested object access) ──

  private static ensureLLMConfig(overrides: PartialRuntimeConfig): NonNullable<PartialRuntimeConfig["llm"]> {
    overrides.llm ??= {};
    return overrides.llm;
  }

  private static ensureRequestGuardsConfig(
    overrides: PartialRuntimeConfig,
  ): NonNullable<PartialRuntimeConfig["requestGuards"]> {
    overrides.requestGuards ??= {};
    return overrides.requestGuards;
  }

  private static ensureRetryConfig(
    overrides: PartialRuntimeConfig,
  ): NonNullable<NonNullable<PartialRuntimeConfig["requestGuards"]>["retry"]> {
    const guards = ConfigLoader.ensureRequestGuardsConfig(overrides);
    guards.retry ??= {};
    return guards.retry;
  }

  private static ensureTimeoutConfig(
    overrides: PartialRuntimeConfig,
  ): NonNullable<NonNullable<PartialRuntimeConfig["requestGuards"]>["timeout"]> {
    const guards = ConfigLoader.ensureRequestGuardsConfig(overrides);
    guards.timeout ??= {};
    return guards.timeout;
  }

  private static ensureRateLimiterConfig(
    overrides: PartialRuntimeConfig,
  ): NonNullable<NonNullable<PartialRuntimeConfig["requestGuards"]>["rateLimiter"]> {
    const guards = ConfigLoader.ensureRequestGuardsConfig(overrides);
    guards.rateLimiter ??= {};
    return guards.rateLimiter;
  }

  // ── Parse Helpers ──

  private static mustParseInt(value: string, name: string): number {
    const n = Number.parseInt(value, 10);
    if (Number.isNaN(n)) throw new Error(`${name} must be an integer`);
    return n;
  }

  private static mustParseFloat(value: string, name: string): number {
    const n = Number.parseFloat(value);
    if (Number.isNaN(n)) throw new Error(`${name} must be a number`);
    return n;
  }

  private static mustParseBool(value: string, name: string): boolean {
    if (value === "true" || value === "1") return true;
    if (value === "false" || value === "0") return false;
    throw new Error(`${name} must be true/false or 1/0`);
  }
  
  private validate(config: RuntimeConfig): void {
    const { llm, requestGuards } = config;

    if (!llm.provider || llm.provider.trim() === "") {
      throw new Error("llm.provider is required. Set it in autocommit.toml or via --provider");
    }
    if (!llm.model || llm.model.trim() === "") {
      throw new Error("llm.model is required. Set it in autocommit.toml or via --model");
    }

    // For remote providers, apiKey is required
    // Ollama/local providers may not need an API key
    const provider = llm.provider.toLowerCase();
    const requiresApiKey = provider !== "ollama" && provider !== "local";
    if (requiresApiKey && (!llm.apiKey || llm.apiKey.trim() === "")) {
      throw new Error(
        `llm.apiKey is required for provider "${llm.provider}". ` +
        "Set env AUTOCOMMIT_API_KEY.",
      );
    }

    // const p = llm.modelParams;
    if (llm.temperature !== undefined && (llm.temperature < 0 || llm.temperature > 2)) {
      throw new Error("temperature must be between 0 and 2");
    }
    if (llm.topP !== undefined && (llm.topP < 0 || llm.topP > 1)) {
      throw new Error("topP must be between 0 and 1");
    }
    if (llm.maxTokens !== undefined && llm.maxTokens <= 0) {
      throw new Error("maxTokens must be > 0");
    }

    // requestGuards
    if (requestGuards.retry.maxRetries < 0) {
      throw new Error("retry.maxRetries must be >= 0");
    }
    if (requestGuards.retry.initialDelayMs < 0) {
      throw new Error("retry.initialDelayMs must be >= 0");
    }
    if (requestGuards.retry.maxDelayMs < 0) {
      throw new Error("retry.maxDelayMs must be >= 0");
    }
    if (requestGuards.retry.factor <= 0) {
      throw new Error("retry.factor must be > 0");
    }
    if (requestGuards.timeout.timeoutMs <= 0) {
      throw new Error("timeout.timeoutMs must be > 0");
    }
    if (requestGuards.rateLimiter.maxRequestsPerMinute <= 0) {
      throw new Error("rateLimiter.maxRequestsPerMinute must be > 0");
    }
    if (requestGuards.rateLimiter.maxQueueSize <= 0) {
      throw new Error("rateLimiter.maxQueueSize must be > 0");
    }
    if (requestGuards.rateLimiter.requestTimeout <= 0) {
      throw new Error("rateLimiter.requestTimeout must be > 0");
    }
  }

  private errorMsg(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

}

