import { Command, Option } from "commander";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  type RuntimeConfig,
  type PartialRuntimeConfig,
  type InternalRuntimeConfig,
  type ResolvedProviderConfig,
} from "../../shared/types/index";
import { DEFAULT_LLM_CONFIG, DEFAULT_REQUEST_GUARDS_CONFIG } from "../../shared/constants/index";
import { ConfigError, ConfigErrorCode } from "../../shared/exceptions/index";
import {
  assertOnlyKnownKeys,
  assertIsBool,
  assertIsFloat,
  assertIsInt,
  assertIsObject,
  assertIsString,
  assertIsTomlBool,
  assertIsTomlFloat,
  assertIsTomlInt,
  validateConfig,
  deepMerge,
  ensureLLMConfig,
  ensureRateLimiterConfig,
  ensureRetryConfig,
  ensureTimeoutConfig,
  toProviderConfig,
} from "../utils/config/index";


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

  public async load(): Promise<ResolvedProviderConfig> {
    // Create deault config
    const defaultConfig: RuntimeConfig = this.createDefaultConfig();

    // Load partial configs from toml config file
    const tomlPartialConfig: PartialRuntimeConfig = await this.loadTomlFile();

    // Load environment variables: 
    // requestGuards + LLMProviderConfig overrides (provider/model/baseUrl/apiKey) 
    const envPartialConfig: PartialRuntimeConfig = this.loadEnvGuardOverrides();

    // Load CLI arguments: LLMGenerationConfig params + verbose flag
    const cliPartialConfig: PartialRuntimeConfig = this.loadCliOverrides();

    // Merge all configs
    const runtimeConfig: RuntimeConfig = this.mergeConfigs(
      defaultConfig,
      tomlPartialConfig,
      envPartialConfig,
      cliPartialConfig,
    );

    // Validates the provided runtime configuration object
    validateConfig(runtimeConfig);

    // Inject retryableErrors into RuntimeConfig object 
    const internalRuntimeConfig: InternalRuntimeConfig = this.injectRetryableErrors(runtimeConfig);
    
    // Transform into ResolvedProviderConfig object to fit provier model
    const resolvedProviderConfig: ResolvedProviderConfig = toProviderConfig(internalRuntimeConfig);

    return resolvedProviderConfig;
  }

  // ── Layer 0: Defaults ──

  private createDefaultConfig(): RuntimeConfig {
    return {
      llm: DEFAULT_LLM_CONFIG,
      requestGuards: DEFAULT_REQUEST_GUARDS_CONFIG,
      verbose: false,
    };
  }

  // ── Layer 1: TOML File ──

  private async loadTomlFile(): Promise<PartialRuntimeConfig> {
    const file = Bun.file(this.configFilePath);
    if (!(await file.exists())) {
      return {};
    }

    let content: string;
    try {
      content = await file.text();
    } catch (error) {
      throw new ConfigError({
        code: ConfigErrorCode.FILE_READ_FAILED,
        source: "file",
        message: `Failed to read config file ${this.configFilePath}: ${this.errorMsg(error)}`,
        key: this.configFilePath,
        cause: error,
      });
    }

    let parsedContent: object;
    try {
      parsedContent = Bun.TOML.parse(content);
    } catch (error) {
      throw new ConfigError({
        code: ConfigErrorCode.FILE_PARSE_FAILED,
        source: "file",
        message: `Failed to parse config file ${this.configFilePath}: ${this.errorMsg(error)}`,
        key: this.configFilePath,
        cause: error,
      });
    }

    try {
      return this.normalizeTomlConfig(parsedContent);
    } catch (error) {
      throw new ConfigError({
        code: ConfigErrorCode.FILE_SCHEMA_INVALID,
        source: "file",
        message: "Invalid config in " + this.configFilePath + ": " + this.errorMsg(error),
        key: this.configFilePath,
        cause: error,
      });
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
        const llm = ensureLLMConfig(overrides);
        llm.provider = value.trim().toLowerCase();
      },
    },
    {
      key: "AUTOCOMMIT_MODEL",
      apply: (overrides, value) => {
        const llm = ensureLLMConfig(overrides);
        llm.model = value.trim();
      },
    },
    {
      key: "AUTOCOMMIT_BASE_URL",
      apply: (overrides, value) => {
        const llm = ensureLLMConfig(overrides);
        llm.baseUrl = value.trim();
      },
    },
    {
      key: "AUTOCOMMIT_API_KEY",
      apply: (overrides, value) => {
        const llm = ensureLLMConfig(overrides);
        llm.apiKey = value.trim();
      },
    },

    // ── RequestGuards.retry ──
    {
      key: "AUTOCOMMIT_RETRY_MAX_RETRIES",
      apply: (overrides, value) => {
        const retry = ensureRetryConfig(overrides);
        retry.maxRetries = assertIsInt(value, "AUTOCOMMIT_RETRY_MAX_RETRIES");
      },
    },
    {
      key: "AUTOCOMMIT_RETRY_INITIAL_DELAY_MS",
      apply: (overrides, value) => {
        const retry = ensureRetryConfig(overrides);
        retry.initialDelayMs = assertIsInt(value, "AUTOCOMMIT_RETRY_INITIAL_DELAY_MS");
      },
    },
    {
      key: "AUTOCOMMIT_RETRY_MAX_DELAY_MS",
      apply: (overrides, value) => {
        const retry = ensureRetryConfig(overrides);
        retry.maxDelayMs = assertIsInt(value, "AUTOCOMMIT_RETRY_MAX_DELAY_MS");
      },
    },
    {
      key: "AUTOCOMMIT_RETRY_FACTOR",
      apply: (overrides, value) => {
        const retry = ensureRetryConfig(overrides);
        retry.factor = assertIsFloat(value, "AUTOCOMMIT_RETRY_FACTOR");
      },
    },
    {
      key: "AUTOCOMMIT_RETRY_JITTER",
      apply: (overrides, value) => {
        const retry = ensureRetryConfig(overrides);
        retry.jitter = assertIsBool(value, "AUTOCOMMIT_RETRY_JITTER");
      },
    },

    // ── RequestGuards.timeout ──
    {
      key: "AUTOCOMMIT_TIMEOUT_MS",
      apply: (overrides, value) => {
        const timeout = ensureTimeoutConfig(overrides);
        timeout.timeoutMs = assertIsInt(value, "AUTOCOMMIT_TIMEOUT_MS");
      },
    },

    // ── RequestGuards.rateLimiter ──
    {
      key: "AUTOCOMMIT_RATE_LIMITER_MAX_RPM",
      apply: (overrides, value) => {
        const rl = ensureRateLimiterConfig(overrides);
        rl.maxRequestsPerMinute = assertIsInt(value, "AUTOCOMMIT_RATE_LIMITER_MAX_RPM");
      },
    },
    {
      key: "AUTOCOMMIT_RATE_LIMITER_MAX_QUEUE_SIZE",
      apply: (overrides, value) => {
        const rl = ensureRateLimiterConfig(overrides);
        rl.maxQueueSize = assertIsInt(value, "AUTOCOMMIT_RATE_LIMITER_MAX_QUEUE_SIZE");
      },
    },
    {
      key: "AUTOCOMMIT_RATE_LIMITER_REQUEST_TIMEOUT",
      apply: (overrides, value) => {
        const rl = ensureRateLimiterConfig(overrides);
        rl.requestTimeout = assertIsInt(value, "AUTOCOMMIT_RATE_LIMITER_REQUEST_TIMEOUT");
      },
    },
  ];

  // Default set of error patterns considered retryable.
  private static readonly DEFAULT_RETRYABLE_ERRORS: RegExp[] = [
    /ECONNRESET/i,          // Connection reset by peer
    /ETIMEDOUT/i,           // Operation timed out
    /ECONNREFUSED/i,        // Connection refused
    /socket hang up/i,      // Socket hang up
    /\b429\b/,              // Too Many Requests (rate limiting)
    /\b503\b/,              // Service Unavailable
    /\b502\b/,              // Bad Gateway
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
        throw new ConfigError({
          code: ConfigErrorCode.ENV_VAR_INVALID,
          source: "env",
          message: `Invalid env var ${entry.key}="${value}": ${this.errorMsg(error)}`,
          key: entry.key,
          cause: error,
        });
      }
    }

    return overrides;
  }

  // ── Layer 3: CLI Arguments ──
  // Handles LLMGenerationConfig overrides (--temperature, --max-tokens, etc.)
  // and the --verbose flag.
  // verbose is intentionally CLI-only: cannot be set via TOML or ENV.
  
  private loadCliOverrides(): PartialRuntimeConfig {
    const program = new Command();

    program
      .exitOverride()
      .allowUnknownOption(false)
      .allowExcessArguments(false)
      // ── LLMGenerationConfig params ──
      .addOption(new Option("--temperature <n>", "Sampling temperature (0-2)"))
      .addOption(new Option("--max-tokens <n>", "Max output tokens"))
      .addOption(new Option("--top-p <n>", "Nucleus sampling probability (0-1)"))
      .addOption(new Option("--frequency-penalty <n>", "Frequency penalty"))
      .addOption(new Option("--presence-penalty <n>", "Presence penalty"))
      // ── General flags ──
      .addOption(new Option("--verbose", "Print detailed logs during execution"));

    const argvForCommander =
      this.argv.length > 0 && this.argv[0]!.startsWith("--")
        ? this.argv
        : this.argv.slice(1);

    try {
      program.parse(argvForCommander, { from: "user" });
    } catch (error) {
      throw new ConfigError({
        code: ConfigErrorCode.CLI_ARG_INVALID,
        source: "cli",
        message: "Invalid CLI arguments: " + this.errorMsg(error),
        cause: error,
      });
    }

    const opts = program.opts<{
      temperature?: string;
      maxTokens?: string;
      topP?: string;
      frequencyPenalty?: string;
      presencePenalty?: string;
      verbose?: boolean;
    }>();

    const overrides: PartialRuntimeConfig = {};

    // ── Numeric LLM generation params ──
    // assertIsFloat/assertIsInt may throw; wrap into ConfigError.
    try {
      if (opts.temperature !== undefined) {
        ensureLLMConfig(overrides).temperature = assertIsFloat(opts.temperature, "--temperature");
      }
      if (opts.maxTokens !== undefined) {
        ensureLLMConfig(overrides).maxTokens = assertIsInt(opts.maxTokens, "--max-tokens");
      }
      if (opts.topP !== undefined) {
        ensureLLMConfig(overrides).topP = assertIsFloat(opts.topP, "--top-p");
      }
      if (opts.frequencyPenalty !== undefined) {
        ensureLLMConfig(overrides).frequencyPenalty = assertIsFloat(opts.frequencyPenalty, "--frequency-penalty");
      }
      if (opts.presencePenalty !== undefined) {
        ensureLLMConfig(overrides).presencePenalty = assertIsFloat(opts.presencePenalty, "--presence-penalty");
      }
    } catch (error) {
      throw new ConfigError({
        code: ConfigErrorCode.CLI_ARG_INVALID,
        source: "cli",
        message: "Invalid CLI arguments: " + this.errorMsg(error),
        cause: error,
      });
    }

    // ── Boolean flags ──
    // verbose is a presence flag: --verbose → true, absent → key absent from overrides.
    // Default false is guaranteed by createDefaultConfig(); no parsing needed here.
    if (opts.verbose === true) {
      overrides.verbose = true;
    }

    return overrides;
  }

  private mergeConfigs(
    defaultConfig: RuntimeConfig, 
    ...partialConfig: PartialRuntimeConfig[]
  ): RuntimeConfig {
    let mergedConfig: RuntimeConfig = deepMerge(defaultConfig, {});
    for (const partial of partialConfig) {
      mergedConfig = deepMerge(mergedConfig, partial);
    }

    return mergedConfig;
  }

  private normalizeTomlConfig(rawToml: unknown): PartialRuntimeConfig {
    const root = assertIsObject(rawToml, "root");
    const normalizedRoot: Record<string, unknown> = { ...root };

    assertOnlyKnownKeys(
      normalizedRoot,
      ["llm", "requestGuards"],
      "root",
    );

    const out: PartialRuntimeConfig = {};

    if (normalizedRoot.llm !== undefined) {
      out.llm = this.normalizeTomlLLM(normalizedRoot.llm);
    }

    if (normalizedRoot.requestGuards !== undefined) {
      out.requestGuards = this.normalizeTomlRequestGuards(normalizedRoot.requestGuards);
    }
    return out;
  }

  private normalizeTomlLLM(rawLLM: unknown): NonNullable<PartialRuntimeConfig["llm"]> {
    const llm: Record<string, unknown> = assertIsObject(rawLLM, "llm");
    assertOnlyKnownKeys(
      llm,
      [
        "provider", "model", "baseUrl",
        "temperature", "maxTokens", "topP",
        "frequencyPenalty", "presencePenalty", "apiKey",
      ],
      "llm",
    );
    const normalizeOptionalString = (value: unknown, name: string) => {
      const str = assertIsString(value, name).trim();
      return str !== "" ? str : undefined;
    };

    const out: NonNullable<PartialRuntimeConfig["llm"]> = {};

    if (llm.provider !== undefined) {
      const value = normalizeOptionalString(llm.provider, "llm.provider");
      if (value !== undefined) out.provider = value.toLowerCase();
    }
    if (llm.model !== undefined) {
      const value = normalizeOptionalString(llm.model, "llm.model");
      if (value !== undefined) out.model = value;
    }
    if (llm.apiKey !== undefined) {
      console.warn(
        "Warning: 'apiKey' is set in the TOML config file. " +
        "This is a security risk as sensitive credentials may be accidentally committed to version control or exposed in logs. " +
        "Consider using the environment variable 'AUTOCOMMIT_API_KEY' instead for better security."
      );
      const value = normalizeOptionalString(llm.apiKey, "llm.apiKey");
      if (value !== undefined) out.apiKey = value;
    }
    if (llm.baseUrl !== undefined) {
      const value = normalizeOptionalString(llm.baseUrl, "llm.baseUrl");
      if (value !== undefined) out.baseUrl = value;
    }
    if (llm.temperature !== undefined) {
      out.temperature = assertIsTomlFloat(llm.temperature, "llm.temperature");
    }
    if (llm.maxTokens !== undefined) {
      out.maxTokens = assertIsTomlInt(llm.maxTokens, "llm.maxTokens");
    }
    if (llm.topP !== undefined) {
      out.topP = assertIsTomlFloat(llm.topP, "llm.topP");
    }
    if (llm.frequencyPenalty !== undefined) {
      out.frequencyPenalty = assertIsTomlFloat(
        llm.frequencyPenalty,
        "llm.frequencyPenalty",
      );
    }
    if (llm.presencePenalty !== undefined) {
      out.presencePenalty = assertIsTomlFloat(
        llm.presencePenalty,
        "llm.presencePenalty",
      );
    }

    return out;
  }

  private normalizeTomlRequestGuards(
    rawRequestGuards: unknown,
  ): NonNullable<PartialRuntimeConfig["requestGuards"]> {
    const requestGuards: Record<string, unknown> = assertIsObject(rawRequestGuards, "requestGuards");
    assertOnlyKnownKeys(
      requestGuards,
      ["retry", "timeout", "rateLimiter"],
      "requestGuards",
    );

    const out: NonNullable<PartialRuntimeConfig["requestGuards"]> = {};

    if (requestGuards.retry !== undefined) {
      const retry = assertIsObject(requestGuards.retry, "requestGuards.retry");
      assertOnlyKnownKeys(
        retry,
        ["maxRetries", "initialDelayMs", "maxDelayMs", "factor", "jitter"],
        "requestGuards.retry",
      );

      const retryOut: NonNullable<
        NonNullable<PartialRuntimeConfig["requestGuards"]>["retry"]
      > = {};

      if (retry.maxRetries !== undefined) {
        retryOut.maxRetries = assertIsTomlInt(
          retry.maxRetries,
          "requestGuards.retry.maxRetries",
        );
      }
      if (retry.initialDelayMs !== undefined) {
        retryOut.initialDelayMs = assertIsTomlInt(
          retry.initialDelayMs,
          "requestGuards.retry.initialDelayMs",
        );
      }
      if (retry.maxDelayMs !== undefined) {
        retryOut.maxDelayMs = assertIsTomlInt(
          retry.maxDelayMs,
          "requestGuards.retry.maxDelayMs",
        );
      }
      if (retry.factor !== undefined) {
        retryOut.factor = assertIsTomlFloat(
          retry.factor,
          "requestGuards.retry.factor",
        );
      }
      if (retry.jitter !== undefined) {
        retryOut.jitter = assertIsTomlBool(
          retry.jitter,
          "requestGuards.retry.jitter",
        );
      }

      out.retry = retryOut;
    }

    if (requestGuards.timeout !== undefined) {
      const timeout = assertIsObject(requestGuards.timeout, "requestGuards.timeout");
      assertOnlyKnownKeys(timeout, ["timeoutMs"], "requestGuards.timeout");

      const timeoutOut: NonNullable<
        NonNullable<PartialRuntimeConfig["requestGuards"]>["timeout"]
      > = {};

      if (timeout.timeoutMs !== undefined) {
        timeoutOut.timeoutMs = assertIsTomlInt(
          timeout.timeoutMs,
          "requestGuards.timeout.timeoutMs",
        );
      }
      out.timeout = timeoutOut;
    }

    if (requestGuards.rateLimiter !== undefined) {
      const rateLimiter = assertIsObject(requestGuards.rateLimiter, "requestGuards.rateLimiter");
      assertOnlyKnownKeys(
        rateLimiter,
        ["maxRequestsPerMinute", "maxQueueSize", "requestTimeout"],
        "requestGuards.rateLimiter",
      );

      const rateLimiterOut: NonNullable<
        NonNullable<PartialRuntimeConfig["requestGuards"]>["rateLimiter"]
      > = {};

      if (rateLimiter.maxRequestsPerMinute !== undefined) {
        rateLimiterOut.maxRequestsPerMinute = assertIsTomlInt(
          rateLimiter.maxRequestsPerMinute,
          "requestGuards.rateLimiter.maxRequestsPerMinute",
        );
      }
      if (rateLimiter.maxQueueSize !== undefined) {
        rateLimiterOut.maxQueueSize = assertIsTomlInt(
          rateLimiter.maxQueueSize,
          "requestGuards.rateLimiter.maxQueueSize",
        );
      }
      if (rateLimiter.requestTimeout !== undefined) {
        rateLimiterOut.requestTimeout = assertIsTomlInt(
          rateLimiter.requestTimeout,
          "requestGuards.rateLimiter.requestTimeout",
        );
      }

      out.rateLimiter = rateLimiterOut;
    }
    return out;
  }

  private injectRetryableErrors(config: RuntimeConfig): InternalRuntimeConfig {
    return {
      ...config,
      requestGuards: {
        ...config.requestGuards,
        retry: {
          ...config.requestGuards.retry,
          retryableErrors: this.createRetryableErrors(),
        }
      }
    };
  }

  private createRetryableErrors(): RegExp[] {
    return ConfigLoader.DEFAULT_RETRYABLE_ERRORS.map(
      (pattern) => new RegExp(pattern.source, pattern.flags),
    );
  }

  private errorMsg(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
