import { 
  type RuntimeConfig, 
  type PartialRuntimeConfig,
  type InternalRuntimeConfig,
  type ResolvedProviderConfig,
} from "../../../shared/types/index";
import { ConfigError, ConfigErrorCode } from "../../../shared/exceptions/index";
import { DEFAULT_SUPPORTED_PROVIDERS } from "../../../shared/constants/index";

type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function deepClone<T>(value: T): T {
  if (!isPlainObject(value)) return value;

  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) {
    out[k] = deepClone(v);
  }
  return out as T;
}

function deepMergeObject<T extends object>(
  base: T, overrides: DeepPartial<T>
): T {
  const result = deepClone<T>(base) as Record<string, unknown>;

  for (const [key, value] of Object.entries(
    overrides as Record<string, unknown>,
  )) {
    if (value === undefined) continue;

    const baseValue = result[key];
    if (isPlainObject(baseValue) && isPlainObject(value)) {
      result[key] = deepMergeObject(
        baseValue,
        value as DeepPartial<typeof baseValue>,
      );
    } else {
      result[key] = deepClone(value);
    }
  }

  return result as T;
}

export function deepMerge(
  baseConfig: RuntimeConfig, 
  partialConfig: PartialRuntimeConfig,
): RuntimeConfig {
  return deepMergeObject(baseConfig, partialConfig as DeepPartial<RuntimeConfig>);
}

// ── Safe config initializers ──

export function ensureLLMConfig(overrides: PartialRuntimeConfig): NonNullable<PartialRuntimeConfig["llm"]> {
  overrides.llm ??= {};
  return overrides.llm;
}

export function ensureRequestGuardsConfig(
  overrides: PartialRuntimeConfig,
): NonNullable<PartialRuntimeConfig["requestGuards"]> {
  overrides.requestGuards ??= {};
  return overrides.requestGuards;
}

export function ensureRetryConfig(
  overrides: PartialRuntimeConfig,
): NonNullable<NonNullable<PartialRuntimeConfig["requestGuards"]>["retry"]> {
  const requestGuards = ensureRequestGuardsConfig(overrides);
  requestGuards.retry ??= {};
  return requestGuards.retry;
}

export function ensureTimeoutConfig(
  overrides: PartialRuntimeConfig,
): NonNullable<NonNullable<PartialRuntimeConfig["requestGuards"]>["timeout"]> {
  const requestGuards = ensureRequestGuardsConfig(overrides);
  requestGuards.timeout ??= {};
  return requestGuards.timeout;
}

export function ensureRateLimiterConfig(
  overrides: PartialRuntimeConfig,
): NonNullable<NonNullable<PartialRuntimeConfig["requestGuards"]>["rateLimiter"]> {
  const requestGuards = ensureRequestGuardsConfig(overrides);
  requestGuards.rateLimiter ??= {};
  return requestGuards.rateLimiter;
}

function validationError(message: string): ConfigError {
  return new ConfigError({
    code: ConfigErrorCode.VALIDATION_FAILED,
    source: "validation",
    message,
  });
}

export function validateConfig(config: RuntimeConfig): void {
  const { llm, requestGuards } = config;

  if (!llm.provider || llm.provider.trim() === "") {
    throw validationError(
      "llm.provider is required. Set llm.provider in [autocommit.toml] or export AUTOCOMMIT_PROVIDER.",
    );
  }

  const provider = llm.provider.trim().toLowerCase();
  if (!DEFAULT_SUPPORTED_PROVIDERS.includes(provider)) {
    throw validationError(
      `llm.provider must be one of: ${DEFAULT_SUPPORTED_PROVIDERS.join(", ")}, got "${llm.provider}".`,
    );
  }

  if (!llm.model || llm.model.trim() === "") {
    throw validationError(
      "llm.model is required. Set llm.model in [autocommit.toml] or export AUTOCOMMIT_MODEL.",
    );
  }

  if (llm.frequencyPenalty !== undefined && !Number.isFinite(llm.frequencyPenalty)) {
    throw validationError("frequencyPenalty must be a finite number");
  }
  if (llm.presencePenalty !== undefined && !Number.isFinite(llm.presencePenalty)) {
    throw validationError("presencePenalty must be a finite number");
  }

  const apiKey = llm.apiKey?.trim();

  // openai/deepseek require real apiKey
  if (provider !== "ollama") {
    if (!apiKey) {
      throw validationError(
        `llm.apiKey is required for provider "${llm.provider}". Set env AUTOCOMMIT_API_KEY.`,
      );
    }
    if (apiKey.toLowerCase() === "ollama") {
      throw validationError(
        `llm.apiKey for provider "${llm.provider}" cannot be "ollama".`,
      );
    }
  }

  if (llm.temperature !== undefined && (llm.temperature < 0 || llm.temperature > 2)) {
    throw validationError("temperature must be between 0 and 2");
  }
  if (llm.topP !== undefined && (llm.topP < 0 || llm.topP > 1)) {
    throw validationError("topP must be between 0 and 1");
  }
  if (llm.maxTokens !== undefined && llm.maxTokens <= 0) {
    throw validationError("maxTokens must be > 0");
  }

  if (requestGuards.retry.maxRetries < 0) {
    throw validationError("retry.maxRetries must be >= 0");
  }
  if (requestGuards.retry.initialDelayMs < 0) {
    throw validationError("retry.initialDelayMs must be >= 0");
  }
  if (requestGuards.retry.maxDelayMs < 0) {
    throw validationError("retry.maxDelayMs must be >= 0");
  }
  if (requestGuards.retry.factor <= 0) {
    throw validationError("retry.factor must be > 0");
  }
  if (requestGuards.timeout.timeoutMs <= 0) {
    throw validationError("timeout.timeoutMs must be > 0");
  }
  if (requestGuards.rateLimiter.maxRequestsPerMinute <= 0) {
    throw validationError("rateLimiter.maxRequestsPerMinute must be > 0");
  }
  if (requestGuards.rateLimiter.maxQueueSize <= 0) {
    throw validationError("rateLimiter.maxQueueSize must be > 0");
  }
  if (requestGuards.rateLimiter.requestTimeout <= 0) {
    throw validationError("rateLimiter.requestTimeout must be > 0");
  }
}


export function toProviderConfig(config: InternalRuntimeConfig): ResolvedProviderConfig {
  const provider = config.llm.provider!;
  const model = config.llm.model!;

  return {
    provider,
    model,
    apiKey: config.llm.apiKey,
    baseUrl: config.llm.baseUrl,
    verbose: config.verbose,
    generationConfig: {
      temperature: config.llm.temperature,
      maxTokens: config.llm.maxTokens,
      topP: config.llm.topP,
      frequencyPenalty: config.llm.frequencyPenalty,
      presencePenalty: config.llm.presencePenalty,
    },
    requestGuardsConfig: config.requestGuards,
  };
}