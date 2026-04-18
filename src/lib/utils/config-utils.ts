import { 
  type RuntimeConfig, 
  type PartialRuntimeConfig 
} from "../../shared/types/index";

type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function deepMergeObject<T extends object>(
  base: T, overrides: DeepPartial<T>
): T {
  const result: Record<string, unknown> = {
    ...(base as Record<string, unknown>),
  };

  for (const [key, overrideValue] of Object.entries(
    overrides as Record<string, unknown>,
  )) {
    if (overrideValue === undefined) continue;

    const baseValue = result[key];
    if (isPlainObject(baseValue) && isPlainObject(overrideValue)) {
      result[key] = deepMergeObject(
        baseValue,
        overrideValue as DeepPartial<Record<string, unknown>>,
      );
    } else {
      result[key] = overrideValue;
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