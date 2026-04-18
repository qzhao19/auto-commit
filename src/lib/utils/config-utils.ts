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