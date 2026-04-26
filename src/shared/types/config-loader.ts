import { type LLMConfig, type PartialLLMConfig } from "./model-settings";
import { 
  type RequestGuardsConfig, 
  type PartialRequestGuardsConfig,
  type InternalRequestGuardsConfig,
} from "./request-guards";

/**
 * Complete runtime configuration; 
 * all fields must have values for the CLI logic to execute properly
 */
export interface RuntimeConfig {
  llm: LLMConfig;
  requestGuards: RequestGuardsConfig;
  verbose: boolean;
}

/**
 * Partial runtime configuration (for intermediate steps)
 * Indicates an incomplete configuration read from a source (TOML/ENV/CLI)
 * For example, the user has only set the `model` in TOML, while other fields are undefined
 */
export type PartialRuntimeConfig = {
  llm?: PartialLLMConfig;
  requestGuards?: PartialRequestGuardsConfig;
  verbose?: boolean;
};

/**
 * Internal configuration output by ConfigLoader (with retryableErrors injected)
 */
export interface InternalRuntimeConfig extends Omit<RuntimeConfig, "requestGuards"> {
  requestGuards: InternalRequestGuardsConfig;
}