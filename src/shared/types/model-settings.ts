/**
 * LLMGenerationConfig defines the configuration options for a Large Language Model (LLM).
 * 
 * Properties:
 * - model: Default model identifier or name
 * - temperature: Controls randomness in generation. Lower values make output more deterministic.
 * - topP: Nucleus sampling parameter for diversity (0-1).
 * - frequencyPenalty: Penalizes repeated tokens based on frequency.
 * - presencePenalty: Penalizes repeated topics or ideas.
 * - maxTokens: Maximum number of tokens to generate in the output.
 */
export interface LLMGenerationConfig {
  temperature?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  maxTokens?: number;
}

/**
 * Configuration identifying the LLM provider
 *
 * Properties:
 * - model: Default model identifier or name
 * - provider: The provider name (e.g., "openai")
 * - baseUrl: Custom API endpoint (optional; for self-hosted or proxy)
 * - apiKey: API key (optional in type, recommended from env at runtime)
 */
export interface LLMProviderConfig {
  model?: string;
  provider?: string;
  baseUrl?: string;
}

export interface LLMSecretConfig {
  apiKey?: string;
}

/**
 * Complete LLM configuration combining provider and generation settings.
 */
export interface LLMConfig extends LLMProviderConfig, LLMGenerationConfig, LLMSecretConfig {};

/**
 * Partial LLM configuration for incremental config loading.
 * Used to represent incomplete configurations from TOML, env vars, or CLI args.
 * 
 * Example: A user might only set temperature in the config file,
 * leaving all other fields undefined until they're merged with defaults.
 */
export type PartialLLMConfig = Partial<LLMConfig>;