/**
 * LLModelParams defines the configuration options for a Large Language Model (LLM).
 * 
 * Properties:
 * - model: Default model identifier or name
 * - temperature: Controls randomness in generation. Lower values make output more deterministic.
 * - topP: Nucleus sampling parameter for diversity (0-1).
 * - frequencyPenalty: Penalizes repeated tokens based on frequency.
 * - presencePenalty: Penalizes repeated topics or ideas.
 * - maxTokens: Maximum number of tokens to generate in the output.
 */
export interface LLModelParams {
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
 * - provider：The provider name (e.g., "openai")
 */
export interface LLMProviderConfig {
  model: string;
  provider: string;
}

/**
 * Complete configuration for an LLM generation request，
 * including provider/model selection and generation parameters
 */
export interface LLMCompletionConfig {
  provider: string;
  model: string;
  modelParams: LLModelParams;
}