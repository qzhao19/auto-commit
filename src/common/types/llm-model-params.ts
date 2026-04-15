/**
 * LLMModelParams defines the configuration options for a Large Language Model (LLM).
 * 
 * Properties:
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