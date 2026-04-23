
// Names of providers that support OpenAI-compatible APIs
export const OPENAI_COMPATIBLE_PROVIDERS = ["openai", "deepseek", "ollama"];
type OpenAICompatibleProvider = (typeof OPENAI_COMPATIBLE_PROVIDERS)[number];

// Default baseURL for each provider 
export const DEFAULT_BASE_URLS: Partial<Record<OpenAICompatibleProvider, string>> = {
  deepseek: "https://api.deepseek.com/v1",
  ollama: "http://localhost:11434/v1",
  openai: "https://api.openai.com/v1",
};