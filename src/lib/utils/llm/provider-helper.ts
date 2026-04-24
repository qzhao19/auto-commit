import { DEFAULT_SUPPORTED_PROVIDERS } from "../../../shared/constants/index";

export function isOpenAICompatibleProvider(provider: string): provider is string {
  const p: string = provider.toLowerCase();
  return DEFAULT_SUPPORTED_PROVIDERS.includes(p);
}