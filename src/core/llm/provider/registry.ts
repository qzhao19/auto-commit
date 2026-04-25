import { BaseProvider } from "./base-provider";
import { ProviderError, ProviderErrorCode } from "../../../shared/exceptions/index";
import { OpenAIProvider } from "./adapters/index";

export async function createProvider(
  options?: Parameters<typeof BaseProvider.loadConfig>[0],
): Promise<BaseProvider> {
  const config = await BaseProvider.loadConfig(options);
  const provider = config.provider.toLowerCase();

  if (provider === "openai" || provider === "deepseek" || provider === "ollama") {
    return OpenAIProvider.createFromConfig(config);
  }

  if (provider === "anthropic") {
    throw new ProviderError({
      code: ProviderErrorCode.ADAPTER_NOT_IMPLEMENTED,
      provider: config.provider,
      message: "Provider adapter not implemented yet",
    });
  }

  throw new ProviderError({
    code: ProviderErrorCode.UNSUPPORTED_PROVIDER,
    provider: config.provider,
    message: "Unsupported provider",
  });
}