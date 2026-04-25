import { OpenAI } from "openai";
import { BaseProvider } from "../base-provider";
import {
  type ProviderInvokeOptions,
  type ResolvedProviderConfig,
} from "../../../../shared/types/index";
import {
  ProviderError,
  ProviderErrorCode,
} from "../../../../shared/exceptions/index";
import { 
  DEFAULT_BASE_URLS, 
  DEFAULT_SUPPORTED_PROVIDERS, 
} from "../../../../shared/constants/index";
import { isOpenAICompatibleProvider } from "../../../../lib/utils/llm/index";

export class OpenAIProvider extends BaseProvider {
  private readonly client: OpenAI;

  constructor(config: ResolvedProviderConfig) {
    super(config);

    const provider = config.provider.toLowerCase();

    if (!(DEFAULT_SUPPORTED_PROVIDERS as readonly string[]).includes(provider)) {
      throw new ProviderError({
        code: ProviderErrorCode.CONFIG_INVALID,
        provider: config.provider,
        message:
          `Provider "${config.provider}" is not supported by OpenAIProvider. ` +
          `Supported: ${DEFAULT_SUPPORTED_PROVIDERS.join(", ")}`,
      });
    }

    // Ollama's local deployment does not require a real API key,
    // a placeholder can be used instead
    if (provider !== "ollama" && (!config.apiKey || config.apiKey.trim() === "")) {
      throw new ProviderError({
        code: ProviderErrorCode.CONFIG_INVALID,
        provider: config.provider,
        message: `${config.provider} provider requires AUTOCOMMIT_API_KEY`,
      });
    }

    let baseUrl: string | undefined = this.config.baseUrl;
    if (!baseUrl && isOpenAICompatibleProvider(this.config.provider)) {
      baseUrl = DEFAULT_BASE_URLS[provider];
    }

    this.client = new OpenAI({
      apiKey:  config.apiKey || "ollama",
      baseURL: baseUrl,
    });
  }

  public static createFromConfig(config: ResolvedProviderConfig): OpenAIProvider {
    return new OpenAIProvider(config);
  }

  public static async create(
    options?: Parameters<typeof BaseProvider.loadConfig>[0],
  ): Promise<OpenAIProvider> {
    const config = await BaseProvider.loadConfig(options);
    return OpenAIProvider.createFromConfig(config);
  }

  protected override async doInvoke(options: ProviderInvokeOptions): Promise<string> {
    const { prompt, model, generationConfig, signal } = options;
    const provider = this.config.provider;

    let response: Awaited<ReturnType<OpenAI["chat"]["completions"]["create"]>>;
    try {
      response = await this.client.chat.completions.create(
        {
          model,
          messages: [{ role: "user", content: prompt }],
          temperature: generationConfig.temperature,
          top_p: generationConfig.topP,
          max_completion_tokens: generationConfig.maxTokens,
          frequency_penalty: generationConfig.frequencyPenalty,
          presence_penalty: generationConfig.presencePenalty,
        },
        { signal },
      );
    } catch (error) {
      throw ProviderError.fromUnknown(error, {
        code: ProviderErrorCode.REQUEST_FAILED,
        provider,
        message: `${provider} request failed`,
      });
    }

    const content = response.choices[0]?.message?.content;
    if (typeof content !== "string" || content.trim() === "") {
      throw new ProviderError({
        code: ProviderErrorCode.EMPTY_RESPONSE,
        provider,
        message: `${provider} returned empty content`,
      });
    }

    return content;
  }
}