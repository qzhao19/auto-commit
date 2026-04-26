import { RateLimiter } from "../../../lib/guards/rate-limit/index";
import { Retry } from "../../../lib/guards/retry/index";
import { Timeout } from "../../../lib/guards/timeout/index";
import {
  type LLMGenerationConfig,
  type ProviderInvokeOptions,
  type ResolvedProviderConfig,
} from "../../../shared/types/index";
import { ProviderError, ProviderErrorCode } from "../../../shared/exceptions/index";
import { ConfigLoader } from "../../../lib/config/index";

export abstract class BaseProvider {
  protected readonly config: ResolvedProviderConfig;

  private readonly retry: Retry;
  private readonly timeout: Timeout;
  private readonly rateLimiter: RateLimiter;
  private static defaultConfigPromise?: Promise<ResolvedProviderConfig>;

  /**
   * The constructor remains synchronous and receives the parsed configuration.
   * Subclasses call this via `super(config)`.
   */
  protected constructor(config: ResolvedProviderConfig) {
    this.config = config;

    const requestGuards = config.requestGuardsConfig;
    this.retry = new Retry(requestGuards.retry);
    this.timeout = new Timeout(requestGuards.timeout);
    this.rateLimiter = new RateLimiter(requestGuards.rateLimiter);
  }

  /**
   * A factory entry point for asynchronously loading configurations.
   * Subclasses can call this method to retrieve the configuration and 
   * then pass it to their constructor.
   */
  public static async loadConfig(
    options?: ConstructorParameters<typeof ConfigLoader>[0],
  ): Promise<ResolvedProviderConfig> {

    if (options === undefined) {
      if (!BaseProvider.defaultConfigPromise) {
        const configLoader = new ConfigLoader();
        BaseProvider.defaultConfigPromise = configLoader.load().catch((error) => {
          BaseProvider.defaultConfigPromise = undefined;
          throw error;
        });
      }
      return BaseProvider.defaultConfigPromise;
    }

    const configLoader = new ConfigLoader(options);
    return configLoader.load();
  }

  /**
   * Unified public API for invoking the LLM.
   * Internally wraps the execution in Retry → RateLimiter → Timeout → doInvoke.
   */
  public async invoke(
    prompt: string,
    generationConfigOverrides?: Partial<LLMGenerationConfig>,
    signal?: AbortSignal,
  ): Promise<string> {
    const provider: string = this.config.provider;
    const startedAt: number = Date.now();
    let attempts: number = 0;

    const model = this.config.model;
    const effectiveGenerationConfig: LLMGenerationConfig = {
      ...this.config.generationConfig,
      ...(generationConfigOverrides ?? {}),
    };

    if (signal !== undefined && signal.aborted) {
      throw ProviderError.fromUnknown(
        signal.reason ?? new Error("Request aborted"), {
          code: ProviderErrorCode.REQUEST_FAILED,
          provider: this.config.provider,
          message: "Provider invocation aborted before start",
      });
    }

    const executeOnce = async (): Promise<string> => {
      attempts++;
      const rateWaitStartedAt: number = Date.now();
      await this.waitForRateLimit(signal);
      const rateWaitMs = Date.now() - rateWaitStartedAt;

      if (this.config.verbose) {
        console.info(
          "[provider] " +
            JSON.stringify({
              event: "rate_limit_acquired",
              provider,
              attempt: attempts,
              waitMs: rateWaitMs,
            }),
        );
      }

      return this.timeout.execute(
        (timeoutSignal) => {
          const combinedSignal = signal
            ? AbortSignal.any([signal, timeoutSignal])
            : timeoutSignal;
          return this.doInvoke({ 
            prompt, 
            model, 
            generationConfig: effectiveGenerationConfig, 
            signal: combinedSignal 
          });
        } 
      );
    };

    try {
      const result = await this.retry.execute(executeOnce);

      if (this.config.verbose) {
        console.info(
          "[provider] " +
            JSON.stringify({
              event: "invoke_success",
              provider,
              attempts,
              durationMs: Date.now() - startedAt,
            }),
        );
      }

      return result;
    } catch (error) {
      const normalized = ProviderError.fromUnknown(error, {
        code: ProviderErrorCode.REQUEST_FAILED,
        provider,
        message: "Provider invocation failed",
        details: {
          attempts,
          durationMs: Date.now() - startedAt,
        },
      });

      console.error(
        "[provider] " +
          JSON.stringify({
            event: "invoke_failed",
            provider,
            code: normalized.code,
            message: normalized.message,
            attempts,
            durationMs: Date.now() - startedAt,
          }),
      );
      throw normalized;
    }
  }

  /**
   * Subclasses ONLY implement the HTTP call. All guard logic is handled above.
   */
  protected abstract doInvoke(request: ProviderInvokeOptions): Promise<string>;

  private async waitForRateLimit(signal?: AbortSignal): Promise<void> {
    if (!signal) {
      await this.rateLimiter.acquire();
      return;
    }

    const makeAbortError = (): ProviderError =>
      new ProviderError({
        code: ProviderErrorCode.REQUEST_FAILED,
        provider: this.config.provider,
        message: "Request aborted during rate limit wait",
        cause: signal.reason instanceof Error
          ? signal.reason
          : new Error(String(signal.reason ?? "Request aborted")),
      });

    if (signal.aborted) {
      throw makeAbortError();
    }

    let onAbort!: () => void;
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => reject(makeAbortError());
      signal.addEventListener("abort", onAbort, { once: true });
    });

    try {
      await Promise.race([this.rateLimiter.acquire(), aborted]);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }
}