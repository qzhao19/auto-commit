import { GuardError, GuardErrorCode } from "./guard-error";

export enum ProviderErrorCode {
  CONFIG_INVALID          = "PROVIDER_CONFIG_INVALID",
  ADAPTER_NOT_IMPLEMENTED = "PROVIDER_ADAPTER_NOT_IMPLEMENTED",
  UNSUPPORTED_PROVIDER    = "PROVIDER_UNSUPPORTED_PROVIDER",
  REQUEST_FAILED          = "PROVIDER_REQUEST_FAILED",
  EMPTY_RESPONSE          = "PROVIDER_EMPTY_RESPONSE",
  RATE_LIMITED            = "PROVIDER_RATE_LIMITED",
  REQUEST_TIMEOUT         = "PROVIDER_REQUEST_TIMEOUT",
  RETRY_EXHAUSTED         = "PROVIDER_RETRY_EXHAUSTED",
}

export class ProviderError extends Error {
  public readonly code: ProviderErrorCode;
  public readonly provider?: string;
  public readonly details?: Record<string, unknown>;
  public override readonly cause?: unknown;

  constructor(options: {
    code: ProviderErrorCode;
    message: string;
    provider?: string;
    details?: Record<string, unknown>;
    cause?: unknown;
  }) {
    super(options.message);
    this.name = "ProviderError";
    this.code = options.code;
    this.provider = options.provider;
    this.details = options.details;
    this.cause = options.cause;
  }

  public static fromUnknown(
    error: unknown,
    fallback: {
      code: ProviderErrorCode;
      provider?: string;
      message: string;
      details?: Record<string, unknown>;
    },
  ): ProviderError {
    if (error instanceof ProviderError) {
      return error;
    }

    // Mapping to GuardError
    if (error instanceof GuardError) {
      switch (error.code) {
        case GuardErrorCode.RETRY_EXHAUSTED:
          // When the retry ultimately fails, if the underlying error is a `ProviderError`, 
          // it indicates that the error itself is the root cause of the problem.
          if (error.cause instanceof ProviderError) {
            return error.cause;
          }
          return new ProviderError({
            code: ProviderErrorCode.RETRY_EXHAUSTED,
            provider: fallback.provider,
            message: error.message,
            details: fallback.details,
            cause: error,
          });
        case GuardErrorCode.RATE_LIMIT_QUEUE_FULL:
        case GuardErrorCode.RATE_LIMIT_TIMEOUT:
          return new ProviderError({
            code: ProviderErrorCode.RATE_LIMITED,
            provider: fallback.provider,
            message: error.message,
            details: fallback.details,
            cause: error,
          });
        case GuardErrorCode.REQUEST_TIMEOUT:
          return new ProviderError({
            code: ProviderErrorCode.REQUEST_TIMEOUT,
            provider: fallback.provider,
            message: error.message,
            details: fallback.details,
            cause: error,
          });
      }
    }

    const message = error instanceof Error ? error.message : String(error);
    return new ProviderError({
      code: fallback.code,
      provider: fallback.provider,
      message: fallback.message + ": " + message,
      details: fallback.details,
      cause: error,
    });
  }
}