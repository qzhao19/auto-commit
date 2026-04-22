export enum GuardErrorCode {
  RATE_LIMIT_INVALID_PARAMETER = "GUARD_RATE_LIMIT_INVALID_PARAMETER",
  RATE_LIMIT_QUEUE_FULL        = "GUARD_RATE_LIMIT_QUEUE_FULL",
  RATE_LIMIT_TIMEOUT           = "GUARD_RATE_LIMIT_TIMEOUT",
  REQUEST_TIMEOUT              = "GUARD_REQUEST_TIMEOUT",
  RETRY_EXHAUSTED              = "GUARD_RETRY_EXHAUSTED",
}

export class GuardError extends Error {
  public readonly code: GuardErrorCode;
  public override readonly cause?: unknown;

  constructor(options: { code: GuardErrorCode; message: string; cause?: unknown }) {
    super(options.message);
    this.name = "GuardError";
    this.code = options.code;
    this.cause = options.cause;
  }
}