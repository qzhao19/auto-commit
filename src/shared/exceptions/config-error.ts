export enum ConfigErrorCode {
  FILE_READ_FAILED    = "CONFIG_FILE_READ_FAILED",
  FILE_PARSE_FAILED   = "CONFIG_FILE_PARSE_FAILED",
  FILE_SCHEMA_INVALID = "CONFIG_FILE_SCHEMA_INVALID",
  ENV_VAR_INVALID     = "CONFIG_ENV_VAR_INVALID",
  CLI_ARG_INVALID     = "CONFIG_CLI_ARG_INVALID",
  VALIDATION_FAILED   = "CONFIG_VALIDATION_FAILED",
}

type ConfigSource = "file" | "env" | "cli" | "validation";

export class ConfigError extends Error {
  public readonly code: ConfigErrorCode;
  public readonly source: ConfigSource;
  public readonly key?: string;
  public override readonly cause?: unknown;

  constructor(options: {
    code: ConfigErrorCode;
    source: ConfigSource;
    message: string;
    key?: string;
    cause?: unknown;
  }) {
    super(options.message);
    this.name = "ConfigError";
    this.code = options.code;
    this.source = options.source;
    this.key = options.key;
    this.cause = options.cause;
  }
}
