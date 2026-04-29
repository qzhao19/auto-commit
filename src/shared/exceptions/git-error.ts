export const GitCode = {
  NOT_A_REPO:         "GIT_NOT_A_REPO",
  LOCK_FILE_EXISTS:   "GIT_LOCK_FILE_EXISTS",
  STAGING_EMPTY:      "GIT_STAGING_EMPTY",
  BISECT_IN_PROGRESS: "GIT_BISECT_IN_PROGRESS",
  COMMAND_FAILED:     "GIT_COMMAND_FAILED",
  COMMIT_FAILED:      "GIT_COMMIT_FAILED",
  BARE_REPO_UNSUPPORTED: "GIT_BARE_REPO_UNSUPPORTED",
  NOTHING_TO_COMMIT:  "GIT_NOTHING_TO_COMMIT",
} as const;

export type GitErrorCode = (typeof GitCode)[keyof typeof GitCode];

export class GitError extends Error {
  public readonly code: GitErrorCode;
  public readonly details?: Record<string, unknown>;
  public override readonly cause?: unknown;

  constructor(options: {
    code: GitErrorCode;
    message: string;
    details?: Record<string, unknown>;
    cause?: unknown;
  }) {
    super(options.message);
    this.name = "GitError";
    this.code = options.code;
    this.details = options.details;
    this.cause = options.cause;
  }
}