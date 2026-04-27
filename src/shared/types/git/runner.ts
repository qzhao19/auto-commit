/**
 * Type definitions for the GitRunner module
 * Low-level abstraction responsible for executing Git commands
 */
export interface GitRunOptions {
  cwd?: string;
  env?: Record<string, string>;
  allowedExitCodes?: readonly number[];
}

export interface GitRunResult {
  args: string[];
  command: string;
  cwd: string;
  exitCode: number;
  stdout: string;
  stderr: string;
}