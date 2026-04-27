/**
 * Type definitions for the RepoChecker module
 * Context and result models for environment pre-checks (Phase 0)
 */
export type PrecheckStep =
  | "is-repo"
  | "lock-check"
  | "staging-check"
  | "resolve-worktree"
  | "initial-commit-check"
  | "detached-head-check";

export type StagingState =
  | {
      status: "has-staged-changes";
    }
  | {
      status: "staging-empty-with-unstaged";
      unstagedFiles: string[];
    }
  | {
      status: "nothing-to-commit";
    };

export interface RepoPrecheckContext {
  gitDir: string;
  workTree: string;
  isInitialCommit: boolean;
  isDetachedHead: boolean;
  currentBranch: string | null;
  stagingState: StagingState;
}

export interface RepoPrecheckResult {
  success: true;
  finalStep: PrecheckStep;
  completedSteps: PrecheckStep[];
  context: RepoPrecheckContext;
}
