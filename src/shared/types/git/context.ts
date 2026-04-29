/**
 * Type definitions for the RepoChecker module
 * Context and result models for environment pre-checks (Phase 0)
 */
export type GitPrecheckStep =
  | "is-repo"
  | "lock-check"
  | "staging-check"
  | "resolve-worktree"
  | "initial-commit-check"
  | "detached-head-check";

export interface GitRepoPrecheckContext {
  gitDir: string;
  workTree: string;
  isInitialCommit: boolean;
  isDetachedHead: boolean;
  currentBranch: string | null;
}

export interface GitRepoPrecheckResult {
  success: true;
  finalStep: GitPrecheckStep;
  completedSteps: GitPrecheckStep[];
  context: GitRepoPrecheckContext;
}

/**
 * Step identifiers for Phase 1: Git internal operation state detection.
 */
export type GitStateDetectStep =
  | "merge-detect"        // Detect if in merge state by reading .git/MERGE_HEAD
  | "squash-detect"       // Detect squash-merge state by reading .git/SQUASH_MSG
  | "cherry-pick-detect"  // Detect cherry-pick state by reading .git/CHERRY_PICK_HEAD
  | "revert-detect"       // Detect revert state by reading .git/REVERT_HEAD
  | "rebase-detect"       // Detect rebase state by checking .git/rebase-merge or .git/rebase-apply
  | "bisect-detect";      // Detect bisect state by reading .git/BISECT_LOG (hard exit if found)

/**
 * Current Git internal operation state.
 */
export type GitInternalOpState =
  | {
      status: "clean";             // Normal state: no special Git operations in progress.
    }
  | {
      status: "merge";
      mergeHead: string;           // Full commit hash from MERGE_HEAD file
      mergeMessage: string | null; // Content of MERGE_MSG file, used for merge commit messages
    }
  | {
      status: "squash-merge";
      squashMessage: string;       // Full content of SQUASH_MSG file
    }
  | {
      status: "cherry-pick";
      cherryPickHead: string;       // Commit hash from CHERRY_PICK_HEAD file
      originalTitle: string | null; // Original commit title from the cherry-picked commit 
    }
  | {
      status: "revert";
      revertHead: string;           // Commit hash from REVERT_HEAD file
      originalTitle: string | null; // Original commit title from the reverted commit
    }
  | {
      status: "rebase";
      rebaseType: "merge" | "apply"; // rebase-merge dir vs rebase-apply dir
      originalMessage: string | null;
    };

export interface GitInternalOpDetectResult {
  success: true;
  finalStep: GitStateDetectStep;
  completedSteps: GitStateDetectStep[];
  state: GitInternalOpState;
}
