// ── Shared types (belongs in src/shared/types/git/diff.ts) ──────────────────

/**
 * Maps directly to git's --name-status output codes.
 * "type-changed" covers T (e.g. regular file ↔ symlink).
 */
export type StagedChangeType =
  | "added"         // A
  | "modified"      // M
  | "deleted"       // D
  | "renamed"       // R — git appends similarity score: R95, R100 …
  | "copied"        // C — git appends similarity score: C80 …
  | "type-changed"; // T


/**
 * All information about a single staged file, gathered from two git commands
 * (--name-status and --numstat) keyed by path and merged in memory.
 */
export interface StagedFileChange {
  path: string;
  oldPath: string | null;
  changeType: StagedChangeType;
  similarityScore: number | null; // Rename/copy similarity score 0–100 (e.g. R95 → 95).
  isBinary: boolean;
  isSubmodule: boolean;
  insertions: number | null;
  deletions: number | null;
  diff: string | null;
}

/**
 * Aggregate view across all staged files.
 * Phase 3 reads this to decide the LLM transmission strategy before
 * requesting any full diff content.
 */
export interface StagedDiffSummary {
  totalFiles: number;
  totalInsertions: number;
  totalDeletions: number;
  hasBinaryFiles: boolean;
  hasSubmodules: boolean;
  files: StagedFileChange[];
}


/**
 * Returned by DiffCollector.collect() on success.
 * Mirrors the Result shape used by RepoChecker and StateDetector.
 */
export interface DiffCollectResult {
  success: true;
  summary: StagedDiffSummary;
}


