import { type StagedFileChange } from "./diff";

/**
 * Three cases cover the entire former D-layer:
 *   - "binary"      → isBinary: true (images, fonts, executables, …)
 *   - "submodule"   → isSubmodule: true
 *   - "lfs-pointer" → diff head matches LFS_POINTER_MAGIC (blob is stored in Git LFS)
 */
export type FileNoiseCategory = "binary" | "submodule" | "lfs-pointer";

export interface NoiseFile {
  readonly file: StagedFileChange;
  readonly isNoise: true;
  readonly noiseCategory: FileNoiseCategory;
}

/**
 * Two non-noise categories:
 *   - "source"   → regular source code file
 *   - "lockfile" → auto-generated dependency lock file (package-lock.json, Cargo.lock, …)
 */
export type FileNonNoiseCategory = "source" | "lockfile";

export interface NonNoiseFile {
  readonly file: StagedFileChange;
  readonly isNoise: false;
  readonly nonNoiseCategory: FileNonNoiseCategory;
}

export type ClassifiedFile = NoiseFile | NonNoiseFile;

export interface FileClassificationResult {
  readonly noiseCount: number;
  readonly nonNoiseCount: number;
  readonly files: readonly ClassifiedFile[];
}
