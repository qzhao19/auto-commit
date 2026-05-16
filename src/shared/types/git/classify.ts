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

export type FileContentCategory = "source" | "lockfile";

export interface ContentFile {
  readonly file: StagedFileChange;
  readonly isNoise: false;
  readonly contentCategory: FileContentCategory;
}

export type ClassifiedFile = NoiseFile | ContentFile;

export interface FileClassificationResult {
  readonly noiseCount: number;
  readonly contentCount: number;
  readonly files: readonly ClassifiedFile[];
}
