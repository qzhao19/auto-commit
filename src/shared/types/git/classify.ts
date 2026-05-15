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
  readonly noiseKind: FileNoiseCategory;
}

export interface ContentFile {
  readonly file: StagedFileChange;
  readonly isNoise: false;
  // No extra fields here — budget estimation is a separate sub-step
}

export type ClassifiedFile = NoiseFile | ContentFile;

export interface FileClassificationResult {
  readonly noiseCount: number;
  readonly contentCount: number;
  readonly files: readonly ClassifiedFile[];
}

/** First line of every Git LFS pointer file */
export const LFS_POINTER_MAGIC = "version https://git-lfs.github.com/objects/";