import { basename } from "node:path";
import {
  type ClassifiedFile,
  type FileClassificationResult,
  type FileNonNoiseCategory,
  type FileNoiseCategory,
  type StagedDiffSummary,
  type StagedFileChange,
} from "../../../shared/types/index";
import {
  LFS_POINTER_MAGIC,
  LOCK_FILE_BASENAMES,
  LOCK_FILE_PATH_PATTERNS,
} from "../../../shared/constants/index";
import { type GitRunner } from "../runner/index";

/** LFS pointer spec caps pointer size at 1024 bytes — 10 lines is a safe upper bound. */
const LFS_POINTER_MAX_LINES = 10;

export class FileClassifier {
  private readonly runner: GitRunner;

  constructor(runner: GitRunner) {
    this.runner = runner;
  }

  public async classify(
    summary: StagedDiffSummary,
  ): Promise<FileClassificationResult> {
    const files = await Promise.all(
      summary.files.map((file) => this.classifyFile(file)),
    );

    let noiseCount = 0;
    let nonNoiseCount = 0;
    for (const f of files) {
      if (f.isNoise) noiseCount++;
      else nonNoiseCount++;
    }

    return { noiseCount, nonNoiseCount, files };
  }

  private async classifyFile(file: StagedFileChange): Promise<ClassifiedFile> {
    const noiseCategory = await this.detectNoiseCategory(file);
    if (noiseCategory !== null) {
      return { file, isNoise: true, noiseCategory };
    }
    const nonNoiseCategory = this.detectNonNoiseCategory(file);
    return { file, isNoise: false, nonNoiseCategory };
  }

  private async detectNoiseCategory(
    file: StagedFileChange,
  ): Promise<FileNoiseCategory | null> {
    if (file.isSubmodule) return "submodule";
    if (file.isBinary) return "binary";
    if (this.isLfsCandidate(file) && (await this.isLfsPointer(file.path))) {
      return "lfs-pointer";
    }
    return null;
  }

  /**
   * LFS pointer spec guarantees pointer files are < 1024 bytes, meaning at most
   * ~10 lines. Skip the blob read for files that exceed this threshold.
   */
  private isLfsCandidate(file: StagedFileChange): boolean {
    if (file.changeType === "deleted") return false;

    // Empty files or modifications with only deletions cannot be LFS pointers
    if (file.insertions === null || file.insertions === 0) return false;
    return file.insertions <= LFS_POINTER_MAX_LINES;
  }

  /** Reads the staged blob and checks for the Git LFS pointer header. */
  private async isLfsPointer(path: string): Promise<boolean> {
    try {
      const result = await this.runner.run(["cat-file", "blob", `:${path}`]);
      return result.stdout.startsWith(LFS_POINTER_MAGIC);
    } catch {
      // cat-file may fail for paths not present in the index — not an LFS pointer
      return false;
    }
  }

  private detectNonNoiseCategory(
    file: StagedFileChange
  ): FileNonNoiseCategory {
    const fileBasename = basename(file.path);

    // Exact match basename
    if (LOCK_FILE_BASENAMES.has(fileBasename)) return "lockfile";

    // Regex matchs fully pathname
    for (const pattern of LOCK_FILE_PATH_PATTERNS) {
      if (pattern.test(file.path)) return "lockfile";
    }

    return "source";
  }
}
