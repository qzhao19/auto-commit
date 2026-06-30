import { describe, expect, jest, test } from "bun:test";
import type { GitRunResult, StagedFileChange, StagedDiffSummary } from "../../src/shared/types/index";
import type { GitRunner } from "../../src/core/git/runner/index";
import { FileClassifier } from "../../src/core/git/diff/file-classifier";
import { LFS_POINTER_MAGIC } from "../../src/shared/constants/index";

// ── Constants ────────────────────────────────────────────────────────────────

const REPO_CWD = "/repo";

/**
 * A realistic LFS pointer blob (3 lines, < 200 bytes).
 * The first line must start with LFS_POINTER_MAGIC exactly.
 */
const LFS_BLOB =
  `${LFS_POINTER_MAGIC}\n` +
  "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n" +
  "size 12345\n";

// ── Helpers ──────────────────────────────────────────────────────────────────

function r(overrides: Partial<GitRunResult> = {}): GitRunResult {
  return {
    args: [],
    command: "git",
    cwd: REPO_CWD,
    exitCode: 0,
    stdout: "",
    stderr: "",
    ...overrides,
  };
}

/**
 * Runner whose run() always returns the given stdout.
 * Use for tests where a cat-file call is expected.
 */
function blobRunner(stdout: string): { runner: GitRunner; runMock: jest.Mock } {
  const runMock = jest.fn(async () => r({ stdout }));
  return { runner: { run: runMock } as unknown as GitRunner, runMock };
}

/**
 * Runner whose run() should never be called.
 * Returns empty stdout so the try/catch in isLfsPointer does not swallow
 * an unexpected invocation silently.
 */
function noopRunner(): { runner: GitRunner; runMock: jest.Mock } {
  const runMock = jest.fn(async () => r());
  return { runner: { run: runMock } as unknown as GitRunner, runMock };
}

/**
 * Runner whose run() always throws — simulates a git index miss.
 * isLfsPointer() catches this and returns false.
 */
function failingRunner(): GitRunner {
  return {
    run: jest.fn(async () => { throw new Error("fatal: path not in the working tree"); }),
  } as unknown as GitRunner;
}

/** Builds a minimal StagedFileChange. Defaults to a plain text modified source file. */
function makeFile(overrides: Partial<StagedFileChange> = {}): StagedFileChange {
  return {
    path: "src/index.ts",
    oldPath: null,
    changeType: "modified",
    similarityScore: null,
    isBinary: false,
    isSubmodule: false,
    insertions: 20,   // > LFS_POINTER_MAX_LINES — skips LFS blob read by default
    deletions: 5,
    diff: null,
    ...overrides,
  };
}

function makeSummary(...files: StagedFileChange[]): StagedDiffSummary {
  return {
    totalFiles: files.length,
    totalInsertions: 0,
    totalDeletions: 0,
    hasBinaryFiles: files.some(f => f.isBinary),
    hasSubmodules: files.some(f => f.isSubmodule),
    files,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// classify() — empty staging area
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — empty staging area", () => {
  test("returns zero counts and an empty files array", async () => {
    const { runner } = noopRunner();
    const result = await new FileClassifier(runner).classify(makeSummary());

    expect(result.noiseCount).toBe(0);
    expect(result.nonNoiseCount).toBe(0);
    expect(result.files).toHaveLength(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// classify() — noise detection: binary
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — noise detection: binary", () => {
  test("isBinary:true → isNoise:true with noiseCategory 'binary'", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "assets/logo.png", isBinary: true, insertions: null });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    expect(result.noiseCount).toBe(1);
    expect(result.nonNoiseCount).toBe(0);
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(true);
    if (cf.isNoise) expect(cf.noiseCategory).toBe("binary");
  });

  test("binary file never triggers a cat-file call", async () => {
    const { runner, runMock } = noopRunner();
    await new FileClassifier(runner).classify(
      makeSummary(makeFile({ isBinary: true, insertions: null })),
    );
    expect(runMock.mock.calls).toHaveLength(0);
  });

  test("multiple binary files are all classified as noise", async () => {
    const { runner } = noopRunner();
    const files = [
      makeFile({ path: "a.png", isBinary: true, insertions: null }),
      makeFile({ path: "b.mp4", isBinary: true, insertions: null }),
      makeFile({ path: "c.pdf", isBinary: true, insertions: null }),
    ];
    const result = await new FileClassifier(runner).classify(makeSummary(...files));

    expect(result.noiseCount).toBe(3);
    expect(result.nonNoiseCount).toBe(0);
    result.files.forEach(cf => {
      expect(cf.isNoise).toBe(true);
      if (cf.isNoise) expect(cf.noiseCategory).toBe("binary");
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// classify() — noise detection: submodule
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — noise detection: submodule", () => {
  test("isSubmodule:true → isNoise:true with noiseCategory 'submodule'", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "vendor/lib", isSubmodule: true, insertions: null });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    expect(result.noiseCount).toBe(1);
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(true);
    if (cf.isNoise) expect(cf.noiseCategory).toBe("submodule");
  });

  test("submodule never triggers a cat-file call", async () => {
    const { runner, runMock } = noopRunner();
    await new FileClassifier(runner).classify(
      makeSummary(makeFile({ isSubmodule: true, insertions: null })),
    );
    expect(runMock.mock.calls).toHaveLength(0);
  });

  test("isSubmodule:true wins over isBinary:true — noiseCategory is 'submodule'", async () => {
    // submodule is checked before binary in detectNoiseCategory()
    const { runner } = noopRunner();
    const file = makeFile({ isSubmodule: true, isBinary: true, insertions: null });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    const cf = result.files[0]!;
    if (cf.isNoise) expect(cf.noiseCategory).toBe("submodule");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// classify() — noise detection: lfs-pointer
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — noise detection: lfs-pointer", () => {
  test("non-binary file with LFS blob → isNoise:true with noiseCategory 'lfs-pointer'", async () => {
    const { runner } = blobRunner(LFS_BLOB);
    const file = makeFile({ path: "weights/model.bin", isBinary: false, insertions: 3 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    expect(result.noiseCount).toBe(1);
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(true);
    if (cf.isNoise) expect(cf.noiseCategory).toBe("lfs-pointer");
  });

  test("passes correct cat-file args including the ':' index prefix", async () => {
    const { runner, runMock } = blobRunner(LFS_BLOB);
    const file = makeFile({ path: "assets/video.mp4", insertions: 3 });
    await new FileClassifier(runner).classify(makeSummary(file));

    expect(runMock.mock.calls).toHaveLength(1);
    expect(runMock.mock.calls[0]![0]).toEqual(["cat-file", "blob", ":assets/video.mp4"]);
  });

  test("blob content that does not start with LFS magic → falls through to NonNoiseFile", async () => {
    const { runner } = blobRunner("const x = 1;\nconst y = 2;\nconst z = 3;\n");
    const file = makeFile({ insertions: 3 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    expect(result.noiseCount).toBe(0);
    expect(result.nonNoiseCount).toBe(1);
    expect(result.files[0]!.isNoise).toBe(false);
  });

  test("LFS magic in the middle of blob content is not detected (must be at start)", async () => {
    const { runner } = blobRunner(`// header\n${LFS_POINTER_MAGIC}\noid sha256:abc\nsize 100\n`);
    const file = makeFile({ insertions: 4 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    expect(result.files[0]!.isNoise).toBe(false);
  });

  test("cat-file throws (index miss) → not classified as lfs-pointer", async () => {
    const result = await new FileClassifier(failingRunner()).classify(
      makeSummary(makeFile({ insertions: 3 })),
    );

    expect(result.noiseCount).toBe(0);
    expect(result.nonNoiseCount).toBe(1);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// classify() — LFS candidate gating (isLfsCandidate)
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — LFS candidate gating", () => {
  test("changeType:'deleted' → skips cat-file regardless of insertion count", async () => {
    const { runner, runMock } = noopRunner();
    // insertions:3 would normally qualify, but deleted files have no staged blob
    const file = makeFile({ changeType: "deleted", insertions: 3 });
    await new FileClassifier(runner).classify(makeSummary(file));

    expect(runMock.mock.calls).toHaveLength(0);
  });

  test("insertions:null → skips cat-file", async () => {
    const { runner, runMock } = noopRunner();
    const file = makeFile({ insertions: null, isBinary: false, isSubmodule: false });
    await new FileClassifier(runner).classify(makeSummary(file));

    expect(runMock.mock.calls).toHaveLength(0);
  });

  test("insertions:11 (> LFS_POINTER_MAX_LINES) → skips cat-file", async () => {
    const { runner, runMock } = noopRunner();
    await new FileClassifier(runner).classify(
      makeSummary(makeFile({ insertions: 11 })),
    );
    expect(runMock.mock.calls).toHaveLength(0);
  });

  test("insertions:10 (= LFS_POINTER_MAX_LINES, boundary) → calls cat-file", async () => {
    const { runner, runMock } = blobRunner("not lfs");
    await new FileClassifier(runner).classify(
      makeSummary(makeFile({ insertions: 10 })),
    );
    expect(runMock.mock.calls).toHaveLength(1);
  });

  test("insertions:1 (minimum possible) → calls cat-file", async () => {
    const { runner, runMock } = blobRunner("not lfs");
    await new FileClassifier(runner).classify(
      makeSummary(makeFile({ insertions: 1 })),
    );
    expect(runMock.mock.calls).toHaveLength(1);
  });

  test("added file with insertions:3 (typical LFS pointer) → calls cat-file", async () => {
    const { runner, runMock } = blobRunner(LFS_BLOB);
    await new FileClassifier(runner).classify(
      makeSummary(makeFile({ changeType: "added", insertions: 3 })),
    );
    expect(runMock.mock.calls).toHaveLength(1);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// classify() — content detection: lockfile (exact basename)
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — content detection: lockfile (exact basename)", () => {
  const knownLockfiles = [
    "yarn.lock",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "pnpm-lock.yaml",
    "bun.lock",
    "Cargo.lock",
    "go.sum",
    "Gemfile.lock",
    "poetry.lock",
    "Podfile.lock",
    "composer.lock",
    "pubspec.lock",
    "mix.lock",
    "flake.lock",
    "renv.lock",
  ];

  for (const name of knownLockfiles) {
    test(`${name} → nonNoiseCategory:'lockfile'`, async () => {
      const { runner } = noopRunner();
      const file = makeFile({ path: name, insertions: 200 });
      const result = await new FileClassifier(runner).classify(makeSummary(file));

      const cf = result.files[0]!;
      expect(cf.isNoise).toBe(false);
      if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("lockfile");
    });
  }

  test("lockfile nested in a subdirectory — basename extracted correctly", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "packages/app/yarn.lock", insertions: 500 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("lockfile");
  });

  test("name that contains 'lock' but does not match exactly → 'source'", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "my-yarn.lock.bak", insertions: 50 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("source");
  });

  test("case-sensitive: Cargo.lock matches, cargo.lock does not", async () => {
    const { runner } = noopRunner();
    const wrong = makeFile({ path: "cargo.lock", insertions: 50 });
    const result = await new FileClassifier(runner).classify(makeSummary(wrong));

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("source");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// classify() — content detection: lockfile (path pattern)
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — content detection: lockfile (path pattern)", () => {
  test("gradle/dependency-locks/*.lockfile → nonNoiseCategory:'lockfile'", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "gradle/dependency-locks/compileClasspath.lockfile", insertions: 50 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(false);
    if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("lockfile");
  });

  test("nested subproject gradle dependency lock → also matches", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "services/api/gradle/dependency-locks/runtimeClasspath.lockfile", insertions: 50 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("lockfile");
  });

  test("*.opam.locked → nonNoiseCategory:'lockfile'", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "mylib.opam.locked", insertions: 30 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("lockfile");
  });

  test("*.opam.locked in subdirectory → nonNoiseCategory:'lockfile'", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "opam/packages/mylib.opam.locked", insertions: 30 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("lockfile");
  });

  test("bare *.lockfile suffix without gradle path → 'source' (pattern does not match)", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "something.lockfile", insertions: 50 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("source");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// classify() — content detection: source
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — content detection: source", () => {
  const sourcePaths = [
    "src/index.ts",
    "README.md",
    "Dockerfile",
    ".env.example",
    "tsconfig.json",
    "src/components/Button.tsx",
    ".github/workflows/ci.yml",
    "Makefile",
    "src/proto/user.proto",
  ];

  for (const path of sourcePaths) {
    test(`${path} → isNoise:false, nonNoiseCategory:'source'`, async () => {
      const { runner } = noopRunner();
      const file = makeFile({ path, insertions: 50 });
      const result = await new FileClassifier(runner).classify(makeSummary(file));

      const cf = result.files[0]!;
      expect(cf.isNoise).toBe(false);
      if (!cf.isNoise) expect(cf.nonNoiseCategory).toBe("source");
    });
  }
});

// ═══════════════════════════════════════════════════════════════════════════
// classify() — aggregate counts and result shape
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier.classify() — aggregate counts and result shape", () => {
  test("all noise → noiseCount equals files.length, nonNoiseCount:0", async () => {
    const { runner } = noopRunner();
    const files = [
      makeFile({ path: "a.png",  isBinary: true,    insertions: null }),
      makeFile({ path: "b.mp4", isBinary: true,    insertions: null }),
      makeFile({ path: "sub",   isSubmodule: true, insertions: null }),
    ];
    const result = await new FileClassifier(runner).classify(makeSummary(...files));

    expect(result.noiseCount).toBe(3);
    expect(result.nonNoiseCount).toBe(0);
    expect(result.files).toHaveLength(3);
  });

  test("all content → nonNoiseCount equals files.length, noiseCount:0", async () => {
    const { runner } = noopRunner();
    const files = [
      makeFile({ path: "src/a.ts",  insertions: 50 }),
      makeFile({ path: "yarn.lock", insertions: 200 }),
      makeFile({ path: "README.md", insertions: 20 }),
    ];
    const result = await new FileClassifier(runner).classify(makeSummary(...files));

    expect(result.noiseCount).toBe(0);
    expect(result.nonNoiseCount).toBe(3);
  });

  test("mixed: noiseCount + nonNoiseCount === files.length", async () => {
    const { runner } = noopRunner();
    const files = [
      makeFile({ path: "src/a.ts",   insertions: 50 }),
      makeFile({ path: "logo.png",   isBinary: true,    insertions: null }),
      makeFile({ path: "yarn.lock",  insertions: 300 }),
      makeFile({ path: "vendor/sub", isSubmodule: true, insertions: null }),
    ];
    const result = await new FileClassifier(runner).classify(makeSummary(...files));

    expect(result.noiseCount).toBe(2);
    expect(result.nonNoiseCount).toBe(2);
    expect(result.noiseCount + result.nonNoiseCount).toBe(result.files.length);
  });

  test("output files array preserves original order", async () => {
    const { runner } = noopRunner();
    const files = [
      makeFile({ path: "src/a.ts",  insertions: 50 }),
      makeFile({ path: "logo.png",  isBinary: true, insertions: null }),
      makeFile({ path: "yarn.lock", insertions: 200 }),
    ];
    const result = await new FileClassifier(runner).classify(makeSummary(...files));

    expect(result.files[0]!.file.path).toBe("src/a.ts");
    expect(result.files[1]!.file.path).toBe("logo.png");
    expect(result.files[2]!.file.path).toBe("yarn.lock");
  });

  test("each ClassifiedFile holds a reference to the original StagedFileChange", async () => {
    const { runner } = noopRunner();
    const file = makeFile({ path: "src/a.ts", insertions: 50 });
    const result = await new FileClassifier(runner).classify(makeSummary(file));

    expect(result.files[0]!.file).toBe(file);
  });
});