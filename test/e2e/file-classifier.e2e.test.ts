/**
 * E2E tests for FileClassifier — exercises the full DiffCollector → FileClassifier
 * pipeline against real git repositories on disk using the real GitRunner.
 *
 * No git commands are mocked. Git LFS pointer detection is exercised by staging
 * a file whose content matches the pointer spec, without requiring git-lfs installed.
 *
 * Requires: git >= 2.28 (for `git init -b <branch>`)
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DiffCollector } from "../../src/core/git/diff/diff-collector";
import { FileClassifier } from "../../src/core/git/diff/file-classifier";
import { GitRunner } from "../../src/core/git/runner/git-runner";
import { LFS_POINTER_MAGIC } from "../../src/shared/constants/index";
import type { FileClassificationResult } from "../../src/shared/types/index";

// ── Constants ────────────────────────────────────────────────────────────────

/**
 * Valid 3-line LFS pointer content (matches the v1 spec exactly).
 * 3 insertions → passes isLfsCandidate(); blob starts with LFS_POINTER_MAGIC.
 */
const LFS_POINTER_CONTENT =
  `${LFS_POINTER_MAGIC}\n` +
  "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n" +
  "size 104857600\n";

/** Minimal PNG magic header — git numstat will flag this as binary. */
const PNG_MAGIC = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00]);

// ── Helpers ───────────────────────────────────────────────────────────────────

async function git(args: string[], cwd: string): Promise<string> {
  const proc = Bun.spawn(["git", ...args], {
    cwd,
    env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    proc.exited,
  ]);
  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    throw new Error(`git ${args.join(" ")} failed (exit ${exitCode}): ${stderr.trim()}`);
  }
  return stdout.trim();
}

async function initRepo(dir: string, branch = "main"): Promise<void> {
  await git(["init", "-b", branch], dir);
  await git(["config", "user.email", "test@example.com"], dir);
  await git(["config", "user.name", "Test User"], dir);
}

async function stageFile(gitRoot: string, relPath: string, content: string): Promise<void> {
  const fullPath = join(gitRoot, relPath);
  await mkdir(join(fullPath, ".."), { recursive: true });
  await writeFile(fullPath, content);
  await git(["add", relPath], gitRoot);
}

async function stageBinary(gitRoot: string, relPath: string, data: Uint8Array): Promise<void> {
  const fullPath = join(gitRoot, relPath);
  await mkdir(join(fullPath, ".."), { recursive: true });
  await writeFile(fullPath, data);
  await git(["add", relPath], gitRoot);
}

async function commitFile(
  gitRoot: string,
  relPath: string,
  content: string,
  message = "chore: test commit",
): Promise<void> {
  await stageFile(gitRoot, relPath, content);
  await git(["commit", "-m", message], gitRoot);
}

/** Runs the full DiffCollector → FileClassifier pipeline against repoDir. */
async function run(cwd: string): Promise<FileClassificationResult> {
  const runner = new GitRunner({ cwd });
  const summary = await new DiffCollector(runner).collect();
  return new FileClassifier(runner).classify(summary);
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

let repoDir: string;

beforeEach(async () => {
  const tmp = await mkdtemp(join(tmpdir(), "file-classifier-e2e-"));
  repoDir = await realpath(tmp);
});

afterEach(async () => {
  await rm(repoDir, { recursive: true, force: true });
});

// ═══════════════════════════════════════════════════════════════════════════
// Empty staging area
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier e2e — empty staging area", () => {
  test("returns zero noise, zero content when nothing is staged", async () => {
    await initRepo(repoDir);
    await commitFile(repoDir, "seed.ts", "export {};\n");

    const result = await run(repoDir);

    expect(result.noiseCount).toBe(0);
    expect(result.contentCount).toBe(0);
    expect(result.files).toHaveLength(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Source files (content / source)
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier e2e — source files", () => {
  test("newly staged source file → isNoise:false, contentCategory:'source'", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "src/index.ts", "export const x = 1;\n");

    const result = await run(repoDir);

    expect(result.contentCount).toBe(1);
    expect(result.noiseCount).toBe(0);
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(false);
    if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
  });

  test("modified source file → content/source", async () => {
    await initRepo(repoDir);
    await commitFile(repoDir, "src/util.ts", "export const a = 1;\n");
    await stageFile(repoDir, "src/util.ts", "export const a = 2;\n");

    const result = await run(repoDir);

    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(false);
    if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
  });

  test("deleted source file → content/source (no staged blob, skips LFS check)", async () => {
    await initRepo(repoDir);
    await commitFile(repoDir, "src/old.ts", "dead code\n");
    await git(["rm", "src/old.ts"], repoDir);

    const result = await run(repoDir);

    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(false);
    if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
  });

  test("multiple source files: all content/source, contentCount matches", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "a.ts", "a\n");
    await stageFile(repoDir, "b.ts", "b\n");
    await stageFile(repoDir, "README.md", "# readme\n");

    const result = await run(repoDir);

    expect(result.contentCount).toBe(3);
    expect(result.noiseCount).toBe(0);
    result.files.forEach(cf => {
      expect(cf.isNoise).toBe(false);
      if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Lock files — exact basename
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier e2e — lock files (exact basename)", () => {
  const cases: Array<[string, string]> = [
    ["yarn.lock",          "LOCKFILE"],
    ["package-lock.json",  "LOCKFILE"],
    ["pnpm-lock.yaml",     "LOCKFILE"],
    ["Cargo.lock",         "LOCKFILE"],
    ["go.sum",             "LOCKFILE"],
    ["Gemfile.lock",       "LOCKFILE"],
    ["poetry.lock",        "LOCKFILE"],
    ["Podfile.lock",       "LOCKFILE"],
    ["composer.lock",      "LOCKFILE"],
    ["bun.lock",           "LOCKFILE"],
    ["flake.lock",         "LOCKFILE"],
  ];

  for (const [filename] of cases) {
    test(`${filename} staged at repo root → content/lockfile`, async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, filename, "# synthetic lock\nv1\n");

      const result = await run(repoDir);

      const cf = result.files[0]!;
      expect(cf.isNoise).toBe(false);
      if (!cf.isNoise) expect(cf.contentCategory).toBe("lockfile");
    });
  }

  test("yarn.lock nested in subdirectory — basename extracted correctly", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "packages/frontend/yarn.lock", "LOCKFILE\n");

    const result = await run(repoDir);

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.contentCategory).toBe("lockfile");
  });

  test("file with similar name that is not a lock file → content/source", async () => {
    await initRepo(repoDir);
    // "yarn.lock.bak" is NOT in LOCK_FILE_BASENAMES
    await stageFile(repoDir, "yarn.lock.bak", "backup\n");

    const result = await run(repoDir);

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
  });

  test("case mismatch: cargo.lock (lowercase) is not a lock file → content/source", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "cargo.lock", "# not Cargo.lock\n");

    const result = await run(repoDir);

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Lock files — path patterns
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier e2e — lock files (path patterns)", () => {
  test("gradle/dependency-locks/*.lockfile → content/lockfile", async () => {
    await initRepo(repoDir);
    await stageFile(
      repoDir,
      "gradle/dependency-locks/compileClasspath.lockfile",
      "empty=\n",
    );

    const result = await run(repoDir);

    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(false);
    if (!cf.isNoise) expect(cf.contentCategory).toBe("lockfile");
  });

  test("nested gradle dependency-locks inside a subproject → content/lockfile", async () => {
    await initRepo(repoDir);
    await stageFile(
      repoDir,
      "services/api/gradle/dependency-locks/runtimeClasspath.lockfile",
      "empty=\n",
    );

    const result = await run(repoDir);

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.contentCategory).toBe("lockfile");
  });

  test("*.opam.locked → content/lockfile", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "mylib.opam.locked", "opam-version: \"2.0\"\n");

    const result = await run(repoDir);

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.contentCategory).toBe("lockfile");
  });

  test("bare *.lockfile at top level (no gradle path) → content/source", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "something.lockfile", "custom\n");

    const result = await run(repoDir);

    const cf = result.files[0]!;
    if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Noise: binary files
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier e2e — noise: binary", () => {
  test("PNG file (binary magic header) → isNoise:true, noiseCategory:'binary'", async () => {
    await initRepo(repoDir);
    await stageBinary(repoDir, "assets/logo.png", PNG_MAGIC);

    const result = await run(repoDir);

    expect(result.noiseCount).toBe(1);
    expect(result.contentCount).toBe(0);
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(true);
    if (cf.isNoise) expect(cf.noiseCategory).toBe("binary");
  });

  test("file containing NUL bytes is detected as binary", async () => {
    await initRepo(repoDir);
    // NUL byte is the canonical git binary marker
    await stageBinary(repoDir, "model.bin", new Uint8Array([0x00, 0x01, 0x02, 0x03]));

    const result = await run(repoDir);

    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(true);
    if (cf.isNoise) expect(cf.noiseCategory).toBe("binary");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Noise: submodule
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier e2e — noise: submodule", () => {
  test("git submodule entry → isNoise:true, noiseCategory:'submodule'", async () => {
    const subDir = await mkdtemp(join(tmpdir(), "sub-e2e-"));
    try {
      // Prepare the submodule source repo
      await initRepo(subDir);
      await commitFile(subDir, "lib.ts", "export const x = 1;\n");

      // Set up parent repo with an initial commit then add submodule
      await initRepo(repoDir);
      await commitFile(repoDir, "initial.md", "# Project\n");
      await git(["-c", "protocol.file.allow=always", "submodule", "add", subDir, "vendor/lib"], repoDir);

      const result = await run(repoDir);

      const submoduleFile = result.files.find(f => f.file.isSubmodule);
      expect(submoduleFile).toBeDefined();
      expect(submoduleFile!.isNoise).toBe(true);
      if (submoduleFile!.isNoise) {
        expect(submoduleFile!.noiseCategory).toBe("submodule");
      }
    } finally {
      await rm(subDir, { recursive: true, force: true });
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Noise: LFS pointer (simulated without git-lfs)
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier e2e — noise: lfs-pointer (simulated)", () => {
  test("staged file with valid LFS pointer content (3 lines) → noise/lfs-pointer", async () => {
    await initRepo(repoDir);
    // Stage a text file whose content matches the LFS pointer spec exactly.
    // cat-file will return this content; startsWith(LFS_POINTER_MAGIC) → true.
    await stageFile(repoDir, "weights/model.bin", LFS_POINTER_CONTENT);

    const result = await run(repoDir);

    expect(result.noiseCount).toBe(1);
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(true);
    if (cf.isNoise) expect(cf.noiseCategory).toBe("lfs-pointer");
  });

  test("file starting with LFS magic but with 11+ lines → skips cat-file, classified as content/source", async () => {
    await initRepo(repoDir);
    // 11 lines: LFS magic on line 1, then 10 extra lines
    const oversized = LFS_POINTER_MAGIC + "\n" + Array.from({ length: 10 }, (_, i) => `extra-line-${i}`).join("\n") + "\n";
    await stageFile(repoDir, "fake-pointer.bin", oversized);

    const result = await run(repoDir);

    // insertions > LFS_POINTER_MAX_LINES → isLfsCandidate() = false → no cat-file call
    expect(result.noiseCount).toBe(0);
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(false);
    if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
  });

  test("deleted LFS pointer → skips cat-file, classified as content/source", async () => {
    await initRepo(repoDir);
    // Commit the LFS pointer file, then delete it
    await commitFile(repoDir, "weights/model.bin", LFS_POINTER_CONTENT);
    await git(["rm", "weights/model.bin"], repoDir);

    const result = await run(repoDir);

    // changeType=deleted → isLfsCandidate() = false → no cat-file → falls through to content
    expect(result.noiseCount).toBe(0);
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(false);
    if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
  });

  test("modified LFS pointer (1 insertion, 1 deletion) is still a candidate and detected", async () => {
    await initRepo(repoDir);
    // Commit with one OID, update to a new OID (simulates re-tracking the pointer)
    const oldPointer =
      `${LFS_POINTER_MAGIC}\n` +
      "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n" +
      "size 100\n";
    const newPointer =
      `${LFS_POINTER_MAGIC}\n` +
      "oid sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n" +
      "size 200\n";
    await commitFile(repoDir, "data.bin", oldPointer);
    await stageFile(repoDir, "data.bin", newPointer);

    const result = await run(repoDir);

    // insertions=1 (changed OID line) ≤ 10 → candidate; blob starts with magic → lfs-pointer
    const cf = result.files[0]!;
    expect(cf.isNoise).toBe(true);
    if (cf.isNoise) expect(cf.noiseCategory).toBe("lfs-pointer");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Mixed staging area
// ═══════════════════════════════════════════════════════════════════════════

describe("FileClassifier e2e — mixed staging area", () => {
  test("typical dependency update: package.json (source) + yarn.lock (lockfile)", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "package.json", "{\"name\":\"app\",\"version\":\"1.1.0\"}\n");
    await stageFile(repoDir, "yarn.lock", "# yarn lockfile v1\npackage-a@^1.0.0:\n  version \"1.0.1\"\n");

    const result = await run(repoDir);

    expect(result.noiseCount).toBe(0);
    expect(result.contentCount).toBe(2);

    const byPath = Object.fromEntries(result.files.map(f => [f.file.path, f]));
    const pkg = byPath["package.json"]!;
    const lock = byPath["yarn.lock"]!;

    expect(pkg.isNoise).toBe(false);
    if (!pkg.isNoise) expect(pkg.contentCategory).toBe("source");

    expect(lock.isNoise).toBe(false);
    if (!lock.isNoise) expect(lock.contentCategory).toBe("lockfile");
  });

  test("feature commit: source files only, no noise", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "src/feature.ts",  "export function f() {}\n");
    await stageFile(repoDir, "src/feature.test.ts", "import './feature';\n");
    await stageFile(repoDir, "README.md",       "## New feature\n");

    const result = await run(repoDir);

    expect(result.noiseCount).toBe(0);
    expect(result.contentCount).toBe(3);
    result.files.forEach(cf => {
      expect(cf.isNoise).toBe(false);
      if (!cf.isNoise) expect(cf.contentCategory).toBe("source");
    });
  });

  test("asset commit: source + binary, noiseCount reflects binary count", async () => {
    await initRepo(repoDir);
    await stageFile(repoDir, "src/icon-loader.ts", "export const load = () => {};\n");
    await stageBinary(repoDir, "assets/icon.png", PNG_MAGIC);
    await stageBinary(repoDir, "assets/font.woff2", new Uint8Array([0x77, 0x4f, 0x46, 0x32, 0x00]));

    const result = await run(repoDir);

    expect(result.noiseCount).toBe(2);
    expect(result.contentCount).toBe(1);

    const noisy = result.files.filter(f => f.isNoise);
    noisy.forEach(cf => {
      if (cf.isNoise) expect(cf.noiseCategory).toBe("binary");
    });
  });

  test("noiseCount + contentCount always equals total staged files", async () => {
    await initRepo(repoDir);
    await commitFile(repoDir, "existing.ts", "old\n");

    await stageFile(repoDir, "src/new.ts",         "code\n");
    await stageFile(repoDir, "Cargo.lock",          "lock\n");
    await stageFile(repoDir, "weights/model.bin",   LFS_POINTER_CONTENT);
    await stageBinary(repoDir, "assets/bg.png",     PNG_MAGIC);
    await stageFile(repoDir, "existing.ts",         "new\n");

    const result = await run(repoDir);

    expect(result.noiseCount + result.contentCount).toBe(result.files.length);
    expect(result.files).toHaveLength(5);
  });
});