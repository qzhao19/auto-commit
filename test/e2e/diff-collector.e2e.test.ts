/**
 * E2E tests for DiffCollector — exercises staging-area information collection
 * against real git repositories on disk, using the real GitRunner.
 *
 * No git commands are mocked. Every scenario is constructed by running
 * actual git commands so assertions reflect true git behavior.
 *
 * Requires: git >= 2.28 (for `git init -b <branch>`)
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GitCode, GitError } from "../../src/shared/exceptions/index";
import { DiffCollector } from "../../src/core/git/diff/diff-collector";
import { GitRunner } from "../../src/core/git/runner/git-runner";

// ── Helpers ───

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

/**
 * Writes a file and stages it (does NOT commit).
 */
async function stageFile(
  gitRoot: string,
  relPath: string,
  content: string,
): Promise<void> {
  const fullPath = join(gitRoot, relPath);
  await mkdir(join(fullPath, ".."), { recursive: true }); // ← 新增
  await writeFile(fullPath, content);
  await git(["add", relPath], gitRoot);
}

/**
 * Writes a file, stages it, and commits it.
 */
async function commitFile(
  gitRoot: string,
  relPath: string,
  content: string,
  message = "chore: test commit",
): Promise<void> {
  await stageFile(gitRoot, relPath, content);
  await git(["commit", "-m", message], gitRoot);
}

// ── Fixtures ───

let repoDir: string;

beforeEach(async () => {
  const tmp = await mkdtemp(join(tmpdir(), "diff-collector-e2e-"));
  repoDir = await realpath(tmp); // resolve macOS /var → /private/var symlink
});

afterEach(async () => {
  await rm(repoDir, { recursive: true, force: true });
});

function makeCollector(): DiffCollector {
  return new DiffCollector(new GitRunner({ cwd: repoDir }));
}

// ── Tests ───

describe("DiffCollector e2e", () => {

  // ── Empty staging area ──

  describe("collect() — empty staging area", () => {
    test("returns success=true with totalFiles=0 when nothing is staged", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "seed.txt", "seed");

      const result = await makeCollector().collect();

      expect(result.success).toBe(true);
      expect(result.summary.totalFiles).toBe(0);
      expect(result.summary.totalInsertions).toBe(0);
      expect(result.summary.totalDeletions).toBe(0);
      expect(result.summary.files).toHaveLength(0);
    });

    test("hasBinaryFiles and hasSubmodules are false on an empty staging area", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "seed.txt", "seed");

      const result = await makeCollector().collect();

      expect(result.summary.hasBinaryFiles).toBe(false);
      expect(result.summary.hasSubmodules).toBe(false);
    });
  });

  // ── Added files ──

  describe("collect() — added files (initial commit)", () => {
    test("staged new file has changeType=added and oldPath=null", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "src/index.ts", "export const x = 1;\n");

      const result = await makeCollector().collect();

      expect(result.summary.totalFiles).toBe(1);
      const file = result.summary.files[0]!;
      expect(file.path).toBe("src/index.ts");
      expect(file.changeType).toBe("added");
      expect(file.oldPath).toBeNull();
      expect(file.similarityScore).toBeNull();
    });

    test("insertions count matches actual lines added", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "three-lines.ts", "line1\nline2\nline3\n");

      const result = await makeCollector().collect();

      const file = result.summary.files[0]!;
      expect(file.insertions).toBe(3);
      expect(file.deletions).toBe(0);
    });

    test("totalInsertions equals sum of insertions across all staged files", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "a.ts", "aa\nbb\n");   // 2 lines
      await stageFile(repoDir, "b.ts", "cc\n");         // 1 line

      const result = await makeCollector().collect();

      expect(result.summary.totalFiles).toBe(2);
      expect(result.summary.totalInsertions).toBe(3);
    });

    test("diff field is null in collect() result (deferred loading)", async () => {
      await initRepo(repoDir);
      await stageFile(repoDir, "file.ts", "content\n");

      const result = await makeCollector().collect();

      expect(result.summary.files[0]!.diff).toBeNull();
    });
  });

  // ── Modified files ───

  describe("collect() — modified files", () => {
    test("modified file has changeType=modified and accurate line counts", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "file.ts", "line1\nline2\nline3\n");
      // stage a modification: replace line2 with two new lines
      await stageFile(repoDir, "file.ts", "line1\nnew-a\nnew-b\nline3\n");

      const result = await makeCollector().collect();

      const file = result.summary.files[0]!;
      expect(file.changeType).toBe("modified");
      expect(file.insertions).toBe(2);
      expect(file.deletions).toBe(1);
    });

    test("isBinary and isSubmodule are false for a normal text modification", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "file.ts", "original\n");
      await stageFile(repoDir, "file.ts", "changed\n");

      const result = await makeCollector().collect();

      const file = result.summary.files[0]!;
      expect(file.isBinary).toBe(false);
      expect(file.isSubmodule).toBe(false);
    });
  });

  // ── Deleted files ──

  describe("collect() — deleted files", () => {
    test("deleted file has changeType=deleted, insertions=0, deletions=N", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "to-delete.ts", "a\nb\nc\n");
      await git(["rm", "to-delete.ts"], repoDir);

      const result = await makeCollector().collect();

      const file = result.summary.files[0]!;
      expect(file.changeType).toBe("deleted");
      expect(file.insertions).toBe(0);
      expect(file.deletions).toBe(3);
    });

    test("deleted file contributes only to totalDeletions, not totalInsertions", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "gone.ts", "x\ny\n");
      await git(["rm", "gone.ts"], repoDir);

      const result = await makeCollector().collect();

      expect(result.summary.totalInsertions).toBe(0);
      expect(result.summary.totalDeletions).toBe(2);
    });
  });

  // ── Renamed files ──

  describe("collect() — renamed files", () => {
    test("renamed file has changeType=renamed, path=newName, oldPath=oldName", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "old-name.ts", "content\n");
      await git(["mv", "old-name.ts", "new-name.ts"], repoDir);

      const result = await makeCollector().collect();

      const file = result.summary.files[0]!;
      expect(file.changeType).toBe("renamed");
      expect(file.path).toBe("new-name.ts");
      expect(file.oldPath).toBe("old-name.ts");
    });

    test("pure rename has similarityScore=100", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "original.ts", "content\n");
      await git(["mv", "original.ts", "moved.ts"], repoDir);

      const result = await makeCollector().collect();

      expect(result.summary.files[0]!.similarityScore).toBe(100);
    });

    test("renamed file across directories: path and oldPath are full relative paths", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "src/old.ts", "content\n");
      await mkdir(join(repoDir, "lib"), { recursive: true });
      await git(["mv", "src/old.ts", "lib/new.ts"], repoDir);

      const result = await makeCollector().collect();

      const file = result.summary.files[0]!;
      expect(file.path).toBe("lib/new.ts");
      expect(file.oldPath).toBe("src/old.ts");
    });
  });

  // ── Mixed staging area ──

  describe("collect() — mixed file types in staging area", () => {
    test("mix of added, modified, deleted: totalFiles and totals are correct", async () => {
      await initRepo(repoDir);
      // Seed committed files
      await commitFile(repoDir, "modify-me.ts", "old\n");
      await commitFile(repoDir, "delete-me.ts", "x\ny\nz\n");
      // Now stage: 1 added, 1 modified, 1 deleted
      await stageFile(repoDir, "new-file.ts", "hello\nworld\n"); // +2
      await stageFile(repoDir, "modify-me.ts", "new\n");           // +1 -1
      await git(["rm", "delete-me.ts"], repoDir);                  // +0 -3

      const result = await makeCollector().collect();

      expect(result.summary.totalFiles).toBe(3);
      expect(result.summary.totalInsertions).toBe(3);   // 2 + 1 + 0
      expect(result.summary.totalDeletions).toBe(4);    // 0 + 1 + 3
    });

    test("files array preserves the same set of paths as staged", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "a.ts", "a\n");
      await commitFile(repoDir, "b.ts", "b\n");
      await stageFile(repoDir, "c.ts", "c\n");          // added
      await stageFile(repoDir, "a.ts", "a-modified\n"); // modified
      await git(["rm", "b.ts"], repoDir);                // deleted

      const result = await makeCollector().collect();

      const paths = result.summary.files.map((f) => f.path).sort();
      expect(paths).toEqual(["a.ts", "b.ts", "c.ts"]);
    });
  });

  // ── collectDiff() ────

  describe("collectDiff() — deferred full diff", () => {
    test("returns empty Map for empty paths array without calling git", async () => {
      await initRepo(repoDir);
      // No git commands for the collector needed at all
      const result = await makeCollector().collectDiff([]);

      expect(result).toBeInstanceOf(Map);
      expect(result.size).toBe(0);
    });

    test("returns Map entry keyed by staged file path", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "foo.ts", "original\n");
      await stageFile(repoDir, "foo.ts", "modified\n");

      const result = await makeCollector().collectDiff(["foo.ts"]);

      expect(result.has("foo.ts")).toBe(true);
    });

    test("diff text contains the expected hunk header and changed lines", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "target.ts", "alpha\n");
      await stageFile(repoDir, "target.ts", "beta\n");

      const result = await makeCollector().collectDiff(["target.ts"]);

      const diff = result.get("target.ts")!;
      expect(diff).toContain("@@");
      expect(diff).toContain("-alpha");
      expect(diff).toContain("+beta");
    });

    test("multi-file: each path gets its own diff section", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "a.ts", "aaa\n");
      await commitFile(repoDir, "b.ts", "bbb\n");
      await stageFile(repoDir, "a.ts", "AAA\n");
      await stageFile(repoDir, "b.ts", "BBB\n");

      const result = await makeCollector().collectDiff(["a.ts", "b.ts"]);

      expect(result.size).toBe(2);
      expect(result.get("a.ts")).toContain("-aaa");
      expect(result.get("b.ts")).toContain("-bbb");
      // Cross-contamination check
      expect(result.get("a.ts")).not.toContain("bbb");
      expect(result.get("b.ts")).not.toContain("aaa");
    });

    test("paths not in the staging area produce no Map entry", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "real.ts", "content\n");
      await stageFile(repoDir, "real.ts", "changed\n");

      const result = await makeCollector().collectDiff(["real.ts", "nonexistent.ts"]);

      expect(result.has("real.ts")).toBe(true);
      expect(result.has("nonexistent.ts")).toBe(false);
    });

    test("collect() then collectDiff() round-trip: paths from summary work as input", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "x.ts", "line\n");
      await stageFile(repoDir, "x.ts", "changed-line\n");

      const collector = makeCollector();
      const summary = await collector.collect();
      const paths = summary.summary.files.map((f) => f.path);
      const diffs = await collector.collectDiff(paths);

      expect(diffs.size).toBe(paths.length);
      for (const path of paths) {
        expect(diffs.has(path)).toBe(true);
      }
    });
  });

  // ── collect() + collectDiff() agreement ──

  describe("collect() and collectDiff() data agreement", () => {
    test("insertions from collect() match hunk line count in collectDiff()", async () => {
      await initRepo(repoDir);
      await commitFile(repoDir, "count.ts", "a\nb\nc\n");
      await stageFile(repoDir, "count.ts", "a\nb\nc\nd\ne\n"); // +2 insertions

      const collector = makeCollector();
      const summary = await collector.collect();
      const diffs = await collector.collectDiff(["count.ts"]);

      const fileInfo = summary.summary.files[0]!;
      const diff = diffs.get("count.ts")!;

      expect(fileInfo.insertions).toBe(2);
      // The diff hunk must show +d and +e
      expect(diff).toContain("+d");
      expect(diff).toContain("+e");
    });
  });

  // ── Error handling ───

  describe("collect() — error handling", () => {
    test("throws GitError when cwd is not a git repository", async () => {
      // repoDir is a plain temp dir with no git init
      await expect(makeCollector().collect()).rejects.toBeInstanceOf(GitError);
    });

    test("thrown error has code COMMAND_FAILED for a non-repo directory", async () => {
      await expect(makeCollector().collect()).rejects.toMatchObject({
        code: GitCode.COMMAND_FAILED,
      });
    });
  });

  describe("collectDiff() — error handling", () => {
    test("throws GitError when cwd is not a git repository", async () => {
      await expect(
        makeCollector().collectDiff(["file.ts"]),
      ).rejects.toBeInstanceOf(GitError);
    });
  });
});