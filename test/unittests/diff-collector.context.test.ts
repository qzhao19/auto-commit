import { describe, expect, jest, test } from "bun:test";
import { GitCode, GitError } from "../../src/shared/exceptions/index";
import type { GitRunResult } from "../../src/shared/types/index";
import type { GitRunner } from "../../src/core/git/runner/index";
import { DiffCollector } from "../../src/core/git/diff/diff-collector";

// ── Constants ────────────────────────────────────────────────────────────────

const REPO_CWD = "/repo";

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

function makeRunner(...responses: Array<GitRunResult | Error>): GitRunner {
  let idx = 0;
  return {
    run: jest.fn(async () => {
      const res = responses[idx++];
      if (res === undefined) {
        throw new Error(`makeRunner: unexpected extra run() call (#${idx})`);
      }
      if (res instanceof Error) throw res;
      return res;
    }),
  } as unknown as GitRunner;
}

/**
 * Builds a runner for collect() — delivers three responses in order:
 * [nameStatus, numstat, raw] matching Promise.all call order.
 */
function collectRunner(nameStatus: string, numstat: string, raw: string): GitRunner {
  return makeRunner(
    r({ stdout: nameStatus }),
    r({ stdout: numstat }),
    r({ stdout: raw }),
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// collect() — empty staging area
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collect() — empty staging area", () => {
  test("returns success:true with a zero-count summary when all outputs are empty", async () => {
    const collector = new DiffCollector(collectRunner("", "", ""));
    const result = await collector.collect();

    expect(result.success).toBe(true);
    expect(result.summary.totalFiles).toBe(0);
    expect(result.summary.totalInsertions).toBe(0);
    expect(result.summary.totalDeletions).toBe(0);
    expect(result.summary.hasBinaryFiles).toBe(false);
    expect(result.summary.hasSubmodules).toBe(false);
    expect(result.summary.files).toHaveLength(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collect() — basic file types
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collect() — basic file types", () => {
  test("added file: changeType=added, oldPath=null, similarityScore=null, diff=null", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0src/new.ts",
      "10\t0\tsrc/new.ts",
      ":000000 100644 0000000 abc1234 A\tsrc/new.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.files).toHaveLength(1);
    const file = result.summary.files[0]!;
    expect(file.path).toBe("src/new.ts");
    expect(file.oldPath).toBeNull();
    expect(file.changeType).toBe("added");
    expect(file.similarityScore).toBeNull();
    expect(file.insertions).toBe(10);
    expect(file.deletions).toBe(0);
    expect(file.isBinary).toBe(false);
    expect(file.isSubmodule).toBe(false);
    expect(file.diff).toBeNull();
  });

  test("modified file: changeType=modified, correct line counts", async () => {
    const collector = new DiffCollector(collectRunner(
      "M\0src/existing.ts",
      "5\t3\tsrc/existing.ts",
      ":100644 100644 abc1234 def5678 M\tsrc/existing.ts",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.changeType).toBe("modified");
    expect(file.insertions).toBe(5);
    expect(file.deletions).toBe(3);
  });

  test("deleted file: changeType=deleted, insertions=0, deletions=N", async () => {
    const collector = new DiffCollector(collectRunner(
      "D\0src/gone.ts",
      "0\t20\tsrc/gone.ts",
      ":100644 000000 abc1234 0000000 D\tsrc/gone.ts",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.changeType).toBe("deleted");
    expect(file.insertions).toBe(0);
    expect(file.deletions).toBe(20);
  });

  test("type-changed file: changeType=type-changed", async () => {
    const collector = new DiffCollector(collectRunner(
      "T\0src/link.ts",
      "0\t0\tsrc/link.ts",
      ":100644 120000 abc1234 def5678 T\tsrc/link.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.changeType).toBe("type-changed");
  });

  test("unknown status code defaults to changeType=modified", async () => {
    const collector = new DiffCollector(collectRunner(
      "X\0src/unknown.ts",
      "2\t1\tsrc/unknown.ts",
      ":100644 100644 abc1234 def5678 X\tsrc/unknown.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.changeType).toBe("modified");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collect() — renamed files
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collect() — renamed files", () => {
  test("R95: changeType=renamed, oldPath set, similarityScore=95, path=newPath", async () => {
    const collector = new DiffCollector(collectRunner(
      "R95\0src/old.ts\0src/new.ts",
      "0\t0\t{src/old.ts => src/new.ts}",
      ":100644 100644 abc1234 def5678 R95\tsrc/old.ts\tsrc/new.ts",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.changeType).toBe("renamed");
    expect(file.path).toBe("src/new.ts");
    expect(file.oldPath).toBe("src/old.ts");
    expect(file.similarityScore).toBe(95);
  });

  test("R100: similarityScore=100, perfect rename", async () => {
    const collector = new DiffCollector(collectRunner(
      "R100\0old.ts\0new.ts",
      "0\t0\t{old.ts => new.ts}",
      ":100644 100644 abc1234 def5678 R100\told.ts\tnew.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.similarityScore).toBe(100);
  });

  test("rename with common prefix: src/{lib => utils}/index.ts — stats keyed by new path", async () => {
    const collector = new DiffCollector(collectRunner(
      "R100\0src/lib/index.ts\0src/utils/index.ts",
      "3\t3\tsrc/{lib => utils}/index.ts",
      ":100644 100644 abc1234 def5678 R100\tsrc/lib/index.ts\tsrc/utils/index.ts",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.path).toBe("src/utils/index.ts");
    expect(file.oldPath).toBe("src/lib/index.ts");
    expect(file.insertions).toBe(3);
    expect(file.deletions).toBe(3);
  });

  test("rename missing newPath token in name-status output is skipped", async () => {
    // "R95\0src/old.ts" — newPath token absent → entry is skipped
    const collector = new DiffCollector(collectRunner(
      "R95\0src/old.ts",
      "",
      "",
    ));
    const result = await collector.collect();

    expect(result.summary.files).toHaveLength(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collect() — copied files
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collect() — copied files", () => {
  test("C80: changeType=copied, path=dest, oldPath=source, similarityScore=80", async () => {
    const collector = new DiffCollector(collectRunner(
      "C80\0src/orig.ts\0src/copy.ts",
      "8\t0\tsrc/copy.ts",
      ":100644 100644 abc1234 abc1234 C80\tsrc/orig.ts\tsrc/copy.ts",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.changeType).toBe("copied");
    expect(file.path).toBe("src/copy.ts");
    expect(file.oldPath).toBe("src/orig.ts");
    expect(file.similarityScore).toBe(80);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collect() — binary files
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collect() — binary files", () => {
  test("binary file: isBinary=true, insertions=null, deletions=null", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0assets/logo.png",
      "-\t-\tassets/logo.png",
      ":000000 100644 0000000 abc1234 A\tassets/logo.png",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.isBinary).toBe(true);
    expect(file.insertions).toBeNull();
    expect(file.deletions).toBeNull();
  });

  test("binary file sets hasBinaryFiles=true in summary", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0assets/logo.png",
      "-\t-\tassets/logo.png",
      ":000000 100644 0000000 abc1234 A\tassets/logo.png",
    ));
    const result = await collector.collect();

    expect(result.summary.hasBinaryFiles).toBe(true);
  });

  test("binary file does not contribute to totalInsertions/totalDeletions", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0assets/logo.png\0M\0src/index.ts",
      "-\t-\tassets/logo.png\n5\t3\tsrc/index.ts",
      ":000000 100644 0000000 abc1234 A\tassets/logo.png\n:100644 100644 abc1234 def5678 M\tsrc/index.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.totalInsertions).toBe(5);
    expect(result.summary.totalDeletions).toBe(3);
  });

  test("binary file with only '-' in insertions column is detected as binary", async () => {
    // Some git versions only put '-' in the insertions column
    const collector = new DiffCollector(collectRunner(
      "M\0data.bin",
      "-\t0\tdata.bin",
      ":100644 100644 abc1234 def5678 M\tdata.bin",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.isBinary).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collect() — submodule files
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collect() — submodule files", () => {
  test("submodule (mode 160000): isSubmodule=true, insertions=null, deletions=null", async () => {
    const collector = new DiffCollector(collectRunner(
      "M\0vendor/lib",
      "1\t1\tvendor/lib",
      ":160000 160000 abc1234 def5678 M\tvendor/lib",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.isSubmodule).toBe(true);
    expect(file.insertions).toBeNull();
    expect(file.deletions).toBeNull();
  });

  test("submodule sets hasSubmodules=true in summary", async () => {
    const collector = new DiffCollector(collectRunner(
      "M\0vendor/lib",
      "1\t1\tvendor/lib",
      ":160000 160000 abc1234 def5678 M\tvendor/lib",
    ));
    const result = await collector.collect();

    expect(result.summary.hasSubmodules).toBe(true);
  });

  test("added submodule (oldMode=000000, newMode=160000) is detected", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0extern/dep",
      "-\t-\textern/dep",
      ":000000 160000 0000000 abc1234 A\textern/dep",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.isSubmodule).toBe(true);
  });

  test("deleted submodule (oldMode=160000, newMode=000000) is detected", async () => {
    const collector = new DiffCollector(collectRunner(
      "D\0extern/dep",
      "-\t-\textern/dep",
      ":160000 000000 abc1234 0000000 D\textern/dep",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.isSubmodule).toBe(true);
  });

  test("renamed submodule: isSubmodule=true using the new (last) path from raw output", async () => {
    const collector = new DiffCollector(collectRunner(
      "R100\0old-sub\0new-sub",
      "0\t0\t{old-sub => new-sub}",
      ":160000 160000 abc1234 def5678 R100\told-sub\tnew-sub",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.isSubmodule).toBe(true);
    expect(file.path).toBe("new-sub");
  });

  test("normal file (mode 100644) is NOT marked as submodule", async () => {
    const collector = new DiffCollector(collectRunner(
      "M\0src/file.ts",
      "2\t1\tsrc/file.ts",
      ":100644 100644 abc1234 def5678 M\tsrc/file.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.isSubmodule).toBe(false);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collect() — summary aggregation
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collect() — summary aggregation", () => {
  test("totalFiles equals the number of staged entries", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0a.ts\0M\0b.ts\0D\0c.ts",
      "5\t0\ta.ts\n3\t2\tb.ts\n0\t10\tc.ts",
      ":000000 100644 0000000 aaa A\ta.ts\n:100644 100644 bbb ccc M\tb.ts\n:100644 000000 ddd 0000000 D\tc.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.totalFiles).toBe(3);
  });

  test("totalInsertions and totalDeletions are summed across all text files", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0a.ts\0M\0b.ts",
      "10\t0\ta.ts\n3\t5\tb.ts",
      ":000000 100644 0000000 aaa A\ta.ts\n:100644 100644 bbb ccc M\tb.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.totalInsertions).toBe(13);
    expect(result.summary.totalDeletions).toBe(5);
  });

  test("hasBinaryFiles=false when no binary files are staged", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0a.ts",
      "5\t0\ta.ts",
      ":000000 100644 0000000 aaa A\ta.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.hasBinaryFiles).toBe(false);
  });

  test("hasSubmodules=false when no submodules are staged", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0a.ts",
      "5\t0\ta.ts",
      ":000000 100644 0000000 aaa A\ta.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.hasSubmodules).toBe(false);
  });

  test("diff field on every file is null after collect() (deferred loading)", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0a.ts\0M\0b.ts",
      "5\t0\ta.ts\n2\t1\tb.ts",
      "",
    ));
    const result = await collector.collect();

    for (const file of result.summary.files) {
      expect(file.diff).toBeNull();
    }
  });

  test("files array in summary is ordered as returned by --name-status", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0first.ts\0D\0second.ts\0M\0third.ts",
      "1\t0\tfirst.ts\n0\t1\tsecond.ts\n2\t2\tthird.ts",
      "",
    ));
    const result = await collector.collect();

    const paths = result.summary.files.map((f) => f.path);
    expect(paths).toEqual(["first.ts", "second.ts", "third.ts"]);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collect() — error handling
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collect() — error handling", () => {
  test("re-throws GitError as-is without wrapping", async () => {
    const original = new GitError({
      code: GitCode.NOT_A_REPO,
      message: "not a git repository",
    });
    const collector = new DiffCollector(makeRunner(original));

    let caught: unknown;
    try { await collector.collect(); } catch (e) { caught = e; }

    expect(caught).toBe(original);
  });

  test("wraps non-GitError as GitError(COMMAND_FAILED) preserving message and cause", async () => {
    const plain = new TypeError("network failure");
    const collector = new DiffCollector(makeRunner(plain));

    let caught: unknown;
    try { await collector.collect(); } catch (e) { caught = e; }

    expect(caught).toBeInstanceOf(GitError);
    const e = caught as GitError;
    expect(e.code).toBe(GitCode.COMMAND_FAILED);
    expect(e.message).toBe("network failure");
    expect(e.cause).toBe(plain);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collectDiff() — empty paths
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collectDiff() — empty paths", () => {
  test("returns empty Map immediately without calling runner.run()", async () => {
    const runMock = jest.fn(async () => {
      throw new Error("runner.run() must not be called for empty paths");
    });
    const collector = new DiffCollector({ run: runMock } as unknown as GitRunner);

    const result = await collector.collectDiff([]);

    expect(result).toBeInstanceOf(Map);
    expect(result.size).toBe(0);
    expect(runMock.mock.calls).toHaveLength(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collectDiff() — single and multi-file diffs
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collectDiff() — diff splitting", () => {
  test("single-file diff: Map contains the file keyed by its current path", async () => {
    const diffOutput = [
      "diff --git a/src/foo.ts b/src/foo.ts",
      "index abc123..def456 100644",
      "--- a/src/foo.ts",
      "+++ b/src/foo.ts",
      "@@ -1,2 +1,3 @@",
      " const x = 1;",
      "+const y = 2;",
      " export { x };",
    ].join("\n");

    const collector = new DiffCollector(makeRunner(r({ stdout: diffOutput })));
    const result = await collector.collectDiff(["src/foo.ts"]);

    expect(result.has("src/foo.ts")).toBe(true);
    expect(result.get("src/foo.ts")).toContain("@@ -1,2 +1,3 @@");
  });

  test("multi-file diff: each file gets its own Map entry", async () => {
    const diffOutput = [
      "diff --git a/src/a.ts b/src/a.ts",
      "index aaa..bbb 100644",
      "--- a/src/a.ts",
      "+++ b/src/a.ts",
      "@@ -1 +1,2 @@",
      " line1",
      "+line2",
      "diff --git a/src/b.ts b/src/b.ts",
      "index ccc..ddd 100644",
      "--- a/src/b.ts",
      "+++ b/src/b.ts",
      "@@ -1 +1 @@",
      "-old",
      "+new",
    ].join("\n");

    const collector = new DiffCollector(makeRunner(r({ stdout: diffOutput })));
    const result = await collector.collectDiff(["src/a.ts", "src/b.ts"]);

    expect(result.size).toBe(2);
    expect(result.has("src/a.ts")).toBe(true);
    expect(result.has("src/b.ts")).toBe(true);
  });

  test("each file section contains only its own diff content", async () => {
    const diffOutput = [
      "diff --git a/a.ts b/a.ts",
      "index aaa..bbb 100644",
      "--- a/a.ts",
      "+++ b/a.ts",
      "@@ -1 +1,2 @@",
      " old",
      "+a-specific-line",
      "diff --git a/b.ts b/b.ts",
      "index ccc..ddd 100644",
      "--- a/b.ts",
      "+++ b/b.ts",
      "@@ -1 +1 @@",
      "-b-old",
      "+b-new",
    ].join("\n");

    const collector = new DiffCollector(makeRunner(r({ stdout: diffOutput })));
    const result = await collector.collectDiff(["a.ts", "b.ts"]);

    expect(result.get("a.ts")).not.toContain("b-old");
    expect(result.get("b.ts")).not.toContain("a-specific-line");
  });

  test("empty diff output (binary/submodule) returns empty Map", async () => {
    const collector = new DiffCollector(makeRunner(r({ stdout: "" })));
    const result = await collector.collectDiff(["assets/image.png"]);

    expect(result.size).toBe(0);
  });

  test("path with spaces: lastIndexOf ' b/' extracts the correct path", async () => {
    const diffOutput = [
      "diff --git a/my file.ts b/my file.ts",
      "index abc..def 100644",
      "--- a/my file.ts",
      "+++ b/my file.ts",
      "@@ -1 +1 @@",
      "-old",
      "+new",
    ].join("\n");

    const collector = new DiffCollector(makeRunner(r({ stdout: diffOutput })));
    const result = await collector.collectDiff(["my file.ts"]);

    expect(result.has("my file.ts")).toBe(true);
  });

  test("malformed section without ' b/' in header is skipped", async () => {
    const diffOutput = "diff --git malformed-header\nsome content";
    const collector = new DiffCollector(makeRunner(r({ stdout: diffOutput })));
    const result = await collector.collectDiff(["something"]);

    expect(result.size).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// collectDiff() — error handling
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector.collectDiff() — error handling", () => {
  test("re-throws GitError as-is", async () => {
    const original = new GitError({
      code: GitCode.COMMAND_FAILED,
      message: "command failed",
    });
    const collector = new DiffCollector(makeRunner(original));

    let caught: unknown;
    try { await collector.collectDiff(["a.ts"]); } catch (e) { caught = e; }

    expect(caught).toBe(original);
  });

  test("wraps non-GitError as GitError(COMMAND_FAILED) with cause", async () => {
    const plain = new Error("I/O error");
    const collector = new DiffCollector(makeRunner(plain));

    let caught: unknown;
    try { await collector.collectDiff(["a.ts"]); } catch (e) { caught = e; }

    expect(caught).toBeInstanceOf(GitError);
    const e = caught as GitError;
    expect(e.code).toBe(GitCode.COMMAND_FAILED);
    expect(e.cause).toBe(plain);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// parseNameStatus — edge cases via collect()
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector — parseNameStatus edge cases", () => {
  test("non-rename entry with missing path token is skipped gracefully", async () => {
    // "A" with no following path token
    const collector = new DiffCollector(collectRunner("A", "", ""));
    const result = await collector.collect();

    expect(result.summary.files).toHaveLength(0);
  });

  test("multiple entries are all parsed in declaration order", async () => {
    const collector = new DiffCollector(collectRunner(
      "A\0a.ts\0D\0b.ts\0R90\0c.ts\0d.ts",
      "1\t0\ta.ts\n0\t5\tb.ts\n0\t0\t{c.ts => d.ts}",
      "",
    ));
    const result = await collector.collect();

    expect(result.summary.files).toHaveLength(3);
    expect(result.summary.files[0]!.changeType).toBe("added");
    expect(result.summary.files[1]!.changeType).toBe("deleted");
    expect(result.summary.files[2]!.changeType).toBe("renamed");
    expect(result.summary.files[2]!.similarityScore).toBe(90);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// expandBracePath — edge cases via collect() numstat lookup
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector — expandBracePath edge cases", () => {
  test("root-level {old.ts => new.ts} expands to new path as map key", async () => {
    const collector = new DiffCollector(collectRunner(
      "R100\0old.ts\0new.ts",
      "0\t0\t{old.ts => new.ts}",
      "",
    ));
    const result = await collector.collect();

    // Stats for "new.ts" were found via expanded key
    const file = result.summary.files[0]!;
    expect(file.path).toBe("new.ts");
    expect(file.insertions).toBe(0);
    expect(file.deletions).toBe(0);
  });

  test("no brace notation: path passed through unchanged as map key", async () => {
    const collector = new DiffCollector(collectRunner(
      "M\0src/utils/helper.ts",
      "2\t1\tsrc/utils/helper.ts",
      ":100644 100644 abc def M\tsrc/utils/helper.ts",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.insertions).toBe(2);
    expect(result.summary.files[0]!.deletions).toBe(1);
  });

  test("prefix+suffix rename: src/{foo => bar}/index.ts expands to src/bar/index.ts", async () => {
    const collector = new DiffCollector(collectRunner(
      "R100\0src/foo/index.ts\0src/bar/index.ts",
      "4\t2\tsrc/{foo => bar}/index.ts",
      "",
    ));
    const result = await collector.collect();

    const file = result.summary.files[0]!;
    expect(file.path).toBe("src/bar/index.ts");
    expect(file.insertions).toBe(4);
    expect(file.deletions).toBe(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// parseSubmodulePaths — edge cases via collect()
// ═══════════════════════════════════════════════════════════════════════════

describe("DiffCollector — parseSubmodulePaths edge cases", () => {
  test("lines not starting with ':' are ignored", async () => {
    const rawOutput = "not a raw line\n:160000 160000 abc def M\tsubmod";
    const collector = new DiffCollector(collectRunner(
      "M\0submod",
      "1\t1\tsubmod",
      rawOutput,
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.isSubmodule).toBe(true);
  });

  test("line with no tab separator is ignored", async () => {
    const rawOutput = ":100644 100644 abc def M"; // no tab → no path segment
    const collector = new DiffCollector(collectRunner(
      "M\0src/file.ts",
      "2\t1\tsrc/file.ts",
      rawOutput,
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.isSubmodule).toBe(false);
  });

  test("empty raw output: no submodule detected", async () => {
    const collector = new DiffCollector(collectRunner(
      "M\0src/file.ts",
      "2\t1\tsrc/file.ts",
      "",
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.isSubmodule).toBe(false);
  });

  test("meta section with fewer than 2 space-parts is skipped", async () => {
    // Malformed — only one field before the tab
    const rawOutput = ":160000\tsubmod";
    const collector = new DiffCollector(collectRunner(
      "M\0submod",
      "1\t1\tsubmod",
      rawOutput,
    ));
    const result = await collector.collect();

    expect(result.summary.files[0]!.isSubmodule).toBe(false);
  });
});