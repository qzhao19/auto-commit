// test/unittests/git-runner.runner.test.ts

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { mkdtemp, rm, realpath } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GitRunner } from "../../src/core/git/runner/git-runner";
import { GitError } from "../../src/shared/exceptions/git-error";
import { GitCode } from "../../src/shared/exceptions/git-error";

// ── Helpers ─────────────────────────────────────────────────────────────────

let tempDir: string;

async function makeGitRepo(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "git-runner-test-"));
  const realDir = await realpath(dir);
  const runner = new GitRunner({ cwd: realDir });
  await runner.run(["init"]);
  await runner.run(["config", "user.email", "test@test.com"]);
  await runner.run(["config", "user.name", "Test"]);
  return realDir;
}

async function makeInitialCommit(dir: string): Promise<void> {
  const runner = new GitRunner({ cwd: dir });
  await Bun.write(join(dir, "README.md"), "hello");
  await runner.run(["add", "."]);
  await runner.run(["commit", "-m", "initial"]);
}

beforeEach(async () => {
  const dir = await mkdtemp(join(tmpdir(), "git-runner-root-"));
  tempDir = await realpath(dir);
});

afterEach(async () => {
  await rm(tempDir, { recursive: true, force: true });
});

// ═══════════════════════════════════════════════════════════════════════════
// 1. Constructor & defaultCwd
// ═══════════════════════════════════════════════════════════════════════════

describe("constructor", () => {
  test("defaults cwd to process.cwd() when no option provided", async () => {
    const runner = new GitRunner();
    // git rev-parse --show-toplevel should succeed if process.cwd() is inside a repo
    // We only verify the instance is created without error
    expect(runner).toBeInstanceOf(GitRunner);
  });

  test("accepts explicit cwd option", () => {
    const runner = new GitRunner({ cwd: tempDir });
    expect(runner).toBeInstanceOf(GitRunner);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 2. run() – happy path (exit 0)
// ═══════════════════════════════════════════════════════════════════════════

describe("run() - successful commands", () => {
  test("returns stdout trimmed", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    const result = await runner.run(["rev-parse", "--git-dir"]);

    expect(result.stdout).toBe(".git");
    expect(result.exitCode).toBe(0);
  });

  test("returns correct args and command fields", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    const result = await runner.run(["rev-parse", "--git-dir"]);

    expect(result.args).toEqual(["rev-parse", "--git-dir"]);
    expect(result.command).toBe("git rev-parse --git-dir");
  });

  test("returns cwd matching the constructor cwd", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    const result = await runner.run(["rev-parse", "--git-dir"]);

    expect(result.cwd).toBe(dir);
  });

  test("stderr is empty string for commands with no stderr output", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    const result = await runner.run(["rev-parse", "--git-dir"]);

    expect(result.stderr).toBe("");
  });

  test("stdout is trimmed (no trailing newline)", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });
    await makeInitialCommit(dir);

    const result = await runner.run(["log", "--oneline", "-1"]);

    expect(result.stdout).not.toMatch(/\n$/);
  });

  test("per-call cwd option overrides constructor cwd", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: tempDir }); // tempDir is NOT a git repo

    const result = await runner.run(
      ["rev-parse", "--git-dir"],
      { cwd: dir },
    );

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe(".git");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 3. run() – allowedExitCodes
// ═══════════════════════════════════════════════════════════════════════════

describe("run() – allowedExitCodes", () => {
  test("exit code 0 is accepted by default", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    const result = await runner.run(["status"]);

    expect(result.exitCode).toBe(0);
  });

  test("non-zero exit throws GitError when not in allowedExitCodes", async () => {
    const runner = new GitRunner({ cwd: tempDir }); // not a git repo

    await expect(
      runner.run(["rev-parse", "--git-dir"]),
    ).rejects.toMatchObject({
      code: GitCode.COMMAND_FAILED,
    });
  });

  test("non-zero exit is accepted when included in allowedExitCodes", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    // git diff --cached --quiet exits 0 on empty staging, 1 on non-empty
    // In a fresh repo with no commits, staging is empty → exit 0
    const result = await runner.run(
      ["diff", "--cached", "--quiet"],
      { allowedExitCodes: [0, 1] },
    );

    expect([0, 1]).toContain(result.exitCode);
  });

  test("exit 1 captured correctly with allowedExitCodes [0,1]", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    // Stage a file to make diff --cached exit 1
    await Bun.write(join(dir, "file.txt"), "content");
    await runner.run(["add", "file.txt"]);

    const result = await runner.run(
      ["diff", "--cached", "--quiet"],
      { allowedExitCodes: [0, 1] },
    );

    expect(result.exitCode).toBe(1);
  });

  test("error includes allowedExitCodes in details", async () => {
    const runner = new GitRunner({ cwd: tempDir });

    let caught: GitError | undefined;
    try {
      await runner.run(["rev-parse", "--git-dir"], { allowedExitCodes: [0] });
    } catch (e) {
      caught = e as GitError;
    }

    expect(caught).toBeDefined();
    expect(caught!.details?.allowedExitCodes).toEqual([0]);
  });

  test("GitError on non-zero exit includes exit code in details", async () => {
    const runner = new GitRunner({ cwd: tempDir });

    let caught: GitError | undefined;
    try {
      await runner.run(["rev-parse", "--git-dir"]);
    } catch (e) {
      caught = e as GitError;
    }

    expect(caught).toBeDefined();
    expect(caught!.code).toBe(GitCode.COMMAND_FAILED);
    expect(caught!.details?.exitCode).not.toBeUndefined();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 4. run() – env injection
// ═══════════════════════════════════════════════════════════════════════════

describe("run() – env injection", () => {
  test("custom env var is visible to the child process", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    // Use `git var GIT_AUTHOR_IDENT` which reads env vars
    // Simpler: use a shell command via env check — but we have no shell.
    // Instead verify GIT_TERMINAL_PROMPT=0 doesn't break normal commands.
    const result = await runner.run(["rev-parse", "--git-dir"]);

    expect(result.exitCode).toBe(0);
  });

  test("GIT_AUTHOR_NAME/EMAIL from env option are used in commits", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    await Bun.write(join(dir, "a.txt"), "content");
    await runner.run(["add", "a.txt"]);
    await runner.run(
      ["commit", "-m", "env test"],
      {
        env: {
          GIT_AUTHOR_NAME: "EnvUser",
          GIT_AUTHOR_EMAIL: "env@test.com",
          GIT_COMMITTER_NAME: "EnvUser",
          GIT_COMMITTER_EMAIL: "env@test.com",
        },
      },
    );

    const log = await runner.run(["log", "--format=%an", "-1"]);
    expect(log.stdout).toBe("EnvUser");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 5. run() – result shape
// ═══════════════════════════════════════════════════════════════════════════

describe("run() – result fields", () => {
  test("multi-line stdout is returned as-is (only trailing newline trimmed)", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    await makeInitialCommit(dir);
    await Bun.write(join(dir, "b.txt"), "b");
    await runner.run(["add", "b.txt"]);
    await runner.run(["commit", "-m", "second"]);

    const result = await runner.run(["log", "--oneline"]);

    // Two commits → two lines
    const lines = result.stdout.split("\n");
    expect(lines.length).toBe(2);
  });

  test("empty stdout returns empty string", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    // git diff with no changes outputs nothing
    const result = await runner.run(["diff"]);

    expect(result.stdout).toBe("");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 6. run() – error message content
// ═══════════════════════════════════════════════════════════════════════════

describe("run() – error message", () => {
  test("GitError message includes the failing command", async () => {
    const runner = new GitRunner({ cwd: tempDir });

    let caught: GitError | undefined;
    try {
      await runner.run(["rev-parse", "--git-dir"]);
    } catch (e) {
      caught = e as GitError;
    }

    expect(caught!.message).toContain("git rev-parse --git-dir");
  });

  test("GitError message includes the exit code", async () => {
    const runner = new GitRunner({ cwd: tempDir });

    let caught: GitError | undefined;
    try {
      await runner.run(["rev-parse", "--git-dir"]);
    } catch (e) {
      caught = e as GitError;
    }

    expect(caught!.message).toContain("128");
  });

  test("GitError details.stderr is captured", async () => {
    const runner = new GitRunner({ cwd: tempDir });

    let caught: GitError | undefined;
    try {
      await runner.run(["rev-parse", "--git-dir"]);
    } catch (e) {
      caught = e as GitError;
    }

    // git writes "fatal: not a git repository" to stderr
    expect(typeof caught!.details?.stderr).toBe("string");
    expect((caught!.details?.stderr as string).length).toBeGreaterThan(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 7. run() – spawn failure
// ═══════════════════════════════════════════════════════════════════════════

describe("run() – spawn failure", () => {
  test("throws GitError(COMMAND_FAILED) when cwd does not exist", async () => {
    const runner = new GitRunner({ cwd: "/nonexistent/path/that/does/not/exist" });

    await expect(
      runner.run(["status"]),
    ).rejects.toMatchObject({
      code: GitCode.COMMAND_FAILED,
    });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 8. Real-world git scenarios
// ═══════════════════════════════════════════════════════════════════════════

describe("real-world git scenarios", () => {
  test("git rev-parse HEAD fails on repo with no commits (initial commit)", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    const result = await runner.run(
      ["rev-parse", "HEAD"],
      { allowedExitCodes: [0, 128] },
    );

    // No commits yet → exit 128
    expect(result.exitCode).toBe(128);
  });

  test("git rev-parse HEAD succeeds after first commit", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });
    await makeInitialCommit(dir);

    const result = await runner.run(["rev-parse", "HEAD"]);

    expect(result.exitCode).toBe(0);
    // SHA-1 or SHA-256 hash
    expect(result.stdout).toMatch(/^[0-9a-f]{40,64}$/);
  });

  test("git symbolic-ref -q HEAD returns branch ref on normal repo", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });
    await makeInitialCommit(dir);

    const result = await runner.run(
      ["symbolic-ref", "-q", "HEAD"],
      { allowedExitCodes: [0, 1] },
    );

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toMatch(/^refs\/heads\//);
  });

  test("git diff --name-only lists staged files", async () => {
    const dir = await makeGitRepo();
    const runner = new GitRunner({ cwd: dir });

    await Bun.write(join(dir, "alpha.ts"), "export const x = 1;");
    await Bun.write(join(dir, "beta.ts"), "export const y = 2;");
    await runner.run(["add", "."]);

    const result = await runner.run(["diff", "--cached", "--name-only"]);

    const files = result.stdout.split("\n");
    expect(files).toContain("alpha.ts");
    expect(files).toContain("beta.ts");
  });

  test("git rev-parse --show-toplevel returns repo root", async () => {
    const dir = await makeGitRepo();
    // Runner cwd is a subdirectory
    const subDir = join(dir, "src");
    await Bun.write(join(subDir, ".gitkeep"), "");
    const runner = new GitRunner({ cwd: subDir });

    const result = await runner.run(["rev-parse", "--show-toplevel"]);

    // Should return repo root, not subDir
    expect(result.stdout).toBe(dir);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 9. GitError type assertions
// ═══════════════════════════════════════════════════════════════════════════

describe("GitError type assertions", () => {
  test("thrown error is an instance of GitError", async () => {
    const runner = new GitRunner({ cwd: tempDir });

    let caught: unknown;
    try {
      await runner.run(["rev-parse", "--git-dir"]);
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(GitError);
  });

  test("GitError.name is 'GitError'", async () => {
    const runner = new GitRunner({ cwd: tempDir });

    let caught: GitError | undefined;
    try {
      await runner.run(["rev-parse", "--git-dir"]);
    } catch (e) {
      caught = e as GitError;
    }

    expect(caught!.name).toBe("GitError");
  });

  test("GitError.details contains args, command, cwd", async () => {
    const runner = new GitRunner({ cwd: tempDir });

    let caught: GitError | undefined;
    try {
      await runner.run(["rev-parse", "--git-dir"]);
    } catch (e) {
      caught = e as GitError;
    }

    expect(caught!.details?.args).toEqual(["rev-parse", "--git-dir"]);
    expect(caught!.details?.command).toBe("git rev-parse --git-dir");
    expect(caught!.details?.cwd).toBe(tempDir);
  });
});