import { GitCode, GitError } from "../../../shared/exceptions/index";
import {
  type StagedChangeType,
  type StagedFileChange,
  type StagedDiffSummary,
  type DiffCollectResult,
} from "../../../shared/types/index";
import { type GitRunner } from "../runner/index";

export class DiffCollector {
  private readonly runner: GitRunner;

  constructor(runner: GitRunner) {
    this.runner = runner;
  }

  /**
   * Collects the staged diff summary in three parallel git commands.
   * Full diff text is NOT loaded — call collectDiff() on demand.
   */
  public async collect(): Promise<DiffCollectResult> {
    try {
      const [nameStatusResult, numstatResult, rawResult] = await Promise.all([
        // NUL-delimited: authoritative path/type/similarity
        this.runner.run(["diff", "--cached", "--name-status", "-z", "-M"]),
        // Line-by-line with brace expansion: accurate line counts + binary marker
        this.runner.run(["diff", "--cached", "--numstat", "-M"]),
        // Line-by-line: mode 160000 ↔ submodule
        this.runner.run(["diff", "--cached", "--raw", "-M"]),
      ]);

      const nameStatusEntries = this.parseNameStatus(nameStatusResult.stdout);
      const numstatMap        = this.parseNumstat(numstatResult.stdout);
      const submodulePaths    = this.parseSubmodulePaths(rawResult.stdout);

      const files: StagedFileChange[] = nameStatusEntries.map((entry) => {
        const stats       = numstatMap.get(entry.path);
        const isBinary    = stats?.isBinary ?? false;
        const isSubmodule = submodulePaths.has(entry.path);

        return {
          path:            entry.path,
          oldPath:         entry.oldPath,
          changeType:      entry.changeType,
          similarityScore: entry.similarityScore,
          isBinary,
          isSubmodule,
          // Binary and submodule entries carry no meaningful line count
          insertions: isBinary || isSubmodule ? null : (stats?.insertions ?? null),
          deletions:  isBinary || isSubmodule ? null : (stats?.deletions  ?? null),
          diff: null,
        };
      });

      return { success: true, summary: this.buildSummary(files) };

    } catch (error) {
      if (error instanceof GitError) throw error;
      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message: error instanceof Error ? error.message : String(error),
        cause: error,
      });
    }
  }

  /**
   * Fetches the full unified diff for the specified paths.
   * Returns Map<currentPath, diffText>. Binary / submodule paths
   * will have no entry (git emits no hunk text for them).
   */
  public async collectDiff(paths: readonly string[]): Promise<Map<string, string>> {
    if (paths.length === 0) return new Map();

    try {
      const result = await this.runner.run(["diff", "--cached", "--", ...paths]);
      return this.splitDiffByFile(result.stdout);
    } catch (error) {
      if (error instanceof GitError) throw error;
      throw new GitError({
        code: GitCode.COMMAND_FAILED,
        message: error instanceof Error ? error.message : String(error),
        cause: error,
      });
    }
  }

  // ── Parser: --name-status -z -M ──

  /**
   * Parses NUL-delimited output from `git diff --cached --name-status -z -M`.
   *
   * Token sequences per entry:
   *   A/M/D/T  → status \0 path \0
   *   R{n}/C{n} → status \0 oldPath \0 newPath \0
   */
  private parseNameStatus(raw: string): Array<{
    path: string;
    oldPath: string | null;
    changeType: StagedChangeType;
    similarityScore: number | null;
  }> {
    if (!raw) return [];

    const tokens  = raw.split("\0").filter((t) => t.length > 0);
    const entries: ReturnType<typeof this.parseNameStatus> = [];
    let i = 0;

    while (i < tokens.length) {
      const status = tokens[i];
      if (status === undefined) break;

      if (status.startsWith("R") || status.startsWith("C")) {
        const score   = parseInt(status.slice(1), 10);
        const oldPath = tokens[i + 1];
        const newPath = tokens[i + 2];

        if (oldPath === undefined || newPath === undefined) { i++; continue; }

        entries.push({
          path:            newPath,
          oldPath,
          changeType:      status.startsWith("R") ? "renamed" : "copied",
          similarityScore: Number.isNaN(score) ? null : score,
        });
        i += 3;
      } else {
        const path = tokens[i + 1];
        if (path === undefined) { i++; continue; }

        entries.push({
          path,
          oldPath:         null,
          changeType:      this.toChangeType(status),
          similarityScore: null,
        });
        i += 2;
      }
    }

    return entries;
  }

  private toChangeType(code: string): StagedChangeType {
    switch (code[0]) {
      case "A": return "added";
      case "M": return "modified";
      case "D": return "deleted";
      case "T": return "type-changed";
      default:  return "modified";
    }
  }

  // ── Parser: --numstat -M ──

  /**
   * Parses line-by-line output from `git diff --cached --numstat -M`.
   *
   * Line formats:
   *   <ins>\t<del>\t<path>              — regular file
   *   -\t-\t<path>                      — binary file
   *   <ins>\t<del>\t{old => new}        — renamed at root level
   *   <ins>\t<del>\tprefix/{a=>b}/rest  — renamed with common prefix/suffix
   *
   * Map is keyed by the CURRENT (new) path so it aligns with name-status entries.
   */
  private parseNumstat(raw: string): Map<string, {
    insertions: number | null;
    deletions:  number | null;
    isBinary:   boolean;
  }> {
    const map = new Map<string, { insertions: number | null; deletions: number | null; isBinary: boolean }>();
    if (!raw) return map;

    for (const line of raw.split("\n")) {
      if (!line) continue;

      const tabIdx1 = line.indexOf("\t");
      const tabIdx2 = line.indexOf("\t", tabIdx1 + 1);
      if (tabIdx1 === -1 || tabIdx2 === -1) continue;

      const insStr  = line.slice(0, tabIdx1);
      const delStr  = line.slice(tabIdx1 + 1, tabIdx2);
      const rawPath = line.slice(tabIdx2 + 1);

      const isBinary  = insStr === "-" || delStr === "-";
      const insertions = isBinary ? null : parseInt(insStr, 10);
      const deletions  = isBinary ? null : parseInt(delStr, 10);

      // Expand brace notation to extract the current (new) path as the map key
      const { newPath } = this.expandBracePath(rawPath);

      map.set(newPath, { insertions, deletions, isBinary });
    }

    return map;
  }

  /**
   * Expands git's brace-compressed rename notation.
   *
   * Examples:
   *   "{old.ts => new.ts}"           → oldPath="old.ts",         newPath="new.ts"
   *   "src/{lib => utils}/index.ts"  → oldPath="src/lib/index.ts", newPath="src/utils/index.ts"
   *   "plain/file.ts"                → oldPath=newPath="plain/file.ts"  (no rename)
   */
  private expandBracePath(raw: string): { oldPath: string; newPath: string } {
    const match = /^(.*?)\{(.+?) => (.+?)\}(.*)$/.exec(raw);
    if (!match) return { oldPath: raw, newPath: raw };

    const prefix  = match[1] ?? "";
    const oldPart = match[2] ?? "";
    const newPart = match[3] ?? "";
    const suffix  = match[4] ?? "";

    return {
      oldPath: prefix + oldPart + suffix,
      newPath: prefix + newPart + suffix,
    };
  }

  // ── Parser: --raw -M (submodule detection) ──

  /**
   * Parses output from `git diff --cached --raw -M` and returns the set of
   * paths whose mode is 160000 (submodule object type).
   *
   * Line format:
   *   :<oldmode> <newmode> <oldhash> <newhash> <status>\t<path>
   *   :<oldmode> <newmode> <oldhash> <newhash> R<n>\t<oldpath>\t<newpath>
   */
  private parseSubmodulePaths(raw: string): Set<string> {
    const submodules = new Set<string>();
    if (!raw) return submodules;

    for (const line of raw.split("\n")) {
      if (!line.startsWith(":")) continue;

      const tabIdx = line.indexOf("\t");
      if (tabIdx === -1) continue;

      const meta  = line.slice(0, tabIdx);  // ":<oldmode> <newmode> ..."
      const paths = line.slice(tabIdx + 1).split("\t");

      const metaParts = meta.split(" ");
      if (metaParts.length < 2) continue;

      const oldMode = metaParts[0]!.slice(1); // strip leading ":"
      const newMode = metaParts[1];

      if (oldMode === "160000" || newMode === "160000") {
        // For renames paths = [oldpath, newpath]; take the current (last) path
        const currentPath = paths[paths.length - 1];
        if (currentPath !== undefined) {
          submodules.add(currentPath);
        }
      }
    }

    return submodules;
  }

  // ── Parser: full diff splitter ──

  /**
   * Splits the output of `git diff --cached -- <paths>` into per-file sections.
   * Keyed by the current (b/) path from each diff header.
   */
  private splitDiffByFile(raw: string): Map<string, string> {
    const result = new Map<string, string>();
    if (!raw) return result;

    // Each file section begins with "diff --git "; split while keeping the delimiter
    const sections = raw.split(/(?=^diff --git )/m);

    for (const section of sections) {
      if (!section.trim()) continue;

      const firstLine = section.slice(0, section.indexOf("\n"));
      // "diff --git a/<oldpath> b/<newpath>" — use lastIndexOf to handle spaces
      const bIdx = firstLine.lastIndexOf(" b/");
      if (bIdx === -1) continue;

      const currentPath = firstLine.slice(bIdx + 3).trim();
      result.set(currentPath, section.trimEnd());
    }

    return result;
  }

  // ── Summary builder ──

  private buildSummary(files: StagedFileChange[]): StagedDiffSummary {
    let totalInsertions = 0;
    let totalDeletions  = 0;
    let hasBinaryFiles  = false;
    let hasSubmodules   = false;

    for (const file of files) {
      totalInsertions += file.insertions ?? 0;
      totalDeletions  += file.deletions  ?? 0;
      if (file.isBinary)    hasBinaryFiles = true;
      if (file.isSubmodule) hasSubmodules  = true;
    }

    return {
      totalFiles: files.length,
      totalInsertions,
      totalDeletions,
      hasBinaryFiles,
      hasSubmodules,
      files,
    };
  }
}