import type {
  LLMMessage,
  AssembledPrompt,
  PromptAssemblyInput,
} from "../../../shared/types/llm/prompt";
import type { GitInternalOpState } from "../../../shared/types/git/context";
import type { FileDiffPlan } from "../../../shared/types/git/planning";

/** Rough tokens-per-character ratio used for budget estimation. */
const CHARS_PER_TOKEN = 4;

/** Assembler-private join: one file's budget plan + its resolved diff text. */
interface FileView {
  readonly plan: FileDiffPlan;
  readonly diffText: string | null;
}

export class PromptAssembler {
  /**
   * Converts pipeline results into an ordered LLM message array plus a token
   * estimate that includes the system message overhead absent from diffPlan.estimate.
   */
  public assemble(input: PromptAssemblyInput): AssembledPrompt {
    const messages: readonly LLMMessage[] = [
      { role: "system", content: this.buildSystemMessage() },
      { role: "user",   content: this.buildUserMessage(input) },
    ];

    return { messages, tokenEstimate: this.estimateTokens(messages) };
  }

  // ── System message ──

  private buildSystemMessage(): string {
    return [
      "You are an expert software developer writing Git commit messages.",
      "Follow the Conventional Commits 1.0.0 specification.",
      "",
      "Format:",
      "  <type>[optional scope]: <description>",
      "  [optional blank line + body]",
      "",
      "Rules:",
      "  - type: feat | fix | docs | style | refactor | perf | test | chore | build | ci",
      "  - description: ≤72 characters, imperative mood, lowercase, no trailing period",
      "  - body: explain WHY not WHAT; wrap at 72 characters",
      "  - merge/squash: summarise what was merged and why",
      "  - cherry-pick/revert: reference the original commit subject",
      "  - Output ONLY the commit message — no explanation, no code fences",
    ].join("\n");
  }

  // ── User message ─

  private buildUserMessage(input: PromptAssemblyInput): string {
    const sections: string[] = [];

    sections.push(this.buildRepoSection(input));

    const opSection = this.buildGitOpSection(input.gitState);
    if (opSection !== null) sections.push(opSection);

    sections.push(this.buildSummarySection(input));
    sections.push(this.buildFileManifest(input));

    const diffSection = this.buildDiffSection(input);
    if (diffSection !== null) sections.push(diffSection);

    return sections.join("\n\n");
  }

  // ── Section builders 

  private buildRepoSection(input: PromptAssemblyInput): string {
    const { repoContext } = input;
    const branch = repoContext.currentBranch ?? "(detached HEAD)";

    const lines = [
      "## Repository",
      `Branch: ${branch}`,
    ];

    if (repoContext.isInitialCommit) {
      lines.push("Initial commit: yes (repository has no prior commits)");
    }
    if (repoContext.isDetachedHead) {
      lines.push("HEAD: detached");
    }

    return lines.join("\n");
  }

  private buildGitOpSection(state: GitInternalOpState): string | null {
    switch (state.status) {
      case "clean":
        return null;

      case "merge": {
        const lines = ["## Git operation: merge"];
        if (state.mergeMessage !== null) {
          lines.push(`Merge message:\n${this.indent(state.mergeMessage)}`);
        }
        return lines.join("\n");
      }

      case "squash-merge":
        return [
          "## Git operation: squash merge",
          `Squash message:\n${this.indent(state.squashMessage)}`,
        ].join("\n");

      case "cherry-pick": {
        const lines = ["## Git operation: cherry-pick"];
        if (state.originalTitle !== null) {
          lines.push(`Original commit: ${state.originalTitle}`);
        }
        return lines.join("\n");
      }

      case "revert": {
        const lines = ["## Git operation: revert"];
        if (state.originalTitle !== null) {
          lines.push(`Reverted commit: ${state.originalTitle}`);
        }
        return lines.join("\n");
      }

      case "rebase": {
        const lines = [`## Git operation: rebase (${state.rebaseType})`];
        if (state.originalMessage !== null) {
          lines.push(`Original message:\n${this.indent(state.originalMessage)}`);
        }
        return lines.join("\n");
      }

      default: {
        // Exhaustiveness check — TypeScript will flag this if a new status is added
        const _: never = state;
        return null;
      }
    }
  }

  private buildSummarySection(input: PromptAssemblyInput): string {
    const { diffSummary, diffPlan: { estimate, fullDiffCount, degradedCount } } = input;

    const lines = [
      "## Summary",
      `${diffSummary.totalFiles} file(s) changed — +${diffSummary.totalInsertions} insertions, -${diffSummary.totalDeletions} deletions`,
    ];

    if (estimate.contentFiles > 0 && estimate.noiseFiles > 0) {
      lines.push(
        `${estimate.contentFiles} content file(s), ${estimate.noiseFiles} noise file(s) ` +
        "(binary/submodule/lfs — diff omitted)",
      );
    } else if (estimate.noiseFiles > 0) {
      lines.push(`${estimate.noiseFiles} noise file(s) (binary/submodule/lfs — diff omitted)`);
    }

    if (estimate.renamedNoContentChangeCount > 0) {
      lines.push(`${estimate.renamedNoContentChangeCount} rename(s) with no content change`);
    }

    if (!estimate.isWithinBudget) {
      lines.push(
        `Token budget exceeded: ${fullDiffCount} file(s) with full diff, ` +
        `${degradedCount} file(s) omitted`,
      );
    }

    if (diffSummary.hasBinaryFiles) lines.push("Contains binary files");
    if (diffSummary.hasSubmodules)  lines.push("Contains submodule changes");

    return lines.join("\n");
  }

  private buildFileManifest(input: PromptAssemblyInput): string {
    const lines = ["## Files"];

    for (const plan of input.diffPlan.plans) {
      const sf = plan.file.file;

      const modeLabel = plan.mode === "full"
        ? "[full diff below]"
        : `[omitted: ${plan.degradationReason ?? "degraded"}]`;

      const pathLabel = sf.oldPath !== null
        ? `${sf.oldPath} → ${sf.path}`
        : sf.path;

      const statParts: string[] = [sf.changeType];
      if (sf.insertions !== null || sf.deletions !== null) {
        statParts.push(`+${sf.insertions ?? 0} -${sf.deletions ?? 0}`);
      }
      if (sf.similarityScore !== null) {
        statParts.push(`${sf.similarityScore}% similar`);
      }

      const categoryLabel = plan.file.isNoise
        ? plan.file.noiseCategory
        : plan.file.contentCategory;

      lines.push(`${modeLabel} ${pathLabel}  (${statParts.join(", ")})  [${categoryLabel}]`);
    }

    return lines.join("\n");
  }

  private buildDiffSection(input: PromptAssemblyInput): string | null {
    const views = this.resolveFileViews(input);
    const withDiff = views.filter(v => v.diffText !== null && v.diffText.length > 0);

    if (withDiff.length === 0) return null;

    const parts = ["## Diffs"];
    for (const { plan, diffText } of withDiff) {
      parts.push(`### ${plan.file.file.path}\n\`\`\`diff\n${diffText!}\n\`\`\``);
    }

    return parts.join("\n\n");
  }

  // ── Helpers 

  /**
   * Joins every full-mode FileDiffPlan with the resolved diff text from diffTexts.
   * A plan that has no entry in diffTexts (e.g. pure rename, zero content change)
   * gets diffText: null and will be omitted from the diff section.
   */
  private resolveFileViews(input: PromptAssemblyInput): FileView[] {
    return input.diffPlan.plans
      .filter(plan => plan.mode === "full")
      .map(plan => ({
        plan,
        diffText: input.diffTexts.get(plan.file.file.path) ?? null,
      }));
  }

  private indent(text: string, prefix = "  "): string {
    return text.split("\n").map(line => prefix + line).join("\n");
  }

  /**
   * Rough token estimate: total characters ÷ 4, plus a small per-message
   * overhead for role labels and API framing tokens.
   */
  private estimateTokens(messages: readonly LLMMessage[]): number {
    const chars = messages.reduce(
      (sum, msg) => sum + msg.content.length + msg.role.length + 8,
      0,
    );
    return Math.ceil(chars / CHARS_PER_TOKEN);
  }
}