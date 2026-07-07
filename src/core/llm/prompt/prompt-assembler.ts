import type {
  LLMMessage,
  AssembledPrompt,
  PromptAssemblyInput,
} from "../../../shared/types/llm/index";
import type {
  GitInternalOpState,
} from "../../../shared/types/git/index";

/**
 * Pure English ≈ 4.0 chars/token, diffs with special characters ≈ 2.0–2.5.
 * Using 3.0 provides a safety margin so actual token count does not exceed
 * the estimate at LLM API call time.
 */
const CHARS_PER_TOKEN = 3;

type FullInput = Extract<PromptAssemblyInput, { route: "full" }>;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
type InterruptedInput = Extract<PromptAssemblyInput, { route: "interrupted" }>;

export class PromptAssembler {
  /**
   * Converts pipeline results into an ordered LLM message array plus a token
   * estimate that includes the system message overhead absent from diffPlan.estimate.
   */
  public assemble(input: PromptAssemblyInput): AssembledPrompt {
    const messages: readonly LLMMessage[] = [
      { role: "system", content: this.buildSystemMessage() },
      { role: "user", content: this.buildUserMessage(input) },
    ];

    return { messages, tokenEstimate: this.estimateTokens(messages) };
  }

  // ── System message ──

  private buildSystemMessage(): string {
    return [
      "You are an expert software engineer writing Git commit messages.",
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
      "  - Output ONLY the commit message — no explanation, no code fences",
    ].join("\n");
  }

  // ── User message ──

  private buildUserMessage(input: PromptAssemblyInput): string {
    // Interrupted route: git op context + suggested message for LLM to refine
    if (input.route === "interrupted") {
      const sections: string[] = [
        this.buildGitOpSection(input.gitState),
        `## Suggested message\n${this.indent(input.commitMessage)}`,
        "Refine the suggested message if needed, or output it verbatim if it is already correct.",
      ];

      return sections.join("\n\n");
    }

    // Full route: repo context + diff summary + file manifest + diffs
    const sections: string[] = [
      this.buildRepoSection(input),
      this.buildSummarySection(input),
      this.buildFileManifest(input),
    ];

    const diffSection = this.buildDiffSection(input);
    if (diffSection !== null) sections.push(diffSection);

    return sections.join("\n\n");
  }

  // ── Section builders ──

  private buildRepoSection(input: FullInput): string {
    const { repoContext } = input;
    const branch = repoContext.currentBranch ?? "(detached HEAD)";

    const lines = ["## Repository", `Branch: ${branch}`];

    if (repoContext.isInitialCommit) {
      lines.push("Initial commit: yes (repository has no prior commits)");
    }
    if (repoContext.isDetachedHead) {
      lines.push("HEAD: detached");
    }

    return lines.join("\n");
  }

  /**
   * Builds the git operation context section for interrupted-route prompts.
   * The state is always non-clean here — the interrupted route guarantees it.
   */
  private buildGitOpSection(
    state: Exclude<GitInternalOpState, { status: "clean" }>,
  ): string {
    switch (state.status) {
      case "merge": {
        const lines = ["## Git operation: merge"];
        if (state.mergeMessage !== null) {
          lines.push(`Merge message:\n${this.indent(state.mergeMessage)}`);
        }
        return lines.join("\n");
      }

      // SQUASH_MSG must be written when creating a squash-merge, 
      // it cannot be empty
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
          lines.push(
            `Original message:\n${this.indent(state.originalMessage)}`,
          );
        }
        return lines.join("\n");
      }

      default: {
        const _: never = state;
        throw new Error("Unreachable: unknown special-op state");
      }
    }
  }

  private buildSummarySection(input: FullInput): string {
    const {
      diffSummary,
      diffPlan: { estimate, fullDiffCount, degradedCount },
    } = input;

    const lines = [
      "## Summary",
      `${diffSummary.totalFiles} file(s) changed — +${diffSummary.totalInsertions} insertions, -${diffSummary.totalDeletions} deletions`,
    ];

    if (estimate.nonNoiseFiles > 0 && estimate.noiseFiles > 0) {
      lines.push(
        `${estimate.nonNoiseFiles} content file(s), ${estimate.noiseFiles} noise file(s) ` +
          "(binary/submodule/lfs — diff omitted)",
      );
    } else if (estimate.noiseFiles > 0) {
      lines.push(
        `${estimate.noiseFiles} noise file(s) (binary/submodule/lfs — diff omitted)`,
      );
    }

    if (estimate.renamedNoContentChangeCount > 0) {
      lines.push(
        `${estimate.renamedNoContentChangeCount} rename(s) with no content change`,
      );
    }

    if (!estimate.isWithinBudget) {
      lines.push(
        `Token budget exceeded: ${fullDiffCount} file(s) with full diff, ` +
          `${degradedCount} file(s) omitted`,
      );
    }

    if (diffSummary.hasBinaryFiles) lines.push("Contains binary files");
    if (diffSummary.hasSubmodules) lines.push("Contains submodule changes");

    return lines.join("\n");
  }

  private buildFileManifest(input: FullInput): string {
    const lines = ["## Files"];

    for (const plan of input.diffPlan.plans) {
      const sf = plan.file.file;

      // 1. Check if this file could get full diff
      const modeLabel =
        plan.mode === "full"
          ? "[full diff below]"
          : `[omitted: ${plan.degradationReason ?? "degraded"}]`;

      // 2. Process renaming of filename 
      const pathLabel =
        sf.oldPath !== null ? `${sf.oldPath} → ${sf.path}` : sf.path;

      // 3. Change statistics
      const statParts: string[] = [sf.changeType];
      if (sf.insertions !== null || sf.deletions !== null) {
        statParts.push(`+${sf.insertions ?? 0} -${sf.deletions ?? 0}`);
      }

      // 4. Add possible similarityScore
      if (sf.similarityScore !== null) {
        statParts.push(`${sf.similarityScore}% similar`);
      }

      // 5. Add file classification label
      const categoryLabel = plan.file.isNoise
        ? plan.file.noiseCategory
        : plan.file.nonNoiseCategory;

      lines.push(
        `${modeLabel} ${pathLabel}  (${statParts.join(", ")})  [${categoryLabel}]`,
      );
    }

    return lines.join("\n");
  }

  /**
   * Joins every full-mode FileDiffPlan with the resolved diff text from diffTexts.
   */
  private buildDiffSection(input: FullInput): string | null {
    const parts = ["## Diffs"];

    for (const plan of input.diffPlan.plans) {
      if (plan.mode !== "full") continue;

      const diffText = input.diffTexts.get(plan.file.file.path);
      if (diffText === undefined || diffText.length === 0) continue;

      parts.push(
        `### ${plan.file.file.path}\n\`\`\`diff\n${diffText}\n\`\`\``,
      );
    }

    return parts.length > 1 ? parts.join("\n\n") : null;
  }

  private indent(text: string, prefix = "  "): string {
    return text
      .split("\n")
      .map((line) => prefix + line)
      .join("\n");
  }

  /**
   * Rough token estimate: total characters ÷ 3, plus a small per-message
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
