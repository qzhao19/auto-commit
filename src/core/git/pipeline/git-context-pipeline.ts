import type { GitRunner } from "../runner/index";
import { RepoChecker, StateDetector } from "../context/index";
import { DiffCollector, FileClassifier, BudgetPlanner } from "../diff/index";
import type {
  BudgetThresholds,
  GitInternalOpState,
  GitPipelineResult,
  GitPipelineStep,
  GitRepoPrecheckContext,
} from "../../../shared/types/index";

export class GitContextPipeline {
  private readonly runner: GitRunner;
  private readonly thresholds: BudgetThresholds | undefined;

  constructor(runner: GitRunner, thresholds?: BudgetThresholds) {
    this.runner = runner;
    this.thresholds = thresholds;
  }

  public async execute(): Promise<GitPipelineResult> {
    const completedSteps: GitPipelineStep[] = [];

    const precheckResult = await new RepoChecker(this.runner).check();
    completedSteps.push("repo-precheck");

    const stateResult = await new StateDetector(
      precheckResult.context.gitDir,
      this.runner,
    ).detect();
    completedSteps.push("state-detect");

    if (stateResult.state.status !== "clean") {
      return this.collectInternalOp(
        stateResult.state,
        completedSteps,
      );
    }

    return this.collectCleanContext(precheckResult.context, completedSteps);
  }

  private collectInternalOp(
    state: Exclude<GitInternalOpState, { status: "clean" }>,
    completedSteps: GitPipelineStep[],
  ): GitPipelineResult {
    const commitMessage = this.extractSpecialOpMessage(state);
    return { route: "internal-op", commitMessage, completedSteps };
  }

  private async collectCleanContext(
    repoContext: GitRepoPrecheckContext,
    completedSteps: GitPipelineStep[],
  ): Promise<GitPipelineResult> {
    const diffCollector = new DiffCollector(this.runner);
    const diffSummary = await diffCollector.collect();
    completedSteps.push("diff-collect");

    const classifier = new FileClassifier(this.runner);
    const classified = await classifier.classify(diffSummary);
    completedSteps.push("file-classify");

    const planner = new BudgetPlanner(this.thresholds);
    const diffPlan = planner.plan(classified);
    completedSteps.push("budget-plan");

    const fullPaths = diffPlan.plans
      .filter((plan) => plan.mode === "full")
      .map((plan) => plan.file.file.path);
    const diffTexts = await diffCollector.collectDiff(fullPaths);
    completedSteps.push("diff-fetch");

    return {
      route: "clean",
      repoContext,
      diffSummary,
      diffPlan,
      diffTexts,
      completedSteps,
    };
  }

  private extractSpecialOpMessage(
    state: Exclude<GitInternalOpState, { status: "clean" }>,
  ): string {
    switch (state.status) {
      case "merge":
        return state.mergeMessage ?? `Merge ${state.mergeHead}`;

      case "squash-merge":
        return state.squashMessage;

      case "cherry-pick":
        return state.originalTitle
          ? `cherry-pick: ${state.originalTitle}`
          : `cherry-pick ${state.cherryPickHead}`;

      case "revert":
        return state.originalTitle
          ? `revert: ${state.originalTitle}`
          : `revert ${state.revertHead}`;

      case "rebase":
        return state.originalMessage ?? `rebase (${state.rebaseType})`;

      default:
        state satisfies never;
        throw new Error("Unreachable: unknown special-op state");
    }
  }
}


