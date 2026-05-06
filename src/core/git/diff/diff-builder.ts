import { GitCode, GitError } from "../../../shared/exceptions/index";
import {
	type ClassifiedFile,
	type FileClassificationResult,
} from "../../../shared/types/git/classify";
import {
	DEFAULT_DIFF_BUDGET,
	type DiffBudgetConfig,
	type DiffBudgetEstimate,
	type DiffBuildResult,
	type DiffSelectionMode,
	type DiffSelectionPlan,
	type FileLLMPayload,
} from "../../../shared/types/git/index";
import { type StagedFileChange } from "../../../shared/types/git/index";
import { type DiffCollector } from "./diff-collector";

const ESTIMATED_CHARS_PER_CHANGED_LINE = 48;
const ESTIMATED_CHARS_PER_FILE = 120;
const ESTIMATED_RENAME_ONLY_CHARS = 32;

export class DiffBuilder {
	private readonly collector: DiffCollector;
	private readonly budget: DiffBudgetConfig;

	constructor(collector: DiffCollector, budget: Partial<DiffBudgetConfig> = {}) {
		this.collector = collector;
		this.budget = {
			...DEFAULT_DIFF_BUDGET,
			...budget,
		};
	}

	public async build(classification: FileClassificationResult): Promise<DiffBuildResult> {
		try {
			const plan = this.buildPlan(classification);
			const diffMap = plan.fullDiffPaths.length > 0
				? await this.collector.collectDiff(plan.fullDiffPaths)
				: new Map<string, string>();
			const selectedPaths = new Set(plan.fullDiffPaths);
			const payloads = this.buildPayloads(classification.files, diffMap, selectedPaths);

			return {
				plan,
				payloads,
				totalDiffChars: this.totalDiffChars(payloads),
			};
		} catch (error) {
			if (error instanceof GitError) throw error;
			throw new GitError({
				code: GitCode.COMMAND_FAILED,
				message: error instanceof Error ? error.message : String(error),
				cause: error,
			});
		}
	}

	private buildPlan(classification: FileClassificationResult): DiffSelectionPlan {
		const textFiles = classification.byLayer.text;
		const estimate = this.estimate(textFiles);

		if (classification.isSummaryOnly) {
			return {
				mode: "summary-only",
				estimate,
				classification,
				fullDiffPaths: [],
				summarizedTextPaths: [],
			};
		}

		const mode: DiffSelectionMode = estimate.oversized ? "trimmed" : "full";

		if (mode === "full") {
			return {
				mode,
				estimate,
				classification,
				fullDiffPaths: textFiles.map((file) => file.change.path),
				summarizedTextPaths: [],
			};
		}

		const selectedFiles = this.selectWithinBudget(textFiles);
		const selectedPaths = new Set(selectedFiles.map((file) => file.change.path));
		const fullDiffPaths = textFiles
			.filter((file) => selectedPaths.has(file.change.path))
			.map((file) => file.change.path);
		const summarizedTextPaths = textFiles
			.filter((file) => !selectedPaths.has(file.change.path))
			.map((file) => file.change.path);

		return {
			mode,
			estimate,
			classification,
			fullDiffPaths,
			summarizedTextPaths,
		};
	}

	private estimate(textFiles: readonly ClassifiedFile[]): DiffBudgetEstimate {
		let estimatedChars = 0;
		let totalChangeLines = 0;

		for (const file of textFiles) {
			estimatedChars += this.estimateFileChars(file.change);
			totalChangeLines += this.totalChangeLines(file.change);
		}

		const estimatedTokens = Math.ceil(estimatedChars / 4);

		return {
			estimatedTokens,
			estimatedChars,
			oversized: estimatedTokens > this.budget.softTokenBudget,
			textFileCount: textFiles.length,
			totalChangeLines,
		};
	}

	private selectWithinBudget(textFiles: readonly ClassifiedFile[]): readonly ClassifiedFile[] {
		const ranked = [...textFiles].sort((left, right) => this.comparePriority(left, right));
		const selected: ClassifiedFile[] = [];
		let usedTokens = 0;

		for (const file of ranked) {
			const estimatedTokens = Math.ceil(this.estimateFileChars(file.change) / 4);
			if (selected.length === 0 || usedTokens + estimatedTokens <= this.budget.softTokenBudget) {
				selected.push(file);
				usedTokens += estimatedTokens;
			}
		}

		return selected;
	}

	private comparePriority(left: ClassifiedFile, right: ClassifiedFile): number {
		const leftHasMaterialContent = this.hasMaterialContent(left.change);
		const rightHasMaterialContent = this.hasMaterialContent(right.change);

		if (leftHasMaterialContent !== rightHasMaterialContent) {
			return leftHasMaterialContent ? -1 : 1;
		}

		const lineDelta = this.totalChangeLines(right.change) - this.totalChangeLines(left.change);
		if (lineDelta !== 0) {
			return lineDelta;
		}

		const depthDelta = this.pathDepth(left.change.path) - this.pathDepth(right.change.path);
		if (depthDelta !== 0) {
			return depthDelta;
		}

		return left.change.path.localeCompare(right.change.path);
	}

	private buildPayloads(
		files: readonly ClassifiedFile[],
		diffMap: ReadonlyMap<string, string>,
		selectedPaths: ReadonlySet<string>,
	): FileLLMPayload[] {
		const payloads: FileLLMPayload[] = [];

		for (const file of files) {
			switch (file.layer) {
				case "binary":
					payloads.push({
						kind: "binary",
						path: file.change.path,
						changeType: file.change.changeType,
					});
					break;
				case "submodule":
					payloads.push({
						kind: "submodule",
						path: file.change.path,
						changeType: file.change.changeType,
					});
					break;
				case "lock":
				case "generated":
					payloads.push(this.buildStatsPayload(file, file.annotation));
					break;
				case "text": {
					const diff = selectedPaths.has(file.change.path)
						? diffMap.get(file.change.path) ?? null
						: null;

					if (diff !== null) {
						payloads.push({
							kind: "diff",
							path: file.change.path,
							oldPath: file.change.oldPath,
							changeType: file.change.changeType,
							insertions: file.change.insertions,
							deletions: file.change.deletions,
							diff,
						});
						break;
					}

					payloads.push(this.buildStatsPayload(file, null));
					break;
				}
			}
		}

		return payloads;
	}

	private buildStatsPayload(file: ClassifiedFile, annotation: string | null): FileLLMPayload {
		return {
			kind: "stats",
			path: file.change.path,
			oldPath: file.change.oldPath,
			changeType: file.change.changeType,
			insertions: file.change.insertions,
			deletions: file.change.deletions,
			annotation,
		};
	}

	private totalDiffChars(payloads: readonly FileLLMPayload[]): number {
		let total = 0;

		for (const payload of payloads) {
			if (payload.kind === "diff") {
				total += payload.diff.length;
			}
		}

		return total;
	}

	private estimateFileChars(change: StagedFileChange): number {
		const changeLines = this.totalChangeLines(change);
		const renameOnlyPenalty = change.changeType === "renamed" && changeLines === 0
			? ESTIMATED_RENAME_ONLY_CHARS
			: 0;

		return ESTIMATED_CHARS_PER_FILE +
			(changeLines * ESTIMATED_CHARS_PER_CHANGED_LINE) +
			renameOnlyPenalty;
	}

	private totalChangeLines(change: StagedFileChange): number {
		return (change.insertions ?? 0) + (change.deletions ?? 0);
	}

	private hasMaterialContent(change: StagedFileChange): boolean {
		return !(change.changeType === "renamed" && this.totalChangeLines(change) === 0);
	}

	private pathDepth(filePath: string): number {
		return filePath.split("/").length;
	}
}
