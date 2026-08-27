use crate::core::git::types::{
    BudgetDecision, ChangeType, ClassifiedSnapshot, DiffPayload, DiffStrategy, FileCategory,
};
use crate::infra::git::GitRunner;
use crate::shared::exception::GitError;

const DIFF_PREFIX: &str = "diff --git ";

pub struct DiffExtractor<'a> {
    runner: &'a GitRunner,
}

impl<'a> DiffExtractor<'a> {
    pub fn new(runner: &'a GitRunner) -> Self {
        Self { runner }
    }

    pub async fn extract(
        &self,
        snapshot: &ClassifiedSnapshot,
        decision: &BudgetDecision,
    ) -> Result<DiffPayload, GitError> {
        match decision.strategy {
            DiffStrategy::PathSummaryOnly => Ok(path_summary(snapshot)),
            DiffStrategy::Full => self.fetch_full(snapshot).await,
            DiffStrategy::TruncateLines {
                max_changed_lines_per_file,
            } => {
                self.fetch_and_truncate(snapshot, max_changed_lines_per_file, None)
                    .await
            }
            DiffStrategy::SampleHunks {
                max_hunks_per_file,
                max_changed_lines_per_file,
            } => {
                self.fetch_and_truncate(
                    snapshot,
                    max_changed_lines_per_file,
                    Some(max_hunks_per_file),
                )
                .await
            }
        }
    }

    async fn fetch_full(&self, snapshot: &ClassifiedSnapshot) -> Result<DiffPayload, GitError> {
        let paths = collect_paths(snapshot);
        if paths.is_empty() {
            return Ok(DiffPayload::default());
        }

        let body = self.fetch(&paths).await?;
        let file_count = split_sections(&body).len();

        Ok(DiffPayload {
            body,
            file_count,
            truncated_file_count: 0,
        })
    }

    async fn fetch_and_truncate(
        &self,
        snapshot: &ClassifiedSnapshot,
        max_changed_lines: u64,
        max_hunks: Option<u32>,
    ) -> Result<DiffPayload, GitError> {
        let paths = collect_paths(snapshot);
        if paths.is_empty() {
            return Ok(DiffPayload::default());
        }

        let raw = self.fetch(&paths).await?;
        let sections = split_sections(&raw);
        let file_count = sections.len();
        let mut body = String::new();
        let mut truncated_file_count = 0;

        for section in sections {
            let (text, is_truncated) = truncate_section(section, max_changed_lines, max_hunks);
            if is_truncated {
                truncated_file_count += 1;
            }
            body.push_str(&text);
        }

        Ok(DiffPayload {
            body,
            file_count,
            truncated_file_count,
        })
    }

    async fn fetch(&self, paths: &[&str]) -> Result<String, GitError> {
        let mut args = Vec::with_capacity(9 + paths.len());
        args.extend_from_slice(&[
            "diff",
            "--cached",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "-U1",
            "-M",
            "-C",
            "--",
        ]);
        args.extend_from_slice(paths);

        let result = self.runner.run(&args, None).await?;
        Ok(result.stdout_str().into_owned())
    }
}

fn collect_paths(snapshot: &ClassifiedSnapshot) -> Vec<&str> {
    snapshot
        .files
        .iter()
        .filter(|file| file.category == FileCategory::SemanticText)
        .filter_map(|file| {
            let path = file.path.to_str()?;
            if path.is_empty() || path.contains('\n') {
                None
            } else {
                Some(path)
            }
        })
        .collect()
}

pub fn split_sections(diff: &str) -> Vec<&str> {
    let mut starts = Vec::new();
    let mut offset = 0;

    for line in diff.split_inclusive('\n') {
        let content = line.trim_end_matches(['\r', '\n']);
        if content.starts_with(DIFF_PREFIX) {
            starts.push(offset);
        }
        offset += line.len()
    }

    starts
        .iter()
        .enumerate()
        .map(|(i, &start)| {
            let end = starts.get(i + 1).copied().unwrap_or(diff.len());
            &diff[start..end]
        })
        .collect()
}

pub fn truncate_section(
    section: &str,
    max_changed_lines: u64,
    max_hunks: Option<u32>,
) -> (String, bool) {
    let is_changed_line = |line: &str| -> bool {
        (line.starts_with('+') && !line.starts_with("+++"))
            || (line.starts_with('-') && !line.starts_with("---"))
    };

    let mut kept_body = String::new();
    let mut kept_changed_lines = 0u64;
    let mut kept_hunks = 0u32;
    let mut truncated_changed_lines = 0u64;
    let mut truncating = false;

    for line in section.split_inclusive('\n') {
        let content = line.trim_end_matches(['\r', '\n']);
        let is_hunk = content.starts_with("@@");
        let is_changed = is_changed_line(content);

        if !truncating {
            let hunk_overflow = is_hunk
                && (max_hunks.is_some_and(|max| kept_hunks >= max)
                    || kept_changed_lines >= max_changed_lines);

            let line_overflow = is_changed && kept_changed_lines >= max_changed_lines;
            if hunk_overflow || line_overflow {
                truncating = true;
            }
        }

        if truncating {
            if is_changed {
                truncated_changed_lines += 1;
            }
            continue;
        }

        if is_hunk {
            kept_hunks += 1;
        }
        if is_changed {
            kept_changed_lines += 1;
        }
        kept_body.push_str(line);
    }

    if truncating {
        if !kept_body.ends_with('\n') {
            kept_body.push('\n');
        }
        kept_body.push_str(&format!(
            "... [{truncated_changed_lines} more changed lines truncated]\n"
        ));
    }

    (kept_body, truncating)
}

pub fn path_summary(snapshot: &ClassifiedSnapshot) -> DiffPayload {
    let mut body = String::new();
    let mut file_count = 0usize;

    for file in &snapshot.files {
        if file.category != FileCategory::SemanticText {
            continue;
        }
        file_count += 1;

        let status = match file.change_type {
            ChangeType::Added => 'A',
            ChangeType::Modified => 'M',
            ChangeType::Deleted => 'D',
            ChangeType::Renamed => 'R',
            ChangeType::Copied => 'C',
            ChangeType::TypeChanged => 'T',
        };
        match (file.change_type, file.old_path.as_ref()) {
            (ChangeType::Renamed | ChangeType::Copied, Some(old)) => {
                body.push_str(&format!(
                    "{status}  {} -> {}\n",
                    old.display(),
                    file.path.display()
                ));
            }
            _ => {
                body.push_str(&format!("{status}  {}\n", file.path.display()));
            }
        }
    }

    DiffPayload {
        body,
        file_count,
        truncated_file_count: 0,
    }
}
