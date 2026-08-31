use crate::core::git::types::{
    BudgetDecision, ChangeType, ClassifiedSnapshot, DiffPayload, DiffStrategy, FileCategory,
};
use crate::infra::git::GitRunner;
use crate::shared::exception::GitError;

const DIFF_PREFIX: &str = "diff --git ";

const LOCK_SUMMARY_DEFAULT_CAPACITY: u64 = 64;

const NO_NEWLINE_MARKER: &str = "\\ No newline at end of file";

const LOCK_SIGNAL_ATOMS: &[&str] = &[
    "version",
    "name",
    "revision",
    "specifier",
    "identity",
    "rrev",
    "remotesha",
];

const LOCK_SIGNAL_JSON_KEYS: &[&str] = &["\"rev\"", "\"ref\"", "\"repo\""];

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
        let paths = collect_paths_by_category(snapshot, FileCategory::SemanticText);
        let mut body = String::new();
        let mut file_count = 0;

        if !paths.is_empty() {
            body = self.fetch(&paths).await?;
            file_count = split_sections(&body).len();
        }

        self.append_lock_digest(
            &mut body,
            &mut file_count,
            snapshot,
            LOCK_SUMMARY_DEFAULT_CAPACITY,
        )
        .await?;

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
        let paths = collect_paths_by_category(snapshot, FileCategory::SemanticText);
        let mut body = String::new();
        let mut file_count = 0;
        let mut truncated_file_count = 0;

        if !paths.is_empty() {
            let raw = self.fetch(&paths).await?;
            let sections = split_sections(&raw);
            file_count = sections.len();

            for section in sections {
                let (text, is_truncated) = truncate_section(section, max_changed_lines, max_hunks);
                if is_truncated {
                    truncated_file_count += 1;
                }
                body.push_str(&text);
            }
        }

        self.append_lock_digest(&mut body, &mut file_count, snapshot, max_changed_lines)
            .await?;

        Ok(DiffPayload {
            body,
            file_count,
            truncated_file_count,
        })
    }

    async fn append_lock_digest(
        &self,
        body: &mut String,
        file_count: &mut usize,
        snapshot: &ClassifiedSnapshot,
        capacity: u64,
    ) -> Result<(), GitError> {
        let paths = collect_paths_by_category(snapshot, FileCategory::DependencyLock);
        if paths.is_empty() {
            return Ok(());
        }

        let raw = self.fetch(&paths).await?;
        let digest = summarize_lock_diff(&raw, capacity);
        if digest.is_empty() {
            return Ok(());
        }

        *file_count += split_sections(&digest).len();
        body.push_str(&digest);
        Ok(())
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

pub fn summarize_lock_diff(diff: &str, capacity: u64) -> String {
    let mut kept = String::new();
    let mut kept_changed_lines = 0u64;
    let mut truncated_changed_lines = 0u64;
    let mut truncating = false;
    let mut last_kept_was_changed = false;
    let mut pending_header: Option<&str> = None;
    let mut pending_context: Option<&str> = None;

    for line in diff.split_inclusive('\n') {
        let content = line.trim_end_matches(['\r', '\n']);

        // Get the hunk header of git diff
        if content.starts_with(DIFF_PREFIX) {
            pending_header = Some(line);
            pending_context = None;
            last_kept_was_changed = false;
            continue;
        }

        if content == NO_NEWLINE_MARKER {
            if last_kept_was_changed && !truncating {
                kept.push_str(line);
            }
            continue;
        }

        // Capture the context line (not the modified line)
        if content.starts_with(' ') {
            pending_context = Some(line);
            continue;
        }

        // Process modified lines (+ or -)
        if is_changed_line(content) && is_lock_signal_line(content) {
            if truncating || kept_changed_lines >= capacity {
                truncating = true;
                truncated_changed_lines += 1;
                pending_header = None;
                pending_context = None;
                last_kept_was_changed = false;
                continue;
            }

            if let Some(header) = pending_header.take() {
                kept.push_str(header);
            }
            if let Some(context) = pending_context.take() {
                kept.push_str(context);
            }
            kept.push_str(line);
            kept_changed_lines += 1;
            last_kept_was_changed = true;
            continue;
        }

        pending_context = None;
        last_kept_was_changed = false;
    }

    if truncated_changed_lines > 0 {
        if !kept.is_empty() && !kept.ends_with('\n') {
            kept.push('\n');
        }
        kept.push_str(&format!(
            "... [{truncated_changed_lines} more dependency lines truncated]\n"
        ));
    }

    kept
}

pub fn path_summary(snapshot: &ClassifiedSnapshot) -> DiffPayload {
    let mut body = String::new();
    let mut file_count = 0usize;

    for file in &snapshot.files {
        if !matches!(
            file.category,
            FileCategory::SemanticText | FileCategory::DependencyLock
        ) {
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

// Helper function

fn collect_paths_by_category(snapshot: &ClassifiedSnapshot, categiry: FileCategory) -> Vec<&str> {
    snapshot
        .files
        .iter()
        .filter(|file| file.category == categiry)
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

fn is_changed_line(line: &str) -> bool {
    (line.starts_with('+') && !line.starts_with("++"))
        || (line.starts_with('-') && !line.starts_with("--"))
}

fn contains_dotted_number(line: &str) -> bool {
    line.as_bytes()
        .windows(3)
        .any(|w| w[0].is_ascii_digit() && w[1] == b'.' && w[2].is_ascii_digit())
}

fn diff_payload(line: &str) -> &str {
    match line.as_bytes().first() {
        Some(b'+' | b'-' | b' ') if line.len() > 1 => line[1..].trim_start(),
        _ => line.trim_start(),
    }
}

pub fn is_lock_signal_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();

    if LOCK_SIGNAL_ATOMS.iter().any(|atom| lower.contains(atom)) {
        return true;
    }

    if LOCK_SIGNAL_JSON_KEYS.iter().any(|key| lower.contains(key)) {
        return true;
    }

    if contains_dotted_number(&lower) {
        return true;
    }

    let payload = diff_payload(&lower);
    payload.starts_with("github ")
        || payload.starts_with("git ")
        || payload.starts_with("binary ")
        || payload.contains("{:git")
        || payload.contains("{git,")
        || payload.contains("{ref,")
}
