use crate::core::git::types::{FileCategory, StagedFile, StagedSnapshot};
use crate::infra::git::GitRunner;
use crate::shared::exception::GitError;

use super::parser::{parse_name_status, parse_numstat, parse_submodule_flags};

/// Collects decision-level metadata of the staged area
///
/// Responsibilities (stage 2):
/// 2.1  name-status   → change type + paths + similarity
/// 2.2  numstat       → insertions / deletions + binary marker
/// 2.3  raw           → submodule detection (mode 160000)
/// 2.4  assemble      → StagedSnapshot
///
/// Never fetches full diffs.
pub struct StagedMetadataCollector<'a> {
    runner: &'a GitRunner,
}

impl<'a> StagedMetadataCollector<'a> {
    pub fn new(runner: &'a GitRunner) -> Self {
        Self { runner }
    }

    pub async fn collect(&self) -> Result<StagedSnapshot, GitError> {
        // Run 3 commands oncurrently
        let (name_status, numstat, raw) = tokio::try_join!(
            self.run_diff("--name-status"),
            self.run_diff("--numstat"),
            self.run_diff("--raw")
        )?;
        let files = Self::merge(name_status, numstat, raw)?;

        Ok(StagedSnapshot { files })
    }

    /// Run `git diff --cached <format> -z -M -C` and return its raw stdout
    async fn run_diff(&self, format: &str) -> Result<Vec<u8>, GitError> {
        let result = self
            .runner
            .run(&["diff", "--cached", format, "-z", "-M", "-C"], None)
            .await?;

        Ok(result.stdout)
    }

    fn merge(
        name_status: Vec<u8>,
        numstat: Vec<u8>,
        raw: Vec<u8>,
    ) -> Result<Vec<StagedFile>, GitError> {
        let entries = parse_name_status(&name_status)?;
        let stats = parse_numstat(&numstat, &entries)?;
        let is_submodules = parse_submodule_flags(&raw, &entries)?;

        let mut files = Vec::with_capacity(entries.len());

        for ((entry, stat), is_submodule) in entries.into_iter().zip(stats).zip(is_submodules) {
            //  Submodule > Binary > SemanticText
            let category = if is_submodule {
                FileCategory::Submodule
            } else if stat.is_binary {
                FileCategory::Binary
            } else {
                FileCategory::Unknown
            };

            // Submodules and binaries never carry line counts.
            let (insertions, deletions) = if is_submodule || stat.is_binary {
                (None, None)
            } else {
                (stat.insertions, stat.deletions)
            };

            files.push(StagedFile {
                path: entry.path,
                old_path: entry.old_path,
                change_type: entry.change_type,
                similarity: entry.similarity,
                insertions,
                deletions,
                category,
            });
        }

        Ok(files)
    }
}
