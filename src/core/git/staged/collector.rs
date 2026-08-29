use crate::core::git::types::{FileCategory, StagedFile, StagedSnapshot};
use crate::infra::git::GitRunner;
use crate::shared::exception::GitError;

use super::parser::{parse_numstat, parse_raw_entries};

/// Collects decision-level metadata of the staged area
///
/// Responsibilities (stage 2):
/// 2.1  raw      → change type + paths + similarity + modes (submodule = 160000)
/// 2.2  numstat  → insertions / deletions + binary marker, path-cross-checked
///                 against the raw stream
/// 2.3  assemble → StagedSnapshot
pub struct StagedMetadataCollector<'a> {
    runner: &'a GitRunner,
}

impl<'a> StagedMetadataCollector<'a> {
    pub fn new(runner: &'a GitRunner) -> Self {
        Self { runner }
    }

    pub async fn collect(&self) -> Result<StagedSnapshot, GitError> {
        let (raw, numstat) = tokio::try_join!(self.run_diff("--raw"), self.run_diff("--numstat"))?;
        let files = Self::merge(raw, numstat)?;

        Ok(StagedSnapshot { files })
    }

    async fn run_diff(&self, format: &str) -> Result<Vec<u8>, GitError> {
        let result = self
            .runner
            .run(
                &[
                    "diff",
                    "--cached",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    format,
                    "-z",
                    "-M",
                    "-C",
                ],
                None,
            )
            .await?;

        Ok(result.stdout)
    }

    fn merge(raw: Vec<u8>, numstat: Vec<u8>) -> Result<Vec<StagedFile>, GitError> {
        let entries = parse_raw_entries(&raw)?;
        let stats = parse_numstat(&numstat, &entries)?;

        let mut files = Vec::with_capacity(entries.len());

        for (entry, stat) in entries.into_iter().zip(stats) {
            // Submodule > Binary > Unknown
            let category = if entry.is_submodule {
                FileCategory::Submodule
            } else if stat.is_binary {
                FileCategory::Binary
            } else {
                FileCategory::Unknown
            };

            // Submodules and binaries never carry line counts.
            let (insertions, deletions) = if entry.is_submodule || stat.is_binary {
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
