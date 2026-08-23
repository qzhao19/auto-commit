use std::path::Path;

use crate::core::git::types::{
    ChangeType, ClassifiedSnapshot, FileCategory, StagedFile, StagedSnapshot,
};
use crate::infra::git::GitRunner;
use crate::shared::exception::{GitError, GitErrorCode};

use super::rules::{classify_by_header, classify_by_name};

const BLOB_HEADER_BYTES: usize = 512;

pub struct FileClassifier<'a> {
    runner: &'a GitRunner,
}

impl<'a> FileClassifier<'a> {
    pub fn new(runner: &'a GitRunner) -> Self {
        Self { runner }
    }

    pub async fn classify(
        &self,
        snapchat: &StagedSnapshot,
    ) -> Result<ClassifiedSnapshot, GitError> {
        let mut files = snapchat.files.clone();
        let mut probe_indices = Vec::new();

        // Phase A — cheap short-circuit
        for (idx, file) in files.iter_mut().enumerate() {
            match file.category {
                FileCategory::Submodule | FileCategory::Binary => {
                    // Hard facts from Stage 2
                    continue;
                }
                FileCategory::Unknown
                | FileCategory::DependencyLock
                | FileCategory::Generated
                | FileCategory::SemanticText => {
                    // 3.1 owns the final label for these
                    if let Some(category) = classify_by_name(&file.path) {
                        file.category = category;
                    } else {
                        file.category = FileCategory::Unknown; // pending Phase B
                        probe_indices.push(idx);
                    }
                }
            }
        }

        // Phase B — header probe for residue
        for idx in probe_indices {
            let header = self
                .read_blob_header(&files[idx], BLOB_HEADER_BYTES)
                .await?;
            files[idx].category = match header {
                Some(bytes) => classify_by_header(&bytes).unwrap_or(FileCategory::SemanticText),
                None => FileCategory::SemanticText,
            };
        }

        debug_assert!(
            files.iter().all(|f| f.category != FileCategory::Unknown),
            "FileClassifier must not leave Unknown"
        );

        Ok(ClassifiedSnapshot::from_files(files))
    }

    /// Added/Modified/Rename/Copy/T → index blob `:path`
    /// Deleted → `HEAD:path` (soft-fail → None)
    async fn read_blob_header(
        &self,
        file: &StagedFile,
        max_bytes: usize,
    ) -> Result<Option<Vec<u8>>, GitError> {
        let spec = match file.change_type {
            ChangeType::Deleted => format!("HEAD:{}", file.path.to_string_lossy()),
            ChangeType::Added
            | ChangeType::Modified
            | ChangeType::Renamed
            | ChangeType::Copied
            | ChangeType::TypeChanged => {
                format!(":{}", file.path.to_string_lossy())
            }
        };

        match self.runner.run(&["cat-file", "-p", &spec], None).await {
            Ok(out) => {
                let mut bytes = out.stdout;
                bytes.truncate(max_bytes.min(bytes.len()));
                Ok(Some(bytes))
            }
            Err(_) if file.change_type == ChangeType::Deleted => Ok(None),
            Err(err) => Err(err),
        }
    }
}
