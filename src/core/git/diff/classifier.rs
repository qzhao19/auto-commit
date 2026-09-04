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
        snapshot: &StagedSnapshot,
    ) -> Result<ClassifiedSnapshot, GitError> {
        let mut files = snapshot.files.clone();
        let mut probes: Vec<(usize, String)> = Vec::new();

        // Phase A: path / basename only
        for (index, file) in files.iter_mut().enumerate() {
            match file.category {
                FileCategory::Submodule
                | FileCategory::Binary
                | FileCategory::DependencyLock
                | FileCategory::Generated
                | FileCategory::SemanticText => continue,
                FileCategory::Unknown => {
                    let category = classify_by_name(&file.path)
                        .or_else(|| file.old_path.as_deref().and_then(classify_by_name));
                    if let Some(category) = category {
                        file.category = category
                    } else if let Some(spec) = Self::blob_spec(file) {
                        probes.push((index, spec));
                    } else {
                        file.category = FileCategory::SemanticText;
                    }
                }
            }
        }

        // Phase B: `git cat-file --batch` for all residue
        if !probes.is_empty() {
            let specs: Vec<&str> = probes.iter().map(|(_, s)| s.as_str()).collect();
            let headers = self
                .runner
                .cat_file_header(&specs, BLOB_HEADER_BYTES, None)
                .await?;

            if headers.len() != probes.len() {
                return Err(GitError::new(
                    GitErrorCode::Other,
                    format!(
                        "cat-file --batch returned {} headers for {} specs",
                        headers.len(),
                        probes.len()
                    ),
                ));
            }
            for ((index, _), header) in probes.iter().zip(headers) {
                files[*index].category = match header {
                    Some(bytes) => classify_by_header(&bytes).unwrap_or(FileCategory::SemanticText),
                    None => FileCategory::SemanticText,
                };
            }
        }

        for file in &mut files {
            if file.category == FileCategory::Unknown {
                file.category = FileCategory::SemanticText;
            }
        }

        Ok(ClassifiedSnapshot::from_files(files))
    }

    fn blob_spec(file: &StagedFile) -> Option<String> {
        let path = file.path.to_str()?;
        if path.contains('\n') {
            return None;
        }
        Some(match file.change_type {
            ChangeType::Deleted => format!("HEAD:{path}"),
            ChangeType::Added
            | ChangeType::Modified
            | ChangeType::Renamed
            | ChangeType::Copied
            | ChangeType::TypeChanged => format!(":{path}"),
        })
    }
}
