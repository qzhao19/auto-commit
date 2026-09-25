use std::path::PathBuf;

use crate::core::git::types::ChangeType;
use crate::shared::exception::{GitError, GitErrorCode};

#[derive(Debug)]
pub struct RawEntry {
    pub change_type: ChangeType,
    pub path: PathBuf,
    pub old_path: Option<PathBuf>,
    pub similarity: Option<u8>,
    pub is_submodule: bool,
}

#[derive(Debug)]
pub struct NumstatRow {
    pub insertions: Option<u64>,
    pub deletions: Option<u64>,
    pub is_binary: bool,
}

const SUBMODULE_MODE: &str = "160000";

// 2.1 raw → change type + paths + similarity + submodule (mode)

/// Parse `git diff --cached --raw -z -M -C`
///
/// `:<srcmode> <dstmode> <srcsha> <dstsha> <status>\0<path>\0`                     (A/M/D/T)
/// `:<srcmode> <dstmode> <srcsha> <dstsha> <status><score>\0<oldpath>\0<newpath>\0` (R/C)
pub fn parse_raw_entries(data: &[u8]) -> Result<Vec<RawEntry>, GitError> {
    let mut fields = split_nul(data).into_iter();
    let mut entries = Vec::new();

    while let Some(meta) = fields.next() {
        let meta_str = std::str::from_utf8(meta).map_err(|err| {
            GitError::new(GitErrorCode::Other, format!("invalid UTF-8 in raw: {err}"))
        })?;

        let mut tokens = meta_str.split_whitespace();
        let src_mode = tokens.next().ok_or_else(|| {
            GitError::new(GitErrorCode::Other, format!("empty raw meta: '{meta_str}'"))
        })?;
        let dst_mode = tokens.next().ok_or_else(|| {
            GitError::new(
                GitErrorCode::Other,
                format!("truncated raw meta: '{meta_str}'"),
            )
        })?;

        // shas are skipped positionally
        let _src_sha = tokens.next();
        let _dst_sha = tokens.next();
        let status = tokens.next().ok_or_else(|| {
            GitError::new(
                GitErrorCode::Other,
                format!("raw meta has no status: '{meta_str}'"),
            )
        })?;

        let (change_type, similarity) = parse_status(status)?;

        let (path, old_path) = match change_type {
            ChangeType::Copied | ChangeType::Renamed => {
                let old = fields.next().ok_or_else(|| {
                    GitError::new(GitErrorCode::Other, "truncated raw record for rename/copy")
                })?;

                let new = fields.next().ok_or_else(|| {
                    GitError::new(GitErrorCode::Other, "truncated raw record for rename/copy")
                })?;

                (
                    PathBuf::from(os_str(new)?),
                    Some(PathBuf::from(os_str(old)?)),
                )
            }
            _ => {
                let path = fields
                    .next()
                    .ok_or_else(|| GitError::new(GitErrorCode::Other, "truncated raw record"))?;
                (PathBuf::from(os_str(path)?), None)
            }
        };

        entries.push(RawEntry {
            change_type,
            path,
            old_path,
            similarity,
            is_submodule: is_gitlink(change_type, src_mode, dst_mode),
        });
    }

    Ok(entries)
}

fn parse_status(s: &str) -> Result<(ChangeType, Option<u8>), GitError> {
    if s.is_empty() {
        return Err(GitError::new(GitErrorCode::Other, "empty status fields"));
    }

    let (letter, score) = s.split_at(1);
    let change_type = match letter {
        "A" => ChangeType::Added,
        "M" => ChangeType::Modified,
        "D" => ChangeType::Deleted,
        "R" => ChangeType::Renamed,
        "C" => ChangeType::Copied,
        "T" => ChangeType::TypeChanged,
        other => {
            return Err(GitError::new(
                GitErrorCode::Other,
                format!(
                    "unexpected change type '{other}' (unmerged should have been filtered by stage 1)"
                ),
            ));
        }
    };

    let similarity = if score.is_empty() {
        None
    } else {
        Some(score.parse::<u8>().map_err(|err| {
            GitError::new(
                GitErrorCode::Other,
                format!("invalid similarity score '{score}': {err}"),
            )
        })?)
    };

    Ok((change_type, similarity))
}

// 2.2  numstat → insertions / deletions + binary marker

/// Parse `git diff --cached --numstat -z -M -C`
///
/// `<ins>\t<del>\t<path>\0`                 (A/M/D/T)
/// `-\t-\t<path>\0`                         (binary)
/// `<ins>\t<del>\t\0<oldpath>\0<newpath>\0` (R/C)
pub fn parse_numstat(data: &[u8], entries: &[RawEntry]) -> Result<Vec<NumstatRow>, GitError> {
    let mut fields = split_nul(data).into_iter();
    let mut rows = Vec::with_capacity(entries.len());

    for (index, entry) in entries.iter().enumerate() {
        let counts = fields.next().ok_or_else(|| {
            GitError::new(
                GitErrorCode::Other,
                "numstat ended before raw (record count mismatch)",
            )
        })?;

        if counts.is_empty() {
            return Err(GitError::new(
                GitErrorCode::Other,
                "numstat ended before raw (record count mismatch)",
            ));
        }

        let parts: Vec<&[u8]> = counts.splitn(3, |&b| b == b'\t').collect();
        if parts.len() != 3 {
            return Err(GitError::new(
                GitErrorCode::Other,
                format!(
                    "malformed numstat counts field: '{}'",
                    String::from_utf8_lossy(counts)
                ),
            ));
        }

        let row = parse_numstat_counts(parts[0], parts[1])?;

        let (numstat_path, numstat_old_path) = match entry.change_type {
            ChangeType::Copied | ChangeType::Renamed => {
                if !parts[2].is_empty() {
                    return Err(GitError::new(
                        GitErrorCode::Other,
                        format!("numstat record {index} embeds a path but raw says rename/copy"),
                    ));
                }
                let old = fields.next().ok_or_else(|| {
                    GitError::new(GitErrorCode::Other, "truncated numstat path field")
                })?;

                let new = fields.next().ok_or_else(|| {
                    GitError::new(GitErrorCode::Other, "truncated numstat path field")
                })?;
                (new, Some(old))
            }
            _ => (parts[2], None),
        };

        // ensures that even if raw and numstat come from the same git diff call,
        // they will not be mismatched due to order or missing records
        let paths_align = os_str(numstat_path)? == entry.path.as_os_str()
            && match (entry.old_path.as_deref(), numstat_old_path) {
                (Some(old), Some(bytes)) => old.as_os_str() == os_str(bytes)?,
                (None, None) => true,
                _ => false,
            };
        if !paths_align {
            return Err(GitError::new(
                GitErrorCode::Other,
                format!(
                    "numstat/raw path misalignment at record {index}: numstat '{}' vs raw '{}'",
                    String::from_utf8_lossy(numstat_path),
                    entry.path.display()
                ),
            ));
        }

        rows.push(row);
    }

    if fields.any(|f| !f.is_empty()) {
        return Err(GitError::new(
            GitErrorCode::Other,
            "'numstat' has more records than raw",
        ));
    }
    Ok(rows)
}

fn parse_numstat_counts(ins_col: &[u8], del_col: &[u8]) -> Result<NumstatRow, GitError> {
    let parse_num_column = |col: &[u8], what: &str| -> Result<u64, GitError> {
        let s = std::str::from_utf8(col).map_err(|err| {
            GitError::new(
                GitErrorCode::Other,
                format!("invalid UTF-8 in numstat {what}: {err}"),
            )
        })?;

        s.parse::<u64>().map_err(|err| {
            GitError::new(GitErrorCode::Other, format!("invalid {what} '{s}': {err}"))
        })
    };

    if ins_col == b"-" && del_col == b"-" {
        return Ok(NumstatRow {
            insertions: None,
            deletions: None,
            is_binary: true,
        });
    }

    Ok(NumstatRow {
        insertions: Some(parse_num_column(ins_col, "insertion")?),
        deletions: Some(parse_num_column(del_col, "deletion")?),
        is_binary: false,
    })
}

/// Split the output of `git diff --cached --raw --numstat -z -M -C` into raw block and numstat block
/// [ raw record ... ][ numstat record ... ]
/// raw:     :mode mode sha sha status\0path\0
///      or  :mode mode sha sha R/Cscore\0old\0new\0

/// numstat: ins\t<del>\t<path>\0
///      or  -\t-\t<path>\0
///      or  ins\t<del>\t\0old\0new\0
pub fn split_raw_and_numstat(data: &[u8]) -> Result<(&[u8], &[u8]), GitError> {
    let mut pos = 0;
    while pos < data.len() {
        let meta_end_pos = find_null_offset(data, pos)?;
        let meta = &data[pos..meta_end_pos];

        if meta.first() != Some(&b':') {
            // First field that cannot start a raw record → numstat block
            return Ok((&data[..pos], &data[pos..]));
        }

        let mut cursor = meta_end_pos + 1;
        for _ in 0..raw_record_path_count(meta) {
            cursor = find_null_offset(data, cursor)? + 1;
        }

        pos = cursor;
    }

    Ok((data, &[]))
}

/// `:<srcmode> <dstmode> <srcsha> <dstsha> <status>` is followed by one path
/// field, or two for rename/copy (`R<score>` / `C<score>`)
fn raw_record_path_count(meta: &[u8]) -> usize {
    match meta
        .iter()
        .rposition(|&b| b == b' ')
        .and_then(|i| meta.get(i + 1))
    {
        Some(b'R') | Some(b'C') => 2,
        _ => 1,
    }
}

/// Find the index of the next NUL (`\0`) starting from the `from` position
fn find_null_offset(data: &[u8], from: usize) -> Result<usize, GitError> {
    data[from..]
        .iter()
        .position(|&b| b == 0)
        .map(|offset| from + offset)
        .ok_or_else(|| {
            GitError::new(
                GitErrorCode::Other,
                "truncated NUL-terminated record".to_string(),
            )
        })
}

// Helper function

fn split_nul(data: &[u8]) -> Vec<&[u8]> {
    let mut fields: Vec<&[u8]> = data.split(|&b| b == 0).collect();
    // `-z` output NUL-terminates every record, strip tailing empty
    while let Some(field) = fields.last() {
        if !field.is_empty() {
            break;
        }
        fields.pop();
    }
    fields
}

fn os_str(bytes: &[u8]) -> Result<&std::ffi::OsStr, GitError> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Ok(std::ffi::OsStr::from_bytes(bytes))
    }
    #[cfg(not(unix))]
    {
        let s = std::str::from_utf8(bytes).map_err(|e| {
            GitError::new(GitErrorCode::Other, format!("path is not valid UTF-8: {e}"))
        })?;
        Ok(std::ffi::OsStr::new(s))
    }
}

fn is_gitlink(change_type: ChangeType, src_mode: &str, dst_mode: &str) -> bool {
    let src = src_mode.strip_prefix(':').unwrap_or(src_mode);
    match change_type {
        ChangeType::Deleted => src == SUBMODULE_MODE,
        _ => dst_mode == SUBMODULE_MODE,
    }
}
