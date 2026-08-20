use std::path::PathBuf;

use crate::core::git::types::ChangeType;
use crate::shared::exception::{GitError, GitErrorCode};

/// One record from `git diff --cached --name-status -z -M -C`
#[derive(Debug)]
pub struct NameStatusEntry {
    pub change_type: ChangeType,
    pub path: PathBuf,
    pub old_path: Option<PathBuf>,
    pub similarity: Option<u8>,
}

/// Per-file line stats from `git diff --cached --numstat -z -M -C`.
#[derive(Debug)]
pub struct NumstatRow {
    pub insertions: Option<u64>,
    pub deletions: Option<u64>,
    pub is_binary: bool,
}

const SUBMODULE_MODE: &str = "160000";

// 2.1 name-status -> change type + paths + similarity

/// Parse `git diff --cached --name-status -z -M -C`
///
/// Formats:
///   `STATUS\0path\0`              (A / M / D / T)
///   `STATUS\0oldpath\0newpath\0`  (R / C, STATUS like `R100`)
pub fn parse_name_status(data: &[u8]) -> Result<Vec<NameStatusEntry>, GitError> {
    let mut entries = Vec::new();
    let mut fields = split_nul(data);

    while !fields.is_empty() {
        let status_field = fields.remove(0);
        if status_field.is_empty() {
            continue;
        }

        let status_str = std::str::from_utf8(status_field).map_err(|err| {
            GitError::new(
                GitErrorCode::Other,
                format!("invalid UTF-8 in name-status status: {err}"),
            )
        })?;

        let (change_type, similarity) = parse_status(status_str)?;

        let (path, old_path) = match change_type {
            ChangeType::Renamed | ChangeType::Copied => {
                if fields.len() < 2 {
                    return Err(GitError::new(
                        GitErrorCode::Other,
                        "truncated name-status record for rename/copy",
                    ));
                }
                let old = PathBuf::from(os_str(fields.remove(0))?);
                let new = PathBuf::from(os_str(fields.remove(0))?);
                (new, Some(old))
            }
            _ => {
                if fields.is_empty() {
                    return Err(GitError::new(
                        GitErrorCode::Other,
                        "truncated name-status record",
                    ));
                }
                let path = PathBuf::from(os_str(fields.remove(0))?);
                (path, None)
            }
        };

        entries.push(NameStatusEntry {
            change_type,
            path,
            old_path,
            similarity,
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

// 2.2  numstat ->  insertions / deletions + binary marker

/// Parse `git diff --cached --numstat -z -M -C`
///
/// <insertions>\t<deletions>\t<path>\0                     (A / M / D / T)
/// -\t-\t<path>\0                                          Binary
/// <insertions>\t<deletions>\t\0<oldpath>\0<newpath>\0     (R / C)
pub fn parse_numstat(
    data: &[u8],
    entries: &[NameStatusEntry],
) -> Result<Vec<NumstatRow>, GitError> {
    let mut fields = split_nul(data).into_iter();
    let mut rows = Vec::with_capacity(entries.len());

    for entry in entries {
        // Count fields
        let counts = fields.next().ok_or_else(|| {
            GitError::new(
                GitErrorCode::Other,
                "numstat ended before name-status (record count mismatch)",
            )
        })?;
        if counts.is_empty() {
            return Err(GitError::new(
                GitErrorCode::Other,
                "numstat ended before name-status (record count mismatch)",
            ));
        }

        let row = parse_numstat_counts(counts)?;

        let extra_path_fields = match entry.change_type {
            ChangeType::Renamed | ChangeType::Copied => 2,
            _ => 0,
        };
        for _ in 0..extra_path_fields {
            fields.next().ok_or_else(|| {
                GitError::new(GitErrorCode::Other, "truncated numstat path field")
            })?;
        }

        rows.push(row);
    }

    if fields.any(|f| !f.is_empty()) {
        return Err(GitError::new(
            GitErrorCode::Other,
            format!("'numstat' has more records than name-status"),
        ));
    }
    Ok(rows)
}

fn parse_numstat_counts(field: &[u8]) -> Result<NumstatRow, GitError> {
    let parts: Vec<&[u8]> = field.splitn(3, |&b| b == b'\t').collect();
    if parts.len() != 3 {
        return Err(GitError::new(
            GitErrorCode::Other,
            format!(
                "malformed numstat counts field: '{}'",
                String::from_utf8_lossy(field)
            ),
        ));
    }

    let ins_col = std::str::from_utf8(parts[0]).map_err(|e| {
        GitError::new(
            GitErrorCode::Other,
            format!("invalid UTF-8 in numstat insertions: {e}"),
        )
    })?;
    let del_col = std::str::from_utf8(parts[1]).map_err(|e| {
        GitError::new(
            GitErrorCode::Other,
            format!("invalid UTF-8 in numstat deletions: {e}"),
        )
    })?;

    if ins_col == "-" && del_col == "-" {
        return Ok(NumstatRow {
            insertions: None,
            deletions: None,
            is_binary: true,
        });
    }

    let parse_num = |s: &str, what: &str| {
        s.parse::<u64>().map_err(|err| {
            GitError::new(GitErrorCode::Other, format!("invalid {what} '{s}': {err}"))
        })
    };

    Ok(NumstatRow {
        insertions: Some(parse_num(ins_col, "insertions")?),
        deletions: Some(parse_num(del_col, "deletions")?),
        is_binary: false,
    })
}

//  2.3  raw  →  submodule detection (mode 160000)

/// Parse `git diff --cached --raw -z -M -C`
///
/// :<srcmode> <dstmode> <srcsha> <dstsha> <status>\0<path>\0 (A / M / D / T)
/// :<srcmode> <dstmode> <srcsha> <dstsha> <status><score>\0<oldpath>\0<newpath>\0 (R / C）
pub fn parse_submodule_flags(
    data: &[u8],
    entries: &[NameStatusEntry],
) -> Result<Vec<bool>, GitError> {
    let mut fields = split_nul(data).into_iter();
    let mut flags = Vec::with_capacity(entries.len());

    for entry in entries {
        let meta = fields.next().ok_or_else(|| {
            GitError::new(
                GitErrorCode::Other,
                "raw ended before name-status (record count mismatch)",
            )
        })?;
        if meta.is_empty() {
            return Err(GitError::new(
                GitErrorCode::Other,
                "raw ended before name-status (record count mismatch)",
            ));
        }

        let meta_str = std::str::from_utf8(meta).map_err(|err| {
            GitError::new(GitErrorCode::Other, format!("invalid UTF-8 in raw: {err}"))
        })?;

        let is_submodule = raw_is_submodule(meta_str)?;

        let extra_path_fields = match entry.change_type {
            ChangeType::Copied | ChangeType::Renamed => 2,
            _ => 1,
        };
        for _ in 0..extra_path_fields {
            fields
                .next()
                .ok_or_else(|| GitError::new(GitErrorCode::Other, "truncated raw path field"))?;
        }

        flags.push(is_submodule);
    }

    if fields.any(|f| !f.is_empty()) {
        return Err(GitError::new(
            GitErrorCode::Other,
            "raw has more records than name-status (invariant violated)",
        ));
    }
    Ok(flags)
}

fn raw_is_submodule(meta: &str) -> Result<bool, GitError> {
    let mut tokens = meta.split_whitespace();
    let src = tokens
        .next()
        .ok_or_else(|| GitError::new(GitErrorCode::Other, format!("empty raw meta: '{meta}'")))?;
    let dst = tokens.next().ok_or_else(|| {
        GitError::new(GitErrorCode::Other, format!("truncated raw meta: '{meta}'"))
    })?;

    // src mode carries a leading `:` (e.g. `:160000`).
    let src = src.strip_prefix(':').unwrap_or(src);
    Ok(src == SUBMODULE_MODE || dst == SUBMODULE_MODE)
}

// Helper function

fn split_nul(data: &[u8]) -> Vec<&[u8]> {
    let mut fields: Vec<&[u8]> = data.split(|&b| b == 0).collect();
    // `-z` output NUL-terminates every record, strip the tailing empties
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
