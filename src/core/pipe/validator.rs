use crate::shared::exception::{ALLOWED_TYPES, MAX_DESCRIPTION_CHARS, ValidationError};
use crate::shared::util::message::ValidatedCommitMessage;

const MAX_CHATTER_LINES: usize = 3;

pub fn validate_and_normalize(raw: &str) -> Result<ValidatedCommitMessage, ValidationError> {
    let cleaned = basic_sanitize(raw)?;
    let cleaned = strip_chatter(cleaned)?;
    let (header, body) = split_header_body(&cleaned)?;
    validate_header(&header)?;
    let final_msg = normalize(&header, body.as_deref());
    Ok(ValidatedCommitMessage(final_msg))
}

fn basic_sanitize(raw: &str) -> Result<String, ValidationError> {
    let unified = if raw.contains('\r') {
        raw.replace("\r\n", "\n").replace("\r", "\n")
    } else {
        raw.to_owned()
    };

    let text = unified.trim();
    if text.is_empty() {
        return Err(ValidationError::Empty);
    }

    if text
        .chars()
        .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err(ValidationError::ControlCharacters);
    }

    Ok(text.to_string())
}

fn strip_chatter(text: String) -> Result<String, ValidationError> {
    let text = strip_fences(&text)?;
    let text = strip_leading_chatter(&text);
    Ok(strip_outer_quotes(&text))
}

/// Clean Markdown fence
fn strip_fences(text: &str) -> Result<String, ValidationError> {
    let lines: Vec<&str> = text.lines().collect();
    let fence_line_indices: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.trim_start().starts_with("```"))
        .map(|(i, _)| i)
        .collect();

    match fence_line_indices.as_slice() {
        [] => Ok(text.to_owned()),
        // Exact 2 fences
        [first, last] if *first == 0 && *last == lines.len() - 1 => {
            let inner = lines[1..*last].join("\n").trim().to_owned();
            if inner.is_empty() {
                Err(ValidationError::Empty)
            } else {
                Ok(inner)
            }
        }
        _ => Err(ValidationError::ForbiddenExtraContent),
    }
}

fn strip_leading_chatter(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    match (0..lines.len())
        .take(MAX_CHATTER_LINES + 1)
        .find(|&row| validate_header(lines[row].trim()).is_ok())
    {
        Some(row) if row > 0 => lines[row..].join("\n"),
        _ => text.to_string(),
    }
}

fn strip_outer_quotes(text: &str) -> String {
    for quote in ["\"", "'"] {
        if text.len() >= 2 && text.starts_with(quote) && text.ends_with(quote) {
            let inner = text[1..text.len() - 1].trim();
            if !inner.is_empty() && !inner.contains(quote) {
                return inner.to_string();
            }
        }
    }
    text.to_string()
}

fn split_header_body(msg: &str) -> Result<(String, Option<String>), ValidationError> {
    let lines: Vec<&str> = msg.lines().collect();
    if lines.is_empty() {
        return Err(ValidationError::Empty);
    }

    let header = lines[0].trim().to_string();
    if header.is_empty() {
        return Err(ValidationError::InvalidHeaderFormat);
    }

    if lines.len() == 1 {
        return Ok((header, None));
    }

    if !lines[1].trim().is_empty() {
        return Err(ValidationError::BadHeaderBodySeparator);
    }

    if lines.len() == 2 {
        return Ok((header, None));
    }

    let body = lines[2..].join("\n").trim_end().to_string();

    if body.is_empty() {
        Ok((header, None))
    } else {
        Ok((header, Some(body)))
    }
}

// <type>[(scope)]: <description>
fn validate_header(header: &str) -> Result<(), ValidationError> {
    // Requires the presence of a ": "
    let separator_pos = match header.find(": ") {
        Some(p) => p,
        None => return Err(ValidationError::InvalidHeaderFormat),
    };

    // type or type(scope)
    let left = &header[..separator_pos];
    let desc = &header[separator_pos + 2..];

    // Parse type and optional scope
    let ty = if let Some(open_paren_pos) = left.find('(') {
        if !left.ends_with(')') {
            return Err(ValidationError::InvalidHeaderFormat);
        }
        let ty = &left[..open_paren_pos];
        let scope = &left[open_paren_pos + 1..left.len() - 1];

        if ty.is_empty() || scope.is_empty() {
            return Err(ValidationError::InvalidHeaderFormat);
        }

        // Only lowercase letters and numbers are allowed
        if !(scope.len() <= 32
            && scope
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()))
        {
            return Err(ValidationError::InvalidScope(scope.to_string()));
        }

        ty
    } else {
        if left.contains(')') {
            return Err(ValidationError::InvalidHeaderFormat);
        }
        left
    };

    // type must be in ALLOWED_TYPES
    if !ALLOWED_TYPES.contains(&ty) {
        return Err(ValidationError::UnknownType(ty.to_string()));
    }

    if desc.is_empty() {
        return Err(ValidationError::EmptyDescription);
    }

    let actual = desc.chars().count();
    if actual > MAX_DESCRIPTION_CHARS {
        return Err(ValidationError::DescriptionTooLong { actual });
    }

    if !desc.starts_with(|c: char| c.is_ascii_lowercase()) {
        return Err(ValidationError::DescriptionNotLowercase);
    }

    if desc.ends_with('.') {
        return Err(ValidationError::TrailingPeriod);
    }

    Ok(())
}

fn normalize(header: &str, body: Option<&str>) -> String {
    let Some(body) = body else {
        return header.to_string();
    };

    let mut lines: Vec<&str> = Vec::new();
    let mut pending_blank = false;

    for line in body.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            pending_blank = !lines.is_empty();
            continue;
        }

        if pending_blank {
            lines.push("");
            pending_blank = false;
        }
        lines.push(line);
    }

    if lines.is_empty() {
        header.to_string()
    } else {
        format!("{header}\n\n{}", lines.join("\n"))
    }
}
