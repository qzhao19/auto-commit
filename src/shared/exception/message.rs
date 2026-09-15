use std::fmt;

/// Header description budget
pub const MAX_DESCRIPTION_CHARS: usize = 72;

pub const ALLOWED_TYPES: &[&str] = &[
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "chore", "build", "ci",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    Empty,
    ControlCharacters,
    InvalidHeaderFormat,
    UnknownType(String),
    InvalidScope(String),
    EmptyDescription,
    DescriptionNotLowercase,
    DescriptionTooLong { actual: usize },
    TrailingPeriod,
    BadHeaderBodySeparator,
    ForbiddenExtraContent,
    TooVague,
    SensitiveContent,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty or whitespace-only message"),
            Self::ControlCharacters => write!(f, "contains non-printable / control characters"),
            Self::InvalidHeaderFormat => {
                write!(f, "header does not match conventional commit format")
            }
            Self::UnknownType(t) => write!(
                f,
                "unknown type `{t}`; allowed: {}",
                ALLOWED_TYPES.join("|")
            ),

            Self::InvalidScope(s) => write!(
                f,
                "invalid scope `{s}`: must be a lowercase short noun inside parentheses"
            ),
            Self::EmptyDescription => {
                write!(f, "description after the colon must not be empty")
            }
            Self::DescriptionNotLowercase => {
                write!(f, "description must not start with an uppercase letter")
            }
            Self::DescriptionTooLong { actual } => write!(
                f,
                "description is {actual} chars (limit {})",
                MAX_DESCRIPTION_CHARS
            ),
            Self::TrailingPeriod => write!(f, "description ends with a period"),
            Self::BadHeaderBodySeparator => {
                write!(
                    f,
                    "header and body must be separated by exactly one blank line"
                )
            }
            Self::ForbiddenExtraContent => write!(
                f,
                "contains markdown / prefix / trailing noise that cannot be stripped safely"
            ),
            Self::TooVague => write!(f, "message is too vague / generic"),
            Self::SensitiveContent => {
                write!(
                    f,
                    "potential sensitive information detected (key / token / password)"
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}
