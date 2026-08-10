use std::path::PathBuf;

#[derive(Debug)]
pub enum ConfigError {
    TomlParse {
        path: PathBuf,
        source: toml::de::Error,
    },
    FileRead {
        path: PathBuf,
        source: std::io::Error,
    },
    EnvParse {
        var: String,
        expected: &'static str,
    },
    MissingRequired {
        field: &'static str,
        hint: String,
    },
    InvalidValue {
        field: &'static str,
        reason: String,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::TomlParse { path, source } => {
                write!(
                    f,
                    "failed to parse config file {}: {}",
                    path.display(),
                    source
                )
            }
            ConfigError::FileRead { path, source } => {
                write!(
                    f,
                    "failed to read config file {}: {}",
                    path.display(),
                    source
                )
            }
            ConfigError::EnvParse { var, expected } => {
                write!(f, "invalid value for env var {var}: expected {expected}")
            }
            ConfigError::MissingRequired { field, hint } => {
                write!(f, "missing required field {field}: {hint}")
            }
            ConfigError::InvalidValue { field, reason } => {
                write!(f, "invalid value for {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::TomlParse { source, .. } => Some(source),
            ConfigError::FileRead { source, .. } => Some(source),
            _ => None,
        }
    }
}
