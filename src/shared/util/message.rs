// Validate message

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCommitMessage(pub(crate) String);

impl ValidatedCommitMessage {
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[inline]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for ValidatedCommitMessage {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ValidatedCommitMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
