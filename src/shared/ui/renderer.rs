/// Key of the interactive loop
/// - Accept     = Tab      → commit with the shown message
/// - Regenerate = Enter    → discard and ask the LLM again
/// - Exit       = anything else (incl. Ctrl+C)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserKey {
    Accept,
    Regenerate,
    Exit,
}

/// Renderer callbacks
pub trait Ui {
    fn show_generating(&mut self);
    fn show_message(&mut self, message: &str);
}
