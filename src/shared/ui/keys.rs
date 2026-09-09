/// Key of the interactive loop
/// - Accept     = Tab      → commit with the shown message
/// - Regenerate = Enter    → discard and ask the LLM again
/// - Prev/Next  = ← / →    → navigate the candidate pool
/// - Exit       = anything else (incl. Ctrl+C)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserKey {
    Accept,
    Regenerate,
    Prev,
    Next,
    Exit,
}
