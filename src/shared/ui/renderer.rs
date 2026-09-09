/// One rendered candidate from the poo
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateView<'a> {
    pub message: &'a str,
    pub position: usize,
    pub total: usize,
}

/// Renderer callbacks
pub trait Ui {
    fn show_generating(&mut self);
    fn show_message(&mut self, view: CandidateView<'_>);
}
