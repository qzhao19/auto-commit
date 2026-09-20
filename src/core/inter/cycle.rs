use std::collections::VecDeque;
use std::marker::PhantomData;

use tokio::sync::mpsc;

use crate::infra::cache::CandidatePool;
use crate::shared::exception::LlmError;
use crate::shared::ui::{CandidateView, Ui, UserKey};

const MAX_CANDIDATES: usize = 32;

/// Owns the interactive generate -> show -> decide loop
pub struct InteractiveLoop<G, F, U>
where
    G: FnMut() -> F,
    F: std::future::Future<Output = Result<String, LlmError>>,
    U: Ui,
{
    generate: G,
    keys: mpsc::UnboundedReceiver<UserKey>,
    ui: U,
    pool: CandidatePool,
    /// `F` is only determined by `G`.
    _output: PhantomData<F>,
}

impl<G, F, U> InteractiveLoop<G, F, U>
where
    G: FnMut() -> F,
    F: std::future::Future<Output = Result<String, LlmError>>,
    U: Ui,
{
    /// Create a new loop
    pub fn new(generate: G, keys: mpsc::UnboundedReceiver<UserKey>, ui: U) -> Self {
        Self {
            generate,
            keys,
            ui,
            pool: CandidatePool::with_capacity(MAX_CANDIDATES),
            _output: PhantomData,
        }
    }

    /// Run until the user accepts a candidate or exits.
    pub async fn run(mut self) -> Result<Option<String>, LlmError> {
        let mut queued: VecDeque<UserKey> = VecDeque::new();

        'generate: loop {
            self.ui.show_generating();

            let future = (self.generate)();
            tokio::pin!(future);

            let message = loop {
                tokio::select! {
                    result = &mut future => break result?,
                    key = self.keys.recv() => match key {
                        Some(UserKey::Exit) | None => return Ok(None),
                        Some(other) => {
                            queued.push_back(other);
                            continue;
                        }
                    }
                }
            };
            // Latest candidate becomes current
            self.pool.push(message);

            loop {
                let Some((candidate_msg, position, total)) = self.pool.current() else {
                    return Ok(None);
                };

                self.ui.show_message(CandidateView {
                    message: candidate_msg,
                    position,
                    total,
                });

                let key = match queued.pop_front() {
                    Some(key) => Some(key),
                    None => self.keys.recv().await,
                };

                match key {
                    Some(UserKey::Accept) => return Ok(Some(candidate_msg.to_owned())),
                    Some(UserKey::Regenerate) => continue 'generate,
                    Some(UserKey::Prev) => {
                        self.pool.prev();
                    }
                    Some(UserKey::Next) => {
                        self.pool.next();
                    }
                    Some(UserKey::Exit) | None => return Ok(None),
                }
            }
        }
    }
}
