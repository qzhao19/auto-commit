use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::core::inter::cycle::InteractiveLoop;
use crate::shared::exception::{LlmError, ProviderErrorType};
use crate::shared::ui::{CandidateView, Ui, UserKey};

#[derive(Default)]
struct RecorderUi {
    shown: Vec<String>,
}

impl Ui for RecorderUi {
    fn show_generating(&mut self) {}

    fn show_message(&mut self, view: CandidateView<'_>) {
        self.shown.push(view.message.to_owned());
    }
}

/// Keys pressed while a generation is in flight replay in press order:
/// pre-pressing Accept (Tab) then Next (→) must commit the newly generated
/// message, not silently drop the Accept behind the later Next.
#[tokio::test]
async fn keys_pressed_during_generation_replay_in_press_order() {
    // Scripted generator: each call parks on the channel until the test
    // releases the next reply, giving full control of the "generating" phase.
    let (reply_tx, reply_rx) = mpsc::unbounded_channel::<String>();
    let replies = Arc::new(tokio::sync::Mutex::new(reply_rx));
    let generate = move || {
        let replies = replies.clone();
        async move {
            replies.lock().await.recv().await.ok_or_else(|| {
                LlmError::Provider(ProviderErrorType::Fatal, "script exhausted".into())
            })
        }
    };

    let (key_tx, key_rx) = mpsc::unbounded_channel();

    let task = tokio::spawn(async move {
        InteractiveLoop::new(generate, key_rx, RecorderUi::default())
            .run()
            .await
            .expect("run must not fail")
    });

    // Generation #1 completes -> candidate "A" shown; user regenerates.
    reply_tx.send("A".to_owned()).unwrap();
    key_tx.send(UserKey::Regenerate).unwrap();
    // Let the loop settle into the pending 2nd generation (#[tokio::test]
    // runs a current-thread runtime, so yields are deterministic here).
    tokio::time::sleep(Duration::from_millis(50)).await;

    // While generation #2 is in flight: press Accept, then Next.
    key_tx.send(UserKey::Accept).unwrap();
    key_tx.send(UserKey::Next).unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Release generation #2.
    reply_tx.send("B".to_owned()).unwrap();

    // Failsafe: without the fix the Accept is swallowed by Next and the
    // loop waits for input; Exit ends the test instead of hanging it.
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        let _ = key_tx.send(UserKey::Exit);
    });

    let decision = task.await.unwrap();
    assert_eq!(decision.as_deref(), Some("B"));
}
