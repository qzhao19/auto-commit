use std::io::{self, IsTerminal};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use crossterm::terminal;
use tokio::sync::mpsc;

use crate::shared::ui::UserKey;

/// Raw-mode scope guard: restores the terminal on every exit path
pub struct RawModeGuard;

impl RawModeGuard {
    pub fn enter() -> io::Result<Self> {
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "auto-commit requires an interactive terminal (stdin/stdout must be a TTY)",
            ));
        }
        terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

/// Stops the key listener thread when dropped.
pub struct StopGuard {
    stop: Arc<AtomicBool>,
}

impl Drop for StopGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Background key listener
///
/// Spawns a dedicated thread that reads terminal key events (crossterm's
/// `event::read` is blocking, so it must not run on a tokio worker thread)
/// and forwards them as `UserKey` values through an mpsc channel
pub struct KeyListener {
    _stop: StopGuard,
    rx: mpsc::UnboundedReceiver<UserKey>,
}

impl KeyListener {
    pub fn spawn() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::unbounded_channel();
        let stop_flag = stop.clone();

        std::thread::Builder::new()
            .name("ui-key-listener".into())
            .spawn(move || {
                while !stop_flag.load(Ordering::Relaxed) {
                    match event::poll(Duration::from_millis(50)) {
                        Ok(true) => {
                            if let Ok(Event::Key(key)) = event::read() {
                                if key.kind == KeyEventKind::Press && tx.send(map_key(key)).is_err()
                                {
                                    break;
                                }
                            }
                        }
                        Ok(false) => {}
                        Err(_) => break,
                    }
                }
            })
            .expect("failed to spawn key listener thread");

        Self {
            _stop: StopGuard { stop },
            rx,
        }
    }

    pub fn into_parts(self) -> (StopGuard, mpsc::UnboundedReceiver<UserKey>) {
        (self._stop, self.rx)
    }
}

// Helper function

fn map_key(key: KeyEvent) -> UserKey {
    match key.code {
        KeyCode::Tab => UserKey::Accept,
        KeyCode::Enter => UserKey::Regenerate,
        KeyCode::Left => UserKey::Prev,
        KeyCode::Right => UserKey::Next,
        _ => UserKey::Exit,
    }
}
