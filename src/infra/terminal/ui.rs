use std::io::{self, Write};

use crossterm::cursor;
use crossterm::execute;
use crossterm::terminal;

use crate::shared::ui::CandidateView;
use crate::shared::ui::Ui;

const TITLE: &str = "Auto Commit Message Generation";

pub struct TerminalUi {
    repo: String,
}

impl TerminalUi {
    pub fn new(repo: impl Into<String>) -> Self {
        Self { repo: repo.into() }
    }

    fn header(&self) -> String {
        format!("{TITLE}\r\nRepository: {}\r\n", self.repo)
    }
}

impl Ui for TerminalUi {
    fn show_generating(&mut self) {
        render(&format!("{}\r\nGenerating message...\r\n", self.header()));
    }

    fn show_message(&mut self, view: CandidateView<'_>) {
        let indented = view
            .message
            .lines()
            .map(|line| {
                if line.is_empty() {
                    String::new()
                } else {
                    format!("    {line}")
                }
            })
            .collect::<Vec<_>>()
            .join("\r\n");

        let mut options = vec![
            "Options:".to_string(),
            "    Tab key    - Accept and use this message".to_string(),
            "    Enter key  - Reject and regenerate".to_string(),
        ];

        if view.total > 1 {
            options.push(format!(
                "    \u{2190} / \u{2192} keys - Switch between the {0} previously generated messages (candidate pool: {0})",
                view.total
            ));
        }

        options.push("    Other keys - Exit".to_string());
        let options = options.join("\r\n");

        render(&format!(
            "\r\nGenerated Commit Message (Version {}/{}):\r\n{}\r\n\r\n{}\r\n",
            view.position, view.total, indented, options
        ))
    }
}

// Helper function

fn render(text: &str) {
    let mut out = io::stdout();
    let _ = execute!(
        out,
        terminal::Clear(terminal::ClearType::All),
        cursor::MoveTo(0, 0)
    );
    let _ = write!(out, "{text}");
    let _ = out.flush();
}
