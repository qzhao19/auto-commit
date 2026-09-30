use std::io::{self, Write};

use crossterm::cursor;
use crossterm::execute;
use crossterm::terminal;
use unicode_width::UnicodeWidthChar;

use crate::shared::ui::CandidateView;
use crate::shared::ui::Ui;

const TAB_STOP: usize = 8;

const TITLE: &str = "Auto Commit Message Generation";

pub struct TerminalUi {
    repo: String,
    drawn_rows: usize,
}

impl TerminalUi {
    pub fn new(repo: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            drawn_rows: 0,
        }
    }

    fn header(&self) -> String {
        format!("{TITLE}\r\nRepository: {}\r\n", self.repo)
    }

    /// Erase the previously drawn block
    fn draw(&mut self, text: &str) {
        let mut out = io::stdout();

        // Query before erasing
        let width = match terminal::size() {
            Ok((columns, _)) => (columns as usize).max(1),
            Err(_) => {
                for _ in 0..self.drawn_rows {
                    let _ = execute!(
                        out,
                        cursor::MoveUp(1),
                        terminal::Clear(terminal::ClearType::CurrentLine)
                    );
                }
                self.drawn_rows = 0;
                let _ = out.flush();
                return;
            }
        };

        for _ in 0..self.drawn_rows {
            let _ = execute!(
                out,
                cursor::MoveUp(1),
                terminal::Clear(terminal::ClearType::CurrentLine)
            );
        }

        let rows = row_count(text, width);
        if write!(out, "{text}").and_then(|_| out.flush()).is_err() {
            self.drawn_rows = 0;
            return;
        }
        self.drawn_rows = rows;
    }
}

impl Ui for TerminalUi {
    fn show_generating(&mut self) {
        self.draw(&format!("{}\r\nGenerating message...\r\n", self.header()));
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

        self.draw(&format!(
            "\r\nGenerated Commit Message (Version {}/{}):\r\n{}\r\n\r\n{}\r\n",
            view.position, view.total, indented, options
        ))
    }
}

fn row_count(text: &str, width: usize) -> usize {
    text.strip_suffix("\r\n")
        .unwrap_or(text)
        .split("\r\n")
        .map(|line| {
            let mut col = 0usize;
            for c in line.chars() {
                match c {
                    '\t' => col = (col / TAB_STOP + 1) * TAB_STOP,
                    _ => col += UnicodeWidthChar::width(c).unwrap_or(0),
                }
            }
            col.div_ceil(width).max(1)
        })
        .sum()
}
