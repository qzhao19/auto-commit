use std::collections::VecDeque;

/// Default capacity used by new()
const DEFAULT_CAPACITY: usize = 8;

#[derive(Debug)]
pub struct CandidatePool {
    messages: VecDeque<String>,
    // Index of current candidate
    cursor: usize,
    capacity: usize,
}

impl CandidatePool {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "candidate pool capacity must be non-zero");
        Self {
            messages: VecDeque::new(),
            cursor: 0,
            capacity,
        }
    }

    /// Record a generated message and make it current
    pub fn push(&mut self, message: String) -> usize {
        if let Some(index) = self.messages.iter().position(|msg| *msg == message) {
            self.cursor = index;
            return index + 1;
        }

        // Pop the oldest message
        if self.messages.len() == self.capacity {
            self.messages.pop_front();
        }

        self.messages.push_back(message);
        self.cursor = self.messages.len() - 1;
        self.messages.len()
    }

    /// Move to previous candidate
    pub fn prev(&mut self) -> Option<usize> {
        if self.messages.is_empty() {
            return None;
        }

        let len = self.messages.len();
        self.cursor = (self.cursor + len - 1) % len;
        Some(self.cursor + 1)
    }

    /// Move to the next candidate
    pub fn next(&mut self) -> Option<usize> {
        if self.messages.is_empty() {
            return None;
        }

        let len = self.messages.len();
        self.cursor = (self.cursor + 1) % len;
        Some(self.cursor + 1)
    }

    /// Current candidate
    pub fn current(&self) -> Option<(&str, usize, usize)> {
        self.messages
            .get(self.cursor)
            .map(|msg| (msg.as_str(), self.cursor + 1, self.messages.len()))
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for CandidatePool {
    fn default() -> Self {
        Self::new()
    }
}
