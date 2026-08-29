mod collector;
mod parser;

pub use collector::StagedMetadataCollector;

// Parser internals exposed for unit tests only
#[cfg(test)]
pub(crate) use parser::{RawEntry, parse_numstat, parse_raw_entries};
