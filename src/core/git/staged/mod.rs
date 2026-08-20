mod collector;
mod parser;

pub use collector::StagedMetadataCollector;

// Parser internals exposed for unit tests only
#[cfg(test)]
pub(crate) use parser::{NameStatusEntry, parse_name_status, parse_numstat, parse_submodule_flags};
