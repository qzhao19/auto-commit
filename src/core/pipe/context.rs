use crate::core::git::types::{
    BudgetDecision, ClassifiedSnapshot, DiffPayload, Operation, RepositoryContext,
};

#[derive(Debug, Clone, Copy)]
pub enum PromptSource<'a> {
    Seed {
        operation: Operation,
        message: Option<&'a str>,
    },
    Staged {
        snapshot: &'a ClassifiedSnapshot,
        payload: &'a DiffPayload,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTemplate {
    Regular,
    SeedReuse,
    SeedTemplate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledPrompt {
    pub template: PromptTemplate,
    pub prompt: String,
}

#[derive(Debug, Clone, Copy)]
pub struct AssemblerInput<'a> {
    pub repo: &'a RepositoryContext,
    pub source: PromptSource<'a>,
    pub decision: Option<&'a BudgetDecision>,
}
