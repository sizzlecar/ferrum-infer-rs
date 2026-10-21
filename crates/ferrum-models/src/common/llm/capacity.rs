//! Borrowed original-cache bindings used while the executor registry is held.

use ferrum_execution::model_executor::PersistentCapacityInspectionBudget;

#[derive(Clone, Copy, Debug)]
pub struct ModelCapacityResident<'a> {
    pub cache_id: &'a str,
    pub consumed_tokens: usize,
    pub through_tokens: usize,
}

/// Model-side physical check. The executor separately validates exact request
/// incarnations and retains its registry lock throughout this observation.
pub struct ModelPersistentCapacityQuery<'a> {
    pub residents: &'a [ModelCapacityResident<'a>],
    pub through_tokens: &'a [usize],
    pub max_live_sequences: usize,
    pub max_wave_sequences: usize,
    pub inspection_budget: PersistentCapacityInspectionBudget,
}
