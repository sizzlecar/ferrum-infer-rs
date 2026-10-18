//! Bounded assertions over the real driver, enabled only for upper-layer tests.
//!
//! Model/backend construction stays above this crate. These scenarios inspect
//! the original private owners and never expose a session or submission guard.
use super::*;
use crate::token_execution::*;

pub mod execution_facts;
pub mod lifecycle;
pub mod observability;
pub mod sampling;
pub mod startup;
pub mod token_execution;
pub use super::prefix_cache::test_support as prefix;

// The concrete upper fixture supplies this four-token embedding/head program.
const VOCAB: usize = 4;

/// Consume the actual driver and require its plan to release every owner.
/// The caller keeps its model files alive and performs backend-specific
/// residency assertions on the original returned runtime.
pub fn assert_executor_closed<R: DeviceRuntime>(executor: ProgramExecutor<R>) -> Arc<R> {
    let runtime = Arc::clone(&executor.runtime);
    let resources = Arc::clone(&executor.plan_resources);
    drop(executor);
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {}
        Ok(PlanRuntimeCloseOutcome::Referenced {
            strong_count,
            deferred_cleanup,
            ..
        }) => {
            panic!(
                "closed cohort retained plan owners: {strong_count}, cleanup: {deferred_cleanup:?}"
            )
        }
        Err(failure) => panic!("plan close failed: {:?}", failure.failure()),
    }
    runtime
}
