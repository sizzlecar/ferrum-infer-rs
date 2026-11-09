//! Single-request adapter over the shared original expansion owner. Visual
//! input continues through its original one-request prepared graph and phase.
#[cfg(test)]
use super::cohort::{CohortState, ParticipantState as State};
use super::cohort::{ExpansionCohortProgress, ExpansionParticipantResult, NativeExpansionCohort};
use super::*;
use ferrum_execution::executor_lifecycle::ExecutorShutdownObserver;
#[cfg(test)]
use ferrum_execution::executor_lifecycle::ExecutorShutdownPhase;
use ferrum_runtime::HostMemoryScope;
#[cfg(test)]
use std::task::Poll;
use std::task::Waker;
#[cfg(test)]
mod tests;

pub(crate) enum ExpansionPhaseProgress {
    Pending { made_progress: bool },
    Expanded(String),
    Cancelled,
}

pub(crate) struct NativeExpansionPhase {
    cohort: NativeExpansionCohort,
}

impl NativeExpansionPhase {
    pub(crate) fn initialize<R: DeviceRuntime>(
        assets: Arc<NativeTextExpansion>,
        composition: &RuntimeComposition<R>,
        prepared: PreparedTextExpansion,
        phase: HostMemoryScope,
    ) -> ExecutorBuildResult<Self> {
        Self::initialize_with_diagnostics(assets, composition, prepared, phase, None)
    }

    pub(crate) fn initialize_with_diagnostics<R: DeviceRuntime>(
        assets: Arc<NativeTextExpansion>,
        composition: &RuntimeComposition<R>,
        prepared: PreparedTextExpansion,
        phase: HostMemoryScope,
        diagnostics: Option<super::DiagnosticContext>,
    ) -> ExecutorBuildResult<Self> {
        NativeExpansionCohort::initialize_single(assets, composition, prepared, phase, diagnostics)
            .map(|cohort| Self { cohort })
    }

    pub(crate) fn cleanup(&self) -> ExecutorShutdownObserver {
        self.cohort.cleanup()
    }

    pub(crate) fn advance(
        &mut self,
        cancelled: bool,
        wake: &Waker,
    ) -> Result<ExpansionPhaseProgress> {
        match self.cohort.advance(&[cancelled], wake)? {
            ExpansionCohortProgress::Pending { made_progress } => {
                Ok(ExpansionPhaseProgress::Pending { made_progress })
            }
            ExpansionCohortProgress::Retired(mut results) => {
                if results.len() != 1 {
                    return Err(FerrumError::internal(
                        "single expansion phase has another cohort extent",
                    ));
                }
                match results.pop().expect("one result checked") {
                    ExpansionParticipantResult::Expanded(text) => {
                        Ok(ExpansionPhaseProgress::Expanded(text))
                    }
                    ExpansionParticipantResult::Cancelled => Ok(ExpansionPhaseProgress::Cancelled),
                    ExpansionParticipantResult::Failed(error) => Err(error),
                }
            }
        }
    }
}
