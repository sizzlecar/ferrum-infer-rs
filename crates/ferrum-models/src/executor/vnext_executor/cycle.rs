//! Coarse, opt-in host boundaries. Diagnostic witnesses retain no resources.

use super::*;
use ferrum_interfaces::{
    DecodeCycleCallContext, DecodeCycleCaller, DecodeCycleOutcome, DecodeCycleParticipantWitness,
    DecodeCycleScope, DecodeCycleStage, DecodeCycleSubmissionWitness, DecodeCycleTraceError,
    DecodeCycleWaveContext, DecodeCycleWaveKind,
};

pub(super) struct VNextCycleCall {
    trace: DecodeCycleTrace,
    callers: Vec<DecodeCycleCaller>,
    context: Option<DecodeCycleWaveContext>,
    submission: Option<Arc<DecodeCycleSubmissionWitness>>,
}

impl VNextCycleCall {
    pub(super) fn prepare<R: DeviceRuntime>(
        &mut self,
        wave: &PreparedStepSubmissionWave<R>,
        kind: VNextExecutionWaveKind,
    ) {
        self.context = None;
        self.submission = None;
        let kind = match kind {
            VNextExecutionWaveKind::Prefill => DecodeCycleWaveKind::Prefill,
            VNextExecutionWaveKind::Decode => DecodeCycleWaveKind::Decode,
            VNextExecutionWaveKind::Mixed => DecodeCycleWaveKind::Mixed,
        };
        let witness = self.trace.wave_witness(wave, kind, self.callers.clone());
        match witness.and_then(|witness| DecodeCycleWaveContext::new(self.trace.clone(), witness)) {
            Ok(context) => {
                context.mark(DecodeCycleStage::PrepareEnd);
                context.mark(DecodeCycleStage::WavePrepared);
                self.context = Some(context);
            }
            Err(error) => self.trace.invalidate(error),
        }
    }

    pub(super) fn context(&self) -> Option<&DecodeCycleWaveContext> {
        self.context.as_ref()
    }

    pub(super) fn bind<R: DeviceRuntime>(&mut self, completion: &CompletionHandle<R>) {
        let Some(context) = self.context.as_ref() else {
            return;
        };
        let bound =
            DecodeCycleSubmissionWitness::bind(Arc::clone(context.witness()), completion.receipt());
        match bound {
            Ok(submission) => match DecodeCycleWaveContext::for_submission(
                self.trace.clone(),
                Arc::clone(&submission),
            ) {
                Ok(context) => {
                    context.mark(DecodeCycleStage::SubmissionBound);
                    self.context = Some(context);
                    self.submission = Some(submission);
                }
                Err(error) => self.trace.invalidate(error),
            },
            Err(error) => self.trace.invalidate(error),
        }
    }

    pub(super) fn mark(&self, stage: DecodeCycleStage) {
        if let Some(context) = &self.context {
            context.mark(stage);
        }
    }

    pub(super) fn failed(&self) {
        if let Some(context) = &self.context {
            self.trace.mark_with_outcome(
                DecodeCycleStage::FailureObserved,
                context.scope(),
                DecodeCycleOutcome::Failed,
            );
        }
    }

    pub(super) fn participant(&self, canonical: usize) -> Option<DecodeCycleParticipantWitness> {
        let submission = self.submission.as_ref()?;
        let result = u32::try_from(canonical)
            .map_err(|_| DecodeCycleTraceError::InvalidCallerMap)
            .and_then(|ordinal| submission.participant(ordinal));
        match result {
            Ok(witness) => Some(witness),
            Err(error) => {
                self.trace.invalidate(error);
                None
            }
        }
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    /// A call boundary can precede canonical admission without inventing a wave.
    pub(super) fn cycle_prepare_start(&self, context: Option<DecodeCycleCallContext>) {
        if !self.decode_cycle_enabled.load(Ordering::Acquire) {
            return;
        }
        let Some(context) = context else {
            return;
        };
        if let Some(trace) = self.decode_cycle_trace.read().as_ref() {
            trace.mark(
                DecodeCycleStage::PrepareStart,
                DecodeCycleScope::Call(context),
            );
        }
    }

    pub(super) fn cycle_call(
        &self,
        callers: impl FnOnce() -> std::result::Result<Vec<DecodeCycleCaller>, DecodeCycleTraceError>,
    ) -> Option<VNextCycleCall> {
        if !self.decode_cycle_enabled.load(Ordering::Acquire) {
            return None;
        }
        let trace = self.decode_cycle_trace.read().clone()?;
        let callers = match callers() {
            Ok(callers) => callers,
            Err(error) => {
                trace.invalidate(error);
                return None;
            }
        };
        Some(VNextCycleCall {
            trace,
            callers,
            context: None,
            submission: None,
        })
    }
}

pub(super) fn caller<R: DeviceRuntime>(
    sequence: &VNextSequence<R>,
    ordinal: usize,
    context: Option<DecodeCycleCallContext>,
) -> std::result::Result<DecodeCycleCaller, DecodeCycleTraceError> {
    let ordinal = u32::try_from(ordinal).map_err(|_| DecodeCycleTraceError::InvalidCallerMap)?;
    Ok(
        DecodeCycleCaller::new(sequence.request_id().clone(), ordinal, context)
            .with_request_identity(
                sequence
                    .session
                    .resources()
                    .request_resources()
                    .request_id()
                    .clone(),
            ),
    )
}

/// The ordinary disabled timing specialization with a separate coarse observer.
/// It must not turn on per-node clocks, attribution, or GPU timing events.
pub(super) struct CycleTiming<'a> {
    pub(super) context: Option<&'a DecodeCycleWaveContext>,
}

impl DeviceSubmissionTimingSink for CycleTiming<'_> {
    const ENABLED: bool = false;

    fn decode_cycle_context(&self) -> Option<&DecodeCycleWaveContext> {
        self.context
    }

    fn record_device_submission(&self, _stage: DeviceSubmissionStage, _elapsed: Duration) {
        unreachable!("disabled fine submission timing cannot record")
    }
}

impl SubmissionWaveDispatchTimingSink for CycleTiming<'_> {
    fn record(&self, _stage: SubmissionWaveDispatchStage, _elapsed: Duration) {
        unreachable!("disabled fine dispatch timing cannot record")
    }
}
