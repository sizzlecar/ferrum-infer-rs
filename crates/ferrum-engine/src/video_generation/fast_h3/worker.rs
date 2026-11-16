//! Concrete FastH3 adaptation to the existing iterative worker. Callbacks are
//! bound by the composition root to its original assets and stage consumers;
//! they are not backend providers or a second scheduling/execution protocol.
use super::denoising::{DenoisingStage, DenoisingStageError, DenoisingSuspendReceipt};
use crate::{
    iterative_service::{
        BatchError, BatchPreparation, BatchWorker, DenoisingBatch, DenoisingGenerator,
        DenoisingRequest, GeneratorError, InitializationOutcome, InitializationProgress,
        PreparationResources, RequestInitialization,
    },
    video_generation::{RawVideoOutput, VideoGenerationParameters, VideoGenerationRequest},
};
use ferrum_execution::TensorSerialActor;
use ferrum_models::fast_h3::FastH3DenoisingRequest;
use ferrum_runtime::{
    DeviceRuntime, HostMemoryAccount, HostMemoryScope, RequestIdentity, SharedComponentWeights,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Per-request context is a concrete composition-owned type (layout, source
/// views and scope). Independent continuations never acquire permanent slots.
pub(super) struct ReadyFastH3Request<C> {
    pub continuation: FastH3DenoisingRequest,
    pub parameters: VideoGenerationParameters,
    pub host: HostMemoryAccount,
    pub context: C,
}
/// Cancelled is legal only after every original native stage has closed. It
/// carries no submitted owner; errors with such owners must remain Fatal.
pub(super) enum ConditioningOutcome<C> {
    Ready(ReadyFastH3Request<C>),
    Cancelled,
}
struct QuiescentCancelledInitialization;
struct Ready<C> {
    value: ReadyFastH3Request<C>,
    cancelled: Arc<AtomicBool>,
}
struct Prepared<P> {
    value: P,
    cancelled: Option<Arc<AtomicBool>>,
}
struct Membership {
    requests: Vec<RequestIdentity>,
    scope: Option<HostMemoryScope>,
    paused: bool,
}

type Result<T> = std::result::Result<T, BatchError>;
/// Private typed injection of actual stage consumers. The composition supplies
/// P and C; there is no unchecked downcast or map of request ids to model state.
pub(super) struct FastH3WorkerStages<R: DeviceRuntime, P, C> {
    pub prepare: Box<dyn FnMut(&VideoGenerationRequest) -> Result<P> + Send>,
    pub admit: Box<dyn FnMut(&mut P, PreparationResources) -> Result<bool> + Send>,
    /// Admission resolves actual immutable Host reuse before this decision.
    pub needs_auxiliary_model: Box<dyn Fn(&P) -> bool + Send>,
    pub condition: Box<
        dyn FnMut(
                P,
                &VideoGenerationRequest,
                &AtomicBool,
                &mut TensorSerialActor<R>,
            ) -> Result<ConditioningOutcome<C>>
            + Send,
    >,
    pub create_stage: Box<dyn FnMut(&ReadyFastH3Request<C>) -> Result<DenoisingStage<R>> + Send>,
    /// False defers an optional join without submitting or replacing owners.
    /// Every prepared domain must still admit its single-request plan.
    pub batch_capacity: Box<
        dyn FnMut(&ReadyFastH3Request<C>, usize, Option<&SharedComponentWeights<R>>) -> Result<bool>
            + Send,
    >,
    /// Add a demanded cardinality to the same semantic domain. The stage owns
    /// any installation; this callback cannot replace a live request or batch.
    pub extend_stage:
        Box<dyn FnMut(&mut DenoisingStage<R>, &ReadyFastH3Request<C>, usize) -> Result<()> + Send>,
    pub restore:
        Box<dyn FnMut(&mut DenoisingStage<R>, &ReadyFastH3Request<C>) -> Result<()> + Send>,
    pub batch_scope: Box<
        dyn FnMut(&[&ReadyFastH3Request<C>], &mut Option<HostMemoryScope>) -> Result<bool> + Send,
    >,
    pub preflight: Box<
        dyn FnMut(
                &mut DenoisingStage<R>,
                &mut TensorSerialActor<R>,
                &HostMemoryScope,
                &mut [&mut ReadyFastH3Request<C>],
            ) -> Result<bool>
            + Send,
    >,
    pub advance: Box<
        dyn FnMut(
                &mut DenoisingStage<R>,
                &mut TensorSerialActor<R>,
                &HostMemoryAccount,
                &mut [&mut ReadyFastH3Request<C>],
            ) -> Result<Vec<bool>>
            + Send,
    >,
    /// Evaluated only after idle dynamic backing has retired at the finish fence.
    pub can_retain_for_decode: Box<dyn FnMut(&ReadyFastH3Request<C>) -> Result<bool> + Send>,
    pub decode: Box<
        dyn FnMut(
                &mut ReadyFastH3Request<C>,
                &AtomicBool,
                &mut TensorSerialActor<R>,
            ) -> Result<Option<RawVideoOutput>>
            + Send,
    >,
}

pub(super) struct FastH3Worker<R: DeviceRuntime, P, C> {
    actor: TensorSerialActor<R>,
    stages: FastH3WorkerStages<R, P, C>,
    denoising: Option<DenoisingStage<R>>,
    // Keep only the latest real evidence; no history grows with served jobs.
    last_suspend: Option<DenoisingSuspendReceipt>,
}
impl<R: DeviceRuntime, P: Send + 'static, C: Send + 'static> FastH3Worker<R, P, C> {
    pub(super) fn new(actor: TensorSerialActor<R>, stages: FastH3WorkerStages<R, P, C>) -> Self {
        Self {
            actor,
            stages,
            denoising: None,
            last_suspend: None,
        }
    }
    fn suspend(&mut self) -> Result<()> {
        if let Some(stage) = &mut self.denoising {
            if stage.is_resident() {
                self.last_suspend = Some(stage.suspend().map_err(fatal)?);
            } else if !stage.is_suspended() {
                return Err(protocol_failure(
                    "FastH3 stage has unresolved native retirement",
                ));
            }
        }
        Ok(())
    }
    fn restore(&mut self, request: &ReadyFastH3Request<C>) -> Result<()> {
        let stage = self
            .denoising
            .as_mut()
            .ok_or_else(|| rejected("missing prepared denoising stage"))?;
        if stage.is_suspended() {
            (self.stages.restore)(stage, request)?;
        }
        if !stage.is_resident() {
            return Err(protocol_failure(
                "FastH3 restore did not publish an actual resident installation",
            ));
        }
        Ok(())
    }
}
impl<R: DeviceRuntime, P: Send + 'static, C: Send + 'static> BatchWorker for FastH3Worker<R, P, C> {
    type Request = VideoGenerationRequest;
    type Output = RawVideoOutput;
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn DenoisingGenerator<Request = Self::Request, Output = Self::Output>> {
        Some(self)
    }
    fn execute(&mut self, _: BatchPreparation, _: Vec<Self::Request>) -> Result<Vec<Self::Output>> {
        Err(rejected(
            "FastH3 generation requires the existing iterative request lifecycle",
        ))
    }
    fn compatible(&self, a: &Self::Request, b: &Self::Request) -> bool {
        // Metadata collection only. The actual model plan separately validates
        // text/tag layout, original sources, numerical profile and region owner.
        a.parameters.width == b.parameters.width
            && a.parameters.height == b.parameters.height
            && a.parameters.num_frames == b.parameters.num_frames
            && a.parameters.frame_rate == b.parameters.frame_rate
            && a.references
                .iter()
                .map(|r| r.kind())
                .eq(b.references.iter().map(|r| r.kind()))
    }
    fn maximum_batch_size(&self, _: &Self::Request, configured: usize) -> usize {
        configured.min(2)
    }
    fn shutdown(&mut self) -> std::result::Result<(), GeneratorError> {
        self.suspend().map_err(|e| Box::new(e) as GeneratorError)
    }
    fn shutdown_evidence(
        &mut self,
    ) -> std::result::Result<Option<Box<dyn std::any::Any + Send>>, GeneratorError> {
        self.shutdown()?;
        Ok(self
            .last_suspend
            .take()
            .map(|receipt| Box::new(receipt) as Box<dyn std::any::Any + Send>))
    }
}
impl<R: DeviceRuntime, P: Send + 'static, C: Send + 'static> DenoisingGenerator
    for FastH3Worker<R, P, C>
{
    type Request = VideoGenerationRequest;
    type Output = RawVideoOutput;
    fn initialization_uses_auxiliary_model(&mut self) -> bool {
        true
    }
    fn remaining_denoising_steps(&mut self, request: &DenoisingRequest) -> Option<usize> {
        request
            .get::<Ready<C>>()
            .ok()
            .map(|r| r.value.continuation.remaining_evaluations())
    }
    fn supports_request(&mut self, request: &Self::Request) -> bool {
        super::REFERENCE_LIMITS
            .validate_kinds(request.references.iter().map(|r| r.kind()))
            .is_ok()
    }
    fn prepare_request(&mut self, request: &Self::Request) -> Result<BatchPreparation> {
        Ok(BatchPreparation::new(Prepared {
            value: (self.stages.prepare)(request)?,
            cancelled: None,
        }))
    }
    fn bind_request_cancellation(
        &mut self,
        prepared: &mut BatchPreparation,
        cancelled: Arc<AtomicBool>,
    ) {
        if let Ok(prepared) = prepared.get_mut::<Prepared<P>>() {
            prepared.cancelled = Some(cancelled);
        }
    }
    fn try_admit_request(
        &mut self,
        prepared: &mut BatchPreparation,
        resources: PreparationResources,
    ) -> Result<bool> {
        (self.stages.admit)(&mut prepared.get_mut::<Prepared<P>>()?.value, resources)
    }
    fn try_admit_request_at_boundary(
        &mut self,
        prepared: &mut BatchPreparation,
        resources: PreparationResources,
        _: Option<&mut DenoisingBatch>,
        _: &mut [&mut DenoisingRequest],
    ) -> Result<bool> {
        // A refusal must not discard the live installation. Suspend only after
        // actual Host admission, at the completed boundary supplied by Core.
        if !self.try_admit_request(prepared, resources)? {
            return Ok(false);
        }
        if (self.stages.needs_auxiliary_model)(&prepared.get_mut::<Prepared<P>>()?.value) {
            self.suspend()?;
        }
        Ok(true)
    }
    fn batch_is_suspended(&mut self, _: &DenoisingBatch) -> bool {
        self.denoising
            .as_ref()
            .is_some_and(DenoisingStage::is_suspended)
    }
    fn restore_suspended_batch(
        &mut self,
        batch: &mut DenoisingBatch,
        requests: &mut [&mut DenoisingRequest],
    ) -> Result<()> {
        check_members::<C>(batch.get::<Membership>()?, requests)?;
        let first = requests
            .first()
            .ok_or_else(|| rejected("empty restore membership"))?
            .get::<Ready<C>>()?;
        self.restore(&first.value)
    }
    fn start_request(
        &mut self,
        prepared: BatchPreparation,
        request: &Self::Request,
    ) -> Result<DenoisingRequest> {
        match self.begin_initialization(prepared, request)? {
            InitializationOutcome::Ready(ready) => Ok(ready),
            InitializationOutcome::Pending(_) => Err(rejected(
                "cancelled initialization requires the iterative cancellation boundary",
            )),
        }
    }
    fn begin_initialization(
        &mut self,
        prepared: BatchPreparation,
        request: &Self::Request,
    ) -> Result<InitializationOutcome> {
        let prepared = prepared.into_inner::<Prepared<P>>()?;
        let cancelled = prepared
            .cancelled
            .ok_or_else(|| rejected("missing original job cancellation authority"))?;
        if (self.stages.needs_auxiliary_model)(&prepared.value) {
            self.suspend()?;
        }
        match (self.stages.condition)(prepared.value, request, &cancelled, &mut self.actor)? {
            ConditioningOutcome::Ready(value) => {
                Ok(InitializationOutcome::Ready(DenoisingRequest::new(Ready {
                    value,
                    cancelled,
                })))
            }
            ConditioningOutcome::Cancelled if cancelled.load(Ordering::Acquire) => {
                Ok(InitializationOutcome::Pending(RequestInitialization::new(
                    QuiescentCancelledInitialization,
                )))
            }
            ConditioningOutcome::Cancelled => Err(protocol_failure(
                "conditioning claimed cancellation without the original job flag",
            )),
        }
    }
    fn advance_initialization(
        &mut self,
        initialization: &mut RequestInitialization,
        _: &Self::Request,
        _: bool,
        _: &std::task::Waker,
    ) -> Result<InitializationProgress> {
        initialization.get::<QuiescentCancelledInitialization>()?;
        Ok(InitializationProgress::Cancelled)
    }
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut DenoisingRequest],
        previous: Option<&mut DenoisingBatch>,
    ) -> Result<Option<DenoisingBatch>> {
        if requests.is_empty() || requests.len() > 2 {
            return Err(rejected("FastH3 batch is outside B1/B2 domain"));
        }
        let mut ready = requests
            .iter_mut()
            .map(|r| r.get_mut::<Ready<C>>())
            .collect::<Result<Vec<_>>>()?;
        let mut prior = previous.map(|p| p.get_mut::<Membership>()).transpose()?;
        if prior
            .as_ref()
            .is_some_and(|p| !p.paused || p.scope.is_none())
        {
            return Err(rejected(
                "batch replacement requires its original fenced Host authority",
            ));
        }
        if let Some(stage) = self.denoising.as_mut() {
            if stage.is_resident()
                && !stage.prepared_batch_counts().any(|n| n == ready.len())
                && stage.compatible(vec![&mut ready[0].value.continuation])
            {
                // An earlier B1 preflight may own idle pools even before its
                // first forward. Retire those actual temporaries at this fence
                // before forecasting a new cardinality. Weights, continuations
                // and the previous Host scope remain owned; refusal can still
                // resume B1 through its ordinary backing admission.
                stage.release_idle_backing().map_err(fatal)?;
            }
        }
        // Borrow the original installation only when this request can attach
        // to its existing B1 domain. The compiler separately checks the actual
        // candidate's runtime, physical layout and admission contract.
        let weights = self
            .denoising
            .as_ref()
            .filter(|stage| stage.compatible(vec![&mut ready[0].value.continuation]))
            .and_then(DenoisingStage::resident_weights);
        if !(self.stages.batch_capacity)(&ready[0].value, ready.len(), weights)? {
            return Ok(None);
        }
        if let Some(stage) = self.denoising.as_mut() {
            // B1 proves this is the existing domain before an optional plan is
            // attached. A different request domain still uses normal suspend.
            if stage.compatible(vec![&mut ready[0].value.continuation]) {
                (self.stages.extend_stage)(stage, &ready[0].value, ready.len())?;
            }
        }
        let compatible = self.denoising.as_ref().is_some_and(|stage| {
            stage.compatible(
                ready
                    .iter_mut()
                    .map(|r| &mut r.value.continuation)
                    .collect(),
            )
        });
        if !compatible {
            if prior.is_some() {
                return Ok(None);
            }
            self.suspend()?;
            let new = (self.stages.create_stage)(&ready[0].value)?;
            if !new.is_suspended() {
                return Err(rejected("cold stage factory returned a live installation"));
            }
            if !new.compatible(
                ready
                    .iter_mut()
                    .map(|r| &mut r.value.continuation)
                    .collect(),
            ) {
                return Ok(None);
            }
            self.denoising = Some(new);
        }
        let mut scope = prior.as_mut().and_then(|p| p.scope.take());
        let proposal = (|| {
            let members = ready.iter().map(|r| &r.value).collect::<Vec<_>>();
            if !(self.stages.batch_scope)(&members, &mut scope)? {
                return Ok(false);
            }
            let account = scope
                .as_ref()
                .ok_or_else(|| rejected("scope preparation did not retain authority"))?
                .account();
            if members.iter().any(|r| !account.same_parent(&r.host)) {
                return Err(rejected("batch scope belongs to another Host account"));
            }
            self.restore(&ready[0].value)?;
            (self.stages.preflight)(
                self.denoising.as_mut().unwrap(),
                &mut self.actor,
                scope.as_ref().expect("checked batch scope"),
                &mut ready.iter_mut().map(|r| &mut r.value).collect::<Vec<_>>(),
            )
        })();
        match proposal {
            Ok(true) => Ok(Some(DenoisingBatch::new(Membership {
                requests: ready
                    .iter()
                    .map(|r| r.value.continuation.request().clone())
                    .collect(),
                scope,
                paused: true,
            }))),
            other => {
                // Exchange preserves the old restart credit. A device refusal
                // returns that actual guard to the unchanged old membership.
                if let Some(prior) = prior {
                    prior.scope = scope;
                }
                match other {
                    Ok(false) => Ok(None),
                    Err(e) => Err(e),
                    Ok(true) => unreachable!(),
                }
            }
        }
    }
    fn pause_batch(
        &mut self,
        batch: &mut DenoisingBatch,
        requests: &mut [&mut DenoisingRequest],
    ) -> Result<()> {
        let membership = batch.get_mut::<Membership>()?;
        check_members::<C>(membership, requests)?;
        // advance() returns only after original completion fences and commits.
        membership.paused = true;
        Ok(())
    }
    fn advance_batch(
        &mut self,
        batch: &mut DenoisingBatch,
        requests: &mut [&mut DenoisingRequest],
    ) -> Result<Vec<bool>> {
        let membership = batch.get_mut::<Membership>()?;
        check_members::<C>(membership, requests)?;
        let mut ready = requests
            .iter_mut()
            .map(|r| r.get_mut::<Ready<C>>())
            .collect::<Result<Vec<_>>>()?;
        self.restore(&ready[0].value)?;
        membership.paused = false;
        let output = (self.stages.advance)(
            self.denoising.as_mut().unwrap(),
            &mut self.actor,
            &membership
                .scope
                .as_ref()
                .ok_or_else(|| rejected("membership lost its Host authority"))?
                .account(),
            &mut ready.iter_mut().map(|r| &mut r.value).collect::<Vec<_>>(),
        )?;
        if output.len() != ready.len()
            || output
                .iter()
                .zip(&ready)
                .any(|(&done, r)| done != r.value.continuation.is_complete())
        {
            return Err(protocol_failure(
                "FastH3 completion differs from actual request continuations",
            ));
        }
        membership.paused = true;
        Ok(output)
    }
    fn finish_request(
        &mut self,
        request: &mut DenoisingRequest,
        batch: &mut DenoisingBatch,
    ) -> Result<Self::Output> {
        self.try_finish_request(request, batch)?.ok_or_else(|| {
            rejected("cancelled decoding requires the iterative cancellation boundary")
        })
    }
    fn try_finish_request(
        &mut self,
        request: &mut DenoisingRequest,
        batch: &mut DenoisingBatch,
    ) -> Result<Option<Self::Output>> {
        let ready = request.get_mut::<Ready<C>>()?;
        let membership = batch.get::<Membership>()?;
        // Core may already have compacted the surviving cohort. The completed
        // request retains its own continuation and is not a row of that cohort.
        if !membership.paused || !ready.value.continuation.is_complete() {
            return Err(rejected(
                "decode requires a completed original member at its fence",
            ));
        }
        if let Some(stage) = &mut self.denoising {
            if stage.is_resident() {
                stage.release_idle_backing().map_err(fatal)?;
                if !(self.stages.can_retain_for_decode)(&ready.value)? {
                    self.suspend()?;
                }
            } else if !stage.is_suspended() {
                return Err(protocol_failure(
                    "decode found unresolved denoiser retirement",
                ));
            }
        }
        let output = (self.stages.decode)(&mut ready.value, &ready.cancelled, &mut self.actor)?;
        if output.is_none() && !ready.cancelled.load(Ordering::Acquire) {
            return Err(protocol_failure(
                "decoder returned cancellation without the original job flag",
            ));
        }
        Ok(output)
    }
}
fn check_members<C: Send + 'static>(
    membership: &Membership,
    requests: &[&mut DenoisingRequest],
) -> Result<()> {
    let actual = requests
        .iter()
        .map(|r| r.get::<Ready<C>>().map(|r| r.value.continuation.request()))
        .collect::<Result<Vec<_>>>()?;
    if !membership.requests.iter().eq(actual) {
        return Err(rejected(
            "step membership changed without actual batch preparation",
        ));
    }
    Ok(())
}
fn fatal<R: DeviceRuntime>(error: DenoisingStageError<R>) -> BatchError {
    BatchError::Fatal(Box::new(error))
}
fn rejected(message: &str) -> BatchError {
    BatchError::Rejected(message.into())
}
fn protocol_failure(message: &'static str) -> BatchError {
    BatchError::Fatal(Box::new(std::io::Error::other(message)))
}
