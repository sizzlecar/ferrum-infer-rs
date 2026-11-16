//! One bounded set of denoising batch plans sharing an actual Core installation.
//! Independent model continuations are borrowed only for one completed step.
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentInitializationError,
        ComponentWeightSource, TensorComponentPolicy, TensorComponentSession,
    },
    TensorExecutionError, TensorExecutionOutput, TensorSerialActor,
};
use ferrum_kernels::native_ops::sage_vsa_fp4::CheckedVsaRegions;
use ferrum_models::fast_h3::*;
use ferrum_runtime::*;
use std::{collections::BTreeMap, fmt, sync::Arc};

mod handoff;
mod preflight;
mod prepared;
mod residency;
mod step;
pub use handoff::DevicePredictionFailure;
pub use prepared::{PreparedDenoisingBatch, PreparedPrediction};
pub use residency::DenoisingSuspendReceipt;
pub use step::{PredictionFailure, PredictionInvocation, PredictionPhase};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DenoisingWorkspacePolicy {
    /// Keep workspace only for the executing phase. After the joint forward
    /// retires and its hidden output reaches Host, return its idle backing
    /// before admitting a prediction head. Static shared weights stay resident.
    /// Cardinality changes also release the previous batch's idle backing.
    ActivePhaseOnly,
}

/// Optional retention never grants extra capacity. Auxiliary stages may force
/// the original fenced retirement before their own ordinary admission.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DenoisingResidencyPolicy {
    #[default]
    ReleaseBeforeDecode,
    RetainIfCapacityAllows,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PredictionHandoffPolicy {
    #[default]
    Host,
    DeviceIfCapacityAllows,
}

/// Partition only independent FFN rows; this never changes a reduction axis
/// or the model's arithmetic profile. Selection is a forecast, not admission.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum FeedForwardPartitionPolicy {
    /// Prefer the compact whole-row graph when its additional peak fits the
    /// current device account. A compatible live installation is already paid;
    /// cold plans include their static weights and native reservations remain
    /// charged to the original account.
    #[default]
    IfCapacityNeeded,
    /// Bound every FFN fragment, useful for explicitly constrained workloads.
    Always,
}

/// Caller-owned compiled plans and immutable source views outlive this stage.
/// Source bytes are borrowed during cold restore only; compatible batch plans
/// attach without reading or uploading them a second time.
pub struct DenoisingStage<R: DeviceRuntime> {
    runtime: Arc<R>,
    host: HostMemoryAccount,
    batches: BTreeMap<usize, PreparedDenoisingBatch<R>>,
    predictions: Vec<PreparedPrediction<R>>,
    sessions: BTreeMap<usize, TensorComponentSession<R>>,
    weights: Option<SharedComponentWeights<R>>,
    failed: bool,
    workspace: DenoisingWorkspacePolicy,
    active_batch: Option<usize>,
    handoff_capacity: Option<(DeviceCapacityOwner, u64)>,
    // Declared last so the original Stored bytes outlive session/installation
    // owners even during drop; normal use still requires explicit retirement.
    sources: Vec<Arc<dyn WeightComponentSource>>,
}
impl<R: DeviceRuntime> DenoisingStage<R> {
    pub fn new(
        runtime: Arc<R>,
        host: HostMemoryAccount,
        batches: Vec<PreparedDenoisingBatch<R>>,
        predictions: Vec<PreparedPrediction<R>>,
        sources: Vec<Arc<dyn WeightComponentSource>>,
        workspace: DenoisingWorkspacePolicy,
    ) -> Result<Self, VNextError> {
        if batches.is_empty()
            || batches.len() > 2
            || predictions.is_empty()
            || sources.len() != predictions.len() + 1
        {
            return Err(invalid(
                "requires B1 and optionally B2, and explicit PDD heads",
            ));
        }
        let mut indexed = BTreeMap::new();
        for batch in batches {
            let count = batch.plan.configuration().backbone().batches() as usize;
            if !(1..=2).contains(&count) || indexed.insert(count, batch).is_some() {
                return Err(invalid("duplicate or unsupported batch cardinality"));
            }
        }
        if !indexed.contains_key(&1)
            || predictions.iter().enumerate().any(|(i, p)| {
                p.config
                    .projection(FastH3OutputModality::Video)
                    .evaluation()
                    != i
            })
        {
            return Err(invalid(
                "requires the B1 source plan and ordered PDD evaluations",
            ));
        }
        Ok(Self {
            runtime,
            host,
            batches: indexed,
            predictions,
            sources,
            sessions: BTreeMap::new(),
            weights: None,
            failed: false,
            workspace,
            active_batch: None,
            handoff_capacity: None,
        })
    }
    pub fn prepared_batch_counts(&self) -> impl Iterator<Item = usize> + '_ {
        self.batches.keys().copied()
    }
    pub fn is_resident(&self) -> bool {
        !self.failed && self.weights.is_some() && self.sessions.len() == self.batches.len()
    }
    pub fn with_device_prediction_handoff(
        mut self,
        capacity: DeviceCapacityOwner,
        device_bytes: u64,
    ) -> Self {
        self.handoff_capacity = Some((capacity, device_bytes));
        self
    }
    pub fn is_suspended(&self) -> bool {
        !self.failed && self.weights.is_none() && self.sessions.is_empty()
    }
    pub fn shared_weights_evidence(&self) -> Option<&SharedComponentWeightsEvidence> {
        self.weights.as_ref().map(SharedComponentWeights::evidence)
    }
    #[cfg(feature = "cuda-fast-h3")]
    pub(super) fn resident_weights(&self) -> Option<&SharedComponentWeights<R>> {
        self.weights.as_ref().filter(|_| self.is_resident())
    }
    /// A successful probe verifies the same transient routing/source contract
    /// that will bind the actual submitted step, without rebuilding the graph.
    pub fn compatible(&self, requests: Vec<&mut FastH3DenoisingRequest>) -> bool {
        self.batches
            .get(&requests.len())
            .is_some_and(|p| p.plan.bind(requests).is_ok())
    }
}

/// Native failures retain their original Core owners. The service must retain
/// this error together with the failed stage; it must not stringify and resume.
#[must_use]
pub enum DenoisingStageError<R: DeviceRuntime> {
    DeviceMemory(R::Error),
    Rejected(VNextError),
    Initialize(ComponentInitializationError<R>),
    Execute(TensorExecutionError<R>),
    PlanClose(Result<PlanRuntimeCloseOutcome<R>, PlanRuntimeCloseFailure<R>>),
    WeightsClose(
        Result<SharedComponentWeightsCloseOutcome<R>, SharedComponentWeightsCloseFailure<R>>,
    ),
    Prediction(PredictionFailure<R>),
    Solver(anyhow::Error),
    IdleBusy(IdleDynamicReleaseBusy),
}
impl<R: DeviceRuntime> fmt::Display for DenoisingStageError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceMemory(e) => write!(f, "device memory observation: {e}"),
            Self::Rejected(e) => write!(f, "denoising preflight: {e}"),
            Self::Initialize(e) => write!(f, "denoising initialization retains its owner: {e}"),
            Self::Execute(e) => write!(f, "denoising execution retains its owner: {e}"),
            Self::PlanClose(_) => {
                f.write_str("denoising plan close retains original retirement owner")
            }
            Self::WeightsClose(_) => {
                f.write_str("denoising installation close retains original retirement owner")
            }
            Self::Prediction(error) => write!(
                f,
                "PDD prediction retains its original stage failure: {error}"
            ),
            Self::Solver(e) => write!(f, "PDD successor preparation: {e}"),
            Self::IdleBusy(e) => write!(f, "completed phase backing is still owned: {e:?}"),
        }
    }
}
impl<R: DeviceRuntime> fmt::Debug for DenoisingStageError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for DenoisingStageError<R> {}
impl<R: DeviceRuntime> From<VNextError> for DenoisingStageError<R> {
    fn from(value: VNextError) -> Self {
        Self::Rejected(value)
    }
}
fn invalid(reason: impl fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 denoising stage: {reason}"),
    }
}
