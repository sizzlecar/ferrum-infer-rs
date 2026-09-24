//! Read-only, whole-wave cost declarations. These values grant no execution,
//! allocation, upload, or readback permission.
use crate::execution_cost::{
    ActualRowWork, ActualWaveKind, CanonicalWaveCostShape, HostCostFeaturesV1,
};
use crate::model_executor::LogitsReturnPolicy;
use crate::vnext::{ResourcePlanningState, ResourcePlanningUnknown, ResourcePlanningView};
use std::sync::Arc;

mod core;
mod masks;
mod uploads;
pub use core::{append_complete_eager_cost_route, EagerCoreWaveCostQuery};
pub use masks::{
    selection_mask_bytes_match, EagerCoreTokenMaskInput, ProductTokenMaskContent,
    ProductTokenMaskResidencyEntry, ProductTokenMaskResidencySnapshot, ProductTokenMaskSelection,
};
pub use uploads::{EagerCoreInputUpload, EagerCoreReadback};

/// A runtime declaration for its actual eager core encoder. Unknown runtimes
/// cannot borrow another backend's transfer or synchronization behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCoreCostCapabilities {
    pub upload_native_operation: &'static str,
    pub zero_native_operation: &'static str,
    /// Each upload/zero produces exactly one transfer and no compute dispatch.
    pub single_transfer_commands: bool,
    /// No program-binding merge changes their number or order.
    pub preserves_program_bindings: bool,
    /// Shared contiguous output can stage a host terminal read without a new
    /// device command. Capacity still needs an independent live read.
    pub staged_host_readback_without_commands: bool,
    /// One scalar transfer per proven contiguous physical readback. None means
    /// this backend has not declared command-based staging, not zero work.
    pub staged_host_readback_native_operation: Option<&'static str>,
    pub fallback_readback: crate::execution_cost::CoreReadbackRoute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionCostRouteUnknown {
    Unsupported,
    InvalidInput,
    Capacity,
    BudgetExhausted,
    StaleView,
    Resource(ResourcePlanningUnknown),
    ExecutionPolicy,
    ProviderRoute,
    CoreLayout,
    InitializationState,
    ReadbackState,
    OutputBranch,
}

#[derive(Debug, Clone)]
pub enum ExecutionCostRouteAvailability<T> {
    Known(T),
    Unknown(ExecutionCostRouteUnknown),
}

#[derive(Debug, Clone, Copy)]
pub enum FutureCostOutput<'a> {
    Prefill { final_logits: bool },
    Decode { policy: &'a LogitsReturnPolicy },
}

#[derive(Debug, Clone, Copy)]
pub struct FutureWaveCostRow<'a> {
    /// Index into the view's original request order, never a physical grant.
    pub participant_index: usize,
    pub work: ActualRowWork,
    pub host_policy_signature: [u8; 32],
    pub host_features: Option<HostCostFeaturesV1>,
    pub output: FutureCostOutput<'a>,
}

#[derive(Debug)]
pub struct FutureWaveCostQuery<'a> {
    pub kind: ActualWaveKind,
    pub rows: &'a [FutureWaveCostRow<'a>],
}

/// A numeric snapshot. The private fence owns no session, buffer, lease, or
/// executable plan. A new candidate replay starts from `initial_state()`.
#[derive(Debug, Clone)]
pub struct ExecutionCostRouteView {
    pub(crate) fence: Arc<()>,
    pub(crate) resources: ResourcePlanningView,
    pub(crate) initial_frontiers: Vec<u64>,
    pub(crate) readback_available_bytes: u64,
    pub(crate) lane_id: crate::vnext::ExecutionLaneId,
    pub(crate) token_masks: Option<ProductTokenMaskResidencySnapshot>,
    pub(crate) graph_stream_state: Option<crate::vnext::DeviceCostGraphStreamState>,
}

impl ExecutionCostRouteView {
    pub const fn graph_stream_state(&self) -> Option<crate::vnext::DeviceCostGraphStreamState> {
        self.graph_stream_state
    }

    pub fn initial_state(&self) -> ExecutionCostRouteState {
        ExecutionCostRouteState {
            fence: Arc::clone(&self.fence),
            resources: self.resources.initial_state(),
            frontiers: self.initial_frontiers.clone(),
            initialized: vec![false; self.initial_frontiers.len()],
            token_masks: self.token_masks.clone(),
            last_token_mask_uploads: None,
        }
    }

    pub fn resource_view(&self) -> &ResourcePlanningView {
        &self.resources
    }

    pub fn with_token_mask_residency(
        mut self,
        snapshot: ProductTokenMaskResidencySnapshot,
    ) -> Self {
        // States from a previous version of this view cannot borrow new product
        // evidence, including when a caller retained a clone of the old view.
        self.fence = Arc::new(());
        self.token_masks = Some(snapshot);
        self
    }

    pub fn participant_count(&self) -> usize {
        self.initial_frontiers.len()
    }

    pub fn participant_authority(&self, index: usize) -> Option<crate::vnext::SequenceAuthorityId> {
        self.resources
            .participants()
            .get(index)
            .map(|participant| participant.authority())
    }

    pub fn lane_id(&self) -> crate::vnext::ExecutionLaneId {
        self.lane_id
    }

    pub fn same_live_evidence(&self, other: &Self) -> bool {
        self.resources.same_live_evidence(&other.resources)
            && self.initial_frontiers == other.initial_frontiers
            && self.readback_available_bytes == other.readback_available_bytes
            && self.lane_id == other.lane_id
            && self.token_masks == other.token_masks
            && self.graph_stream_state == other.graph_stream_state
    }
}

/// Private per-witness state. Persistent initialization/frontiers advance only
/// after a modeled successful whole-wave receipt. Never reuse a beam sibling's
/// state, or infer its history from its final token count.
#[derive(Debug, Clone)]
pub struct ExecutionCostRouteState {
    pub(crate) fence: Arc<()>,
    pub(crate) resources: ResourcePlanningState,
    pub(crate) frontiers: Vec<u64>,
    pub(crate) initialized: Vec<bool>,
    pub(crate) token_masks: Option<ProductTokenMaskResidencySnapshot>,
    pub(crate) last_token_mask_uploads: Option<Vec<bool>>,
}

impl ExecutionCostRouteState {
    pub fn projected_waves(&self) -> usize {
        self.resources.projected_waves()
    }
    pub fn last_token_mask_uploads(&self) -> Option<&[bool]> {
        self.last_token_mask_uploads.as_deref()
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionCostRouteProjection {
    pub statistical_evidence: Option<crate::execution_cost::StatisticalWaveEvidenceV1>,
    pub shape: CanonicalWaveCostShape,
    pub state: ExecutionCostRouteState,
}
