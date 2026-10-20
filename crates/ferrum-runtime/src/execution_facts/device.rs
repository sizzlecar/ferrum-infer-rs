//! Pure descriptions supplied by the backend that submitted the original fence.
//! These DTOs grant no execution authority. Core validates and binds them only
//! after arming that fence. The coverage domain is provider compute commands;
//! initialization, bindings, transfers, status copies and terminal tails are out
//! of scope. A projection may contain several kernels.
use super::{DeviceExecutionFactsCoverage, DeviceExecutionFactsUnavailableReason};
use crate::SubmittedOperationReceipt;
use crate::{
    DeviceExecutionPath, DeviceNativeOperationId, ElementType, HadamardTransformSpec, WeightId,
};
use serde::Serialize;
use std::{ops::Range, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceProjectionRole {
    OutputHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProjectionDimensions {
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProjectionLayout {
    pub input_stride: u32,
    pub output_stride: u32,
    pub output_offset: u32,
    pub input_type: ElementType,
    pub output_type: ElementType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProjectionLaunchGeometry {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PackedProjectionGeometry {
    pub padded_k: u32,
    pub padded_n: u32,
    pub row_tile: u32,
    pub column_tile: u32,
    pub blocks: u32,
    pub threads: u32,
    pub shared_bytes: u32,
    pub fixup: bool,
    pub guard_blocks: u32,
    pub pack_abi: u32,
    pub packed_bytes: u64,
    pub fixup_bytes: u64,
    /// Total workspace extent; packed and optional fixup regions are disjoint.
    /// Other staging buffers and alignment can make this larger than their sum.
    pub workspace_bytes: u64,
}

/// Describes the actual chosen implementation, never a routing request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProjectionArithmetic {
    Direct {
        implementation: DeviceNativeOperationId,
        row_tile: u32,
        launch: ProjectionLaunchGeometry,
    },
    PackedQuantized {
        implementation: DeviceNativeOperationId,
        geometry: PackedProjectionGeometry,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProjectionInputTransform {
    pub spec: HadamardTransformSpec,
    pub implementation: DeviceNativeOperationId,
    pub launch: ProjectionLaunchGeometry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParticipantTokenSelector {
    LastSourceToken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParticipantRowSelector {
    pub participant_index: u32,
    pub source: ParticipantTokenSelector,
    pub physical_rows: Range<u32>,
}

/// Immutable execution selection. Cached templates contain participant indices,
/// never a prior request's identity, source offset or resource owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProjectionInvocationTemplate {
    pub ordinal: u32,
    pub role: DeviceProjectionRole,
    pub component: WeightId,
    pub dimensions: ProjectionDimensions,
    pub layout: ProjectionLayout,
    pub arithmetic: ProjectionArithmetic,
    pub input_transforms: Box<[ProjectionInputTransform]>,
    pub rows: Box<[ParticipantRowSelector]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum ProjectionCommandCoverage {
    Covered(#[serde(serialize_with = "serialize_arc_slice")] Arc<[ProjectionInvocationTemplate]>),
    Unavailable(DeviceExecutionFactsUnavailableReason),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceCommandExecutionFacts {
    /// Index in the original core DeviceCommandBatch, not a kernel/launch count.
    pub submission_command_index: u32,
    /// Ordinal in the actual reused compute executable, when one was launched.
    /// Adaptive candidates start at zero and continue across contiguous original
    /// batch entries. Zero may start a distinct adjacent candidate. Direct
    /// segments are checked against their original declared node range.
    pub replay_command_ordinal: Option<u32>,
    pub node_index: Option<u32>,
    pub path: DeviceExecutionPath,
    pub coverage: ProjectionCommandCoverage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceSubmissionExecutionFacts {
    batch_command_count: u32,
    #[serde(serialize_with = "serialize_arc_slice")]
    compute_commands: Arc<[DeviceCommandExecutionFacts]>,
}

fn serialize_arc_slice<T: Serialize, S: serde::Serializer>(
    values: &Arc<[T]>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    values.as_ref().serialize(serializer)
}

impl DeviceSubmissionExecutionFacts {
    /// Store pure observations without introducing a failure after enqueue.
    /// Structural validation belongs to Core after the original fence is armed.
    pub fn new(
        batch_command_count: u32,
        compute_commands: Vec<DeviceCommandExecutionFacts>,
    ) -> Self {
        Self {
            batch_command_count,
            compute_commands: compute_commands.into(),
        }
    }
    pub const fn batch_command_count(&self) -> u32 {
        self.batch_command_count
    }
    pub fn compute_commands(&self) -> &[DeviceCommandExecutionFacts] {
        &self.compute_commands
    }
}

/// Current-work resolution of one cached selector. These ranges describe the
/// original submitted node, not a cached request or a newly issued authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedProjectionRow {
    pub participant_index: u32,
    pub source_token_range: Range<u64>,
    pub packed_token_range: Range<u64>,
    pub physical_rows: Range<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundProjectionRows {
    pub submission_command_index: u32,
    pub replay_command_ordinal: Option<u32>,
    pub node_index: u32,
    pub projection_ordinal: u32,
    pub rows: Box<[ResolvedProjectionRow]>,
}

/// Validated observation sharing the original receipt and immutable backend
/// templates. It retains no device resources or completion authority.
#[derive(Debug, PartialEq, Eq)]
pub struct BoundSubmissionDeviceFacts {
    submission: SubmittedOperationReceipt,
    device: DeviceSubmissionExecutionFacts,
    projection_rows: Box<[BoundProjectionRows]>,
    coverage: DeviceExecutionFactsCoverage,
}

impl BoundSubmissionDeviceFacts {
    pub(crate) fn new(
        submission: SubmittedOperationReceipt,
        device: DeviceSubmissionExecutionFacts,
        projection_rows: Box<[BoundProjectionRows]>,
        coverage: DeviceExecutionFactsCoverage,
    ) -> Self {
        Self {
            submission,
            device,
            projection_rows,
            coverage,
        }
    }
    pub fn submission(&self) -> &SubmittedOperationReceipt {
        &self.submission
    }
    pub fn device(&self) -> &DeviceSubmissionExecutionFacts {
        &self.device
    }
    pub fn projection_rows(&self) -> &[BoundProjectionRows] {
        &self.projection_rows
    }
    pub const fn coverage(&self) -> DeviceExecutionFactsCoverage {
        self.coverage
    }
}
