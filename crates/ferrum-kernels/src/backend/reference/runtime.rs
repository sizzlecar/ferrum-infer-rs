use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferrum_runtime::{
    BufferDescriptor, CopyRegion, DefinitelyNotSubmitted, DeviceAllocationPermit,
    DeviceBatchingForm, DeviceBufferRetention, DeviceClass, DeviceCommandBatch,
    DeviceCommandLogicalWork, DeviceCommandPhase, DeviceComputePathRequirement, DeviceDescriptor,
    DeviceErrorReport, DeviceExecutionPath, DeviceNativeOperationId, DeviceNativeWorkAttribution,
    DeviceRuntime, DeviceSubmissionAttribution, DeviceTerminal, DeviceTerminalReceipt,
    DeviceTimingMode, ElementType, FenceIndeterminate, FenceQuery, HostTransferLayout,
    ManagedHostBuffer, ManagedHostBytes, StreamState, VNextError, DEVICE_COPY_NATIVE_OPERATION_ID,
    DEVICE_ZERO_NATIVE_OPERATION_ID, HOST_UPLOAD_NATIVE_OPERATION_ID,
};
use half::{bf16, f16};

mod row_transport;
pub(crate) use row_transport::{
    checked_concat_bytes, checked_gather_bytes, ReferenceBatchedRowConcatLaunch,
    ReferenceRowGatherLaunch,
};
mod layer_norm;
pub(crate) use layer_norm::ReferenceLayerNormLaunch;
mod zero_centered_rms_norm;
pub(crate) use zero_centered_rms_norm::{
    ReferenceZeroCenteredRmsNormLaunch, ReferenceZeroCenteredRmsNormShape,
};
mod gelu_tanh;
mod head_rms_norm;
pub(crate) use head_rms_norm::{ReferenceHeadRmsNormLaunch, ReferenceHeadRmsNormShape};
mod structured_attention;
pub(crate) use structured_attention::ReferenceStructuredAttentionLaunch;
mod adjacent_rope;
pub(crate) use adjacent_rope::{ReferenceAdjacentRopeLaunch, ReferenceAdjacentRopeShape};
mod pointwise;
pub(crate) use pointwise::{
    ReferenceGatedResidualLaunch, ReferenceModulationScaleLaunch, ReferencePointwiseShape,
};
mod transcendental;
#[cfg(test)]
mod weightless_initialization;
#[cfg(test)]
pub(super) use transcendental::{silu_bf16_bits, silu_mul_bf16_bits, tanh_bf16_bits};
pub(crate) use transcendental::{
    ReferenceActivationShape, ReferenceSiluMulLaunch, ReferenceUnaryKind, ReferenceUnaryLaunch,
};

static NEXT_RUNTIME_INSTANCE: AtomicU64 = AtomicU64::new(1);
static NEXT_STREAM_INSTANCE: AtomicU64 = AtomicU64::new(1);
static NEXT_FENCE_INSTANCE: AtomicU64 = AtomicU64::new(1);

pub(crate) struct ReferenceDeviceRuntimeConfig {
    pub(crate) descriptor: DeviceDescriptor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceDeviceRuntimeError {
    message: String,
}

impl ReferenceDeviceRuntimeError {
    pub(crate) fn contract(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ReferenceDeviceRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ReferenceDeviceRuntimeError {}

struct ReferenceAllocation {
    bytes: Mutex<Box<[u8]>>,
    logical_offset_bytes: usize,
    logical_size_bytes: usize,
    live_allocations: Arc<AtomicU64>,
}

impl ReferenceAllocation {
    fn lock(&self) -> MutexGuard<'_, Box<[u8]>> {
        self.bytes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for ReferenceAllocation {
    fn drop(&mut self) {
        let prior = self.live_allocations.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(prior > 0, "reference allocation accounting underflow");
    }
}

/// One core-authorized reference allocation.
pub struct ReferenceDeviceBuffer {
    descriptor: BufferDescriptor,
    runtime_instance: u64,
    allocation: Arc<ReferenceAllocation>,
}

impl fmt::Debug for ReferenceDeviceBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReferenceDeviceBuffer")
            .field("descriptor", &self.descriptor)
            .field("runtime_instance", &self.runtime_instance)
            .finish_non_exhaustive()
    }
}

impl ReferenceDeviceBuffer {
    fn region(
        &self,
        range: Range<u64>,
    ) -> Result<ReferenceBufferRegion, ReferenceDeviceRuntimeError> {
        self.region_with_retention(range, None)
    }

    pub(crate) fn retained_region(
        &self,
        range: Range<u64>,
        retention: DeviceBufferRetention,
    ) -> Result<ReferenceBufferRegion, ReferenceDeviceRuntimeError> {
        self.region_with_retention(range, Some(retention))
    }

    fn region_with_retention(
        &self,
        range: Range<u64>,
        retention: Option<DeviceBufferRetention>,
    ) -> Result<ReferenceBufferRegion, ReferenceDeviceRuntimeError> {
        if range.start >= range.end || range.end > self.descriptor.size_bytes {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference buffer region is empty or exceeds its admitted allocation",
            ));
        }
        let offset_bytes = usize::try_from(range.start).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference buffer offset exceeds usize")
        })?;
        let offset_bytes = self
            .allocation
            .logical_offset_bytes
            .checked_add(offset_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference buffer physical offset overflows usize",
                )
            })?;
        let length_bytes = usize::try_from(range.end - range.start).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference buffer length exceeds usize")
        })?;
        Ok(ReferenceBufferRegion {
            allocation: Arc::clone(&self.allocation),
            runtime_instance: self.runtime_instance,
            offset_bytes,
            length_bytes,
            element_type: self.descriptor.element_type,
            _retention: retention,
        })
    }
}

#[derive(Clone)]
pub(crate) struct ReferenceBufferRegion {
    allocation: Arc<ReferenceAllocation>,
    runtime_instance: u64,
    offset_bytes: usize,
    length_bytes: usize,
    element_type: ElementType,
    _retention: Option<DeviceBufferRetention>,
}

impl ReferenceBufferRegion {
    pub(crate) const fn length_bytes(&self) -> usize {
        self.length_bytes
    }

    pub(crate) const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub(crate) fn same_physical_region(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
            && self.offset_bytes == other.offset_bytes
            && self.length_bytes == other.length_bytes
            && self.element_type == other.element_type
    }

    fn validate_runtime(&self, runtime_instance: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        if self.runtime_instance != runtime_instance {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference command contains a buffer from another runtime",
            ));
        }
        self.validate_range()
    }

    fn validate_range(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let end = self.offset_bytes.checked_add(self.length_bytes);
        let allocation_end = self
            .allocation
            .logical_offset_bytes
            .checked_add(self.allocation.logical_size_bytes);
        if self.length_bytes == 0
            || self.offset_bytes < self.allocation.logical_offset_bytes
            || end
                .zip(allocation_end)
                .is_none_or(|(end, limit)| end > limit)
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference command region exceeds its allocation",
            ));
        }
        Ok(())
    }

    fn overlaps(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.allocation, &other.allocation)
            && self.offset_bytes < other.offset_bytes + other.length_bytes
            && other.offset_bytes < self.offset_bytes + self.length_bytes
    }

    fn read(&self) -> Vec<u8> {
        let bytes = self.allocation.lock();
        bytes[self.offset_bytes..self.offset_bytes + self.length_bytes].to_vec()
    }

    fn write(&self, source: &[u8]) {
        assert_eq!(source.len(), self.length_bytes);
        let mut bytes = self.allocation.lock();
        bytes[self.offset_bytes..self.offset_bytes + self.length_bytes].copy_from_slice(source);
    }

    fn zero(&self) {
        let mut bytes = self.allocation.lock();
        bytes[self.offset_bytes..self.offset_bytes + self.length_bytes].fill(0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferenceDenseScalar {
    F16,
    Bf16,
}

impl ReferenceDenseScalar {
    pub(crate) const fn element_type(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::Bf16 => ElementType::Bf16,
        }
    }

    fn decode(self, bits: u16) -> f32 {
        match self {
            Self::F16 => f16::from_bits(bits).to_f32(),
            Self::Bf16 => bf16::from_bits(bits).to_f32(),
        }
    }

    fn encode(self, value: f32) -> u16 {
        match self {
            Self::F16 => f16::from_f32(value).to_bits(),
            Self::Bf16 => bf16::from_f32(value).to_bits(),
        }
    }
}

pub(crate) struct ReferenceDenseLinearLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) weight: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) rows: usize,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) scalar: ReferenceDenseScalar,
}

impl ReferenceDenseLinearLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        if self.rows == 0 || self.in_features == 0 || self.out_features == 0 {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference dense-linear dimensions must be positive",
            ));
        }
        for (region, left, right) in [
            (&self.input, self.rows, self.in_features),
            (&self.weight, self.out_features, self.in_features),
            (&self.output, self.rows, self.out_features),
        ] {
            region.validate_range()?;
            let expected = left.checked_mul(right).and_then(|n| n.checked_mul(2));
            if expected != Some(region.length_bytes)
                || region.element_type != self.scalar.element_type()
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "reference dense-linear shape, range, alignment or scalar type differs",
                ));
            }
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.weight) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference dense-linear output overlaps a read operand",
            ));
        }
        Ok(())
    }
}

enum ReferenceCommandKind {
    Copy {
        source: ReferenceBufferRegion,
        destination: ReferenceBufferRegion,
    },
    Upload {
        source: Box<[u8]>,
        destination: ReferenceBufferRegion,
    },
    ManagedUpload {
        source: ManagedHostBytes,
        destination: ReferenceBufferRegion,
    },
    Zero {
        destination: ReferenceBufferRegion,
    },
    DenseLinear {
        launches: Box<[ReferenceDenseLinearLaunch]>,
    },
    LayerNorm {
        launch: ReferenceLayerNormLaunch,
    },
    ZeroCenteredRmsNorm {
        launch: ReferenceZeroCenteredRmsNormLaunch,
    },
    RowGather {
        launch: ReferenceRowGatherLaunch,
    },
    BatchedRowConcat {
        launch: ReferenceBatchedRowConcatLaunch,
    },
    HeadRmsNorm {
        launch: ReferenceHeadRmsNormLaunch,
    },
    StructuredAttention {
        launch: ReferenceStructuredAttentionLaunch,
    },
    AdjacentRope {
        launch: ReferenceAdjacentRopeLaunch,
    },
    ModulationScale {
        launch: ReferenceModulationScaleLaunch,
    },
    GatedResidual {
        launch: ReferenceGatedResidualLaunch,
    },
    UnaryActivation {
        launch: ReferenceUnaryLaunch,
    },
    SiluMul {
        launch: ReferenceSiluMulLaunch,
    },
}

/// An owned, fully validated command for the synchronous reference runtime.
pub struct ReferenceDeviceCommand {
    operation: &'static str,
    logical_work: Option<DeviceCommandLogicalWork>,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
    kind: ReferenceCommandKind,
}

impl fmt::Debug for ReferenceDeviceCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            ReferenceCommandKind::Copy { .. } => "copy",
            ReferenceCommandKind::Upload { .. } => "upload",
            ReferenceCommandKind::ManagedUpload { .. } => "managed_upload",
            ReferenceCommandKind::Zero { .. } => "zero",
            ReferenceCommandKind::DenseLinear { .. } => "dense_linear",
            ReferenceCommandKind::LayerNorm { .. } => "layer_norm",
            ReferenceCommandKind::ZeroCenteredRmsNorm { .. } => "zero_centered_rms_norm",
            ReferenceCommandKind::RowGather { .. } => "row_gather",
            ReferenceCommandKind::BatchedRowConcat { .. } => "batched_row_concat",
            ReferenceCommandKind::HeadRmsNorm { .. } => "head_rms_norm",
            ReferenceCommandKind::StructuredAttention { .. } => "structured_attention",
            ReferenceCommandKind::AdjacentRope { .. } => "adjacent_rope",
            ReferenceCommandKind::ModulationScale { .. } => "modulation_scale",
            ReferenceCommandKind::GatedResidual { .. } => "gated_residual",
            ReferenceCommandKind::UnaryActivation { launch: ref unary } => match unary.kind {
                ReferenceUnaryKind::Silu => "silu",
                ReferenceUnaryKind::Tanh => "tanh",
                ReferenceUnaryKind::GeluTanh => "gelu_tanh",
            },
            ReferenceCommandKind::SiluMul { .. } => "silu_mul",
        };
        formatter
            .debug_struct("ReferenceDeviceCommand")
            .field("kind", &kind)
            .field("operation", &self.operation)
            .field("logical_work", &self.logical_work)
            .finish()
    }
}

impl ReferenceDeviceCommand {
    pub(crate) fn dense_linear(
        launches: Vec<ReferenceDenseLinearLaunch>,
        batching_form: DeviceBatchingForm,
        participant_count: u32,
        token_count: u64,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        if launches.is_empty() || participant_count == 0 || token_count == 0 {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference dense-linear attribution has no participants or work",
            ));
        }
        for launch in &launches {
            launch.validate()?;
        }
        let compute_dispatch_count = u64::try_from(launches.len()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract(
                "reference dense-linear dispatch count exceeds u64",
            )
        })?;
        Ok(Self {
            operation: "vnext_dense_linear",
            logical_work: Some(
                DeviceCommandLogicalWork::new(batching_form, participant_count, token_count)
                    .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?,
            ),
            compute_dispatch_count,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::DenseLinear {
                launches: launches.into_boxed_slice(),
            },
        })
    }

    pub(crate) fn dense_tensor(
        launch: ReferenceDenseLinearLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        let rows = u64::try_from(launch.rows).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference tensor rows exceed u64")
        })?;
        let in_features = u64::try_from(launch.in_features).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference tensor input width exceeds u64")
        })?;
        let out_features = u64::try_from(launch.out_features).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference tensor output width exceeds u64")
        })?;
        let element_count = rows.checked_mul(out_features).ok_or_else(|| {
            ReferenceDeviceRuntimeError::contract("reference tensor element count overflows")
        })?;
        let arithmetic_operations = element_count
            .checked_mul(in_features)
            .and_then(|operations| operations.checked_mul(2))
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference tensor arithmetic operation count overflows",
                )
            })?;
        let logical_work =
            DeviceCommandLogicalWork::tensor(element_count, arithmetic_operations)
                .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        Ok(Self {
            operation: "vnext_dense_linear",
            logical_work: Some(logical_work),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::DenseLinear {
                launches: vec![launch].into_boxed_slice(),
            },
        })
    }

    pub(crate) fn structured_attention_tensor(
        launch: ReferenceStructuredAttentionLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_structured_attention_bf16",
            logical_work: Some(launch.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::StructuredAttention { launch },
        })
    }

    pub(crate) fn adjacent_rope_tensor(
        launch: ReferenceAdjacentRopeLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_adjacent_rope_bf16",
            logical_work: Some(launch.shape.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::AdjacentRope { launch },
        })
    }

    pub(crate) fn zero_centered_rms_norm_tensor(
        launch: ReferenceZeroCenteredRmsNormLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_zero_centered_rms_norm_bf16",
            logical_work: Some(launch.shape.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::ZeroCenteredRmsNorm { launch },
        })
    }

    pub(crate) fn row_gather_tensor(
        launch: ReferenceRowGatherLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_row_gather_bf16",
            logical_work: Some(launch.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::RowGather { launch },
        })
    }
    pub(crate) fn batched_row_concat_tensor(
        launch: ReferenceBatchedRowConcatLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_batched_row_concat_bf16",
            logical_work: Some(launch.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::BatchedRowConcat { launch },
        })
    }

    pub(crate) fn head_rms_norm_tensor(
        launch: ReferenceHeadRmsNormLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_head_rms_norm_bf16",
            logical_work: Some(launch.shape.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::HeadRmsNorm { launch },
        })
    }

    pub(crate) fn layer_norm_tensor(
        launch: ReferenceLayerNormLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        let logical_work = launch.logical_work()?;
        Ok(Self {
            operation: "vnext_layer_norm_bf16",
            logical_work: Some(logical_work),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::LayerNorm { launch },
        })
    }

    pub(crate) fn modulation_scale_tensor(
        launch: ReferenceModulationScaleLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_modulation_scale_bf16",
            logical_work: Some(launch.shape.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::ModulationScale { launch },
        })
    }

    pub(crate) fn gated_residual_tensor(
        launch: ReferenceGatedResidualLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_gated_residual_bf16",
            logical_work: Some(launch.shape.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::GatedResidual { launch },
        })
    }

    pub(crate) fn unary_activation_tensor(
        launch: ReferenceUnaryLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: match launch.kind {
                ReferenceUnaryKind::Silu => "vnext_silu_bf16",
                ReferenceUnaryKind::Tanh => "vnext_tanh_bf16",
                ReferenceUnaryKind::GeluTanh => "vnext_gelu_tanh_bf16",
            },
            logical_work: Some(launch.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::UnaryActivation { launch },
        })
    }

    pub(crate) fn silu_mul_tensor(
        launch: ReferenceSiluMulLaunch,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        launch.validate()?;
        Ok(Self {
            operation: "vnext_silu_mul_bf16",
            logical_work: Some(launch.logical_work()?),
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            kind: ReferenceCommandKind::SiluMul { launch },
        })
    }

    fn transfer(operation: &'static str, kind: ReferenceCommandKind) -> Self {
        Self {
            operation,
            logical_work: None,
            compute_dispatch_count: 0,
            transfer_command_count: 1,
            kind,
        }
    }

    fn bind_core_logical_work(
        mut self,
        logical_work: DeviceCommandLogicalWork,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        if self.logical_work.is_some() {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference core logical work cannot replace provider command attribution",
            ));
        }
        self.logical_work = Some(logical_work);
        Ok(self)
    }

    fn attribution(
        &self,
        command_index: u32,
        node_index: Option<u32>,
        phase: DeviceCommandPhase,
    ) -> Result<DeviceNativeWorkAttribution, ReferenceDeviceRuntimeError> {
        let native_op_id = DeviceNativeOperationId::new(self.operation).ok_or_else(|| {
            ReferenceDeviceRuntimeError::contract(
                "reference command attribution has a non-portable native operation identity",
            )
        })?;
        let attribution = if let Some(logical_work) = self.logical_work {
            DeviceNativeWorkAttribution::with_logical_work(
                command_index,
                node_index,
                phase,
                native_op_id,
                DeviceExecutionPath::Eager,
                logical_work,
                self.compute_dispatch_count,
                self.transfer_command_count,
            )
        } else {
            DeviceNativeWorkAttribution::with_participant_range(
                command_index,
                node_index,
                phase,
                native_op_id,
                DeviceExecutionPath::Eager,
                DeviceBatchingForm::Scalar,
                0,
                0,
                0,
                self.compute_dispatch_count,
                self.transfer_command_count,
                None,
            )
        };
        attribution.ok_or_else(|| {
            ReferenceDeviceRuntimeError::contract(
                "reference command attribution has invalid native work metadata",
            )
        })
    }

    fn validate_runtime(&self, runtime_instance: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        match &self.kind {
            ReferenceCommandKind::Copy {
                source,
                destination,
            } => {
                source.validate_runtime(runtime_instance)?;
                destination.validate_runtime(runtime_instance)?;
                if source.length_bytes != destination.length_bytes
                    || source.element_type != destination.element_type
                {
                    return Err(ReferenceDeviceRuntimeError::contract(
                        "reference copy source and destination disagree",
                    ));
                }
                Ok(())
            }
            ReferenceCommandKind::Upload {
                source,
                destination,
            } => {
                destination.validate_runtime(runtime_instance)?;
                validate_upload_length(source.len(), destination)
            }
            ReferenceCommandKind::ManagedUpload {
                source,
                destination,
            } => {
                destination.validate_runtime(runtime_instance)?;
                validate_upload_length(source.len(), destination)
            }
            ReferenceCommandKind::Zero { destination } => {
                destination.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::DenseLinear { launches } => {
                if launches.is_empty() {
                    return Err(ReferenceDeviceRuntimeError::contract(
                        "reference dense-linear command has no launches",
                    ));
                }
                for launch in launches {
                    launch.input.validate_runtime(runtime_instance)?;
                    launch.weight.validate_runtime(runtime_instance)?;
                    launch.output.validate_runtime(runtime_instance)?;
                    launch.validate()?;
                }
                Ok(())
            }
            ReferenceCommandKind::ZeroCenteredRmsNorm { launch } => {
                launch.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::RowGather { launch } => launch.validate_runtime(runtime_instance),
            ReferenceCommandKind::BatchedRowConcat { launch } => {
                launch.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::HeadRmsNorm { launch } => {
                launch.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::StructuredAttention { launch } => {
                launch.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::AdjacentRope { launch } => {
                launch.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::LayerNorm { launch } => {
                launch.input.validate_runtime(runtime_instance)?;
                launch.output.validate_runtime(runtime_instance)?;
                launch.validate()?;
                launch.logical_work()?;
                Ok(())
            }
            ReferenceCommandKind::ModulationScale { launch } => {
                launch.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::GatedResidual { launch } => {
                launch.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::UnaryActivation { launch } => {
                launch.validate_runtime(runtime_instance)
            }
            ReferenceCommandKind::SiluMul { launch } => launch.validate_runtime(runtime_instance),
        }
    }

    fn execute(&self, counters: &ReferenceRuntimeCounters) {
        match &self.kind {
            ReferenceCommandKind::Copy {
                source,
                destination,
            } => destination.write(&source.read()),
            ReferenceCommandKind::Upload {
                source,
                destination,
            } => {
                destination.write(source);
                counters.uploaded_bytes.fetch_add(
                    u64::try_from(source.len()).expect("reference upload length fits u64"),
                    Ordering::Relaxed,
                );
            }
            ReferenceCommandKind::ManagedUpload {
                source,
                destination,
            } => {
                destination.write(source.bytes());
                counters.uploaded_bytes.fetch_add(
                    u64::try_from(source.len()).expect("reference managed upload length fits u64"),
                    Ordering::Relaxed,
                );
            }
            ReferenceCommandKind::Zero { destination } => destination.zero(),
            ReferenceCommandKind::DenseLinear { launches } => {
                for launch in launches {
                    execute_dense_linear(launch);
                    counters
                        .dense_linear_launches
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            ReferenceCommandKind::LayerNorm { launch } => {
                layer_norm::execute(launch);
                counters.layer_norm_launches.fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::ZeroCenteredRmsNorm { launch } => {
                zero_centered_rms_norm::execute(launch);
                counters
                    .zero_centered_rms_norm_launches
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::RowGather { launch } => {
                row_transport::execute_gather(launch);
                counters.row_gather_launches.fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::BatchedRowConcat { launch } => {
                row_transport::execute_concat(launch);
                counters
                    .batched_row_concat_launches
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::HeadRmsNorm { launch } => {
                head_rms_norm::execute(launch);
                counters
                    .head_rms_norm_launches
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::StructuredAttention { launch } => {
                structured_attention::execute(launch);
                counters
                    .structured_attention_launches
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::AdjacentRope { launch } => {
                adjacent_rope::execute(launch);
                counters
                    .adjacent_rope_launches
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::ModulationScale { launch } => {
                pointwise::execute_modulation(launch);
                counters
                    .modulation_scale_launches
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::GatedResidual { launch } => {
                pointwise::execute_gated_residual(launch);
                counters
                    .gated_residual_launches
                    .fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::UnaryActivation { launch } => {
                transcendental::execute_unary(launch);
                let count = match launch.kind {
                    ReferenceUnaryKind::Silu => &counters.silu_launches,
                    ReferenceUnaryKind::Tanh => &counters.tanh_launches,
                    ReferenceUnaryKind::GeluTanh => &counters.gelu_tanh_launches,
                };
                count.fetch_add(1, Ordering::Relaxed);
            }
            ReferenceCommandKind::SiluMul { launch } => {
                transcendental::execute_silu_mul(launch);
                counters.silu_mul_launches.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn execute_dense_linear(launch: &ReferenceDenseLinearLaunch) {
    // Fixed-size metadata only. Every distinct allocation is locked once, in
    // a global address order, including when two or all three roles share it.
    // Preflight proved that output does not overlap either read operand.
    let mut storage = AllocationLocks::new([&launch.input, &launch.weight, &launch.output]);
    for row in 0..launch.rows {
        for out_feature in 0..launch.out_features {
            let mut sum = 0.0_f32;
            for in_feature in 0..launch.in_features {
                let input_index = row * launch.in_features + in_feature;
                let weight_index = out_feature * launch.in_features + in_feature;
                let input = launch.scalar.decode(storage.read_bits(0, input_index));
                let weight = launch.scalar.decode(storage.read_bits(1, weight_index));
                // Preserve the existing F16 left-to-right FP32 arithmetic.
                // BF16 follows the same order and rounds only at the output.
                sum += input * weight;
            }
            let output_index = row * launch.out_features + out_feature;
            storage.write_bits(2, output_index, launch.scalar.encode(sum));
        }
    }
}

struct AllocationLocks<'a, const N: usize> {
    guards: [Option<MutexGuard<'a, Box<[u8]>>>; N],
    slots: [usize; N],
    offsets: [usize; N],
}

impl<'a, const N: usize> AllocationLocks<'a, N> {
    fn new(regions: [&'a ReferenceBufferRegion; N]) -> Self {
        let mut allocations = regions.map(|region| region.allocation.as_ref());
        allocations.sort_unstable_by_key(|allocation| *allocation as *const _ as usize);
        let mut unique: [Option<&'a ReferenceAllocation>; N] = [None; N];
        let mut count = 0;
        for allocation in allocations {
            if count == 0 || !std::ptr::eq(unique[count - 1].unwrap(), allocation) {
                unique[count] = Some(allocation);
                count += 1;
            }
        }
        let mut guards = std::array::from_fn(|_| None);
        for index in 0..count {
            guards[index] = Some(unique[index].unwrap().lock());
        }
        let slots = regions.map(|region| {
            unique[..count]
                .iter()
                .position(|allocation| {
                    std::ptr::eq(allocation.unwrap(), region.allocation.as_ref())
                })
                .expect("each operand role is present in its allocation table")
        });
        Self {
            guards,
            slots,
            offsets: regions.map(|region| region.offset_bytes),
        }
    }

    fn read_bits(&self, role: usize, index: usize) -> u16 {
        let bytes = self.guards[self.slots[role]].as_ref().unwrap();
        let offset = self.offsets[role] + index * 2;
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    }

    fn read_f32(&self, role: usize, index: usize) -> f32 {
        let bytes = self.guards[self.slots[role]].as_ref().unwrap();
        let offset = self.offsets[role] + index * 4;
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    fn write_bits(&mut self, role: usize, index: usize, bits: u16) {
        let bytes = self.guards[self.slots[role]].as_mut().unwrap();
        let offset = self.offsets[role] + index * 2;
        bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
    }
}

fn validate_upload_length(
    length: usize,
    destination: &ReferenceBufferRegion,
) -> Result<(), ReferenceDeviceRuntimeError> {
    if length != destination.length_bytes {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference upload source and destination lengths disagree",
        ));
    }
    Ok(())
}

fn aligned_storage(
    size_bytes: usize,
    alignment_bytes: usize,
) -> Result<(Box<[u8]>, usize), ReferenceDeviceRuntimeError> {
    if size_bytes == 0 || alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference allocation size or alignment is invalid",
        ));
    }
    let storage_bytes = size_bytes.checked_add(alignment_bytes - 1).ok_or_else(|| {
        ReferenceDeviceRuntimeError::contract("reference aligned allocation size overflows")
    })?;
    let bytes = vec![0_u8; storage_bytes].into_boxed_slice();
    let base = bytes.as_ptr() as usize;
    let aligned = base
        .checked_add(alignment_bytes - 1)
        .map(|address| address & !(alignment_bytes - 1))
        .ok_or_else(|| {
            ReferenceDeviceRuntimeError::contract("reference aligned address overflows")
        })?;
    let logical_offset_bytes = aligned.checked_sub(base).ok_or_else(|| {
        ReferenceDeviceRuntimeError::contract("reference aligned address precedes its allocation")
    })?;
    if logical_offset_bytes
        .checked_add(size_bytes)
        .is_none_or(|end| end > bytes.len())
    {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference aligned logical range exceeds its allocation",
        ));
    }
    Ok((bytes, logical_offset_bytes))
}

pub struct ReferenceDeviceStream {
    runtime_instance: u64,
    stream_instance: u64,
    state: StreamState,
}

impl fmt::Debug for ReferenceDeviceStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReferenceDeviceStream")
            .field("runtime_instance", &self.runtime_instance)
            .field("stream_instance", &self.stream_instance)
            .field("state", &self.state)
            .finish()
    }
}

pub struct ReferenceDeviceFence {
    runtime_instance: u64,
    fence_instance: u64,
    attribution: Option<DeviceSubmissionAttribution>,
}

impl fmt::Debug for ReferenceDeviceFence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReferenceDeviceFence")
            .field("runtime_instance", &self.runtime_instance)
            .field("fence_instance", &self.fence_instance)
            .finish()
    }
}

#[derive(Default)]
struct ReferenceRuntimeCounters {
    allocations: AtomicU64,
    live_allocations: Arc<AtomicU64>,
    submissions: AtomicU64,
    commands: AtomicU64,
    uploaded_bytes: AtomicU64,
    dense_linear_launches: AtomicU64,
    layer_norm_launches: AtomicU64,
    head_rms_norm_launches: AtomicU64,
    row_gather_launches: AtomicU64,
    batched_row_concat_launches: AtomicU64,
    zero_centered_rms_norm_launches: AtomicU64,
    gelu_tanh_launches: AtomicU64,
    adjacent_rope_launches: AtomicU64,
    structured_attention_launches: AtomicU64,
    modulation_scale_launches: AtomicU64,
    gated_residual_launches: AtomicU64,
    silu_launches: AtomicU64,
    silu_mul_launches: AtomicU64,
    tanh_launches: AtomicU64,
    readback_bytes: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceDeviceRuntimeSnapshot {
    pub allocations: u64,
    pub live_allocations: u64,
    pub submissions: u64,
    pub commands: u64,
    pub uploaded_bytes: u64,
    pub dense_linear_launches: u64,
    pub layer_norm_launches: u64,
    pub head_rms_norm_launches: u64,
    pub row_gather_launches: u64,
    pub batched_row_concat_launches: u64,
    pub zero_centered_rms_norm_launches: u64,
    pub gelu_tanh_launches: u64,
    pub adjacent_rope_launches: u64,
    pub structured_attention_launches: u64,
    pub modulation_scale_launches: u64,
    pub gated_residual_launches: u64,
    pub silu_launches: u64,
    pub silu_mul_launches: u64,
    pub tanh_launches: u64,
    pub readback_bytes: u64,
}

/// Synchronous, in-memory vNext runtime used for bounded numerical reference
/// execution. It is never selected implicitly by a product backend.
pub struct ReferenceDeviceRuntime {
    descriptor: DeviceDescriptor,
    runtime_instance: u64,
    counters: ReferenceRuntimeCounters,
}

impl ReferenceDeviceRuntime {
    pub(crate) fn new(
        config: ReferenceDeviceRuntimeConfig,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        config
            .descriptor
            .validate()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        let supported_capabilities = super::composition::reference_vnext_capabilities()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        if config.descriptor.class != DeviceClass::Reference
            || config.descriptor.capabilities != supported_capabilities
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime descriptor overclaims its fixed device class or capabilities",
            ));
        }
        let runtime_instance = NEXT_RUNTIME_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                ReferenceDeviceRuntimeError::contract("reference runtime identity exhausted")
            })?;
        Ok(Self {
            descriptor: config.descriptor,
            runtime_instance,
            counters: ReferenceRuntimeCounters::default(),
        })
    }

    // Shared by DeviceRuntime::submit and focused backend preflight tests.
    // This accepts only already-owned commands, never creates core permits.
    fn submit_entries(
        &self,
        stream: &mut ReferenceDeviceStream,
        entries: Vec<(DeviceCommandPhase, Option<u32>, ReferenceDeviceCommand)>,
        attribution_required: bool,
    ) -> Result<ReferenceDeviceFence, DefinitelyNotSubmitted<ReferenceDeviceRuntimeError>> {
        if stream.runtime_instance != self.runtime_instance || entries.is_empty() {
            return Err(DefinitelyNotSubmitted::new(
                ReferenceDeviceRuntimeError::contract(
                    "reference submission has a foreign stream or no commands",
                ),
            ));
        }
        // Every command and every launch is validated before any upload,
        // compute, zeroing or stream transition takes effect.
        for (_, _, command) in &entries {
            command
                .validate_runtime(self.runtime_instance)
                .map_err(DefinitelyNotSubmitted::new)?;
        }
        let attribution = if attribution_required {
            let rows = entries
                .iter()
                .enumerate()
                .map(|(index, (phase, node, command))| {
                    let index = u32::try_from(index).map_err(|_| {
                        ReferenceDeviceRuntimeError::contract("reference command index exceeds u32")
                    })?;
                    command.attribution(index, *node, *phase)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(DefinitelyNotSubmitted::new)?;
            Some(DeviceSubmissionAttribution::new(rows).ok_or_else(|| {
                DefinitelyNotSubmitted::new(ReferenceDeviceRuntimeError::contract(
                    "reference submission attribution is empty or unordered",
                ))
            })?)
        } else {
            None
        };
        let command_count = u64::try_from(entries.len()).map_err(|_| {
            DefinitelyNotSubmitted::new(ReferenceDeviceRuntimeError::contract(
                "reference command count exceeds u64",
            ))
        })?;
        let fence_instance = NEXT_FENCE_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                DefinitelyNotSubmitted::new(ReferenceDeviceRuntimeError::contract(
                    "reference fence identity space exhausted",
                ))
            })?;
        for (_, _, command) in &entries {
            command.execute(&self.counters);
        }
        self.counters.submissions.fetch_add(1, Ordering::Relaxed);
        self.counters
            .commands
            .fetch_add(command_count, Ordering::Relaxed);
        stream.state = StreamState::Submitted;
        Ok(ReferenceDeviceFence {
            runtime_instance: self.runtime_instance,
            fence_instance,
            attribution,
        })
    }

    pub fn snapshot(&self) -> ReferenceDeviceRuntimeSnapshot {
        ReferenceDeviceRuntimeSnapshot {
            allocations: self.counters.allocations.load(Ordering::Relaxed),
            live_allocations: self.counters.live_allocations.load(Ordering::Relaxed),
            submissions: self.counters.submissions.load(Ordering::Relaxed),
            commands: self.counters.commands.load(Ordering::Relaxed),
            uploaded_bytes: self.counters.uploaded_bytes.load(Ordering::Relaxed),
            dense_linear_launches: self.counters.dense_linear_launches.load(Ordering::Relaxed),
            layer_norm_launches: self.counters.layer_norm_launches.load(Ordering::Relaxed),
            head_rms_norm_launches: self.counters.head_rms_norm_launches.load(Ordering::Relaxed),
            row_gather_launches: self.counters.row_gather_launches.load(Ordering::Relaxed),
            batched_row_concat_launches: self
                .counters
                .batched_row_concat_launches
                .load(Ordering::Relaxed),
            zero_centered_rms_norm_launches: self
                .counters
                .zero_centered_rms_norm_launches
                .load(Ordering::Relaxed),
            gelu_tanh_launches: self.counters.gelu_tanh_launches.load(Ordering::Relaxed),
            adjacent_rope_launches: self.counters.adjacent_rope_launches.load(Ordering::Relaxed),
            structured_attention_launches: self
                .counters
                .structured_attention_launches
                .load(Ordering::Relaxed),
            modulation_scale_launches: self
                .counters
                .modulation_scale_launches
                .load(Ordering::Relaxed),
            gated_residual_launches: self
                .counters
                .gated_residual_launches
                .load(Ordering::Relaxed),
            silu_launches: self.counters.silu_launches.load(Ordering::Relaxed),
            silu_mul_launches: self.counters.silu_mul_launches.load(Ordering::Relaxed),
            tanh_launches: self.counters.tanh_launches.load(Ordering::Relaxed),
            readback_bytes: self.counters.readback_bytes.load(Ordering::Relaxed),
        }
    }
}

impl DeviceRuntime for ReferenceDeviceRuntime {
    type Buffer = ReferenceDeviceBuffer;
    type Stream = ReferenceDeviceStream;
    type Command = ReferenceDeviceCommand;
    type Fence = ReferenceDeviceFence;
    type Error = ReferenceDeviceRuntimeError;

    fn descriptor(&self) -> &DeviceDescriptor {
        &self.descriptor
    }

    fn attention_execution_policy(&self) -> ferrum_types::AttentionExecutionPolicy {
        ferrum_types::AttentionExecutionPolicy::Portable
    }

    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error> {
        let request = permit.into_request();
        let size = usize::try_from(request.size_bytes()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference allocation exceeds usize")
        })?;
        let alignment = usize::try_from(request.alignment_bytes()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference alignment exceeds usize")
        })?;
        let (bytes, logical_offset_bytes) = aligned_storage(size, alignment)?;
        let descriptor = BufferDescriptor {
            resource_id: request.resource_id().clone(),
            size_bytes: request.size_bytes(),
            alignment_bytes: request.alignment_bytes(),
            usage: request.usage(),
            element_type: request.element_type(),
        };
        self.counters.allocations.fetch_add(1, Ordering::Relaxed);
        self.counters
            .live_allocations
            .fetch_add(1, Ordering::Relaxed);
        Ok(ReferenceDeviceBuffer {
            descriptor,
            runtime_instance: self.runtime_instance,
            allocation: Arc::new(ReferenceAllocation {
                bytes: Mutex::new(bytes),
                logical_offset_bytes,
                logical_size_bytes: size,
                live_allocations: Arc::clone(&self.counters.live_allocations),
            }),
        })
    }

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor {
        buffer.descriptor.clone()
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error> {
        let stream_instance = NEXT_STREAM_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| {
                ReferenceDeviceRuntimeError::contract("reference stream identity exhausted")
            })?;
        Ok(ReferenceDeviceStream {
            runtime_instance: self.runtime_instance,
            stream_instance,
            state: StreamState::Ready,
        })
    }

    fn stream_state(&self, stream: &Self::Stream) -> StreamState {
        if stream.runtime_instance == self.runtime_instance {
            stream.state
        } else {
            StreamState::Failed
        }
    }

    fn encode_copy(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: CopyRegion,
    ) -> Result<Self::Command, Self::Error> {
        region
            .validate_bounds(&source.descriptor, &destination.descriptor)
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        if source.descriptor.element_type != destination.descriptor.element_type {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference copy requires matching source and destination element types",
            ));
        }
        let source = source.region(
            region.source_offset_bytes()..region.source_offset_bytes() + region.length_bytes(),
        )?;
        let destination = destination.region(
            region.destination_offset_bytes()
                ..region.destination_offset_bytes() + region.length_bytes(),
        )?;
        Ok(ReferenceDeviceCommand::transfer(
            DEVICE_COPY_NATIVE_OPERATION_ID.as_str(),
            ReferenceCommandKind::Copy {
                source,
                destination,
            },
        ))
    }

    fn encode_upload(
        &self,
        source: &[u8],
        source_layout: HostTransferLayout,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        source_layout
            .validate_bytes(source.len())
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        if source_layout.element_type() != destination.descriptor.element_type {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference upload layout differs from destination element type",
            ));
        }
        let length = source_layout
            .byte_len()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        let end = destination_offset_bytes
            .checked_add(length)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract("reference upload range overflows")
            })?;
        let destination = destination.region(destination_offset_bytes..end)?;
        Ok(ReferenceDeviceCommand::transfer(
            HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
            ReferenceCommandKind::Upload {
                source: source.to_vec().into_boxed_slice(),
                destination,
            },
        ))
    }

    fn supports_managed_tensor_io(&self) -> bool {
        true
    }

    fn encode_managed_upload(
        &self,
        source: ManagedHostBytes,
        source_layout: HostTransferLayout,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
    ) -> Result<Option<Self::Command>, Self::Error> {
        source_layout
            .validate_bytes(source.len())
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        if destination.runtime_instance != self.runtime_instance {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference managed upload belongs to another runtime",
            ));
        }
        if source_layout.element_type() != destination.descriptor.element_type {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference managed upload layout differs from destination element type",
            ));
        }
        let length = source_layout
            .byte_len()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        let end = destination_offset_bytes
            .checked_add(length)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract("reference managed upload range overflows")
            })?;
        let destination = destination.region(destination_offset_bytes..end)?;
        Ok(Some(ReferenceDeviceCommand::transfer(
            HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
            ReferenceCommandKind::ManagedUpload {
                source,
                destination,
            },
        )))
    }

    fn encode_zero(
        &self,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self::Command, Self::Error> {
        let end = destination_offset_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract("reference zero range overflows")
            })?;
        let destination = destination.region(destination_offset_bytes..end)?;
        Ok(ReferenceDeviceCommand::transfer(
            DEVICE_ZERO_NATIVE_OPERATION_ID.as_str(),
            ReferenceCommandKind::Zero { destination },
        ))
    }

    fn submit(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>> {
        if stream.runtime_instance != self.runtime_instance {
            return Err(DefinitelyNotSubmitted::new(
                ReferenceDeviceRuntimeError::contract(
                    "reference stream belongs to another runtime",
                ),
            ));
        }
        if commands.is_empty() {
            return Err(DefinitelyNotSubmitted::new(
                ReferenceDeviceRuntimeError::contract(
                    "reference runtime cannot submit an empty batch",
                ),
            ));
        }
        validate_submission_requirements(
            commands.timing_mode(),
            commands.compute_path_requirement(),
            commands.reusable_execution_capture().is_some(),
        )
        .map_err(DefinitelyNotSubmitted::new)?;
        let attribution_required = commands
            .attribution_requirement()
            .logical_execution_path_required();
        let entries = commands
            .into_entries()
            .into_iter()
            .map(|entry| {
                let (phase, node_index, logical_work, command) = entry.into_parts();
                let command = match logical_work {
                    Some(logical_work) => command.bind_core_logical_work(logical_work)?,
                    None => command,
                };
                Ok((phase, node_index, command))
            })
            .collect::<Result<Vec<_>, ReferenceDeviceRuntimeError>>()
            .map_err(DefinitelyNotSubmitted::new)?;
        self.submit_entries(stream, entries, attribution_required)
    }

    fn submission_attribution(&self, fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        (fence.runtime_instance == self.runtime_instance)
            .then(|| fence.attribution.clone())
            .flatten()
    }

    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error> {
        if fence.runtime_instance != self.runtime_instance {
            return FenceQuery::Indeterminate(ReferenceDeviceRuntimeError::contract(
                "reference fence belongs to another runtime",
            ));
        }
        FenceQuery::Terminal(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }

    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>> {
        if fence.runtime_instance != self.runtime_instance {
            return Err(FenceIndeterminate::new(
                ReferenceDeviceRuntimeError::contract("reference fence belongs to another runtime"),
            ));
        }
        Ok(DeviceTerminalReceipt::unprofiled(DeviceTerminal::Succeeded))
    }

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error> {
        if stream.runtime_instance != self.runtime_instance {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference stream belongs to another runtime",
            ));
        }
        stream.state = StreamState::Ready;
        Ok(())
    }

    fn readback(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error> {
        if stream.runtime_instance != self.runtime_instance
            || source.runtime_instance != self.runtime_instance
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference readback belongs to another runtime",
            ));
        }
        if output_layout.element_type() != source.descriptor.element_type {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference readback layout differs from source element type",
            ));
        }
        let output_bytes = output_layout
            .byte_len()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        let length = region.length_bytes();
        let source_end = region
            .source_offset_bytes()
            .checked_add(length)
            .filter(|end| *end <= source.descriptor.size_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference readback source exceeds its admitted allocation",
                )
            })?;
        let output_end = region
            .destination_offset_bytes()
            .checked_add(length)
            .filter(|end| *end <= output_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference readback destination exceeds its host layout",
                )
            })?;
        self.synchronize(stream)?;
        let source_bytes = source
            .region(region.source_offset_bytes()..source_end)?
            .read();
        let output_len = usize::try_from(output_bytes).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference readback output exceeds usize")
        })?;
        let output_start = usize::try_from(region.destination_offset_bytes()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference readback offset exceeds usize")
        })?;
        let output_end = usize::try_from(output_end).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference readback end exceeds usize")
        })?;
        if source_bytes.len() != output_end - output_start {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference readback source and destination lengths differ",
            ));
        }
        let mut output = vec![0_u8; output_len];
        output[output_start..output_end].copy_from_slice(&source_bytes);
        self.counters
            .readback_bytes
            .fetch_add(length, Ordering::Relaxed);
        Ok(output)
    }

    fn readback_managed_into(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        output_layout: HostTransferLayout,
        destination: &mut ManagedHostBuffer,
    ) -> Result<bool, Self::Error> {
        if stream.runtime_instance != self.runtime_instance
            || source.runtime_instance != self.runtime_instance
        {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference managed readback belongs to another runtime",
            ));
        }
        if output_layout.element_type() != source.descriptor.element_type {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference managed readback layout differs from source element type",
            ));
        }
        let piece_bytes = output_layout
            .byte_len()
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))?;
        if piece_bytes != region.length_bytes() {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference managed readback piece differs from its transfer layout",
            ));
        }
        let output_bytes = u64::try_from(destination.as_ref().len()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract(
                "reference managed readback destination exceeds u64",
            )
        })?;
        let source_end = region
            .source_offset_bytes()
            .checked_add(region.length_bytes())
            .filter(|&end| end <= source.descriptor.size_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference managed readback source exceeds its admitted allocation",
                )
            })?;
        let output_end = region
            .destination_offset_bytes()
            .checked_add(region.length_bytes())
            .filter(|&end| end <= output_bytes)
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference managed readback destination exceeds its managed allocation",
                )
            })?;
        let output_start = usize::try_from(region.destination_offset_bytes()).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference managed readback offset exceeds usize")
        })?;
        let output_end = usize::try_from(output_end).map_err(|_| {
            ReferenceDeviceRuntimeError::contract("reference managed readback end exceeds usize")
        })?;
        let source = source.region(region.source_offset_bytes()..source_end)?;
        if source.length_bytes != output_end - output_start {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference managed readback source and destination lengths differ",
            ));
        }
        // All fallible validation precedes writes. This synchronous path copies
        // directly into admitted host backing and never retains its borrow.
        self.synchronize(stream)?;
        let source_bytes = source.allocation.lock();
        destination.as_mut()[output_start..output_end].copy_from_slice(
            &source_bytes[source.offset_bytes..source.offset_bytes + source.length_bytes],
        );
        self.counters
            .readback_bytes
            .fetch_add(region.length_bytes(), Ordering::Relaxed);
        Ok(true)
    }

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError> {
        DeviceErrorReport::new("reference_runtime", error.to_string(), false)
    }
}

fn validate_submission_requirements(
    timing_mode: DeviceTimingMode,
    compute_path: DeviceComputePathRequirement,
    has_reusable_capture: bool,
) -> Result<(), ReferenceDeviceRuntimeError> {
    if timing_mode != DeviceTimingMode::Off {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference runtime does not provide device timing evidence",
        ));
    }
    if matches!(
        compute_path,
        DeviceComputePathRequirement::ReplayedOnly
            | DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries
    ) {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference runtime cannot satisfy a replay-required compute submission",
        ));
    }
    if has_reusable_capture {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference runtime cannot consume reusable execution capture metadata",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::reference::composition::reference_vnext_runtime_config;
    use ferrum_runtime::{
        BufferUsage, CapabilityId, DeviceId, DeviceLogicalWork, HostMemoryAccount, HostMemoryClass,
        HostMemoryClassLimits, HostMemoryLimits, ResourceId,
    };

    pub(super) fn config() -> ReferenceDeviceRuntimeConfig {
        reference_vnext_runtime_config(
            DeviceId::new("device.reference.runtime-test").expect("valid device id"),
        )
        .expect("valid reference config")
    }

    pub(super) fn buffer(
        runtime: &ReferenceDeviceRuntime,
        resource_id: &str,
        contents: &[u8],
        alignment_bytes: usize,
        element_type: ElementType,
    ) -> ReferenceDeviceBuffer {
        let (bytes, logical_offset_bytes) =
            aligned_storage(contents.len(), alignment_bytes).expect("aligned test storage");
        runtime
            .counters
            .live_allocations
            .fetch_add(1, Ordering::Relaxed);
        let buffer = ReferenceDeviceBuffer {
            descriptor: BufferDescriptor {
                resource_id: ResourceId::new(resource_id).expect("valid test resource id"),
                size_bytes: contents.len() as u64,
                alignment_bytes: alignment_bytes as u64,
                usage: BufferUsage::Transfer,
                element_type,
            },
            runtime_instance: runtime.runtime_instance,
            allocation: Arc::new(ReferenceAllocation {
                bytes: Mutex::new(bytes),
                logical_offset_bytes,
                logical_size_bytes: contents.len(),
                live_allocations: Arc::clone(&runtime.counters.live_allocations),
            }),
        };
        buffer
            .region(0..contents.len() as u64)
            .expect("test buffer region")
            .write(contents);
        buffer
    }

    fn host_account(total: usize) -> HostMemoryAccount {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                total,
                HostMemoryClassLimits {
                    input_preprocess_bytes: total,
                    transfer_staging_bytes: total,
                    output_bytes: total,
                    weight_backing_bytes: total,
                },
            )
            .unwrap(),
        )
    }

    #[test]
    fn aligned_reference_storage_fulfills_descriptor_contract() {
        for alignment in [1, 2, 16, 64, 4096] {
            let (bytes, offset) = aligned_storage(257, alignment).expect("aligned storage");
            assert_eq!((bytes.as_ptr() as usize + offset) % alignment, 0);
            assert!(offset + 257 <= bytes.len());
        }
    }

    #[test]
    fn descriptor_cannot_overclaim_device_class_or_capabilities() {
        ReferenceDeviceRuntime::new(config()).expect("fixed reference descriptor must be valid");

        let mut wrong_class = config();
        wrong_class.descriptor.class = DeviceClass::Accelerator;
        assert!(ReferenceDeviceRuntime::new(wrong_class).is_err());

        let mut extra_capability = config();
        extra_capability.descriptor.capabilities.insert(
            CapabilityId::new("capability.reference.unimplemented")
                .expect("valid synthetic capability id"),
        );
        assert!(ReferenceDeviceRuntime::new(extra_capability).is_err());
    }

    #[test]
    fn submission_requirements_fail_closed_before_reference_execution() {
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::Adaptive,
            false,
        )
        .is_ok());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::EagerOnly,
            false,
        )
        .is_ok());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::ReplayedOnly,
            false,
        )
        .is_err());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries,
            false,
        )
        .is_err());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Completion,
            DeviceComputePathRequirement::Adaptive,
            false,
        )
        .is_err());
        assert!(validate_submission_requirements(
            DeviceTimingMode::Off,
            DeviceComputePathRequirement::Adaptive,
            true,
        )
        .is_err());
    }

    #[test]
    fn foreign_stream_state_fails_closed() {
        let first = ReferenceDeviceRuntime::new(config()).expect("first runtime");
        let second = ReferenceDeviceRuntime::new(config()).expect("second runtime");
        let stream = first.create_stream().expect("first stream");
        assert_eq!(first.stream_state(&stream), StreamState::Ready);
        assert_eq!(second.stream_state(&stream), StreamState::Failed);
    }

    #[test]
    fn readback_honors_host_offset_and_element_type() {
        let runtime = ReferenceDeviceRuntime::new(config()).expect("reference runtime");
        let source = buffer(
            &runtime,
            "resource.reference.readback",
            &[10, 11, 12, 13],
            16,
            ElementType::U8,
        );
        let mut stream = runtime.create_stream().expect("reference stream");
        let region = CopyRegion::new(1, 2, 2).expect("valid readback region");
        let output = runtime
            .readback(
                &mut stream,
                &source,
                region,
                HostTransferLayout::new(ElementType::U8, 6).expect("valid host layout"),
            )
            .expect("offset readback");
        assert_eq!(output, [0, 0, 11, 12, 0, 0]);
        assert!(runtime
            .readback(
                &mut stream,
                &source,
                region,
                HostTransferLayout::new(ElementType::F16, 3).expect("valid mismatched layout"),
            )
            .is_err());
    }

    #[test]
    fn managed_upload_retains_the_actual_sliced_backing_until_command_drop() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        assert!(runtime.supports_managed_tensor_io());
        let host = host_account(8);
        let mut backing = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
            .unwrap();
        backing
            .as_mut()
            .copy_from_slice(&[90, 91, 10, 11, 12, 13, 92, 93]);
        let backing = backing.freeze();
        let source = backing.slice(2..6).unwrap();
        let destination = buffer(
            &runtime,
            "resource.reference.managed-upload",
            &[0; 6],
            16,
            ElementType::U8,
        );
        let command = runtime
            .encode_managed_upload(
                source,
                HostTransferLayout::new(ElementType::U8, 4).unwrap(),
                &destination,
                1,
            )
            .unwrap()
            .unwrap();
        drop(backing);
        assert_eq!(host.usage().unwrap().committed_bytes(), 8);
        assert_eq!(runtime.snapshot().uploaded_bytes, 0);
        command.execute(&runtime.counters);
        assert_eq!(
            destination.region(0..6).unwrap().read(),
            [0, 10, 11, 12, 13, 0]
        );
        assert_eq!(runtime.snapshot().uploaded_bytes, 4);
        assert_eq!(host.usage().unwrap().committed_bytes(), 8);
        drop(command);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn managed_upload_rejects_foreign_runtime_and_overflow_without_a_command() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        let host = host_account(4);
        let source = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
            .unwrap()
            .freeze();
        let local_destination = buffer(
            &runtime,
            "resource.reference.local",
            &[0; 4],
            16,
            ElementType::U8,
        );
        let foreign_destination = buffer(
            &foreign,
            "resource.reference.foreign",
            &[0; 4],
            16,
            ElementType::U8,
        );
        let layout = HostTransferLayout::new(ElementType::U8, 4).unwrap();
        assert!(runtime
            .encode_managed_upload(source.clone(), layout, &foreign_destination, 0)
            .is_err());
        assert!(runtime
            .encode_managed_upload(source.clone(), layout, &local_destination, u64::MAX)
            .is_err());
        assert!(runtime
            .encode_managed_upload(source.clone(), layout, &local_destination, 1)
            .is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 4);
        assert_eq!(runtime.snapshot().uploaded_bytes, 0);
        drop(source);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn managed_readback_preserves_destination_on_validation_failure_and_copies_exact_region() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
        let host = host_account(6);
        let source = buffer(
            &runtime,
            "resource.reference.managed-readback",
            &[10, 11, 12, 13],
            16,
            ElementType::U8,
        );
        let foreign_source = buffer(
            &foreign,
            "resource.reference.foreign-readback",
            &[10, 11, 12, 13],
            16,
            ElementType::U8,
        );
        let mut destination = host.allocate_zeroed(HostMemoryClass::Output, 6).unwrap();
        destination.as_mut().fill(0xa5);
        let mut stream = runtime.create_stream().unwrap();
        // The layout describes this two-byte piece; destination is the complete
        // six-byte host result, and the copy starts at a nonzero host offset.
        let layout = HostTransferLayout::new(ElementType::U8, 2).unwrap();
        let valid_region = CopyRegion::new(1, 2, 2).unwrap();
        for (read_source, region, read_layout) in [
            (&foreign_source, valid_region, layout),
            (&source, CopyRegion::new(3, 0, 2).unwrap(), layout),
            (&source, CopyRegion::new(0, 5, 2).unwrap(), layout),
            (
                &source,
                valid_region,
                HostTransferLayout::new(ElementType::F16, 1).unwrap(),
            ),
            (
                &source,
                valid_region,
                HostTransferLayout::new(ElementType::U8, 1).unwrap(),
            ),
        ] {
            assert!(runtime
                .readback_managed_into(
                    &mut stream,
                    read_source,
                    region,
                    read_layout,
                    &mut destination
                )
                .is_err());
            assert_eq!(destination.as_ref(), &[0xa5; 6]);
            assert_eq!(runtime.snapshot().readback_bytes, 0);
        }
        assert!(runtime
            .readback_managed_into(&mut stream, &source, valid_region, layout, &mut destination)
            .unwrap());
        assert_eq!(destination.as_ref(), &[0xa5, 0xa5, 11, 12, 0xa5, 0xa5]);
        assert_eq!(runtime.snapshot().readback_bytes, 2);
        assert_eq!(host.usage().unwrap().committed_bytes(), 6);
        drop(destination);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn fixed_tensor_dense_reports_tensor_work_and_computes_rectangular_f16_values() {
        let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
        let encode = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
                .collect()
        };
        let input = buffer(
            &runtime,
            "resource.reference.tensor-input",
            &encode(&[1., 2., 3., -1., 0., 2.]),
            16,
            ElementType::F16,
        );
        let weight = buffer(
            &runtime,
            "resource.reference.tensor-weight",
            &encode(&[2., -1., 1., 1., 3., -2.]),
            16,
            ElementType::F16,
        );
        let output = buffer(
            &runtime,
            "resource.reference.tensor-output",
            &[0; 8],
            16,
            ElementType::F16,
        );
        let launch = || ReferenceDenseLinearLaunch {
            input: input.region(0..12).unwrap(),
            weight: weight.region(0..12).unwrap(),
            output: output.region(0..8).unwrap(),
            rows: 2,
            in_features: 3,
            out_features: 2,
            scalar: ReferenceDenseScalar::F16,
        };
        let command = ReferenceDeviceCommand::dense_tensor(launch()).unwrap();
        let attribution = command
            .attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap();
        assert_eq!(attribution.token_count(), None);
        assert_eq!(
            attribution.logical_work(),
            DeviceLogicalWork::Tensor {
                element_count: 4,
                arithmetic_operations: 24
            }
        );
        assert_eq!(attribution.participant_count(), 1);
        assert_eq!(attribution.compute_dispatch_count(), 1);
        command.execute(&runtime.counters);
        assert_eq!(
            output.region(0..8).unwrap().read(),
            encode(&[3., 1., 0., -5.])
        );
        assert_eq!(runtime.snapshot().dense_linear_launches, 1);
        let token_command =
            ReferenceDeviceCommand::dense_linear(vec![launch()], DeviceBatchingForm::Scalar, 1, 2)
                .unwrap();
        assert_eq!(
            token_command
                .attribution(0, Some(0), DeviceCommandPhase::Compute)
                .unwrap()
                .token_count(),
            Some(2)
        );
    }
}

#[cfg(test)]
#[path = "runtime/dense_tests.rs"]
mod dense_tests;
