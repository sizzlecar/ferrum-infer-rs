//! Isolated noncausal segments on the existing admitted cuDNN SDPA owner.
//! Selected only by the explicit cuDNN composition. No model-specific geometry.
use super::cudnn_attention::layout::{ScratchLayout, SCRATCH_ALIGNMENT};
use super::*;
use crate::backend::cuda::vnext_runtime::{
    CudaCudnnSdpaCache, CudnnSdpaBindings, CudnnSdpaExecution, CudnnSdpaPlan, CudnnSdpaSignature,
};
use ferrum_runtime::{
    DynamicStorageRequirement, PhysicalWeightLayout, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    TensorPrimitiveBf16, TensorPrimitiveBf16Geometry, WeightFormatId,
};
use sha2::{Digest, Sha256};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.cudnn920.segmented_attention.bf16.v1";
const NUMERICAL_IDENTITY: &str = concat!(
    "BF16-QKVY-contiguous-rows;D72-or-D128;independent-positive-noncausal-segments;",
    "F32-byvalue-scale;cuDNN9.20-forward-inference;deterministic-engine-filter;",
    "vendor-intermediate-rounding;finite-visible-domain;no-sequential-or-Torch-bitwise-claim;",
    "each-segment-B1-QO-slice-and-stable-KV-pack;no-cross-segment-read;",
    "one-admitted-scratch-reused-in-stream-order;exact-queried-workspace;",
    "shared-cuDNN-handle-cache-opaque-and-binding-limits;prepared-native-plan-digest;",
    "logical-work=mathematical-QK-scale-softmax-PV-scalar-operations;not-hardware-FLOPs"
);
const PROVIDER: &str = "provider.cuda.segmented_attention.bf16.cudnn920";
const ESTIMATOR: &str = "resource-estimator.cuda.segmented_attention.bf16.cudnn920";

pub(super) fn provider(
    device: &DeviceDescriptor,
    kernels: &Arc<RetainedFunctions>,
    cache: Arc<CudaCudnnSdpaCache>,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    let kind = TensorPrimitiveBf16::SegmentedAttention;
    let capability = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks segmented attention capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        cache.fingerprint().as_bytes(),
        include_bytes!("cudnn_segmented.rs"),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new(PROVIDER).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(3),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        cache,
    }))
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    offset_bytes: u64,
    bytes: u64,
    rows: u32,
    signature: CudnnSdpaSignature,
}
struct Schedule {
    geometry: TensorPrimitiveBf16Geometry,
    segments: Vec<Segment>,
    operations: u64,
}
impl Schedule {
    fn new(
        bindings: &[ResolvedValueBinding],
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, String> {
        let geometry = TensorPrimitiveBf16Geometry::new(
            TensorPrimitiveBf16::SegmentedAttention,
            bindings,
            attrs,
        )
        .map_err(|e| e.to_string())?;
        if !matches!(geometry.head_dim, 72 | 128) {
            return Err("cuDNN segmented attention supports D72 or D128".into());
        }
        if geometry.output_bytes > isize::MAX as u64
            || bindings.iter().any(|b| {
                b.weight().is_some_and(|w| {
                    !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. })
                })
            })
        {
            return Err("cuDNN segmented attention requires addressable dense BF16 storage".into());
        }
        let heads = geometry.columns / geometry.head_dim;
        let mut segments = Vec::with_capacity(geometry.lengths.len());
        let mut operations = 0u64;
        for (&start, &length) in geometry.starts.iter().zip(&geometry.lengths) {
            let rows = u32::try_from(length)
                .ok()
                .filter(|&v| v <= i32::MAX as u32)
                .ok_or("cuDNN segment length exceeds positive i32")?;
            let extent = length
                .checked_mul(geometry.columns)
                .ok_or("segment extent overflows")?;
            let offset_bytes = start
                .checked_mul(geometry.columns)
                .and_then(|n| n.checked_mul(2))
                .ok_or("segment offset overflows")?;
            let bytes = extent.checked_mul(2).ok_or("segment bytes overflow")?;
            if offset_bytes % 16 != 0
                || offset_bytes
                    .checked_add(bytes)
                    .is_none_or(|end| end > geometry.output_bytes)
            {
                return Err("segment slice exceeds tensor or native alignment".into());
            }
            let stride = i64::try_from(extent).map_err(|_| "segment stride exceeds i64")?;
            let signature = CudnnSdpaSignature {
                batch: 1,
                heads: i64::try_from(heads).map_err(|_| "segment heads exceed i64")?,
                query_length: i64::from(rows),
                key_value_length: i64::from(rows),
                head_dim: geometry.head_dim as i64,
                query_batch_stride: stride,
                key_value_batch_stride: stride,
                output_batch_stride: stride,
                scale_bits: geometry.scalar.to_bits(),
            };
            signature.validate()?;
            operations = operations
                .checked_add(logical_work(length, heads, geometry.head_dim)?)
                .ok_or("segmented attention work overflows")?;
            segments.push(Segment {
                offset_bytes,
                bytes,
                rows,
                signature,
            });
        }
        Ok(Self {
            geometry,
            segments,
            operations,
        })
    }
}

// Count a mathematical full-softmax graph, not cuDNN's internal instruction
// count: QK+scale=2D per pair; subtract/exp/sum/div=4L-1 per query;
// PV=D(2L-1) per query. Comparisons, conversions and transport are excluded.
fn logical_work(length: u64, heads: u64, dim: u64) -> Result<u64, String> {
    dim.checked_mul(4)
        .and_then(|v| v.checked_add(4))
        .and_then(|v| v.checked_mul(length))
        .and_then(|v| v.checked_sub(dim))
        .and_then(|v| v.checked_sub(1))
        .and_then(|v| v.checked_mul(length))
        .and_then(|v| v.checked_mul(heads))
        .ok_or_else(|| "segmented attention work overflows".into())
}

struct PreparedSegment {
    segment: Segment,
    plan: Arc<CudnnSdpaPlan>,
    scratch: ScratchLayout,
}
fn prepared_fingerprint(stages: &[PreparedSegment]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"ferrum.cuda.segmented-attention.prepared.v1\0");
    digest.update((stages.len() as u64).to_le_bytes());
    for s in stages {
        digest.update(s.segment.offset_bytes.to_le_bytes());
        digest.update(s.segment.bytes.to_le_bytes());
        digest.update(s.segment.rows.to_le_bytes());
        digest.update(s.plan.fingerprint().as_bytes());
    }
    format!("{:x}", digest.finalize())
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    cache: Arc<CudaCudnnSdpaCache>,
}
impl Provider {
    fn stages(
        &self,
        schedule: &Schedule,
        prepare: bool,
    ) -> Result<(Vec<PreparedSegment>, u64), String> {
        let mut result = Vec::with_capacity(schedule.segments.len());
        let mut peak = 0;
        for &segment in &schedule.segments {
            let plan = if prepare {
                self.cache.prepare(segment.signature)
            } else {
                self.cache.lookup(&segment.signature)
            }
            .map_err(|e| e.to_string())?;
            let scratch = ScratchLayout::new(
                1,
                segment.rows,
                schedule.geometry.columns,
                plan.workspace_bytes(),
            )?;
            peak = peak.max(scratch.bytes);
            result.push(PreparedSegment {
                segment,
                plan,
                scratch,
            });
        }
        Ok((result, peak))
    }
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id != *self.descriptor.operation_id()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(
                "cuDNN segmented estimator received another operation",
            ));
        }
        let schedule =
            Schedule::new(request.values(), request.attributes()).map_err(invalid_plan)?;
        let (stages, bytes) = self.stages(&schedule, true).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(bytes)?,
            SCRATCH_ALIGNMENT,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            SCRATCH_ALIGNMENT,
            Some(scratch),
            None,
        )
        .with_prepared_implementation_fingerprint(prepared_fingerprint(&stages))
    }
}
impl OperationProvider<CudaDeviceRuntime> for Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            PROVIDER,
            "fixed tensor work required".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        self.encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| provider_failure(identity, PROVIDER, e))
    }
}
struct BoundSegment {
    segment: Segment,
    scratch: ScratchLayout,
    mask: attention::NativeMask,
    execution: Arc<CudnnSdpaExecution>,
}
impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("segmented attention requires one participant".into());
        };
        if invocation.operation().id.as_str()
            != TensorPrimitiveBf16::SegmentedAttention.operation_id()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("segmented attention requires its fixed tensor contract".into());
        }
        let schedule = Schedule::new(participant.bindings(), participant.attributes())?;
        let (prepared, scratch_bytes) = self.stages(&schedule, false)?;
        let identity = prepared_fingerprint(&prepared);
        if invocation.prepared_implementation_fingerprint() != Some(identity.as_str()) {
            return Err("segmented native plans differ from the admitted implementation".into());
        }
        let mut regions = Vec::with_capacity(5);
        for ordinal in 0..3 {
            regions.push(retained_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
                schedule.geometry.output_bytes,
            )?);
        }
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            schedule.geometry.output_bytes,
        )?;
        for input in &regions {
            nonoverlapping_ranges(
                input.device_ptr(),
                input.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let view = participant
            .scratch_view()
            .ok_or("segmented scratch was not admitted")?;
        let translated = view
            .translate(0, scratch_bytes)
            .map_err(|e| e.to_string())?;
        let mut pieces = translated.iter();
        let piece = pieces.next().ok_or("segmented scratch is empty")?;
        if pieces.next().is_some() {
            return Err("segmented scratch must be contiguous".into());
        }
        let (buffer, range, retention) = piece.buffer_and_physical_range();
        let region = buffer
            .retained_region(range, retention)
            .map_err(|e| e.to_string())?;
        if region.element_type() != ElementType::U8
            || region.length_bytes() != scratch_bytes
            || region.device_ptr() % SCRATCH_ALIGNMENT != 0
        {
            return Err("segmented scratch dtype/extent/alignment mismatch".into());
        }
        for input in &regions {
            nonoverlapping_ranges(
                input.device_ptr(),
                input.length_bytes(),
                region.device_ptr(),
                region.length_bytes(),
            )?;
        }
        let scratch = region.device_ptr();
        regions.push(region);
        let address = |base: u64, offset: u64| {
            base.checked_add(offset)
                .ok_or_else(|| "segmented pointer overflow".to_owned())
        };
        let mut bound = Vec::with_capacity(prepared.len());
        for p in prepared {
            let s = p.scratch;
            let execution = p
                .plan
                .bind(CudnnSdpaBindings {
                    q: address(regions[0].device_ptr(), p.segment.offset_bytes)?,
                    k: scratch,
                    v: address(scratch, s.value_offset)?,
                    o: address(regions[3].device_ptr(), p.segment.offset_bytes)?,
                    seq_len_q: address(scratch, s.query_lengths_offset)?,
                    seq_len_kv: address(scratch, s.key_lengths_offset)?,
                    workspace: if s.workspace_bytes == 0 {
                        0
                    } else {
                        address(scratch, s.workspace_offset)?
                    },
                    workspace_bytes: s.workspace_bytes,
                })
                .map_err(|e| e.to_string())?;
            bound.push(BoundSegment {
                segment: p.segment,
                scratch: s,
                mask: attention::NativeMask::isolated_sequence(p.segment.rows)?,
                execution,
            });
        }
        let elements = schedule.geometry.elements();
        let columns = schedule.geometry.columns;
        let calls = (bound.len() as u64)
            .checked_mul(2)
            .ok_or("segmented call count overflows")?;
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            "vnext_cudnn_segmented_attention_bf16",
            regions,
            move |stream, regions| {
                for b in &bound {
                    let k = regions[1].device_ptr() + b.segment.offset_bytes;
                    let v = regions[2].device_ptr() + b.segment.offset_bytes;
                    let packed_v = scratch + b.scratch.value_offset;
                    let q_lengths = scratch + b.scratch.query_lengths_offset;
                    let k_lengths = scratch + b.scratch.key_lengths_offset;
                    let rows = b.segment.rows;
                    let zero = 0u32;
                    let one = 1u32;
                    let mut launch = stream.launch_builder(kernels.fused_attention_function(1));
                    launch
                        .arg(&k)
                        .arg(&v)
                        .arg(&scratch)
                        .arg(&packed_v)
                        .arg(&q_lengths)
                        .arg(&k_lengths)
                        .arg(&columns)
                        .arg(&rows)
                        .arg(&zero)
                        .arg(&one)
                        .arg(&rows)
                        .arg(&rows)
                        .arg(&rows)
                        .arg(&b.mask);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (
                                (b.segment.bytes / 2)
                                    .div_ceil(u64::from(THREADS))
                                    .min(MAX_BLOCKS) as u32,
                                1,
                                1,
                            ),
                            block_dim: (THREADS, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map_err(|e| CudaDeviceRuntimeError::driver("isolated segment KV pack", e))?;
                    b.execution.execute(stream)?;
                }
                Ok(())
            },
        )
        .and_then(|c| c.with_tensor_work(elements, schedule.operations, calls, 0))
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests;
