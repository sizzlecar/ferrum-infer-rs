//! Vendor SDPA for bidirectional segments, with an exact structured-mask
//! fallback for causal query ranges. All payload temporaries are admitted
//! scratch; native graph/variant-pack metadata has a separate finite quota.

use super::*;
use crate::backend::cuda::vnext_runtime::{
    CudaCudnnSdpaCache, CudnnSdpaBindings, CudnnSdpaExecution, CudnnSdpaPlan, CudnnSdpaSignature,
};
use ferrum_runtime::{
    structured_attention_fused_bf16_contract, DynamicStorageRequirement,
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, StructuredAttentionBf16Geometry, StructuredAttentionBf16Mask,
    STRUCTURED_ATTENTION_FUSED_BF16_CAPABILITY_ID, STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID,
};
use sha2::{Digest, Sha256};

pub(super) mod layout;
use layout::{Method, Schedule, ScratchLayout, Stage, SCRATCH_ALIGNMENT};
#[cfg(test)]
mod tests;

pub const NUMERICAL_PROFILE_ID: &str = "cuda.cudnn920.structured_attention.bf16.segmented.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "BF16-QKVY-BSHD;D128;cuDNN9.20-SDPA-forward-inference;F32-byvalue-scale;",
    "vendor-intermediate-rounding-not-IEEE-online-profile;no-Torch-bitwise-claim;",
    "bidirectional-Q-entire-segment;KV-valid-prefix-stable-order;actual-i32-sequence-lengths;",
    "all-nonempty-batches-use-one-batched-plan;empty-samples-zero-without-QKV-read;",
    "causal-segments-only-use-original-FP32-online-query-range-kernel;",
    "masked-KV-never-read;query-padding-not-removed;finite-visible-domain-for-vendor-SDPA;",
    "NoAlias;one-admitted-scratch-for-KV-lengths-exact-queried-workspace;",
    "sequential-segments-reuse-scratch;prepare-in-estimator-lookup-in-encode;",
    "bounded-native-plan-and-binding-cache;target-numerical-performance-evidence-required"
);
const PROVIDER: &str = "provider.cuda.structured_attention_fused.bf16.cudnn920";
const ESTIMATOR: &str = "resource-estimator.cuda.structured_attention_fused.bf16.cudnn920";

fn flat_launch(elements: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            elements.div_ceil(u64::from(THREADS)).min(MAX_BLOCKS) as u32,
            1,
            1,
        ),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn provider(
    device: &DeviceDescriptor,
    kernels: &Arc<RetainedFunctions>,
    cache: Arc<CudaCudnnSdpaCache>,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    let capability =
        CapabilityId::new(STRUCTURED_ATTENTION_FUSED_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks fused structured attention capability",
        ));
    }
    let contract = structured_attention_fused_bf16_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        cache.fingerprint().as_bytes(),
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
        BTreeSet::new(),
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

struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    cache: Arc<CudaCudnnSdpaCache>,
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Schedule, String> {
    if bindings.len() != 4 {
        return Err("fused attention requires exactly Q/K/V and Y".into());
    }
    Schedule::new(
        StructuredAttentionBf16Geometry::new(
            binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
            binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
            binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
            binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
            attributes,
        )
        .map_err(|e| e.to_string())?,
    )
}

struct PreparedStage {
    stage: Stage,
    sdpa: Option<(Arc<CudnnSdpaPlan>, ScratchLayout)>,
}

/// The ordered executable choices, not just their common workspace extent.
/// Fixed-width framing includes range/fallback identity and the actual native
/// engine/knob/numerical-note digest returned by each prepared cuDNN plan.
fn prepared_stages_fingerprint<'a>(
    stages: impl IntoIterator<Item = (Stage, Option<&'a str>)>,
) -> Result<String, String> {
    let mut digest = Sha256::new();
    digest.update(b"ferrum.cuda.structured-attention.prepared-stages.v1\0");
    let mut count = 0u64;
    for (stage, native) in stages {
        count = count
            .checked_add(1)
            .ok_or("attention prepared stage count overflows")?;
        for value in [
            stage.first_query,
            stage.query_count,
            stage.first_batch,
            stage.batches,
        ] {
            digest.update(value.to_le_bytes());
        }
        match (stage.method, native) {
            (Method::Causal, None) => digest.update([0]),
            (Method::Empty, None) => digest.update([1]),
            (Method::Sdpa { packed_rows }, Some(fingerprint)) => {
                if fingerprint.len() != 64
                    || !fingerprint
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err("attention native plan fingerprint is not canonical SHA-256".into());
                }
                digest.update([2]);
                digest.update(packed_rows.to_le_bytes());
                digest.update(fingerprint.as_bytes());
            }
            _ => return Err("attention prepared stage and native plan identity disagree".into()),
        }
    }
    digest.update(count.to_le_bytes());
    Ok(format!("{:x}", digest.finalize()))
}

fn prepared_fingerprint(stages: &[PreparedStage]) -> Result<String, String> {
    prepared_stages_fingerprint(stages.iter().map(|prepared| {
        (
            prepared.stage,
            prepared.sdpa.as_ref().map(|(plan, _)| plan.fingerprint()),
        )
    }))
}

impl Provider {
    fn stages(
        &self,
        schedule: &Schedule,
        prepare: bool,
    ) -> Result<(Vec<PreparedStage>, u64), String> {
        let mut peak = 0;
        let mut stages = Vec::with_capacity(schedule.stages.len());
        for &stage in &schedule.stages {
            let sdpa = if let Method::Sdpa { packed_rows } = stage.method {
                let signature = schedule.signature(stage)?;
                let plan = if prepare {
                    self.cache.prepare(signature)
                } else {
                    self.cache.lookup(&signature)
                }
                .map_err(|e| e.to_string())?;
                let layout = ScratchLayout::new(
                    stage.batches,
                    packed_rows,
                    schedule.geometry.columns(),
                    plan.workspace_bytes(),
                )?;
                peak = peak.max(layout.bytes);
                Some((plan, layout))
            } else {
                None
            };
            stages.push(PreparedStage { stage, sdpa });
        }
        Ok((stages, peak))
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
                "fused attention estimator received another operation",
            ));
        }
        let schedule = geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        let (stages, bytes) = self.stages(&schedule, true).map_err(invalid_plan)?;
        let prepared = prepared_fingerprint(&stages).map_err(invalid_plan)?;
        let scratch = if bytes == 0 {
            None
        } else {
            Some(ProviderWorkspaceRequirement::from_formula(
                ProviderWorkspaceSizeFormula::fixed(bytes)?,
                SCRATCH_ALIGNMENT,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )?)
        };
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            SCRATCH_ALIGNMENT,
            scratch,
            None,
        )
        .with_prepared_implementation_fingerprint(prepared)
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
            "cuda.structured_attention_fused.bf16.encode",
            "fixed tensor work is required".into(),
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
            .map_err(|e| {
                provider_failure(identity, "cuda.structured_attention_fused.bf16.encode", e)
            })
    }
}

struct BoundStage {
    stage: Stage,
    sdpa: Option<(Arc<CudnnSdpaExecution>, ScratchLayout)>,
}

impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("fused attention requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("fused attention requires its fixed tensor contract".into());
        }
        let schedule = geometry(participant.bindings(), participant.attributes())?;
        let (prepared, scratch_bytes) = self.stages(&schedule, false)?;
        let actual_prepared = prepared_fingerprint(&prepared)?;
        if invocation.prepared_implementation_fingerprint() != Some(actual_prepared.as_str()) {
            return Err(
                "cuDNN attention prepared implementation differs from the admitted plan".into(),
            );
        }
        let mut regions = Vec::with_capacity(5);
        for ordinal in 0..3 {
            regions.push(retained_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
                schedule.bytes,
            )?);
        }
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            schedule.bytes,
        )?;
        for read in &regions {
            nonoverlapping_ranges(
                read.device_ptr(),
                read.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let scratch = if scratch_bytes == 0 {
            if participant.scratch_view().is_some() {
                return Err("fused attention received unneeded scratch".into());
            }
            0
        } else {
            let view = participant
                .scratch_view()
                .ok_or("fused attention scratch was not admitted")?;
            let translated = view
                .translate(0, scratch_bytes)
                .map_err(|e| e.to_string())?;
            let mut pieces = translated.iter();
            let piece = pieces.next().ok_or("fused attention scratch is empty")?;
            if pieces.next().is_some() {
                return Err("fused attention scratch must be contiguous".into());
            }
            let (buffer, range, retention) = piece.buffer_and_physical_range();
            let region = buffer
                .retained_region(range, retention)
                .map_err(|e| e.to_string())?;
            if region.element_type() != ElementType::U8
                || region.length_bytes() != scratch_bytes
                || region.device_ptr() % SCRATCH_ALIGNMENT != 0
            {
                return Err("fused attention scratch dtype, extent or alignment mismatch".into());
            }
            for read in &regions {
                nonoverlapping_ranges(
                    read.device_ptr(),
                    read.length_bytes(),
                    region.device_ptr(),
                    region.length_bytes(),
                )?;
            }
            let pointer = region.device_ptr();
            regions.push(region);
            pointer
        };
        let address = |base: u64, offset: u64| {
            base.checked_add(offset)
                .ok_or_else(|| "fused attention pointer overflow".to_owned())
        };
        let mut bound = Vec::with_capacity(prepared.len());
        let g = &schedule.geometry;
        for stage in prepared {
            let sdpa = if let Some((plan, layout)) = stage.sdpa {
                let first_row = u64::from(stage.stage.first_batch) * g.sequence_length()
                    + u64::from(stage.stage.first_query);
                let offset = first_row
                    .checked_mul(g.columns())
                    .and_then(|n| n.checked_mul(2))
                    .ok_or("fused attention query offset overflow")?;
                let bindings = CudnnSdpaBindings {
                    q: address(regions[0].device_ptr(), offset)?,
                    k: scratch,
                    v: address(scratch, layout.value_offset)?,
                    o: address(regions[3].device_ptr(), offset)?,
                    seq_len_q: address(scratch, layout.query_lengths_offset)?,
                    seq_len_kv: address(scratch, layout.key_lengths_offset)?,
                    workspace: if layout.workspace_bytes == 0 {
                        0
                    } else {
                        address(scratch, layout.workspace_offset)?
                    },
                    workspace_bytes: layout.workspace_bytes,
                };
                Some((plan.bind(bindings).map_err(|e| e.to_string())?, layout))
            } else {
                None
            };
            bound.push(BoundStage {
                stage: stage.stage,
                sdpa,
            });
        }
        let kernels = Arc::clone(&self.kernels);
        let elements = g.element_count();
        let operations = g.logical_arithmetic_operations();
        // Native counts describe our visible launches/calls, not vendor-internal
        // kernels. Pack + cuDNN is two; fallback/empty is one.
        let calls = bound
            .iter()
            .try_fold(0u64, |n, b| {
                n.checked_add(if b.sdpa.is_some() { 2 } else { 1 })
            })
            .ok_or("fused attention native call count overflow")?;
        CudaDeviceCommand::operation(
            "vnext_structured_attention_fused_bf16",
            regions,
            move |stream, regions| {
                let q = regions[0].device_ptr();
                let k = regions[1].device_ptr();
                let v = regions[2].device_ptr();
                let y = regions[3].device_ptr();
                let columns = schedule.geometry.columns();
                let heads = schedule.geometry.heads();
                let sequence = schedule.geometry.sequence_length() as u32;
                let width = 128u32;
                let scale = schedule.geometry.effective_scale();
                for item in &bound {
                    let s = item.stage;
                    match s.method {
                        Method::Causal => {
                            let groups = u64::from(s.batches) * u64::from(s.query_count) * heads;
                            let mut launch =
                                stream.launch_builder(kernels.fused_attention_function(0));
                            launch
                                .arg(&q)
                                .arg(&k)
                                .arg(&v)
                                .arg(&y)
                                .arg(&groups)
                                .arg(&heads)
                                .arg(&sequence)
                                .arg(&width)
                                .arg(&scale)
                                .arg(&schedule.mask)
                                .arg(&s.first_query)
                                .arg(&s.query_count)
                                .arg(&s.first_batch);
                            unsafe {
                                launch.launch(LaunchConfig {
                                    grid_dim: (groups.min(MAX_BLOCKS) as u32, 1, 1),
                                    block_dim: (width, 1, 1),
                                    shared_mem_bytes: width * 4,
                                })
                            }
                            .map_err(|e| {
                                CudaDeviceRuntimeError::driver("attention causal query range", e)
                            })?;
                        }
                        Method::Empty => {
                            let n = u64::from(s.batches) * u64::from(s.query_count) * columns;
                            let mut launch =
                                stream.launch_builder(kernels.fused_attention_function(2));
                            launch
                                .arg(&y)
                                .arg(&columns)
                                .arg(&sequence)
                                .arg(&s.first_batch)
                                .arg(&s.batches)
                                .arg(&s.first_query)
                                .arg(&s.query_count);
                            unsafe { launch.launch(flat_launch(n)) }.map_err(|e| {
                                CudaDeviceRuntimeError::driver("attention empty query range", e)
                            })?;
                        }
                        Method::Sdpa { packed_rows } => {
                            let (call, layout) = item.sdpa.as_ref().expect("prepared SDPA stage");
                            let packed_v = scratch + layout.value_offset;
                            let q_lengths = scratch + layout.query_lengths_offset;
                            let k_lengths = scratch + layout.key_lengths_offset;
                            let visible_end = s.first_query + s.query_count;
                            let n = u64::from(s.batches) * u64::from(packed_rows) * columns;
                            let copy_payload = 1_u32;
                            let mut launch =
                                stream.launch_builder(kernels.fused_attention_function(1));
                            launch
                                .arg(&k)
                                .arg(&v)
                                .arg(&scratch)
                                .arg(&packed_v)
                                .arg(&q_lengths)
                                .arg(&k_lengths)
                                .arg(&columns)
                                .arg(&sequence)
                                .arg(&s.first_batch)
                                .arg(&s.batches)
                                .arg(&s.query_count)
                                .arg(&visible_end)
                                .arg(&packed_rows)
                                .arg(&copy_payload)
                                .arg(&schedule.mask);
                            unsafe { launch.launch(flat_launch(n)) }.map_err(|e| {
                                CudaDeviceRuntimeError::driver("attention stable KV packing", e)
                            })?;
                            call.execute(stream)?;
                        }
                    }
                }
                Ok(())
            },
        )
        .and_then(|command| command.with_tensor_work(elements, operations, calls, 0))
        .map_err(|e| e.to_string())
    }
}
