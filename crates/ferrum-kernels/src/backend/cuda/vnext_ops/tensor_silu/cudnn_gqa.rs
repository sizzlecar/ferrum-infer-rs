//! Compact all-valid causal GQA on the shared cuDNN owner/cache.
use super::cudnn_attention::layout::{ScratchLayout, SCRATCH_ALIGNMENT};
use super::*;
use crate::backend::cuda::vnext_runtime::{
    CudaCudnnSdpaCache, CudnnSdpaBindings, CudnnSdpaPlan, CudnnSdpaSignature,
};
use ferrum_runtime::*;

pub const NUMERICAL_PROFILE_ID: &str = "cuda.cudnn920.structured_gqa.causal.bf16.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "compact-BSHD-QH/KVH;Hq-multiple-Hkv;D128;B1..16;all-valid-single-causal-segment;",
    "top-left-inclusive-causal;no-dropout;F32-scale;BF16-QKVY;",
    "cuDNN9.20-virtual-QK-scale-GEN_INDEX-GE-SELECT-softmax-PV-pattern;",
    "F32-compute-BF16-virtual-probability;vendor-intermediate-rounding;",
    "no-sequential-or-Torch-bitwise-claim;finite-visible-input-domain;",
    "exact-workspace-admission;shared-owner-cache-binding-quota;NoAlias;no-KV-expansion"
);
const PROVIDER: &str = "provider.cuda.structured_gqa_attention_fused.bf16.cudnn920";
const ESTIMATOR: &str = "resource-estimator.cuda.structured_gqa_attention_fused.bf16.cudnn920";

pub(super) fn provider(
    device: &DeviceDescriptor,
    cache: Arc<CudaCudnnSdpaCache>,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(STRUCTURED_GQA_ATTENTION_FUSED_BF16_CAPABILITY_ID)
        .map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks fused GQA capability",
        ));
    }
    let contract = structured_gqa_attention_fused_bf16_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        cache.fingerprint().as_bytes(),
        include_bytes!("cudnn_gqa.rs"),
        include_bytes!("../../vnext_runtime/cudnn/sdpa/native/causal.rs"),
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
    Ok(Box::new(Provider { descriptor, cache }))
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    cache: Arc<CudaCudnnSdpaCache>,
}
struct Shape {
    geometry: StructuredGqaAttentionBf16Geometry,
    signature: CudnnSdpaSignature,
    q_bytes: u64,
    kv_bytes: u64,
    operations: u64,
}
impl Shape {
    fn new(
        bindings: &[ResolvedValueBinding],
        attrs: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, String> {
        if bindings.len() != 4 {
            return Err("fused GQA requires exact Q/K/V/Y ports".into());
        }
        let g = StructuredGqaAttentionBf16Geometry::new(
            binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
            binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
            binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
            binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
            attrs,
        )
        .map_err(|e| e.to_string())?;
        let length = g.sequence_length() as u32;
        if g.head_dim() != 128
            || g.batch() > 16
            || g.mask().segment_ends() != [length]
            || g.mask().segment_bidirectional() != [false]
            || (0..g.batch() as usize).any(|b| {
                g.mask().key_run_ends(b) != Some(&[length][..])
                    || g.mask().key_run_valid(b) != Some(&[true][..])
            })
        {
            return Err("cuDNN GQA requires D128, B1..16 and one all-valid causal segment; padding/holes/bidirectional masks need another explicit profile".into());
        }
        let checked = |n: u64| i64::try_from(n).map_err(|_| "GQA stride exceeds i64".to_owned());
        let signature = CudnnSdpaSignature {
            batch: checked(g.batch())?,
            heads: checked(g.query_heads())?,
            key_value_heads: checked(g.key_value_heads())?,
            causal: true,
            query_length: i64::from(length),
            key_value_length: i64::from(length),
            head_dim: 128,
            query_batch_stride: checked(
                g.sequence_length()
                    .checked_mul(g.query_columns())
                    .ok_or("Q stride overflow")?,
            )?,
            key_value_batch_stride: checked(
                g.sequence_length()
                    .checked_mul(g.key_value_columns())
                    .ok_or("KV stride overflow")?,
            )?,
            output_batch_stride: checked(
                g.sequence_length()
                    .checked_mul(g.query_columns())
                    .ok_or("Y stride overflow")?,
            )?,
            scale_bits: g.effective_scale().to_bits(),
        };
        signature.validate()?;
        let q_bytes = g.element_count().checked_mul(2).ok_or("Q bytes overflow")?;
        let kv_bytes = g
            .rows()
            .checked_mul(g.key_value_columns())
            .and_then(|n| n.checked_mul(2))
            .ok_or("KV bytes overflow")?;
        // Mathematical QK/scale, full softmax and PV count, not GPU instructions.
        let operations = g
            .visible_pairs()
            .checked_mul(4 * g.head_dim() + 4)
            .and_then(|n| n.checked_sub(g.nonempty_rows() * (g.head_dim() + 1)))
            .and_then(|n| n.checked_mul(g.query_heads()))
            .ok_or("GQA logical work overflow")?;
        Ok(Self {
            geometry: g,
            signature,
            q_bytes,
            kv_bytes,
            operations,
        })
    }
}
impl Provider {
    fn plan(
        &self,
        s: &Shape,
        prepare: bool,
    ) -> Result<(Arc<CudnnSdpaPlan>, ScratchLayout), String> {
        let plan = if prepare {
            self.cache.prepare(s.signature)
        } else {
            self.cache.lookup(&s.signature)
        }
        .map_err(|e| e.to_string())?;
        let scratch =
            ScratchLayout::without_payload(s.geometry.batch() as u32, plan.workspace_bytes())?;
        Ok((plan, scratch))
    }
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        r: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if r.operation().id != *self.descriptor.operation_id()
            || r.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan("fused GQA estimator operation differs"));
        }
        let shape = Shape::new(r.values(), r.attributes()).map_err(invalid_plan)?;
        let (plan, scratch) = self.plan(&shape, true).map_err(invalid_plan)?;
        let workspace = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(scratch.bytes)?,
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
            r.input_fingerprint(),
            SCRATCH_ALIGNMENT,
            Some(workspace),
            None,
        )
        .with_prepared_implementation_fingerprint(plan.fingerprint().to_owned())?
        .with_prepared_implementation_retention(plan)
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
        i: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            i.participants()[0].identity().clone(),
            PROVIDER,
            "fixed tensor work required".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        self.encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| provider_failure(identity, PROVIDER, e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("fused GQA requires one physical tensor participant".into());
        };
        if i.operation().id.as_str() != STRUCTURED_GQA_ATTENTION_FUSED_BF16_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("fused GQA requires its fixed contract".into());
        }
        let s = Shape::new(p.bindings(), p.attributes())?;
        let (plan, scratch) = self.plan(&s, false)?;
        if i.prepared_implementation_fingerprint() != Some(plan.fingerprint()) {
            return Err("GQA native plan differs from admitted plan".into());
        }
        let mut regions = Vec::with_capacity(5);
        for (role, ordinal, bytes) in [
            (ResolvedValueRole::Input, 0, s.q_bytes),
            (ResolvedValueRole::Input, 1, s.kv_bytes),
            (ResolvedValueRole::Input, 2, s.kv_bytes),
            (ResolvedValueRole::Output, 0, s.q_bytes),
        ] {
            regions.push(retained_region(
                p,
                binding(p.bindings(), role, ordinal)?,
                bytes,
            )?);
        }
        for region in &regions[..3] {
            nonoverlapping_ranges(
                region.device_ptr(),
                region.length_bytes(),
                regions[3].device_ptr(),
                regions[3].length_bytes(),
            )?;
        }
        let view = p.scratch_view().ok_or("GQA scratch not admitted")?;
        let translated = view
            .translate(0, scratch.bytes)
            .map_err(|e| e.to_string())?;
        let mut iter = translated.iter();
        let piece = iter.next().ok_or("empty GQA scratch")?;
        if iter.next().is_some() {
            return Err("GQA scratch must be contiguous".into());
        }
        let (buffer, range, retention) = piece.buffer_and_physical_range();
        let region = buffer
            .retained_region(range, retention)
            .map_err(|e| e.to_string())?;
        if region.element_type() != ElementType::U8
            || region.length_bytes() != scratch.bytes
            || region.device_ptr() % SCRATCH_ALIGNMENT != 0
        {
            return Err("GQA scratch dtype/extent/alignment differs".into());
        }
        for r in &regions {
            nonoverlapping_ranges(
                r.device_ptr(),
                r.length_bytes(),
                region.device_ptr(),
                region.length_bytes(),
            )?;
        }
        let base = region.device_ptr();
        let address = |offset| {
            base.checked_add(offset)
                .ok_or("GQA scratch address overflow")
        };
        let execution = plan
            .bind(CudnnSdpaBindings {
                q: regions[0].device_ptr(),
                k: regions[1].device_ptr(),
                v: regions[2].device_ptr(),
                o: regions[3].device_ptr(),
                seq_len_q: address(scratch.query_lengths_offset)?,
                seq_len_kv: address(scratch.key_lengths_offset)?,
                workspace: if scratch.workspace_bytes == 0 {
                    0
                } else {
                    address(scratch.workspace_offset)?
                },
                workspace_bytes: scratch.workspace_bytes,
            })
            .map_err(|e| e.to_string())?;
        regions.push(region);
        CudaDeviceCommand::operation("vnext_cudnn_causal_gqa_bf16", regions, move |stream, _| {
            execution.execute(stream)
        })
        .and_then(|c| c.with_tensor_work(s.geometry.element_count(), s.operations, 1, 0))
        .map_err(|e| e.to_string())
    }
}
