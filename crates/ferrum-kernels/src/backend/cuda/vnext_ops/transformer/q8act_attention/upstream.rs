//! MarkerV2 attention projections: cold wave proofs and retained validation leases.
use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::{
    upstream_linear::{
        PreparedUpstreamLeaf, ProjectionPreparation, UpstreamPlanFactory, WeightValidation,
        WeightValidationRegistry,
    },
    CudaNativeBlockKernels,
};
use crate::backend::cuda::vnext_runtime::CudaDeviceCommand;
use crate::native_ops::upstream_linear::DeviceSpan;
use cudarc::driver::{CudaContext, CudaStream};
use ferrum_interfaces::vnext::{
    CompositeNumericalArithmetic, DynamicStorageRequirement, ElementType,
    EncodedRetainedPlanDependency, OperationInvocation, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, UpstreamMarkerV2Profile,
    UpstreamProjectionLayout, UpstreamProjectionWaveFacts, UpstreamScratchEstimate,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub(in crate::backend::cuda::vnext_ops::transformer) struct Runtime {
    plans: UpstreamPlanFactory,
    profile: UpstreamMarkerV2Profile,
    arithmetic: CompositeNumericalArithmetic,
    validation: WeightValidationRegistry,
    fingerprint: String,
    scratch: Mutex<BTreeMap<String, UpstreamScratchEstimate>>,
}
impl Runtime {
    pub fn new(
        context: &Arc<CudaContext>,
        fingerprint: &str,
        profile: UpstreamMarkerV2Profile,
    ) -> Result<Arc<Self>, CudaDeviceRuntimeError> {
        Ok(Arc::new(Self {
            profile,
            arithmetic: profile.arithmetic(),
            plans: UpstreamPlanFactory::new(context, fingerprint.to_owned())
                .map_err(CudaDeviceRuntimeError::contract)?,
            validation: WeightValidationRegistry::default(),
            fingerprint: fingerprint.to_owned(),
            scratch: Mutex::new(BTreeMap::new()),
        }))
    }
    /// Only new-hybrid, provably small complete waves may reuse the original
    /// binding-only encoder. The core still checks the empty dependency seal.
    pub fn g32_binding_only(
        &self,
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<bool, String> {
        if !self.profile.hybrid() {
            return Ok(false);
        }
        if invocation.operation().id.as_str() != self.profile.operation_id() {
            return Err("hybrid binding operation identity mismatch".into());
        }
        if invocation.participant_token_ranges().len() != invocation.participants().len() {
            return Err("hybrid binding participant coverage mismatch".into());
        }
        if !ferrum_interfaces::vnext::G32MmqPrefillPolicy::g32_partition(
            invocation.work_shape().immediate_tokens(),
            invocation
                .participant_token_ranges()
                .iter()
                .map(|r| r.immediate_token_range()),
        )? {
            return Ok(false);
        }
        let first = invocation
            .participants()
            .first()
            .ok_or("empty hybrid binding wave")?;
        let numerics = first
            .projection_numerics()
            .ok_or("missing retained hybrid numerics")?;
        if numerics.contract() != &self.arithmetic
            || invocation
                .participants()
                .iter()
                .any(|p| p.projection_numerics() != Some(numerics))
        {
            return Err("hybrid binding retained contract mismatch".into());
        }
        // Reuse the actual Plan Preserve/U8/prefix/shared-storage validation;
        // G32 does not read these flags, but their admission cannot be bypassed.
        let _retained_flags = persistent_region(invocation, flag_bytes(numerics)?)?;
        Ok(true)
    }
    fn scratch_bytes(
        &self,
        numerics: &PreparedProjectionNumerics,
    ) -> Result<UpstreamScratchEstimate, String> {
        let mut cache = self
            .scratch
            .lock()
            .map_err(|_| "attention upstream scratch cache poisoned")?;
        if let Some(&bytes) = cache.get(numerics.fingerprint()) {
            return Ok(bytes);
        }
        let bytes = self.plans.maximum_scratch(numerics)?;
        cache.insert(numerics.fingerprint().to_owned(), bytes);
        Ok(bytes)
    }
}
#[derive(Clone)]
pub(super) struct State {
    runtime: Arc<Runtime>,
    pub fixed_bytes: u64,
    launches: Vec<Projection>,
    preparation: ProjectionPreparation,
}
impl State {
    pub(super) fn validate_parts(
        &self,
        projection: &PreparedProjection,
        parts: &[weights::MatrixPart],
    ) -> Result<(), String> {
        crate::backend::cuda::vnext_ops::native_blocks::upstream_linear::validate_parts(
            self.runtime.profile,
            projection,
            parts,
        )
    }

    pub(super) fn uses_g32(&self, rows: u32) -> bool {
        self.runtime.profile.uses_g32(rows)
    }
}
#[derive(Clone)]
struct Projection {
    role: ProjectionRole,
    rows: u32,
    input: CudaBufferRegion,
    output: CudaBufferRegion,
    leaves: Vec<Leaf>,
    extra_dispatches: u64,
    transfers: u64,
}
#[derive(Clone)]
struct Leaf {
    part: weights::MatrixPart,
    weight: CudaBufferRegion,
    signs: Option<CudaBufferRegion>,
    stage: Arc<PreparedUpstreamLeaf>,
    validation: Option<Arc<WeightValidation>>,
}
impl<'a> PreparedAttentionProjections<'a> {
    pub fn prepare_upstream(
        kind: Q8ActAttentionProfile,
        values: &[ResolvedValueBinding],
        runtime: Arc<Runtime>,
    ) -> Result<Self, String> {
        if runtime.profile.strict_operation_id()
            != kind.arithmetic().strict_base.operation_id.as_str()
        {
            return Err("upstream attention kind/profile mismatch".into());
        }
        let numerics = PreparedProjectionNumerics::prepare(&runtime.arithmetic, values)?;
        let result = Self::from_upstream_numerics(
            kind,
            Cow::Owned(numerics),
            runtime,
            ProjectionPreparation::Full,
        )?;
        for projection in result.numerics.projections() {
            let value = binding(
                values,
                ResolvedValueRole::Input,
                projection.weight_input_ordinal(),
            )?;
            let parts = weights::matrix_parts(
                value.weight().ok_or("attention weight missing")?,
                value.tensor().dimensions(),
            )?;
            result.validate_parts(projection.role(), &parts)?;
        }
        Ok(result)
    }
    fn from_upstream_numerics(
        kind: Q8ActAttentionProfile,
        numerics: Cow<'a, PreparedProjectionNumerics>,
        runtime: Arc<Runtime>,
        preparation: ProjectionPreparation,
    ) -> Result<Self, String> {
        let mut estimate = runtime.scratch_bytes(&numerics)?;
        if runtime.profile.hybrid() {
            estimate.bytes_per_row = estimate
                .bytes_per_row
                .max(q8act::workspace_per_token(&numerics)?);
        }
        Ok(Self {
            profile: kind,
            numerics,
            bytes_per_token: estimate.bytes_per_row,
            upstream: Some(State {
                runtime,
                fixed_bytes: estimate.fixed_bytes,
                launches: Vec::new(),
                preparation,
            }),
        })
    }
    pub fn from_upstream_invocation(
        kind: Q8ActAttentionProfile,
        invocation: &'a BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        runtime: Arc<Runtime>,
        preparation: ProjectionPreparation,
    ) -> Result<Self, String> {
        let numerics = invocation
            .participants()
            .first()
            .ok_or("empty upstream attention")?
            .projection_numerics()
            .ok_or("missing retained upstream attention numerics")?;
        if numerics.contract() != &runtime.arithmetic
            || invocation
                .participants()
                .iter()
                .any(|p| p.projection_numerics() != Some(numerics))
        {
            return Err("upstream attention participants differ from the explicit contract".into());
        }
        Self::from_upstream_numerics(kind, preparation.numerics(numerics), runtime, preparation)
    }
    pub fn upstream_persistent_requirement(
        &self,
    ) -> Result<Option<ProviderWorkspaceRequirement>, String> {
        if !self.is_upstream() {
            return Ok(None);
        }
        ProviderWorkspaceRequirement::new(
            flag_bytes(&self.numerics)?,
            16,
            ProviderWorkspaceScope::Plan,
            ProviderWorkspaceReusePolicy::Preserve,
            DynamicStorageRequirement::contiguous(),
        )
        .map(Some)
        .map_err(|e| e.to_string())
    }
    /// Resolve actual allocation offsets before capture. All local calls reuse
    /// the same attention scratch prefix; row widths may differ between calls.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_upstream_projection(
        &mut self,
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        role: ProjectionRole,
        parts: &[weights::MatrixPart],
        weight_regions: &[CudaBufferRegion],
        scratch: &CudaBufferRegion,
        input_offset: u64,
        output_offset: u64,
        rows: u32,
    ) -> Result<(), String> {
        self.validate_parts(role, parts)?;
        let projection = self.projection(role)?;
        let k = projection.input_features();
        let n = projection.output_features();
        let first_leaf = self
            .numerics
            .projections()
            .iter()
            .take_while(|p| p.role() != role)
            .map(|p| p.leaves().len() as u64)
            .sum::<u64>();
        let input = scratch
            .subregion(input_offset, mul(mul(u64::from(rows), k)?, 2)?)
            .map_err(|e| e.to_string())?;
        let output = scratch
            .subregion(output_offset, mul(mul(u64::from(rows), n)?, 2)?)
            .map_err(|e| e.to_string())?;
        let persistent = persistent_region(invocation, flag_bytes(&self.numerics)?)?;
        let state = self
            .upstream
            .as_mut()
            .ok_or("not an upstream attention plan")?;
        if state
            .launches
            .iter()
            .any(|p| p.role == role && p.rows == rows)
        {
            return Ok(());
        }
        let mut leaves = Vec::new();
        let mut extra_dispatches = 0;
        let mut transfers = 0;
        for (i, part) in parts.iter().enumerate() {
            let weight = weight_regions
                .get(i)
                .ok_or("missing physical upstream attention leaf")?
                .clone();
            let facts = UpstreamProjectionWaveFacts {
                role,
                component_id: part.component_id.clone(),
                local_rows: rows,
                layout: UpstreamProjectionLayout::Columns,
                input_stride: k,
                output_stride: n,
                input_byte_offset: input.backing_byte_offset(),
                output_byte_offset: output.backing_byte_offset(),
                weight_byte_offset: weight.backing_byte_offset(),
                input_available_bytes: input.length_bytes(),
                output_available_bytes: output.length_bytes(),
                weight_available_bytes: weight.length_bytes(),
                retained_zero_padded_weight_rows: u64::from(part.rows),
            };
            let stage =
                state
                    .runtime
                    .plans
                    .prepare_for(&self.numerics, &facts, state.preparation)?;
            let validation = if let Some(native) = &stage.native {
                let flag = persistent
                    .subregion(
                        flag_offset(first_leaf, i as u64, native.geometry().algorithm)?,
                        4,
                    )
                    .map_err(|e| e.to_string())?;
                let validation = state
                    .runtime
                    .validation
                    .prepare(
                        native.clone(),
                        &state.runtime.fingerprint,
                        weight.clone(),
                        flag,
                    )
                    .map_err(|e| e.to_string())?;
                // Replace one strict leaf launch by convert+pack+dot+cast(+fixup).
                extra_dispatches = add(
                    extra_dispatches,
                    3 + u64::from(native.geometry().fixup != 0),
                )?;
                transfers = add(
                    transfers,
                    1 + u64::from(native.geometry().guard_blocks != 0),
                )?;
                Some(validation)
            } else {
                None
            };
            let signs = part
                .signs_region
                .map(|i| {
                    weight_regions
                        .get(i)
                        .cloned()
                        .ok_or("missing attention transform signs")
                })
                .transpose()?;
            leaves.push(Leaf {
                part: part.clone(),
                weight,
                signs,
                stage,
                validation,
            });
        }
        state.launches.push(Projection {
            role,
            rows,
            input,
            output,
            leaves,
            extra_dispatches,
            transfers,
        });
        Ok(())
    }
    pub fn upstream_bindings(
        &self,
        invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
        mut segment: Option<&mut Vec<super::super::segment_bindings::ValidationRecipe>>,
    ) -> Result<Vec<EncodedRetainedPlanDependency<CudaDeviceCommand>>, String> {
        let mut bindings = Vec::new();
        for projection in self.upstream.iter().flat_map(|s| s.launches.iter()) {
            let retained = self.projection(projection.role)?;
            let first_leaf = self
                .numerics
                .projections()
                .iter()
                .take_while(|p| p.role() != projection.role)
                .map(|p| p.leaves().len() as u64)
                .sum();
            for (i, leaf) in projection.leaves.iter().enumerate() {
                if let Some(validation) = &leaf.validation {
                    let algorithm = leaf
                        .stage
                        .native
                        .as_ref()
                        .ok_or("validation without selected native leaf")?
                        .geometry()
                        .algorithm;
                    if let Some(segment) = segment.as_deref_mut() {
                        segment.push(validation.segment_declaration(
                            retained.weight_input_ordinal(),
                            &leaf.part.component_id,
                            flag_offset(first_leaf, i as u64, algorithm)?,
                        ));
                    }
                    bindings.push(
                        validation
                            .retained_dependency(
                                invocation,
                                retained.weight_input_ordinal(),
                                &leaf.part.component_id,
                                flag_offset(first_leaf, i as u64, algorithm)?,
                            )
                            .map_err(|e| e.to_string())?,
                    );
                }
            }
        }
        Ok(bindings)
    }
    pub fn upstream_work(&self, rows: u32) -> Result<(u64, u64), String> {
        self.upstream
            .iter()
            .flat_map(|s| s.launches.iter())
            .filter(|p| p.rows == rows)
            .try_fold((0, 0), |(d, t), p| {
                Ok((add(d, p.extra_dispatches)?, add(t, p.transfers)?))
            })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn launch_upstream(
        &self,
        native: &CudaNativeBlockKernels,
        stream: &CudaStream,
        role: ProjectionRole,
        input: u64,
        output: u64,
        rows: u32,
        stride: u32,
        scratch: &CudaBufferRegion,
        base_bytes: u64,
        transform: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let state = self
            .upstream
            .as_ref()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("missing upstream attention state"))?;
        let projection = state
            .launches
            .iter()
            .find(|p| p.role == role && p.rows == rows)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("unprepared attention wave"))?;
        if projection.input.device_ptr() != input || projection.output.device_ptr() != output {
            return Err(CudaDeviceRuntimeError::contract(
                "attention wave scratch identity changed after planning",
            ));
        }
        let (address, bytes) = self.workspace(scratch, base_bytes, u64::from(rows))?;
        let span = |region: &CudaBufferRegion| DeviceSpan {
            address: region.device_ptr(),
            bytes: region.length_bytes(),
        };
        for leaf in &projection.leaves {
            if let Some(validation) = &leaf.validation {
                // Captured State owns this Arc and its retained flag/weight leases.
                unsafe {
                    leaf.stage.launch(
                        stream,
                        span(&projection.input),
                        span(&leaf.weight),
                        span(&projection.output),
                        DeviceSpan { address, bytes },
                        validation.flag_span(),
                    )?;
                }
            } else {
                super::super::native_matrix::launch_parts(
                    stream,
                    native,
                    std::iter::once((
                        &leaf.part,
                        leaf.weight.device_ptr(),
                        leaf.signs.as_ref().map_or(0, CudaBufferRegion::device_ptr),
                    )),
                    input,
                    output,
                    i32::try_from(rows).map_err(|_| {
                        CudaDeviceRuntimeError::contract("attention rows exceed i32")
                    })?,
                    i32::try_from(stride).map_err(|_| {
                        CudaDeviceRuntimeError::contract("attention stride exceeds i32")
                    })?,
                    i32::try_from(leaf.part.columns)
                        .map_err(|_| CudaDeviceRuntimeError::contract("attention K exceeds i32"))?,
                    transform,
                )?;
            }
        }
        Ok(())
    }
}
fn flag_offset(first_leaf: u64, index: u64, algorithm: u32) -> Result<u64, String> {
    let bank = match algorithm {
        1 => 0,
        2 => 4,
        _ => return Err("unknown upstream validation arithmetic bank".into()),
    };
    add(mul(add(first_leaf, index)?, 8)?, bank)
}

fn flag_bytes(numerics: &PreparedProjectionNumerics) -> Result<u64, String> {
    numerics
        .projections()
        .iter()
        .try_fold(0u64, |n, p| add(n, p.leaves().len() as u64))
        .and_then(|n| mul(n, 8))
}
fn persistent_region(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let resolve = |p: &OperationInvocation<'_, CudaDeviceBuffer>| {
        let view = p
            .persistent_view()
            .ok_or("upstream attention lacks retained validation storage")?;
        let descriptor = view.descriptor();
        if descriptor.element_type != ElementType::U8 || descriptor.size_bytes < bytes {
            return Err(format!(
                "upstream attention persistent storage requires {bytes} U8 bytes, got {} bytes of {:?}",
                descriptor.size_bytes, descriptor.element_type
            ));
        }
        // Admission may round storage up for alignment. Retain only the flag prefix;
        // trailing allocation padding does not belong to any validation flag.
        let parts = view.translate(0, bytes).map_err(|e| e.to_string())?;
        let mut parts = parts.iter();
        let part = parts.next().ok_or("empty validation storage")?;
        if parts.next().is_some() {
            return Err("validation storage is not contiguous".into());
        }
        let (buffer, range, retention) = part.buffer_and_physical_range();
        buffer
            .retained_region(range, retention)
            .map_err(|e| e.to_string())
    };
    let first = resolve(&invocation.participants()[0])?;
    for p in &invocation.participants()[1..] {
        if !super::super::same_physical_region(&first, &resolve(p)?) {
            return Err("upstream attention validation is not plan-shared".into());
        }
    }
    Ok(first)
}
fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| "upstream attention extent overflows".into())
}
fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "upstream attention extent overflows".into())
}

impl State {
    pub(super) fn append_replay_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), String> {
        for projection in &self.launches {
            for leaf in &projection.leaves {
                let fingerprint = leaf.stage.replay_fingerprint()?;
                bytes.extend_from_slice(&(fingerprint.len() as u64).to_le_bytes());
                bytes.extend_from_slice(fingerprint.as_bytes());
            }
        }
        Ok(())
    }
}

/// Native ABI/source and lease protocol are part of the provider identity.
pub(in crate::backend::cuda::vnext_ops::transformer) fn fingerprint_sources() -> Vec<&'static [u8]>
{
    vec![
        include_bytes!("upstream.rs"),
        include_bytes!("../../native_blocks/upstream_linear.rs"),
        include_bytes!("../../native_blocks/upstream_linear/preparation.rs"),
        include_bytes!("../../native_blocks/upstream_linear/native_plan.rs"),
        include_bytes!("../../../../../native_ops/upstream_q6_f16_linear.rs"),
        include_bytes!("../../../../../native_ops/upstream_q6_f16_linear/ffi.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ferrum-native-ops/src/upstream_q6_f16_linear.rs"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-q6-f32-linear/abi.h"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-q6-f32-linear/mmq.cu"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-q6-f32-linear/f16_adapter.cuh"
        )),
        include_bytes!("../../native_blocks/upstream_linear/weight_validation.rs"),
        include_bytes!("../../../../../native_ops/upstream_linear.rs"),
        include_bytes!("../../../../../native_ops/upstream_linear/ffi.rs"),
        include_bytes!("../../../../../native_ops/upstream_linear/dispatch.rs"),
        include_bytes!("../../../../../native_ops/upstream_linear/extra_ffi.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ferrum-native-ops/src/upstream_extra_linear.rs"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../ferrum-native-ops/src/upstream_extra_linear/prefill.rs"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-extra-linear/mmq.cu"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-extra-linear/mmvq.cu"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-extra-linear/marker.cuh"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-extra-linear/format.h"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-extra-linear/abi.h"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-extra-linear/boundary.h"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-extra-linear/boundary.cu"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-linear/mmq.cu"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-linear/mmvq.cu"
        )),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../native-operators/cuda/upstream-linear/marker.cuh"
        )),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upstream_attention_validation_banks_are_disjoint_and_checked() {
        for leaf in [0, 1, 7, u32::MAX as u64] {
            let mmq = flag_offset(leaf, 0, 1).unwrap();
            let mmvq = flag_offset(leaf, 0, 2).unwrap();
            assert_eq!(mmq % 8, 0);
            assert_eq!(mmq + 4, mmvq);
            assert_eq!(mmvq + 4, flag_offset(leaf, 1, 1).unwrap());
            assert_eq!(
                flag_offset(leaf, 1, 2).unwrap(),
                flag_offset(leaf + 1, 0, 2).unwrap()
            );
        }
        assert!(flag_offset(0, 0, 0).is_err());
        assert!(flag_offset(u64::MAX, 1, 1).is_err());
        assert!(flag_offset(u64::MAX / 8 + 1, 0, 2).is_err());
    }
}
