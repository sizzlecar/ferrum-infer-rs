//! Complete ordered FP16-KV attention work for native projections and the
//! explicit RN-F16 cuBLAS API. Other opaque ABIs and INT8 readback stay unknown. Dynamic context contributes
//! numeric work; only real fixed launch geometry/scalars enter the replay seal.
use super::super::cublas_api::{CublasHandleApiIdentity, GemmF16ApiPlan};
use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::selected as native_selected;
use crate::backend::cuda::vnext_runtime::selected_cost;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1, StatisticalTransferKindV1,
};
use ferrum_types::SloStructuredCostCapture;
use std::sync::OnceLock;

mod template;
use template::{CostBuilder, Query};
pub(super) use template::{CostTemplate, Geometry};

/// The actual projection ABI, never inferred from a cost-table success.
/// DenseF16 is created only after the explicit RN consumer has validated its
/// complete materialized operands and the runtime has frozen a live handle.
#[derive(Clone, Copy)]
pub(super) enum ProjectionWork<'a> {
    Native([&'a [weights::MatrixPart]; 4]),
    DenseF16(CublasHandleApiIdentity),
}
impl ProjectionWork<'_> {
    fn append(
        self,
        builder: &mut CostBuilder<'_>,
        index: usize,
        rows: u64,
        output: u64,
        input: u64,
        scratch: u64,
    ) -> Option<()> {
        match self {
            Self::Native(parts) => projection_work(builder, parts[index], rows, output, scratch),
            Self::DenseF16(identity) => builder.gemm(
                index,
                checked_i32(rows, "causal GEMM rows").ok()?,
                checked_i32(output, "causal GEMM output").ok()?,
                checked_i32(input, "causal GEMM input").ok()?,
                identity,
            ),
        }
    }
    fn extra_dispatches(self, rows: u64) -> Option<u64> {
        match self {
            Self::Native(parts) => parts.iter().try_fold(0_u64, |sum, part| {
                sum.checked_add(native_matrix_extra_dispatches(part, rows).ok()?)
            }),
            Self::DenseF16(_) => Some(0),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Row {
    pub tokens: u64,
    pub sequence: u64,
    pub inplace: bool,
    pub(super) path: CausalAttentionKernelPath,
    pub(super) envelope: CausalAttentionReplayEnvelope,
}
impl Row {
    pub(super) fn new(
        shape: CausalAttentionShape,
        policy: AttentionExecutionPolicy,
        tokens: u64,
        sequence: u64,
        inplace: bool,
    ) -> Option<Self> {
        if tokens == 0 || sequence < tokens || sequence > shape.maximum_context_tokens {
            return None;
        }
        checked_i32(tokens, "selected causal tokens").ok()?;
        checked_i32(sequence, "selected causal context").ok()?;
        let path = CausalAttentionKernelPath::select(policy, shape, tokens, sequence).ok()?;
        let envelope = CausalAttentionReplayEnvelope::new(shape, path, sequence).ok()?;
        Some(Self {
            tokens,
            sequence,
            inplace,
            path,
            envelope,
        })
    }
}
fn class(entry: &str, block: [u32; 3]) -> Option<SelectedAlgorithmClassV1> {
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let numerical = *NUMERICAL.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(b"cuda.causal.actual-native-arithmetic.v1");
        for source in [
            include_bytes!("../causal_attention.rs").as_slice(),
            include_bytes!("precision.rs").as_slice(),
            include_bytes!("batch_decode.rs").as_slice(),
            include_bytes!("launch_geometry.rs").as_slice(),
            include_bytes!("selected.rs").as_slice(),
            include_bytes!("selected/template.rs").as_slice(),
        ] {
            hash.update(source);
        }
        for ptx in [
            crate::ptx::VNEXT_CAUSAL_ATTENTION,
            crate::ptx::PAGED_VARLEN_ATTENTION_VLLM,
            crate::ptx::QK_NORM_ROPE,
            crate::ptx::RMS_NORM,
            crate::ptx::VNEXT_GGUF,
            crate::ptx::RESIDUAL_ADD,
        ] {
            hash.update(ptx.as_bytes());
        }
        #[cfg(feature = "vllm-paged-attn-v2")]
        {
            hash.update(crate::native_ops::CUDA_NATIVE_SOURCE_BUNDLE_ID.as_bytes());
            hash.update(include_bytes!("../../../vllm_paged_attn.rs"));
        }
        hash.finalize().into()
    });
    let mut layout = Sha256::new();
    layout.update(b"cuda.causal.native-abi.geometry.v1");
    for dim in block {
        layout.update(dim.to_le_bytes());
    }
    SelectedAlgorithmClassV1::new(entry, 1, numerical, layout.finalize().into()).ok()
}
fn kernel(
    b: &mut CostBuilder<'_>,
    entry: &str,
    config: LaunchConfig,
    logical: u64,
    inner: u64,
    scratch: u64,
    fixed: &[u64],
) -> Option<()> {
    let block = [config.block_dim.0, config.block_dim.1, config.block_dim.2];
    let algorithm = b.class(entry, block)?;
    b.kernel_with_replay_geometry(
        algorithm,
        KernelNumericWorkV1 {
            logical_units: logical,
            padded_units: logical,
            inner_units_per_logical_unit: inner,
            grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
            scratch_bytes: scratch,
            staged_weight_bytes: 0,
        },
        KernelReplayGeometryV1 {
            block,
            dynamic_shared_bytes: u64::from(config.shared_mem_bytes),
            fixed_parameters: fixed,
        },
    )
    .ok()
}
pub(in crate::backend::cuda::vnext_ops::transformer) fn binding_evidence(
    bytes: impl IntoIterator<Item = u64>,
    tokens: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut b = selected_cost::builder(capture, tokens)?;
    // This entry/block is independent of node, rows, context and addresses.
    // Preserve the same class digest while rebuilding every transfer's bytes.
    static CONTROL_CLASS: OnceLock<Option<SelectedAlgorithmClassV1>> = OnceLock::new();
    let algorithm =
        (*CONTROL_CLASS.get_or_init(|| class("cuda.cuMemcpyHtoDAsync.causal-control", [1, 1, 1])))?;
    for bytes in bytes {
        b.transfer(algorithm, StatisticalTransferKindV1::HostToDevice, bytes)
            .ok()?;
    }
    b.finish().ok()
}
pub(super) fn attach(
    command: ferrum_interfaces::vnext::OperationCostCommand,
    evidence: Option<SelectedCommandCostEvidenceV1>,
) -> ferrum_interfaces::vnext::OperationCostCommand {
    match evidence {
        Some(evidence) => command
            .clone()
            .with_statistical_evidence(evidence)
            .unwrap_or(command),
        None => command,
    }
}

#[test]
fn prepared_causal_control_class_keeps_each_transfer_geometry_and_legacy_evidence() {
    let capture = SloStructuredCostCapture::HostSettledV1;
    for sizes in [vec![32], vec![32, 48, 64]] {
        let mut original = selected_cost::builder(capture, sizes.len() as u64).unwrap();
        for &bytes in &sizes {
            original
                .transfer(
                    class("cuda.cuMemcpyHtoDAsync.causal-control", [1, 1, 1]).unwrap(),
                    StatisticalTransferKindV1::HostToDevice,
                    bytes,
                )
                .unwrap();
        }
        let original = original.finish().unwrap();
        let prepared =
            binding_evidence(sizes.iter().copied(), sizes.len() as u64, capture).unwrap();
        assert_eq!(prepared, original);
        assert_eq!(prepared.algorithm_work(), original.algorithm_work());
        assert_eq!(
            prepared.work().host_to_device_bytes,
            sizes.iter().sum::<u64>()
        );
    }
}
pub(super) fn actual(
    p: &prepared::Prepared,
    policy: AttentionExecutionPolicy,
    precision: CausalPrecision,
    capture: SloStructuredCostCapture,
    library_identity: Option<CublasHandleApiIdentity>,
) -> Option<SelectedCommandCostEvidenceV1> {
    if capture == SloStructuredCostCapture::Disabled {
        return None;
    }
    fn native(weight: &SharedProjectionWeight) -> Option<&[weights::MatrixPart]> {
        match weight {
            SharedProjectionWeight::Native(matrix) => Some(&matrix.parts),
            _ => None,
        }
    }
    let projections = match p.projection {
        CausalProjection::Native { .. } => ProjectionWork::Native([
            native(&p.shared.query_weight)?,
            native(&p.shared.key_weight)?,
            native(&p.shared.value_weight)?,
            native(&p.shared.output_weight)?,
        ]),
        CausalProjection::F16 => {
            if [
                &p.shared.query_weight,
                &p.shared.key_weight,
                &p.shared.value_weight,
                &p.shared.output_weight,
            ]
            .iter()
            .any(|weight| !matches!(weight, SharedProjectionWeight::F16 { .. }))
            {
                return None;
            }
            ProjectionWork::DenseF16(library_identity?)
        }
        #[cfg(feature = "vllm-marlin")]
        _ => return None,
    };
    let rows = p
        .launches
        .iter()
        .map(|launch| {
            let row = Row::new(
                p.shape,
                policy,
                launch.tokens,
                launch.sequence_tokens,
                p.compute_regions[launch.input_region].device_ptr()
                    == p.compute_regions[launch.output_region].device_ptr(),
            )?;
            (row.path == launch.path && row.envelope == launch.replay_topology.envelope())
                .then_some(row)
        })
        .collect::<Option<Vec<_>>>()?;
    compute(
        p.shape,
        precision,
        p.projection,
        policy,
        projections,
        &rows,
        p.total_tokens,
        p.packed.is_some(),
        capture,
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn compute(
    shape: CausalAttentionShape,
    precision: CausalPrecision,
    projection: CausalProjection,
    policy: AttentionExecutionPolicy,
    projections: ProjectionWork<'_>,
    rows: &[Row],
    tokens: u64,
    packed: bool,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    if capture.is_disabled() {
        return None;
    }
    Geometry::new(
        shape,
        precision,
        projection,
        policy,
        shape.cuda_shape().ok()?,
        tokens,
        rows.len(),
        None,
    )
    .ok()?
    .finish(rows, packed, projections)?
    .compute(capture)
}

fn compute_query(
    query: &Query<'_>,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut b = CostBuilder::new(query, capture)?;
    let Geometry {
        shape,
        precision,
        projection,
        cuda,
        layout,
        bindings,
        tokens,
        ..
    } = query.geometry;
    let rows = query.rows;
    let projections = query.projections;
    let packed = query.packed;
    let transform = projection.transform_bytes_per_token().checked_mul(tokens)?;
    let before = |b: &mut CostBuilder<'_>, n: u64| -> Option<()> {
        rms(b, shape, precision, n, layout.required_bytes)?;
        for (index, stride) in [
            shape.query_projection_features,
            shape.kv_features,
            shape.kv_features,
        ]
        .into_iter()
        .enumerate()
        {
            projections.append(b, index, n, stride, shape.hidden_size, transform)?;
        }
        Some(())
    };
    let after = |b: &mut CostBuilder<'_>, n: u64, inplace: bool| -> Option<()> {
        projections.append(b, 3, n, shape.hidden_size, shape.query_features, transform)?;
        if shape.post_attention_norm {
            rms(b, shape, precision, n, layout.required_bytes)?;
        }
        let elements = n.checked_mul(shape.hidden_size)?;
        checked_i32(elements, "selected residual elements").ok()?;
        kernel(
            b,
            if inplace {
                precision
                    .inplace_residual_kernel()
                    .unwrap_or(precision.residual_kernel())
            } else {
                precision.residual_kernel()
            },
            launch_geometry::flat(elements).ok()?,
            elements,
            1,
            layout.required_bytes,
            &[elements],
        )
    };
    if packed {
        // A packed invocation has one actual input/output span, so every row's
        // alias relation must describe the same physical choice.
        if rows.iter().any(|row| row.inplace != rows[0].inplace) {
            return None;
        }
        before(&mut b, tokens)?;
        if query.batch_decode {
            for row in rows {
                if row.tokens != 1 {
                    return None;
                }
                prepare(&mut b, shape, cuda, *row, None, layout.required_bytes)?;
            }
            let count = u32::try_from(rows.len()).ok()?;
            kernel(
                &mut b,
                "vnext_causal_gather_decode_lengths",
                LaunchConfig::for_num_elems(count),
                u64::from(count),
                1,
                layout.required_bytes,
                &[bindings.slot_bytes, u64::from(count)],
            )?;
            for (range, maximum) in &query.groups {
                let group = &rows[range.clone()];
                native_attention(
                    &mut b,
                    shape,
                    group,
                    *maximum,
                    bindings.slot_bytes.checked_div(POINTER_BYTES)?,
                    layout.required_bytes,
                )?;
            }
            if shape.output_gate {
                for row in rows {
                    gate(&mut b, shape, *row, layout.required_bytes)?;
                }
            }
        } else if rows.len() > 1
            && rows[0].path.is_fallback()
            && rows.iter().all(|row| row.path == rows[0].path)
        {
            let fallback = PackedFallbackLaunch {
                token_grid: rows.iter().map(|row| row.tokens).max()?,
                packed_token_grid: tokens,
                participant_grid: u32::try_from(rows.len()).ok()?,
                participant_count_i32: i32::try_from(rows.len()).ok()?,
                binding_slot_bytes: bindings.slot_bytes,
                path: rows[0].path,
            };
            prepare(
                &mut b,
                shape,
                cuda,
                rows[0],
                Some(fallback),
                layout.required_bytes,
            )?;
            fallback_attention(
                &mut b,
                shape,
                cuda,
                rows,
                Some(fallback),
                layout.required_bytes,
            )?;
            if shape.output_gate {
                for row in rows {
                    gate(&mut b, shape, *row, layout.required_bytes)?;
                }
            }
        } else {
            for row in rows {
                prepare(&mut b, shape, cuda, *row, None, layout.required_bytes)?;
                attention(&mut b, shape, cuda, *row, layout.required_bytes)?;
                if shape.output_gate {
                    gate(&mut b, shape, *row, layout.required_bytes)?;
                }
            }
        }
        after(&mut b, tokens, rows[0].inplace)?;
    } else {
        for row in rows {
            before(&mut b, row.tokens)?;
            prepare(&mut b, shape, cuda, *row, None, layout.required_bytes)?;
            attention(&mut b, shape, cuda, *row, layout.required_bytes)?;
            if shape.output_gate {
                gate(&mut b, shape, *row, layout.required_bytes)?;
            }
            after(&mut b, row.tokens, row.inplace)?;
        }
    }
    let evidence = b.finish()?;
    evidence
        .validate_command(tokens, query.dispatches, 0)
        .ok()?;
    Some(evidence)
}

/// Frozen CPU counterpart of the selected actual attention launches. Current
/// context changes only numerical work; the worker never reselects a path or
/// reconstructs a runtime/PSO/buffer view.
pub(in crate::backend::cuda::vnext_ops::transformer) struct Recipe {
    shape: CausalAttentionShape,
    precision: CausalPrecision,
    projection: CausalProjection,
    policy: AttentionExecutionPolicy,
    native: Option<[std::sync::Arc<[weights::MatrixPart]>; 4]>,
    library: Option<CublasHandleApiIdentity>,
    rows: Box<[Row]>,
    tokens: u64,
    packed: bool,
    retained: usize,
    _construction: ferrum_interfaces::vnext::DeviceObservationTemplateReservation,
}
impl Recipe {
    pub(in crate::backend::cuda::vnext_ops::transformer) fn library_identity(
        &self,
    ) -> Option<CublasHandleApiIdentity> {
        self.library
    }
    pub(super) fn from_prepared(
        prepared: &prepared::Prepared,
        policy: AttentionExecutionPolicy,
        precision: CausalPrecision,
        library: Option<CublasHandleApiIdentity>,
        budget: &std::sync::Arc<ferrum_interfaces::vnext::DeviceObservationTemplateBudget>,
    ) -> Option<Self> {
        let weights = [
            &prepared.shared.query_weight,
            &prepared.shared.key_weight,
            &prepared.shared.value_weight,
            &prepared.shared.output_weight,
        ];
        let mut copied = prepared
            .launches
            .len()
            .checked_mul(std::mem::size_of::<Row>())?
            .checked_add(
                4usize.checked_mul(std::mem::size_of::<std::sync::Arc<[weights::MatrixPart]>>())?,
            )?;
        for weight in weights {
            if let SharedProjectionWeight::Native(matrix) = weight {
                copied = copied
                    .checked_add(weights::retained_payload_bytes(&matrix.parts)?)?
                    .checked_add(2 * std::mem::size_of::<usize>())?;
            }
        }
        let construction = budget
            .reserve(std::mem::size_of::<Self>().checked_add(copied.checked_mul(2)?)?)
            .ok()?;
        let (native, library): (
            Option<[std::sync::Arc<[weights::MatrixPart]>; 4]>,
            Option<CublasHandleApiIdentity>,
        ) = match prepared.projection {
            CausalProjection::Native { .. } => {
                let mut parts = Vec::with_capacity(4);
                for weight in weights {
                    let SharedProjectionWeight::Native(matrix) = weight else {
                        return None;
                    };
                    parts.push(std::sync::Arc::clone(&matrix.parts));
                }
                (Some(parts.try_into().ok()?), None)
            }
            CausalProjection::F16
                if weights
                    .iter()
                    .all(|w| matches!(w, SharedProjectionWeight::F16 { .. })) =>
            {
                (None, Some(library?))
            }
            _ => return None,
        };
        let rows: Box<[_]> = prepared
            .launches
            .iter()
            .map(|launch| Row {
                tokens: launch.tokens,
                sequence: launch.sequence_tokens,
                inplace: prepared.compute_regions[launch.input_region].device_ptr()
                    == prepared.compute_regions[launch.output_region].device_ptr(),
                path: launch.path,
                envelope: launch.replay_topology.envelope(),
            })
            .collect();
        if rows.is_empty() {
            return None;
        }
        let mut retained = std::mem::size_of::<Self>()
            .checked_add(rows.len().checked_mul(std::mem::size_of::<Row>())?)?;
        if let Some(parts) = &native {
            for part in parts.iter() {
                retained = retained
                    .checked_add(weights::retained_payload_bytes(part.as_ref())?)?
                    .checked_add(2 * std::mem::size_of::<usize>())?;
            }
        }
        Some(Self {
            _construction: construction,
            shape: prepared.shape,
            precision,
            projection: prepared.projection,
            policy,
            native,
            library,
            rows,
            tokens: prepared.total_tokens,
            packed: prepared.packed.is_some(),
            retained,
        })
    }
    pub(in crate::backend::cuda::vnext_ops::transformer) fn retained_payload_bytes(&self) -> usize {
        self.retained
    }
    pub(in crate::backend::cuda::vnext_ops::transformer) fn projection_scratch_bytes(
        &self,
    ) -> Option<usize> {
        // Fresh numerical rows plus the optional packed path list. Both use
        // exact-size allocations; grouping itself is a borrowed iterator.
        self.rows.len().checked_mul(
            std::mem::size_of::<Row>()
                .checked_add(std::mem::size_of::<CausalAttentionKernelPath>())?,
        )
    }
    pub(in crate::backend::cuda::vnext_ops::transformer) fn project(
        &self,
        current: &ferrum_interfaces::vnext::FrozenObservationInput,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        if current.tokens() != self.tokens
            || current.participant_ranges().len() != self.rows.len()
            || current.source_ranges().len() != self.rows.len()
        {
            return None;
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.rows.len()).ok()?;
        for ((&captured, immediate), source) in self
            .rows
            .iter()
            .zip(current.participant_ranges())
            .zip(current.source_ranges())
        {
            let tokens = immediate.end.checked_sub(immediate.start)?;
            if tokens != captured.tokens
                || source.end.checked_sub(source.start) != Some(tokens)
                || source.end > self.shape.maximum_context_tokens
                || source.end > captured.envelope.sequence_capacity_tokens
            {
                return None;
            }
            rows.push(Row {
                sequence: source.end,
                ..captured
            });
        }

        let projections = match &self.native {
            Some(parts) => ProjectionWork::Native([&parts[0], &parts[1], &parts[2], &parts[3]]),
            None => ProjectionWork::DenseF16(self.library?),
        };
        compute(
            self.shape,
            self.precision,
            self.projection,
            self.policy,
            projections,
            &rows,
            self.tokens,
            self.packed,
            SloStructuredCostCapture::HostSettledV1,
        )
    }
}
fn projection_work(
    b: &mut CostBuilder<'_>,
    parts: &[weights::MatrixPart],
    rows: u64,
    stride: u64,
    scratch: u64,
) -> Option<()> {
    for part in parts {
        native_selected::append_transformed_linear(
            b,
            part,
            u32::try_from(rows).ok()?,
            u32::try_from(stride).ok()?,
            ElementType::F16,
            scratch,
        )?;
    }
    Some(())
}
fn rms(
    b: &mut CostBuilder<'_>,
    s: CausalAttentionShape,
    p: CausalPrecision,
    n: u64,
    scratch: u64,
) -> Option<()> {
    kernel(
        b,
        p.rms_kernel(),
        launch_geometry::rms(n, i32::try_from(s.hidden_size).ok()?).ok()?,
        n,
        s.hidden_size,
        scratch,
        &[s.hidden_size, u64::from(s.epsilon.to_bits())],
    )
}
fn prepare(
    b: &mut CostBuilder<'_>,
    s: CausalAttentionShape,
    c: CudaCausalAttentionShape,
    row: Row,
    packed: Option<PackedFallbackLaunch>,
    scratch: u64,
) -> Option<()> {
    let tokens = packed.map_or(row.tokens, |p| p.token_grid);
    let participants = packed.map_or(1, |p| p.participant_grid);
    let fixed = [
        VNEXT_KV_PAGE_BYTES / 2,
        u64::from(row.path.uses_vllm_layout()),
        s.query_heads,
        s.key_value_heads,
        s.head_dim,
        s.rope_dim,
        s.rope_frequency_denominator,
        s.rope_pair_offset,
        s.query_projection_features,
        s.head_dim.checked_mul(if s.output_gate { 2 } else { 1 })?,
        s.kv_features,
        u64::from(s.epsilon.to_bits()),
        u64::from(s.rope_theta.to_bits()),
        u64::from(s.rope_interleaved),
        u64::from(s.value_rms_norm),
        packed.map_or(0, |p| p.binding_slot_bytes),
    ];
    kernel(
        b,
        PREPARE_FUNCTION,
        launch_geometry::prepare(c, tokens, participants).ok()?,
        packed.map_or(row.tokens, |p| p.packed_token_grid),
        s.query_features
            .checked_add(s.kv_features.checked_mul(2)?)?,
        scratch,
        &fixed,
    )
}
fn gate(b: &mut CostBuilder<'_>, s: CausalAttentionShape, row: Row, scratch: u64) -> Option<()> {
    let elements = row.tokens.checked_mul(s.query_features)?;
    kernel(
        b,
        ATTENTION_GATE_FUNCTION,
        launch_geometry::flat(elements).ok()?,
        elements,
        1,
        scratch,
        &[
            row.tokens,
            s.query_features,
            s.query_projection_features,
            s.head_dim,
        ],
    )
}
// One unit is the complete attention call. inner_units records query rows
// times their current (window-bounded) context extent and Q-head width. This
// is a checked shape coordinate, not measured instruction counts or FLOPs.
fn attention_work(s: CausalAttentionShape, rows: &[Row]) -> Option<u64> {
    rows.iter().try_fold(0_u64, |sum, row| {
        sum.checked_add(
            row.tokens
                .checked_mul(if s.sliding_window_tokens == 0 {
                    row.sequence
                } else {
                    row.sequence.min(s.sliding_window_tokens)
                })?
                .checked_mul(s.query_features)?,
        )
    })
}
fn fallback_attention(
    b: &mut CostBuilder<'_>,
    s: CausalAttentionShape,
    c: CudaCausalAttentionShape,
    rows: &[Row],
    packed: Option<PackedFallbackLaunch>,
    scratch: u64,
) -> Option<()> {
    let first = *rows.first()?;
    let mut fixed = vec![
        VNEXT_KV_PAGE_BYTES / 2,
        u64::from(first.path.uses_vllm_layout()),
        s.query_heads,
        s.key_value_heads,
        s.head_dim,
        s.query_projection_features,
        0,
        u64::from(s.attention_scale.to_bits()),
        s.sliding_window_tokens,
        packed.map_or(0, |p| p.binding_slot_bytes),
    ];
    let (entry, config) =
        if let Some(threads) = packed.and_then(|_| grouped_fallback_block_threads(c)) {
            fixed.push(u64::from(packed?.participant_grid));
            (
                GROUPED_ATTENTION_FUNCTION,
                launch_geometry::grouped(c, packed?.packed_token_grid, threads).ok()?,
            )
        } else {
            (
                ATTENTION_FUNCTION,
                launch_geometry::fallback(
                    c,
                    packed.map_or(first.tokens, |p| p.token_grid),
                    packed.map_or(1, |p| p.participant_grid),
                )
                .ok()?,
            )
        };
    kernel(
        b,
        entry,
        config,
        1,
        attention_work(s, rows)?,
        scratch,
        &fixed,
    )
}
fn attention(
    b: &mut CostBuilder<'_>,
    s: CausalAttentionShape,
    c: CudaCausalAttentionShape,
    row: Row,
    scratch: u64,
) -> Option<()> {
    match row.path {
        CausalAttentionKernelPath::TokenMajorFallback
        | CausalAttentionKernelPath::VllmAddressedFallback => {
            fallback_attention(b, s, c, &[row], None, scratch)
        }
        CausalAttentionKernelPath::VllmAddressedVarlen
        | CausalAttentionKernelPath::VllmAddressedVarlenTiled => {
            let tiled = row.path == CausalAttentionKernelPath::VllmAddressedVarlenTiled;
            let mut fixed = vec![s.query_heads, s.key_value_heads, s.head_dim];
            if tiled {
                fixed.push(row.sequence);
            }
            fixed.push(u64::from(s.attention_scale.to_bits()));
            kernel(
                b,
                if tiled {
                    VARLEN_TILED_ADDRESSED_FUNCTION
                } else {
                    VARLEN_ADDRESSED_FUNCTION
                },
                launch_geometry::varlen(c, row.tokens, row.sequence, tiled).ok()?,
                1,
                attention_work(s, &[row])?,
                scratch,
                &fixed,
            )
        }
        CausalAttentionKernelPath::VllmAddressedDecodeV1
        | CausalAttentionKernelPath::VllmAddressedDecodeV2 => native_attention(
            b,
            s,
            &[row],
            row.envelope.sequence_capacity_tokens,
            u64::try_from(row.envelope.table_capacity_entries).ok()?,
            scratch,
        ),
    }
}
// Fixed ABI of the compiled native source bundle's addressed launcher (FP16,
// 16-token blocks, 128 threads, 512-token V2 partitions). Bundle identity above
// binds these formulas to the actual pinned library, not a family-name guess.
fn native_attention(
    b: &mut CostBuilder<'_>,
    s: CausalAttentionShape,
    rows: &[Row],
    envelope: u64,
    table_stride: u64,
    scratch: u64,
) -> Option<()> {
    #[cfg(not(feature = "vllm-paged-attn-v2"))]
    {
        let _ = (b, s, rows, envelope, table_stride, scratch);
        None
    }
    #[cfg(feature = "vllm-paged-attn-v2")]
    {
        if !matches!(s.head_dim, 128 | 256)
            || rows.is_empty()
            || rows.iter().any(|r| r.tokens != 1)
            || table_stride < envelope.div_ceil(16)
        {
            return None;
        }
        let sequences = u32::try_from(rows.len()).ok()?;
        let heads = u32::try_from(s.query_heads).ok()?;
        let v2 = envelope > 512;
        if rows
            .iter()
            .any(|r| (r.path == CausalAttentionKernelPath::VllmAddressedDecodeV2) != v2)
        {
            return None;
        }
        let partitions = envelope.div_ceil(512);
        let fixed = [
            s.key_value_heads,
            u64::from(s.attention_scale.to_bits()),
            table_stride,
            s.query_features,
            s.key_value_heads.checked_mul(s.head_dim)?.checked_mul(16)?,
            s.head_dim.checked_mul(16)?,
            0,
            0,
            0,
            0,
            0,
        ];
        let config = LaunchConfig {
            grid_dim: (
                heads,
                sequences,
                if v2 {
                    u32::try_from(partitions).ok()?
                } else {
                    1
                },
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: u32::try_from(
                (if v2 {
                    512
                } else {
                    envelope.div_ceil(16).checked_mul(16)?
                })
                .checked_mul(4)?
                .max(s.head_dim.checked_mul(8)?),
            )
            .ok()?,
        };
        let entry = match (v2, s.head_dim) {
            (false, 128) => "vllm.paged_attention_v1_kernel.f16.h128.b16.t128",
            (false, 256) => "vllm.paged_attention_v1_kernel.f16.h256.b16.t128",
            (true, 128) => "vllm.paged_attention_v2_kernel.f16.h128.b16.t128.p512",
            (true, 256) => "vllm.paged_attention_v2_kernel.f16.h256.b16.t128.p512",
            _ => return None,
        };
        kernel(
            b,
            entry,
            config,
            1,
            attention_work(s, rows)?,
            scratch,
            &fixed,
        )?;
        if v2 {
            kernel(
                b,
                if s.head_dim == 128 {
                    "vllm.paged_attention_v2_reduce_kernel.f16.h128.t128.p512"
                } else {
                    "vllm.paged_attention_v2_reduce_kernel.f16.h256.t128.p512"
                },
                LaunchConfig {
                    grid_dim: (heads, sequences, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: u32::try_from(partitions.checked_mul(8)?).ok()?,
                },
                1,
                rows.iter().try_fold(0_u64, |sum, row| {
                    sum.checked_add(row.sequence.div_ceil(512).checked_mul(s.query_features)?)
                })?,
                scratch,
                &[partitions],
            )?;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests;
