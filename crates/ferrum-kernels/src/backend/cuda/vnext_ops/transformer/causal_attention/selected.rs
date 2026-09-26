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
        builder: &mut SelectedCommandCostBuilderV1,
        index: usize,
        rows: u64,
        output: u64,
        input: u64,
        scratch: u64,
    ) -> Option<()> {
        match self {
            Self::Native(parts) => projection_work(builder, parts[index], rows, output, scratch),
            Self::DenseF16(identity) => GemmF16ApiPlan::new(
                checked_i32(rows, "causal GEMM rows").ok()?,
                checked_i32(output, "causal GEMM output").ok()?,
                checked_i32(input, "causal GEMM input").ok()?,
            )
            .ok()?
            .append_selected(builder, identity)
            .ok(),
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
    path: CausalAttentionKernelPath,
    envelope: CausalAttentionReplayEnvelope,
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
    b: &mut SelectedCommandCostBuilderV1,
    entry: &str,
    config: LaunchConfig,
    logical: u64,
    inner: u64,
    scratch: u64,
    fixed: &[u64],
) -> Option<()> {
    let block = [config.block_dim.0, config.block_dim.1, config.block_dim.2];
    b.kernel_with_replay_geometry(
        class(entry, block)?,
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
pub(super) fn binding_evidence(
    bytes: impl IntoIterator<Item = u64>,
    tokens: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut b = selected_cost::builder(capture, tokens)?;
    for bytes in bytes {
        b.transfer(
            class("cuda.cuMemcpyHtoDAsync.causal-control", [1, 1, 1])?,
            StatisticalTransferKindV1::HostToDevice,
            bytes,
        )
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
    let mut b = selected_cost::builder(capture, tokens)?;
    if shape.int8_kv || rows.is_empty() {
        return None;
    }
    match (projection, projections) {
        (CausalProjection::Native { .. }, ProjectionWork::Native(parts))
            if parts.iter().all(|part| !part.is_empty()) => {}
        (CausalProjection::F16, ProjectionWork::DenseF16(_)) => {}
        _ => return None,
    }
    let seen = rows
        .iter()
        .try_fold(0_u64, |sum, row| sum.checked_add(row.tokens))?;
    if seen != tokens || (packed && rows.len() < 2) {
        return None;
    }
    let layout =
        ScratchLayout::for_participants(shape, tokens, rows.len(), projection, policy).ok()?;
    let bindings = BindingLayout::new(shape, rows.len()).ok()?;
    let cuda = shape.cuda_shape().ok()?;
    let transform = projection.transform_bytes_per_token().checked_mul(tokens)?;
    let before = |b: &mut SelectedCommandCostBuilderV1, n: u64| -> Option<()> {
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
    let after = |b: &mut SelectedCommandCostBuilderV1, n: u64, inplace: bool| -> Option<()> {
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
        if batch_decode::eligible(rows.iter().map(|row| row.path), true) {
            let paths = rows.iter().map(|row| row.path).collect::<Vec<_>>();
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
            for range in batch_decode::group_ranges(&paths) {
                let group = &rows[range];
                let maximum = group
                    .iter()
                    .map(|row| row.envelope.sequence_capacity_tokens)
                    .max()?;
                let local_binding = BindingLayout {
                    slot_bytes: bindings.slot_bytes,
                    required_bytes: bindings.slot_bytes.checked_mul(group.len() as u64)?,
                };
                batch_decode::BatchDecode::dimensions(
                    group.len(),
                    local_binding,
                    maximum,
                    shape.query_heads,
                    shape.head_dim,
                )
                .ok()?;
                native_attention(
                    &mut b,
                    shape,
                    group,
                    maximum,
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
    let evidence = b.finish().ok()?;
    let extra = if packed {
        projections.extra_dispatches(tokens)?
    } else {
        rows.iter().try_fold(0_u64, |sum, row| {
            sum.checked_add(projections.extra_dispatches(row.tokens)?)
        })?
    };
    let dispatches = physical_dispatch_count(
        rows.iter().map(|row| row.path),
        shape.output_gate,
        shape.post_attention_norm,
        packed,
    )
    .checked_add(extra)?;
    evidence.validate_command(tokens, dispatches, 0).ok()?;
    Some(evidence)
}
fn projection_work(
    b: &mut SelectedCommandCostBuilderV1,
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
    b: &mut SelectedCommandCostBuilderV1,
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
    b: &mut SelectedCommandCostBuilderV1,
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
fn gate(
    b: &mut SelectedCommandCostBuilderV1,
    s: CausalAttentionShape,
    row: Row,
    scratch: u64,
) -> Option<()> {
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
    b: &mut SelectedCommandCostBuilderV1,
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
    b: &mut SelectedCommandCostBuilderV1,
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
    b: &mut SelectedCommandCostBuilderV1,
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
