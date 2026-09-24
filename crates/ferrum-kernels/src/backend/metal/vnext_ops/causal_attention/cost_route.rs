//! Static eager causal-attention route. Physical page ownership stays with
//! actual admission. In particular, packed grouped decode may depend on page
//! aliasing and is Unknown until the caller can provide that read-only proof.
use super::super::linear::{prepare_leaf_encoding, PreparedLinearPart};
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, DeviceCostBufferRange, OperationCostCommand, OperationCostRoute,
    OperationCostRouteRequest, OperationCostWorkRow, PhysicalWeightLayout, ResolvedWeightBinding,
};

#[derive(Clone, Copy)]
pub(super) struct Capabilities {
    pub kv_type: ElementType,
    pub maximum_attention_simdgroups: u32,
    pub maximum_threadgroup_memory_length: u64,
    pub supports_gqa_tiled_prefill: bool,
    pub batched_grouped: [bool; 2],
}

impl From<&MetalCausalAttentionPipelines> for Capabilities {
    fn from(pipelines: &MetalCausalAttentionPipelines) -> Self {
        Self {
            kv_type: pipelines.kv_type,
            maximum_attention_simdgroups: pipelines.maximum_attention_simdgroups,
            maximum_threadgroup_memory_length: pipelines.maximum_threadgroup_memory_length,
            supports_gqa_tiled_prefill: pipelines.supports_gqa_tiled_prefill,
            batched_grouped: std::array::from_fn(|i| {
                let p = &pipelines.specialized[i];
                p.grouped.is_some() && p.batched_grouped.is_some()
            }),
        }
    }
}

impl Capabilities {
    pub(super) fn simdgroups(self, context: u64) -> u32 {
        self.maximum_attention_simdgroups
            .min(u32::try_from(context).unwrap_or(u32::MAX).max(1))
    }

    pub(super) fn dispatch_plan(self, params: &CausalAttentionParams) -> AttentionDispatchPlan {
        if self.kv_type == ElementType::I8 {
            int8_attention_dispatch_plan(
                params,
                self.maximum_threadgroup_memory_length,
                self.supports_gqa_tiled_prefill,
            )
        } else {
            attention_dispatch_plan_with_memory_limit(
                params,
                self.maximum_threadgroup_memory_length,
            )
        }
    }

    /// Necessary numerical conditions only. True still requires the actual
    /// retained-page disjointness check; it never authorizes a batched kernel.
    pub(super) fn may_batch_grouped<'a>(
        self,
        mut rows: impl ExactSizeIterator<Item = &'a CausalAttentionParams>,
    ) -> bool {
        if self.kv_type != ElementType::F16 || rows.len() < 2 {
            return false;
        }
        let first = rows.next().expect("nonempty rows");
        let available = match first.head_dim {
            128 => self.batched_grouped[0],
            256 => self.batched_grouped[1],
            _ => false,
        };
        available
            && std::iter::once(first).chain(rows).all(|row| {
                self.dispatch_plan(row).kind == AttentionDispatchKind::GroupedDecode
                    && row.head_dim == first.head_dim
                    && row.query_heads == first.query_heads
                    && row.key_value_heads == first.key_value_heads
            })
    }
}

pub(super) fn row_params(
    shape: CausalAttentionShape,
    tokens: u64,
    offset: u64,
    full_input: u64,
    caps: Capabilities,
) -> Result<CausalAttentionParams, String> {
    if !matches!(caps.kv_type, ElementType::F16 | ElementType::I8) {
        return Err("causal cost has unsupported KV type".into());
    }
    let end = shape.context_end(tokens, offset, full_input)?;
    let pages = shape.physical_state_bytes_with_type(end, caps.kv_type)? / VNEXT_KV_PAGE_BYTES;
    if pages == 0 || pages > MAXIMUM_KV_PAGES {
        return Err("causal cost exceeds the fixed KV page table".into());
    }
    let mut params = shape.params(tokens, offset, pages, caps.simdgroups(end))?;
    params.page_elements = checked_u32(
        VNEXT_KV_PAGE_BYTES / caps.kv_type.size_bytes(),
        "causal cost page elements",
    )?;
    checked_u32(
        tokens
            .checked_mul(shape.hidden_size)
            .ok_or("causal cost residual extent overflows")?,
        "causal cost residual elements",
    )?;
    if caps.kv_type == ElementType::I8
        && shape.physical_scale_bytes(end)? / VNEXT_KV_PAGE_BYTES > MAXIMUM_KV_PAGES
    {
        return Err("causal cost exceeds the fixed KV scale page table".into());
    }
    Ok(params)
}

/// The actual encoder and query both consume this physical accounting. The
/// host-only binding slot is retained for ordering but is never device work.
pub(super) fn commands(
    hidden_type: ElementType,
    kv_type: ElementType,
    participants: usize,
    tokens: u64,
    packed: bool,
    grouped_rows: u64,
    batched_grouped: bool,
    extra_projections: u64,
) -> Result<[OperationCostCommand; 2], String> {
    let count = checked_u32(participants as u64, "causal cost participants")?;
    if count == 0
        || tokens == 0
        || !matches!(hidden_type, ElementType::F16 | ElementType::F32)
        || !matches!(kv_type, ElementType::F16 | ElementType::I8)
        || grouped_rows > u64::from(count)
        || (kv_type == ElementType::I8 && grouped_rows != 0)
        || (batched_grouped && (!packed || participants < 2 || grouped_rows != u64::from(count)))
    {
        return Err("causal cost command has inconsistent work".into());
    }
    let collapsed = if batched_grouped {
        2 * (participants - participants.div_ceil(GROUPED_BATCH_ROWS)) as u64
    } else {
        0
    };
    let compute = physical_dispatch_count(participants, packed)
        .checked_add(grouped_rows)
        .and_then(|n| n.checked_sub(collapsed))
        .and_then(|n| n.checked_add(extra_projections))
        .ok_or("causal cost command count overflows")?;
    let label = match (hidden_type, kv_type) {
        (ElementType::F32, ElementType::I8) => "vnext_causal_paged_attention_f32_master_int8_kv",
        (_, ElementType::I8) => "vnext_causal_paged_attention_int8_kv",
        (ElementType::F32, _) => "vnext_causal_paged_attention_f32_master",
        _ => "vnext_causal_paged_attention",
    };
    let row_form = if count == 1 {
        DeviceBatchingForm::Scalar
    } else {
        DeviceBatchingForm::ParticipantLoop
    };
    Ok([
        OperationCostCommand::new(
            "vnext_causal_paged_attention_bindings",
            DeviceCommandPhase::DynamicBinding,
            row_form,
            0,
            count,
            tokens,
            0,
            0,
        )
        .map_err(|e| e.to_string())?,
        OperationCostCommand::new(
            label,
            DeviceCommandPhase::Compute,
            if packed {
                DeviceBatchingForm::Packed
            } else {
                row_form
            },
            0,
            count,
            tokens,
            compute,
            if kv_type == ElementType::I8 {
                u64::from(count)
            } else {
                0
            },
        )
        .map_err(|e| e.to_string())?,
    ])
}

pub(super) fn projection_extra(
    projections: impl IntoIterator<Item = LinearLaunch>,
) -> Result<u64, String> {
    projections
        .into_iter()
        .try_fold(0_u64, |count, projection| {
            projection
                .dispatch_count()
                .checked_sub(1)
                .and_then(|extra| count.checked_add(extra))
                .ok_or_else(|| "causal projection dispatch count overflows".into())
        })
}

pub(super) fn eager_route(
    request: OperationCostRouteRequest<'_>,
    operation_id: &str,
    hidden_type: ElementType,
    pipelines: &MetalCausalAttentionPipelines,
) -> Result<Option<OperationCostRoute>, VNextError> {
    if request.operation_id().as_str() != operation_id {
        return Ok(None);
    }
    let calculate = || -> Result<Option<[OperationCostCommand; 2]>, String> {
        let shape = CausalAttentionShape::from_attributes(request.attributes())?;
        let caps = Capabilities::from(pipelines);
        shape.validate_page_count(caps.kv_type)?;
        validate_bindings(request.bindings(), shape, hidden_type, caps.kv_type)?;
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(4)
            .map_err(|_| "causal weight metadata capacity unavailable")?;
        for (ordinal, output, input) in [
            (2, shape.query_projection_features, shape.hidden_size),
            (3, shape.kv_features, shape.hidden_size),
            (4, shape.kv_features, shape.hidden_size),
            (5, shape.hidden_size, shape.query_features),
        ] {
            let Some(weight) = plain_weight(
                binding(request.bindings(), ResolvedValueRole::Input, ordinal)?,
                output,
                input,
            )?
            else {
                return Ok(None);
            };
            weights.push(weight);
        }
        let weights: [PreparedLinearPart; 4] = weights
            .try_into()
            .map_err(|_| "causal cost projection set is incomplete")?;
        let packed = request.rows().len() > 1
            && request
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
                .map_err(|e| e.to_string())?
            && request
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)
                .map_err(|e| e.to_string())?;
        // Size only, no binding buffer or argument encoder acquired.
        BindingLayout::new(pipelines.binding_slot_bytes()?, request.rows().len())?;
        project(
            shape,
            hidden_type,
            request.rows(),
            request.immediate_tokens(),
            packed,
            weights,
            caps,
            Some(&request),
        )
    };
    calculate()
        .map_err(invalid_plan)?
        .map(|commands| OperationCostRoute::new(Vec::from(commands))?.with_relocatable_binding(0))
        .transpose()
}

fn plain_weight(
    binding: &ResolvedValueBinding,
    output: u64,
    input: u64,
) -> Result<Option<PreparedLinearPart>, String> {
    let weight = binding
        .weight()
        .ok_or("causal cost projection lacks a weight ABI")?;
    if !matches!(
        weight.physical_layout(),
        PhysicalWeightLayout::Dense { .. }
            | PhysicalWeightLayout::Stored { .. }
            | PhysicalWeightLayout::BlockQuantized { .. }
    ) {
        return Ok(None);
    }
    let [component] = weight.components() else {
        return Ok(None);
    };
    let [stored] = binding.storage().components() else {
        return Err("causal cost weight storage differs".into());
    };
    if component.physical_dimensions().len() > 3 {
        return Ok(None);
    }
    if stored.component_id() != Some(component.component_id())
        || stored.element_type() != component.physical_element_type()
        || stored.length_bytes() != component.physical_bytes().map_err(|e| e.to_string())?
    {
        return Err("causal cost weight storage differs from its ABI".into());
    }
    leaf_weight(weight, output, input)
}

fn leaf_weight(
    weight: &ResolvedWeightBinding,
    output: u64,
    input: u64,
) -> Result<Option<PreparedLinearPart>, String> {
    if !matches!(
        weight.physical_layout(),
        PhysicalWeightLayout::Dense { .. }
            | PhysicalWeightLayout::Stored { .. }
            | PhysicalWeightLayout::BlockQuantized { .. }
    ) {
        return Ok(None);
    }
    let [component] = weight.components() else {
        return Ok(None);
    };
    if component.physical_dimensions().len() > 3 {
        return Ok(None);
    }
    let components = [super::super::weights::component_metadata(component)];
    let layout = super::super::weights::resolve_layout(weight)?;
    prepare_leaf_encoding(&components, &layout, output, input, 1, 0).map(Some)
}

#[allow(clippy::too_many_arguments)]
fn project(
    shape: CausalAttentionShape,
    hidden_type: ElementType,
    rows: &[OperationCostWorkRow],
    total: u64,
    packed: bool,
    weights: [PreparedLinearPart; 4],
    caps: Capabilities,
    request: Option<&OperationCostRouteRequest<'_>>,
) -> Result<Option<[OperationCostCommand; 2]>, String> {
    let mut params = Vec::new();
    params
        .try_reserve_exact(rows.len())
        .map_err(|_| "causal cost row metadata capacity unavailable")?;
    for row in rows {
        params.push(row_params(
            shape,
            row.count.get(),
            row.offset,
            row.full_input_tokens.get(),
            caps,
        )?);
    }
    let batched_grouped = if packed && caps.may_batch_grouped(params.iter()) {
        let Some(request) = request else {
            return Ok(None);
        };
        let mut pages = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            let end = row
                .offset
                .checked_add(row.count.get())
                .ok_or("causal cost context overflow")?;
            let Some(mut participant_pages) = request
                .binding_sequence_ranges(
                    ResolvedValueRole::Input,
                    8,
                    index,
                    shape.physical_state_bytes_with_type(end, caps.kv_type)?,
                    VNEXT_KV_PAGE_BYTES,
                )
                .map_err(|e| e.to_string())?
            else {
                return Ok(None);
            };
            if participant_pages
                .iter()
                .any(|range| range.allocation_id().is_none())
            {
                return Ok(None);
            }
            if participant_pages
                .iter()
                .map(|p| p.length() / VNEXT_KV_PAGE_BYTES)
                .sum::<u64>()
                != u64::from(params[index].page_count)
            {
                return Ok(None);
            }
            pages.append(&mut participant_pages);
        }
        pages_are_disjoint(&mut pages)
    } else {
        false
    };
    let layout = ScratchLayout::new_with_storage(shape, total, rows.len(), caps.kv_type)?;
    let mut extra = if packed {
        projection_extra(projected_launches(shape, weights, layout, 0, total)?)?
    } else {
        0
    };
    let mut start = 0_u64;
    for row in rows {
        if !packed {
            extra = extra
                .checked_add(projection_extra(projected_launches(
                    shape,
                    weights,
                    layout,
                    start,
                    row.count.get(),
                )?)?)
                .ok_or("causal cost projection count overflows")?;
        }
        start = start
            .checked_add(row.count.get())
            .ok_or("causal cost token count overflows")?;
    }
    if start != total {
        return Err("causal cost rows differ from total tokens".into());
    }
    let grouped = params
        .iter()
        .filter(|p| caps.dispatch_plan(p).kind == AttentionDispatchKind::GroupedDecode)
        .count() as u64;
    commands(
        hidden_type,
        caps.kv_type,
        rows.len(),
        total,
        packed,
        grouped,
        batched_grouped,
        extra,
    )
    .map(Some)
}

/// Same whole-page exclusion as actual grouped dispatch, including aliases
/// within one row. Unknown identity domains cannot authorize batching.
pub(super) fn pages_are_disjoint(pages: &mut [DeviceCostBufferRange]) -> bool {
    if pages.is_empty() || pages.iter().any(|page| page.allocation_id().is_none()) {
        return false;
    }
    pages.sort_unstable_by_key(|page| (page.allocation_id(), page.start()));
    !pages.windows(2).any(|pair| pair[0].overlaps(pair[1]))
}

fn projected_launches(
    shape: CausalAttentionShape,
    weights: [PreparedLinearPart; 4],
    layout: ScratchLayout,
    start: u64,
    tokens: u64,
) -> Result<[LinearLaunch; 4], String> {
    let end = start
        .checked_add(tokens)
        .ok_or("causal cost projection range overflows")?;
    checked_u32(
        tokens
            .checked_mul(shape.hidden_size)
            .ok_or("causal cost residual extent overflows")?,
        "causal cost residual elements",
    )?;
    let segments = [
        (layout.normalized, shape.hidden_size, layout.query_raw),
        (
            layout.query_raw,
            shape.query_projection_features,
            layout.key_raw,
        ),
        (layout.key_raw, shape.kv_features, layout.value_raw),
        (layout.value_raw, shape.kv_features, layout.query),
        (layout.context, shape.query_features, layout.projected),
        (layout.projected, shape.hidden_size, layout.required_bytes),
    ];
    for (base, width, limit) in segments {
        let bytes = end
            .checked_mul(width)
            .and_then(|v| v.checked_mul(ElementType::F16.size_bytes()))
            .and_then(|n| base.checked_add(n))
            .ok_or("causal cost activation span overflows")?;
        if bytes > limit {
            return Err("causal cost projection crosses its scratch segment".into());
        }
    }
    let normalized = layout.token_offset(layout.normalized, start, shape.hidden_size)?;
    let launch = |weight, width, stride, input, output| {
        linear_launch(weight, 0, 0, tokens, width, stride, input, output)
    };
    Ok([
        launch(
            weights[0],
            shape.hidden_size,
            shape.query_projection_features,
            normalized,
            layout.token_offset(layout.query_raw, start, shape.query_projection_features)?,
        )?,
        launch(
            weights[1],
            shape.hidden_size,
            shape.kv_features,
            normalized,
            layout.token_offset(layout.key_raw, start, shape.kv_features)?,
        )?,
        launch(
            weights[2],
            shape.hidden_size,
            shape.kv_features,
            normalized,
            layout.token_offset(layout.value_raw, start, shape.kv_features)?,
        )?,
        launch(
            weights[3],
            shape.query_features,
            shape.hidden_size,
            layout.token_offset(layout.context, start, shape.query_features)?,
            layout.token_offset(layout.projected, start, shape.hidden_size)?,
        )?,
    ])
}

#[cfg(test)]
mod tests;
