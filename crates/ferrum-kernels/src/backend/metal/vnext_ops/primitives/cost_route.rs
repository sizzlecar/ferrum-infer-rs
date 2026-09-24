//! Pure branch metadata shared by eager primitive encoding and route queries.
use super::*;
use ferrum_interfaces::vnext::{
    DeviceBatchingForm, DeviceCommandPhase, HadamardApplication, HadamardSigns,
    OperationCostCommand, OperationCostRoute, OperationCostRouteRequest, PhysicalStorageLayout,
    PhysicalWeightLayout,
};

#[derive(Clone, Copy)]
pub(super) enum PrimitiveRoute {
    TokenEmbedding { transformed_participants: u32 },
    RmsNorm,
    ResidualAdd,
    MaskedArgmax { vocabulary_size: u32 },
}

pub(super) fn compute_command(
    route: PrimitiveRoute,
    participants: u32,
    immediate_tokens: u64,
) -> Result<OperationCostCommand, VNextError> {
    let (operation, dispatches, tokens, multi_form) = match route {
        PrimitiveRoute::TokenEmbedding {
            transformed_participants,
        } => {
            if transformed_participants > participants {
                return Err(invalid_plan(
                    "cost route has more transforms than participants",
                ));
            }
            (
                "vnext_token_embedding",
                u64::from(participants) + u64::from(transformed_participants),
                immediate_tokens,
                DeviceBatchingForm::ParticipantLoop,
            )
        }
        PrimitiveRoute::RmsNorm => (
            "vnext_rms_norm",
            1,
            immediate_tokens,
            DeviceBatchingForm::Packed,
        ),
        PrimitiveRoute::ResidualAdd => (
            "vnext_residual_add",
            1,
            immediate_tokens,
            DeviceBatchingForm::Packed,
        ),
        PrimitiveRoute::MaskedArgmax { vocabulary_size } => {
            if vocabulary_size == 0 {
                return Err(invalid_plan("cost route vocabulary is empty"));
            }
            (
                "vnext_last_token_masked_argmax",
                u64::from(participants) * masked_argmax_dispatch_count(vocabulary_size),
                u64::from(participants),
                DeviceBatchingForm::ParticipantLoop,
            )
        }
    };
    OperationCostCommand::new(
        operation,
        DeviceCommandPhase::Compute,
        if participants == 1 {
            DeviceBatchingForm::Scalar
        } else {
            multi_form
        },
        0,
        participants,
        tokens,
        dispatches,
        0,
    )
}

/// Metadata proves static ABI and numerical launch bounds. Live storage,
/// aliasing and submission authority still belong to the actual invocation.
pub(super) fn eager_route(
    request: OperationCostRouteRequest<'_>,
    pipelines: &MetalPrimitivePipelines,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let checked = || -> Result<Option<PrimitiveRoute>, String> {
        let get = |role, ordinal| binding(request.bindings(), role, ordinal);
        let attribute = |name| unsigned_attribute(request.attributes(), name);
        Ok(Some(match request.operation_id().as_str() {
            TOKEN_EMBEDDING_OPERATION_ID | TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID => {
                let output_type = if request.operation_id().as_str() == TOKEN_EMBEDDING_OPERATION_ID
                {
                    ElementType::F16
                } else {
                    ElementType::F32
                };
                let hidden = attribute("hidden_size")?;
                let vocabulary = attribute("vocab_size")?;
                let table = get(ResolvedValueRole::Input, 1)?;
                validate_embedding_signature(
                    get(ResolvedValueRole::Input, 0)?,
                    table,
                    get(ResolvedValueRole::Output, 0)?,
                    vocabulary,
                    hidden,
                    output_type,
                )?;
                let (_, _, transform) = embedding_weight_metadata(table, vocabulary, hidden)?;
                if let Some(transform) = transform {
                    pipelines.hadamard.validate_dispatch(
                        transform,
                        checked_u32(hidden, "Metal embedding hidden size")?,
                        ElementType::F32,
                        output_type,
                    )?;
                }
                let mut scratch = 0;
                for row in request.rows() {
                    embedding_params(row.count.get(), hidden, vocabulary)?;
                    if transform.is_some() {
                        scratch = embedding_scratch_bytes(scratch, row.count.get(), hidden)?;
                    }
                }
                PrimitiveRoute::TokenEmbedding {
                    transformed_participants: if transform.is_some() {
                        request.rows().len() as u32
                    } else {
                        0
                    },
                }
            }
            RMS_NORM_OPERATION_ID
            | RMS_NORM_F32_OPERATION_ID
            | RMS_NORM_F32_TO_F16_OPERATION_ID => {
                let (input_type, output_type) = match request.operation_id().as_str() {
                    RMS_NORM_OPERATION_ID => (ElementType::F16, ElementType::F16),
                    RMS_NORM_F32_OPERATION_ID => (ElementType::F32, ElementType::F32),
                    _ => (ElementType::F32, ElementType::F16),
                };
                let hidden = attribute("hidden_size")?;
                if !valid_rms_norm(
                    get(ResolvedValueRole::Input, 0)?,
                    get(ResolvedValueRole::Input, 1)?,
                    get(ResolvedValueRole::Output, 0)?,
                    hidden,
                    input_type,
                    output_type,
                ) {
                    return Err("Metal RMSNorm cost query differs from its signature".into());
                }
                rms_norm_params(
                    request.immediate_tokens(),
                    hidden,
                    rational_attribute(request.attributes(), "epsilon")?,
                )?;
                if request.rows().len() > 1
                    && (!request
                        .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
                        .map_err(|e| e.to_string())?
                        || !request
                            .binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)
                            .map_err(|e| e.to_string())?)
                {
                    return Ok(None);
                }
                PrimitiveRoute::RmsNorm
            }
            RESIDUAL_ADD_OPERATION_ID | RESIDUAL_ADD_F32_F16_OPERATION_ID => {
                let (left, output) = if request.operation_id().as_str() == RESIDUAL_ADD_OPERATION_ID
                {
                    (ElementType::F16, ElementType::F16)
                } else {
                    (ElementType::F32, ElementType::F32)
                };
                let hidden = attribute("hidden_size")?;
                if !valid_residual_add(
                    get(ResolvedValueRole::Input, 0)?,
                    get(ResolvedValueRole::Input, 1)?,
                    get(ResolvedValueRole::Output, 0)?,
                    hidden,
                    left,
                    ElementType::F16,
                    output,
                ) {
                    return Err("Metal residual-add cost query differs from its signature".into());
                }
                residual_add_params(request.immediate_tokens(), hidden)?;
                if request.rows().len() > 1
                    && (!request
                        .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
                        .map_err(|e| e.to_string())?
                        || !request
                            .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 1)
                            .map_err(|e| e.to_string())?
                        || !request
                            .binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)
                            .map_err(|e| e.to_string())?)
                {
                    return Ok(None);
                }
                PrimitiveRoute::ResidualAdd
            }
            LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID | LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID => {
                let logits_type =
                    if request.operation_id().as_str() == LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID {
                        ElementType::F16
                    } else {
                        ElementType::F32
                    };
                let vocabulary = attribute("vocab_size")?;
                let repetition = valid_last_token_masked_argmax(
                    get(ResolvedValueRole::Input, 0)?,
                    get(ResolvedValueRole::Input, 1)?,
                    get(ResolvedValueRole::Input, 2)?,
                    get(ResolvedValueRole::Input, 3)?,
                    get(ResolvedValueRole::Input, 4)?,
                    get(ResolvedValueRole::Output, 0)?,
                    vocabulary,
                    logits_type,
                )
                .ok_or_else(|| {
                    "Metal masked argmax cost query differs from its signature".to_owned()
                })?;
                let params = masked_argmax_params(vocabulary, repetition)?;
                masked_argmax_scratch_stride(vocabulary, logits_type)?
                    .checked_mul(request.rows().len() as u64)
                    .ok_or_else(|| "Metal masked argmax scratch size overflows".to_owned())?;
                PrimitiveRoute::MaskedArgmax {
                    vocabulary_size: params.vocabulary_size,
                }
            }
            _ => return Ok(None),
        }))
    };
    let Some(route) = checked().map_err(invalid_plan)? else {
        return Ok(None);
    };
    OperationCostRoute::new(vec![compute_command(
        route,
        request.rows().len() as u32,
        request.immediate_tokens(),
    )?])
    .map(Some)
}

fn positive_u32(value: u64, label: &str) -> Result<u32, String> {
    if value == 0 {
        return Err(format!("{label} is empty"));
    }
    checked_u32(value, label)
}

pub(super) fn embedding_params(
    tokens: u64,
    hidden: u64,
    vocabulary: u64,
) -> Result<EmbeddingParams, String> {
    Ok(EmbeddingParams {
        token_count: positive_u32(tokens, "Metal embedding token count")?,
        hidden_size: positive_u32(hidden, "Metal embedding hidden size")?,
        vocabulary_size: positive_u32(vocabulary, "Metal embedding vocabulary size")?,
    })
}

pub(super) fn rms_norm_params(
    tokens: u64,
    hidden: u64,
    epsilon: f32,
) -> Result<RmsNormParams, String> {
    Ok(RmsNormParams {
        rows: positive_u32(tokens, "Metal RMSNorm row count")?,
        hidden_size: positive_u32(hidden, "Metal RMSNorm hidden size")?,
        epsilon,
    })
}

pub(super) fn residual_add_params(tokens: u64, hidden: u64) -> Result<ResidualAddParams, String> {
    let elements = tokens
        .checked_mul(hidden)
        .ok_or_else(|| "Metal residual-add element count overflows".to_owned())?;
    Ok(ResidualAddParams {
        elements: positive_u32(elements, "Metal residual-add element count")?,
    })
}

pub(super) fn masked_argmax_params(
    vocabulary: u64,
    repetition_capacity: u32,
) -> Result<LastTokenMaskedArgmaxParams, String> {
    Ok(LastTokenMaskedArgmaxParams {
        vocabulary_size: positive_u32(vocabulary, "Metal masked argmax vocabulary size")?,
        repetition_capacity,
    })
}

pub(super) fn embedding_scratch_bytes(
    existing: u64,
    tokens: u64,
    hidden: u64,
) -> Result<u64, String> {
    hidden
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(15))
        .map(|bytes| bytes & !15)
        .and_then(|stride| stride.checked_mul(tokens))
        .and_then(|bytes| existing.checked_add(bytes))
        .ok_or_else(|| "Metal embedding inverse workspace size overflows".to_owned())
}

/// Equivalent explicit row-major strides are legal resolved value/block storage. The
/// future route must prove the flat ABI; actual encoding retains its original
/// live-layout validation and does not acquire a new storage restriction here.
fn row_major_storage(storage: &PhysicalStorageLayout, dimensions: &[u64]) -> bool {
    match storage {
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact,
        } => true,
        PhysicalStorageLayout::Strided {
            strides_in_elements,
            padding: PhysicalWeightPadding::Exact,
        } if strides_in_elements.len() == dimensions.len() => {
            let mut expected = 1_u64;
            for (&extent, &stride) in dimensions.iter().zip(strides_in_elements).rev() {
                if extent == 0 || (extent > 1 && stride != expected) {
                    return false;
                }
                let Some(next) = expected.checked_mul(extent) else {
                    return false;
                };
                expected = next;
            }
            true
        }
        _ => false,
    }
}

/// Static interpretation of the resolved embedding ABI for the future query. Component
/// indices retain the canonical order used by `resolve_weight`; no GPU region
/// or allocation is created. Actual encoding additionally validates live views.
pub(super) fn embedding_weight_metadata(
    table: &ResolvedValueBinding,
    vocabulary: u64,
    hidden: u64,
) -> Result<(EmbeddingPhysicalFormat, usize, Option<HadamardTransform>), String> {
    if table.tensor().element_type() != ElementType::F16
        || table.tensor().dimensions() != [vocabulary, hidden]
    {
        return Err("Metal embedding logical weight differs from its contract".into());
    }
    let weight = table
        .weight()
        .ok_or_else(|| "Metal embedding lacks physical weight metadata".to_owned())?;
    embedding_weight_layout(weight, vocabulary, hidden)
}

fn embedding_weight_layout(
    weight: &ferrum_interfaces::vnext::ResolvedWeightBinding,
    vocabulary: u64,
    hidden: u64,
) -> Result<(EmbeddingPhysicalFormat, usize, Option<HadamardTransform>), String> {
    let components = weight.components();
    let index = |id: &ferrum_interfaces::vnext::WeightId| {
        components
            .binary_search_by(|component| component.component_id().cmp(id))
            .map_err(|_| "Metal embedding physical component is absent".to_owned())
    };
    let exact = |component: &ferrum_interfaces::vnext::PhysicalWeightComponentBinding| {
        let at = index(&component.component_id)?;
        if !row_major_storage(&component.storage, components[at].physical_dimensions()) {
            return Err("Metal embedding cost requires row-major component storage".to_owned());
        }
        Ok(at)
    };
    let (layout, transform) = match weight.physical_layout() {
        PhysicalWeightLayout::Hadamard { values, transform } => {
            if transform.application != HadamardApplication::AfterEmbeddingLookup {
                return Err("Metal embedding requires an inverse Hadamard transform".into());
            }
            transform.validate(hidden).map_err(|e| e.to_string())?;
            let signs_region = match &transform.signs {
                HadamardSigns::Identity => None,
                HadamardSigns::Explicit(signs) => {
                    // The schema deliberately narrows TransformSigns to this
                    // immutable exact-contiguous ABI, unlike values/blocks.
                    if signs.storage != PhysicalStorageLayout::exact_contiguous() {
                        return Err("Metal Hadamard signs require exact-contiguous storage".into());
                    }
                    let at = exact(signs)?;
                    if components[at].encoding()
                        != &(WeightEncoding::Dense {
                            element_type: ElementType::F32,
                        })
                        || components[at].physical_dimensions() != [hidden]
                    {
                        return Err(
                            "Metal Hadamard signs must be a complete F32 feature row".into()
                        );
                    }
                    Some(at)
                }
            };
            (
                values.as_ref(),
                Some(HadamardTransform {
                    block_size: transform.block_size.get(),
                    signs_region,
                    inverse: true,
                    permutation: None,
                }),
            )
        }
        layout => (layout, None),
    };
    let (component, format) = match layout {
        PhysicalWeightLayout::Dense { component_id } => {
            (index(component_id)?, EmbeddingPhysicalFormat::DenseF16)
        }
        PhysicalWeightLayout::Stored { component } => {
            (exact(component)?, EmbeddingPhysicalFormat::DenseF16)
        }
        PhysicalWeightLayout::BlockQuantized {
            blocks,
            block_axis,
            block_padding,
        } => {
            if *block_axis != 1 || *block_padding != PhysicalWeightPadding::Exact {
                return Err("Metal quantized embedding physical ABI differs".into());
            }
            let at = exact(blocks)?;
            let WeightEncoding::BlockQuantized(spec) = components[at].encoding() else {
                return Err("Metal embedding block component is not block quantized".into());
            };
            let format = match (
                spec.format_id.as_str(),
                spec.logical_values_per_block,
                spec.bytes_per_block,
            ) {
                (Q4_K_FORMAT_ID, 256, 144) => EmbeddingPhysicalFormat::Q4K,
                (Q6_K_FORMAT_ID, 256, 210) => EmbeddingPhysicalFormat::Q6K,
                (Q8_0_FORMAT_ID, 32, 34) => EmbeddingPhysicalFormat::Q8_0,
                (PQ2_0_FORMAT_ID, 128, 34) => EmbeddingPhysicalFormat::Pq2_0,
                _ => return Err("Metal embedding does not support this quantized block ABI".into()),
            };
            if !hidden.is_multiple_of(u64::from(spec.logical_values_per_block))
                || components[at].physical_dimensions()
                    != [
                        vocabulary,
                        hidden / u64::from(spec.logical_values_per_block),
                    ]
            {
                return Err("Metal quantized embedding component shape differs".into());
            }
            (at, format)
        }
        _ => return Err("Metal token embedding does not support this physical layout".into()),
    };
    if format == EmbeddingPhysicalFormat::DenseF16
        && (components[component].encoding()
            != &(WeightEncoding::Dense {
                element_type: ElementType::F16,
            })
            || components[component].physical_dimensions() != [vocabulary, hidden])
    {
        return Err("Metal dense embedding physical ABI differs".into());
    }
    let signs = transform.and_then(|t| t.signs_region);
    if signs == Some(component) || components.len() != 1 + usize::from(signs.is_some()) {
        return Err(
            "Metal token embedding requires one table and its declared transform signs".into(),
        );
    }
    Ok((format, component, transform))
}

#[cfg(test)]
#[path = "cost_route/tests.rs"]
mod boundary_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_work_and_per_participant_reductions_have_distinct_routes() {
        let packed = compute_command(PrimitiveRoute::RmsNorm, 3, 129).unwrap();
        assert_eq!(packed.batching(), DeviceBatchingForm::Packed);
        assert_eq!(packed.token_count(), 129);
        assert_eq!(packed.compute_dispatch_count(), 1);
        let reduce = compute_command(
            PrimitiveRoute::MaskedArgmax {
                vocabulary_size: MASKED_ARGMAX_PARALLEL_MIN_VOCAB,
            },
            3,
            129,
        )
        .unwrap();
        assert_eq!(reduce.batching(), DeviceBatchingForm::ParticipantLoop);
        assert_eq!(reduce.token_count(), 3);
        assert_eq!(reduce.compute_dispatch_count(), 6);
        let scalar = compute_command(PrimitiveRoute::ResidualAdd, 1, 129).unwrap();
        assert_eq!(scalar.batching(), DeviceBatchingForm::Scalar);
        assert_eq!(scalar.token_count(), 129);
    }

    #[test]
    fn argmax_route_tracks_the_actual_partition_boundary_and_rejects_empty_work() {
        let below = compute_command(
            PrimitiveRoute::MaskedArgmax {
                vocabulary_size: MASKED_ARGMAX_PARALLEL_MIN_VOCAB - 1,
            },
            2,
            2,
        )
        .unwrap();
        let at = compute_command(
            PrimitiveRoute::MaskedArgmax {
                vocabulary_size: MASKED_ARGMAX_PARALLEL_MIN_VOCAB,
            },
            2,
            2,
        )
        .unwrap();
        assert_eq!(below.compute_dispatch_count(), 2);
        assert_eq!(at.compute_dispatch_count(), 4);
        assert!(
            compute_command(PrimitiveRoute::MaskedArgmax { vocabulary_size: 0 }, 1, 1).is_err()
        );
        assert!(compute_command(PrimitiveRoute::RmsNorm, 0, 1).is_err());
        assert!(compute_command(PrimitiveRoute::ResidualAdd, 1, 0).is_err());
    }
}
