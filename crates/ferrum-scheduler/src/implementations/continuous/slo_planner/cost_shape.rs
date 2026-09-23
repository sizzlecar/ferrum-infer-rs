//! Representation conversion only. Both observation and planning retain the
//! canonical provider/output key unchanged; this module never invents a hash.
use super::{super::cost_model::*, types::PlanningUnknownReason};
use ferrum_interfaces::execution_cost::*;
use std::num::NonZeroU32;

pub fn canonical_cost_shape(
    shape: &CanonicalWaveCostShape,
) -> Result<WaveExecutionShape, PlanningUnknownReason> {
    project(
        ShapeMetadata {
            numeric_features: shape.numeric_features.clone(),
            host_content_features: shape.host_content_features,
            row_multiset_features: shape.row_multiset_features.clone(),
            kind: shape.kind,
            path: shape.path,
            graph: shape.graph,
            order: shape.row_order,
            provider: shape.provider_signature,
            output: shape.output_policy_signature,
            recurrent: shape.recurrent_state_bytes,
            restore: 0,
            maintenance: 0,
            maintenance_units: 0,
        },
        shape.rows.iter().copied(),
    )
}

/// Observation adapter for completed inference waves. Restore/maintenance
/// observations require their own explicit model and are not inference rows.
pub fn actual_cost_shape(
    shape: &ActualWaveShape,
) -> Result<WaveExecutionShape, PlanningUnknownReason> {
    shape
        .validate(MAX_COST_ROWS)
        .map_err(|_| PlanningUnknownReason::InvalidShapeEvidence)?;
    project(
        ShapeMetadata {
            numeric_features: shape.numeric_features.clone(),
            host_content_features: shape.host_content_features,
            row_multiset_features: shape.row_multiset_features.clone(),
            kind: shape.kind,
            path: shape.path,
            graph: shape.graph,
            order: shape.row_order,
            provider: shape.provider_signature,
            output: shape.output_policy_signature,
            recurrent: shape.recurrent_state_bytes,
            restore: shape.restore_bytes,
            maintenance: shape.maintenance_bytes,
            maintenance_units: shape.maintenance_units,
        },
        shape.rows.iter().map(|row| row.work),
    )
}
struct ShapeMetadata {
    numeric_features: Option<CanonicalWaveCostFeatures>,
    host_content_features: Option<HostContentCostFeaturesV1>,
    row_multiset_features: Option<HostRowMultisetCostFeaturesV2>,
    kind: ActualWaveKind,
    path: ActualWavePath,
    graph: ActualWaveGraphState,
    order: ActualWaveRowOrder,
    provider: [u8; 32],
    output: [u8; 32],
    recurrent: u64,
    restore: u64,
    maintenance: u64,
    maintenance_units: u32,
}
fn project(
    metadata: ShapeMetadata,
    rows: impl ExactSizeIterator<Item = ActualRowWork> + Clone,
) -> Result<WaveExecutionShape, PlanningUnknownReason> {
    if rows.len() == 0 || rows.len() > MAX_COST_ROWS {
        return Err(PlanningUnknownReason::InvalidShapeEvidence);
    }
    if let Some(features) = &metadata.numeric_features {
        features
            .validate(rows.len())
            .map_err(|_| PlanningUnknownReason::InvalidShapeEvidence)?;
    }
    if let Some(features) = &metadata.host_content_features {
        features
            .validate()
            .map_err(|_| PlanningUnknownReason::InvalidShapeEvidence)?;
        if metadata.numeric_features.is_none() {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
    }
    if let Some(features) = &metadata.row_multiset_features {
        features
            .validate(rows.len())
            .map_err(|_| PlanningUnknownReason::InvalidShapeEvidence)?;
        if metadata.numeric_features.is_none()
            || features
                .rows
                .iter()
                .zip(rows.clone())
                .any(|(row, work)| !row.role.matches_work(work))
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
    }
    let decode_count = rows
        .clone()
        .filter(|row| matches!(row, ActualRowWork::Decode { .. }))
        .count();
    let prefill_count = rows.len() - decode_count;
    let mut decode_kv_tokens = Vec::new();
    let mut prefill_chunks = Vec::new();
    decode_kv_tokens
        .try_reserve_exact(decode_count)
        .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
    prefill_chunks
        .try_reserve_exact(prefill_count)
        .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
    for row in rows {
        match row {
            ActualRowWork::Decode { kv_tokens } if kv_tokens > 0 => {
                decode_kv_tokens.push(kv_tokens)
            }
            ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => {
                if offset
                    .checked_add(count)
                    .is_none_or(|end| end > total_prompt_tokens)
                {
                    return Err(PlanningUnknownReason::InvalidShapeEvidence);
                }
                prefill_chunks.push(PrefillShape {
                    offset,
                    count: NonZeroU32::new(count)
                        .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?,
                    total_prompt_tokens: NonZeroU32::new(total_prompt_tokens)
                        .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?,
                });
            }
            _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
        }
    }
    let kind = match (
        metadata.kind,
        decode_kv_tokens.is_empty(),
        prefill_chunks.is_empty(),
    ) {
        (ActualWaveKind::Decode, false, true) => WaveKind::Decode,
        (ActualWaveKind::Prefill, true, false) => WaveKind::Prefill,
        (ActualWaveKind::Mixed, false, false) => WaveKind::Mixed,
        _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
    };
    Ok(WaveExecutionShape {
        numeric_features: metadata.numeric_features,
        host_content_features: metadata.host_content_features,
        row_multiset_features: metadata.row_multiset_features,
        kind,
        path: match metadata.path {
            ActualWavePath::PlanRuntime => WaveExecutionPath::PlanRuntime,
            ActualWavePath::NativeUnified => WaveExecutionPath::NativeUnified,
            ActualWavePath::LegacySplit => WaveExecutionPath::LegacySplit,
            ActualWavePath::UnsupportedFallback => WaveExecutionPath::UnsupportedFallback,
            ActualWavePath::CapacityFallback => WaveExecutionPath::CapacityFallback,
        },
        graph_state: match metadata.graph {
            ActualWaveGraphState::Disabled => WaveGraphState::Disabled,
            ActualWaveGraphState::Cold => WaveGraphState::Cold,
            ActualWaveGraphState::Warm => WaveGraphState::Warm,
        },
        order: match metadata.order {
            ActualWaveRowOrder::Ordered => BatchOrderSemantics::Ordered,
            ActualWaveRowOrder::IndependentRows => BatchOrderSemantics::IndependentRows,
        },
        provider_signature: metadata.provider,
        output_policy_signature: metadata.output,
        decode_kv_tokens,
        prefill_chunks,
        recurrent_state_bytes: metadata.recurrent,
        restore_bytes: metadata.restore,
        maintenance_bytes: metadata.maintenance,
        maintenance_units: metadata.maintenance_units,
    })
}
