//! Statistical exchangeability within actual contiguous role segments.
//! This is an explicit empirical model, never an execution reorder or an
//! IndependentRows capability. Every work/numeric/static coordinate moves as
//! one tuple; the actual observation and immutable physical proof stay intact.
use super::*;
use ferrum_interfaces::execution_cost::{
    CostRowNumericFeatures, HostRowRoleV2, HostRowStaticCostFeaturesV2,
};

pub(super) fn enabled(mode: &CostFeatureModel) -> bool {
    matches!(
        mode,
        CostFeatureModel::EmpiricalRowMultisetV2 { .. }
            | CostFeatureModel::EmpiricalPromptRangeV3 { .. }
    )
}

pub(super) fn validate(shape: &WaveExecutionShape) -> Result<(), CostUnknownReason> {
    let features = shape
        .row_multiset_features
        .as_ref()
        .ok_or(CostUnknownReason::HostContentFeaturesMissing)?;
    let count = shape.decode_kv_tokens.len() + shape.prefill_chunks.len();
    features
        .validate(count)
        .map_err(|_| CostUnknownReason::InvalidShape)?;
    let decodes = features
        .rows
        .iter()
        .filter(|row| row.role == HostRowRoleV2::Decode)
        .count();
    if decodes != shape.decode_kv_tokens.len()
        || count - decodes != shape.prefill_chunks.len()
        || shape
            .numeric_features
            .as_ref()
            .is_none_or(|numeric| numeric.rows.len() != count)
    {
        return Err(CostUnknownReason::InvalidShape);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Work {
    Decode(u32),
    Prefill(PrefillShape),
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Row {
    static_features: HostRowStaticCostFeaturesV2,
    work: Work,
    numeric: CostRowNumericFeatures,
}

/// The caller has validated the row budget, features and work-role counts.
/// Allocate at most one bounded temporary row vector. Both training and query
/// use this same ordering before building keys or joint support coordinates.
pub(super) fn statistical_order(shape: &mut WaveExecutionShape) {
    let features = shape
        .row_multiset_features
        .as_mut()
        .expect("validated V2 features");
    let numeric = shape
        .numeric_features
        .as_mut()
        .expect("validated numeric features");
    let mut decodes = shape.decode_kv_tokens.iter();
    let mut prefills = shape.prefill_chunks.iter();
    let mut rows: Vec<_> = features
        .rows
        .iter()
        .zip(&numeric.rows)
        .map(|(static_features, numeric)| Row {
            static_features: static_features.clone(),
            work: match static_features.role {
                HostRowRoleV2::Decode => {
                    Work::Decode(*decodes.next().expect("validated decode rows"))
                }
                HostRowRoleV2::Prefill => {
                    Work::Prefill(prefills.next().expect("validated prefill rows").clone())
                }
            },
            numeric: numeric.clone(),
        })
        .collect();
    let mut start = 0;
    while start < rows.len() {
        let role = rows[start].static_features.role;
        let end = start
            + rows[start..]
                .iter()
                .take_while(|row| row.static_features.role == role)
                .count();
        rows[start..end].sort_unstable();
        start = end;
    }
    let mut decodes = shape.decode_kv_tokens.iter_mut();
    let mut prefills = shape.prefill_chunks.iter_mut();
    for (index, row) in rows.into_iter().enumerate() {
        features.rows[index] = row.static_features;
        numeric.rows[index] = row.numeric;
        match row.work {
            Work::Decode(kv) => *decodes.next().expect("validated decode rows") = kv,
            Work::Prefill(chunk) => *prefills.next().expect("validated prefill rows") = chunk,
        }
    }
}

#[cfg(test)]
mod tests;
