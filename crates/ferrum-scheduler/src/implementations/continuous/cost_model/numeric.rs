//! Joint, observed support for an explicitly selected empirical numeric model.
//! Categorical/provider branches stay exact. No GPU latency monotonicity theorem
//! is asserted: independent pre-update errors and actual SLOs remain necessary.
use super::*;

pub(super) fn validate_mode(
    shape: &WaveExecutionShape,
    mode: &CostFeatureModel,
) -> Result<(), CostUnknownReason> {
    if !matches!(mode, CostFeatureModel::ExactV1 {}) {
        if shape.numeric_features.is_none() {
            return Err(CostUnknownReason::NumericFeaturesMissing);
        }
        // Numeric rows are physically ordered, unlike the legacy sorted work
        // vectors. Do not sort one without its corresponding host evidence.
        if shape.order != BatchOrderSemantics::Ordered {
            return Err(CostUnknownReason::NumericRowOrderUnsupported);
        }
    }
    if matches!(mode, CostFeatureModel::EmpiricalHostContentV1 { .. })
        && shape
            .host_content_features
            .as_ref()
            .is_none_or(|features| features.schema_version != 1)
    {
        return Err(CostUnknownReason::HostContentFeaturesMissing);
    }
    if multiset::enabled(mode) {
        multiset::validate(shape)?;
    }
    Ok(())
}

pub(super) fn project_bucket(shape: &mut WaveExecutionShape, mode: &CostFeatureModel) {
    // Producers may supply both versions. Existing keys must not acquire a
    // new field merely because a newer binary observed the same actual work.
    if !multiset::enabled(mode) {
        shape.row_multiset_features = None;
    }
    match mode {
        CostFeatureModel::ExactV1 {} => {
            shape.numeric_features = None;
            shape.host_content_features = None;
        }
        CostFeatureModel::BoundedNumericV1 {
            host_history_bucket_tokens,
        }
        | CostFeatureModel::EmpiricalHostContentV1 {
            host_history_bucket_tokens,
        }
        | CostFeatureModel::EmpiricalRowMultisetV2 {
            host_history_bucket_tokens,
        } => {
            // validate_mode runs before lookup/training. This normalized shape
            // is only a private grouping key, never executable work evidence.
            let features = shape
                .numeric_features
                .as_mut()
                .expect("validated numeric mode");
            let signature = if multiset::enabled(mode) {
                shape.host_content_features = None;
                shape
                    .row_multiset_features
                    .as_ref()
                    .expect("validated V2 features")
                    .wave_policy_signature
            } else if matches!(mode, CostFeatureModel::EmpiricalHostContentV1 { .. }) {
                shape
                    .host_content_features
                    .as_ref()
                    .expect("validated host content mode")
                    .output_policy_signature
            } else {
                shape.host_content_features = None;
                features.output_policy_signature
            };
            shape.output_policy_signature = signature;
            // This is only the private grouping key. Actual numeric/exact
            // identities remain in retained observations and public evidence.
            features.output_policy_signature = signature;
            for row in &mut features.rows {
                row.generated_tokens_before /= u64::from(host_history_bucket_tokens.get());
                // max_tokens constrains capacity/termination, not the amount
                // of work of this wave. The canonical static hash separately
                // retains the checked final-limit branch. Actual resource
                // permission is still required before execution.
                row.maximum_output_tokens = 0;
                row.sampling_history_tokens = 0;
                row.repetition_tokens = 0;
                row.decoded_prefix_tokens = 0;
                row.decoded_text_bytes_bound = 0;
                row.decode_scratch_bytes_bound = 0;
            }
        }
    }
}

fn coordinates(shape: &WaveExecutionShape) -> Vec<u64> {
    let features = shape
        .numeric_features
        .as_ref()
        .expect("validated numeric mode");
    let mut values = Vec::with_capacity(
        shape.decode_kv_tokens.len() + shape.prefill_chunks.len() + 6 * features.rows.len(),
    );
    values.extend(shape.decode_kv_tokens.iter().map(|&n| u64::from(n)));
    // Count/total-prompt/final-fragment and recurrent/maintenance work remain
    // exact in the grouping key. Only these declared work axes range here.
    values.extend(
        shape
            .prefill_chunks
            .iter()
            .map(|chunk| u64::from(chunk.offset)),
    );
    for row in &features.rows {
        values.extend([
            row.generated_tokens_before,
            row.sampling_history_tokens,
            row.repetition_tokens,
            row.decoded_prefix_tokens,
            row.decoded_text_bytes_bound,
            row.decode_scratch_bytes_bound,
        ]);
    }
    values
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NumericSupport {
    minima: Box<[u64]>,
    maxima: Box<[u64]>,
    /// Joint support points are bounded by the trainer's samples/rows limits.
    /// Arcs avoid copying that support on each prediction or coverage clone.
    points: Arc<[Box<[u64]>]>,
}

impl NumericSupport {
    pub(super) fn from_samples(samples: &[&CostSample], mode: &CostFeatureModel) -> Option<Self> {
        if matches!(mode, CostFeatureModel::ExactV1 {}) {
            return None;
        }
        let points: Vec<Box<[u64]>> = samples
            .iter()
            .map(|sample| coordinates(&sample.shape).into_boxed_slice())
            .collect();
        let mut minima = points.first()?.clone();
        let mut maxima = minima.clone();
        for point in &points[1..] {
            for ((min, max), value) in minima.iter_mut().zip(maxima.iter_mut()).zip(point.iter()) {
                *min = (*min).min(*value);
                *max = (*max).max(*value);
            }
        }
        Some(Self {
            minima,
            maxima,
            points: points.into(),
        })
    }

    pub(super) fn contains(&self, shape: &WaveExecutionShape) -> bool {
        let values = coordinates(shape);
        if values.len() != self.minima.len()
            || !values
                .iter()
                .zip(self.minima.iter().zip(self.maxima.iter()))
                .all(|(value, (min, max))| min <= value && value <= max)
        {
            return false;
        }
        // One actually measured joint point must cover the entire wave, not
        // a different supporting sample for every request/coordinate.
        self.points.iter().any(|point| {
            point.len() == values.len()
                && point
                    .iter()
                    .zip(&values)
                    .all(|(observed, query)| observed >= query)
        })
    }
}
