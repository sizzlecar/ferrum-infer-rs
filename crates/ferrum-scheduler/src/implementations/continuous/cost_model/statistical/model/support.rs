use super::*;
/// Full observed coordinates, not just the regression axes. Frozen joint
/// evidence is bounded by the original per-bucket sample limit.
pub(super) const AXES: usize = 26;
pub(super) fn coordinates(input: &StatisticalModelInputV1) -> [u64; AXES] {
    let d = input.device();
    let h = input.host_and_sequence();
    [
        d.logical_units,
        d.padded_units,
        d.inner_work_units,
        d.grid_blocks,
        d.peak_scratch_bytes,
        d.staged_weight_bytes,
        d.host_to_device_bytes,
        d.device_to_host_bytes,
        d.device_to_device_bytes,
        d.fill_bytes,
        h.rows,
        h.prefill_tokens,
        h.attention_pairs,
        h.kv_tokens_sum,
        h.kv_tokens_max,
        h.prompt_tokens_sum,
        h.prompt_tokens_max,
        h.generated_tokens_sum,
        h.output_budget_sum,
        h.sampling_history_sum,
        h.sampling_history_max,
        h.repetition_tokens_sum,
        h.decoded_prefix_sum,
        h.decoded_text_bytes_sum,
        h.decode_scratch_bytes_sum,
        h.recurrent_bytes,
    ]
}
#[derive(Debug, Clone)]
pub(super) struct Support {
    minimum: [u64; AXES],
    maximum: [u64; AXES],
    points: Vec<[u64; AXES]>,
}
impl Support {
    pub(super) fn new(points: impl Iterator<Item = [u64; AXES]>) -> Result<Self, ModelUnknown> {
        let points: Vec<_> = points.collect();
        if points.is_empty() || points.len() > 4096 {
            return Err(ModelUnknown::Capacity);
        }
        let mut minimum = [u64::MAX; AXES];
        let mut maximum = [0; AXES];
        for p in &points {
            for i in 0..AXES {
                minimum[i] = minimum[i].min(p[i]);
                maximum[i] = maximum[i].max(p[i]);
            }
        }
        Ok(Self {
            minimum,
            maximum,
            points,
        })
    }
    pub(super) fn contains(&self, query: &[u64; AXES]) -> bool {
        // One complete supporting point is necessary; independent maxima are
        // not permission to combine unobserved worst-case dimensions.
        query
            .iter()
            .enumerate()
            .all(|(i, q)| *q >= self.minimum[i] && *q <= self.maximum[i])
            && self
                .points
                .iter()
                .any(|p| query.iter().zip(p).all(|(q, p)| q <= p))
    }
}
