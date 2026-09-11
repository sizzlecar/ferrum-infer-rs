//! Failure-only test output. All values come from the failing recording or
//! resolution attempt; this diagnostic never samples a new timestamp anchor.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn report(
    cause: &str,
    anchor_start: (u64, u64),
    anchor_end: Option<(u64, u64)>,
    command: Option<u32>,
    page: Option<usize>,
    sample_indices: Option<(u64, u64)>,
    raw_start: Option<u64>,
    raw_end: Option<u64>,
) {
    eprintln!(
        "ferrum_metal_counter_failure {}",
        serde_json::json!({
            "cause": cause,
            "cpu_anchor_start": anchor_start.0,
            "gpu_anchor_start": anchor_start.1,
            "cpu_anchor_end": anchor_end.map(|anchor| anchor.0),
            "gpu_anchor_end": anchor_end.map(|anchor| anchor.1),
            "command_index": command,
            "page_index": page,
            "start_sample_index": sample_indices.map(|indices| indices.0),
            "end_sample_index": sample_indices.map(|indices| indices.1),
            "raw_start": raw_start,
            "raw_end": raw_end,
        })
    );
}

impl MetalCounterCapture {
    pub(super) fn diagnose(
        &self,
        cause: &str,
        anchor_end: Option<(u64, u64)>,
        mapping: Option<&MetalCounterIntervalMapping>,
        page_samples: &[&[u64]],
    ) {
        let samples = mapping.and_then(|mapping| page_samples.get(mapping.page_index));
        let raw = |index: Option<u64>| {
            index
                .and_then(|index| usize::try_from(index).ok())
                .and_then(|index| samples.and_then(|samples| samples.get(index)))
                .copied()
        };
        report(
            cause,
            (self.cpu_anchor_start, self.gpu_anchor_start),
            anchor_end,
            mapping.map(|mapping| mapping.command_index),
            mapping.map(|mapping| mapping.page_index),
            mapping.map(|mapping| (mapping.start_sample_index, mapping.end_sample_index)),
            raw(mapping.map(|mapping| mapping.start_sample_index)),
            raw(mapping.map(|mapping| mapping.end_sample_index)),
        );
    }
}
