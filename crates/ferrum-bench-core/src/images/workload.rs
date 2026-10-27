//! Per-request workload partitions retain failures and the common wall window.

use super::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageWorkloadMetrics {
    pub request_kind: String,
    pub width: u32,
    pub height: u32,
    /// images_per_second is this class's contribution over the complete shared
    /// measurement window, not throughput of an isolated homogeneous workload.
    pub metrics: ImageMetrics,
}

pub fn image_workload_metrics(
    elapsed_seconds: f64,
    records: &[ImageRequestRecord],
    dataset: &ImageDataset,
    latency_slo_p99_ms: Option<f64>,
) -> Result<Vec<ImageWorkloadMetrics>, ImageBenchError> {
    let mut groups = BTreeMap::<(bool, u32, u32), Vec<&ImageRequestRecord>>::new();
    for record in records {
        let sample = dataset
            .samples
            .get(record.sample_index)
            .filter(|sample| sample.id == record.sample_id)
            .ok_or_else(|| invalid("request record does not identify a dataset sample"))?;
        let generate = sample.effective_mode(&dataset.mode).is_generate();
        groups
            .entry((generate, sample.width, sample.height))
            .or_default()
            .push(record);
    }
    Ok(groups
        .into_iter()
        .map(
            |((generate, width, height), records)| ImageWorkloadMetrics {
                request_kind: if generate { "generate" } else { "edit" }.into(),
                width,
                height,
                metrics: ImageMetrics::from_records(elapsed_seconds, records, latency_slo_p99_ms),
            },
        )
        .collect())
}
