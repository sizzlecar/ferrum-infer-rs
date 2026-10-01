use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub samples: usize,
    pub minimum_ns: u64,
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p99_ns: u64,
    pub maximum_ns: u64,
    pub total_ns: u128,
}
pub(super) fn distributions(input: BTreeMap<String, Vec<u64>>) -> BTreeMap<String, Distribution> {
    input
        .into_iter()
        .filter_map(|(key, mut values)| {
            if values.is_empty() {
                return None;
            }
            values.sort_unstable();
            let percentile = |p: usize| values[(values.len() * p).div_ceil(100).saturating_sub(1)];
            Some((
                key,
                Distribution {
                    samples: values.len(),
                    minimum_ns: values[0],
                    p50_ns: percentile(50),
                    p90_ns: percentile(90),
                    p99_ns: percentile(99),
                    maximum_ns: *values.last().unwrap(),
                    total_ns: values.iter().map(|n| u128::from(*n)).sum(),
                },
            ))
        })
        .collect()
}
pub(super) fn count(counters: &mut BTreeMap<String, u64>, key: impl Into<String>, n: u64) {
    *counters.entry(key.into()).or_default() += n;
}
pub(super) fn value(metrics: &mut BTreeMap<String, Vec<u64>>, key: impl Into<String>, n: u64) {
    metrics.entry(key.into()).or_default().push(n);
}
