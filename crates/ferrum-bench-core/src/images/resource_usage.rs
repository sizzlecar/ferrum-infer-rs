//! Hardware-independent parsing and aggregation of sampled resource evidence.
use serde::Serialize;
use std::{collections::BTreeSet, io};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_ticks: u64,
}

pub fn proc_stat(stat: &str, pid: u32) -> io::Result<(ProcessIdentity, u32)> {
    if stat
        .split_once(' ')
        .and_then(|(p, _)| p.parse::<u32>().ok())
        != Some(pid)
    {
        return Err(invalid("proc stat PID differs from requested process"));
    }
    let tail = stat
        .rsplit_once(") ")
        .ok_or_else(|| invalid("invalid proc stat command"))?
        .1;
    let fields = tail.split_whitespace().collect::<Vec<_>>();
    let parent = fields
        .get(1)
        .and_then(|field| field.parse().ok())
        .ok_or_else(|| invalid("invalid proc stat parent"))?;
    let start_ticks = fields
        .get(19)
        .and_then(|field| field.parse().ok())
        .ok_or_else(|| invalid("invalid proc stat starttime"))?;
    Ok((ProcessIdentity { pid, start_ticks }, parent))
}

pub fn proc_start_ticks(stat: &str, pid: u32) -> io::Result<u64> {
    Ok(proc_stat(stat, pid)?.0.start_ticks)
}

pub fn proc_memory_bytes(status: &str, name: &str) -> io::Result<Option<u64>> {
    let mut value = None;
    for line in status.lines().filter(|line| line.starts_with(name)) {
        if value.is_some() {
            return Err(invalid("duplicate proc memory field"));
        }
        let mut fields = line.split_whitespace();
        if fields.next() != Some(name) {
            return Err(invalid("invalid proc memory field"));
        }
        let n = fields
            .next()
            .and_then(|n| n.parse::<u64>().ok())
            .and_then(|n| n.checked_mul(1024));
        if fields.next() != Some("kB") || fields.next().is_some() || n.is_none() {
            return Err(invalid("invalid proc memory count or unit"));
        }
        value = n;
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuIdentity {
    pub index: u32,
    pub uuid: String,
}

pub fn nvidia_memory_csv(text: &str, indices: &[u32]) -> io::Result<(Vec<GpuIdentity>, u64)> {
    let mut found = BTreeSet::new();
    let mut identities = Vec::new();
    let mut sum = 0u64;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(invalid("invalid NVIDIA memory row"));
        }
        let index = fields[0]
            .parse::<u32>()
            .map_err(|_| invalid("invalid GPU index"))?;
        if !indices.contains(&index) {
            return Err(invalid("unexpected GPU index"));
        }
        if !found.insert(index) {
            return Err(invalid("duplicate GPU index"));
        }
        let uuid = fields[1];
        if !uuid.starts_with("GPU-")
            || uuid.len() > 128
            || !uuid.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return Err(invalid("invalid physical GPU UUID"));
        }
        let bytes = fields[2]
            .parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(1024 * 1024))
            .ok_or_else(|| invalid("GPU memory.used is unavailable or invalid"))?;
        sum = sum
            .checked_add(bytes)
            .ok_or_else(|| invalid("GPU memory sum overflow"))?;
        identities.push(GpuIdentity {
            index,
            uuid: uuid.into(),
        });
    }
    if found.len() != indices.len() || indices.is_empty() {
        return Err(invalid("GPU memory sample is incomplete"));
    }
    identities.sort_by_key(|gpu| gpu.index);
    Ok((identities, sum))
}

/// Even complete successful sampling provides a lower bound on the true peak.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SampledMemory {
    pub successful_samples: u64,
    pub failed_samples: u64,
    pub observed_peak_bytes: Option<u64>,
    pub peak_offset_ms: Option<f64>,
}

impl SampledMemory {
    pub fn record(&mut self, bytes: Option<u64>, offset_ms: f64) {
        if let Some(bytes) = bytes {
            self.successful_samples += 1;
            if self.observed_peak_bytes.is_none_or(|old| bytes > old) {
                self.observed_peak_bytes = Some(bytes);
                self.peak_offset_ms = Some(offset_ms);
            }
        } else {
            self.failed_samples += 1;
        }
    }
    pub fn complete(&self) -> bool {
        self.successful_samples > 0 && self.failed_samples == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proc_parsing_preserves_identity_and_rejects_invalid_units() {
        let mut fields: Vec<String> = vec!["S".into(), "7".into()];
        fields.extend((5..22).map(|_| "0".into()));
        fields.push("123".into());
        let text = format!("42 (worker ) spaced) {}", fields.join(" "));
        assert_eq!(
            proc_stat(&text, 42).unwrap(),
            (
                ProcessIdentity {
                    pid: 42,
                    start_ticks: 123
                },
                7
            )
        );
        assert!(proc_stat(&text, 41).is_err());
        assert_eq!(
            proc_memory_bytes("VmRSS: 0 kB\n", "VmRSS:").unwrap(),
            Some(0)
        );
        for text in [
            "VmRSS: 1 bytes",
            "VmRSS: 1 kB\nVmRSS: 2 kB",
            "VmRSS: 18446744073709551615 kB",
        ] {
            assert!(proc_memory_bytes(text, "VmRSS:").is_err());
        }
    }
    #[test]
    fn gpu_memory_is_exact_requested_device_sum_with_explicit_missing_errors() {
        let (identities, sum) = nvidia_memory_csv("1, GPU-b, 2\n0, GPU-a, 0\n", &[0, 1]).unwrap();
        assert_eq!(sum, 2 * 1024 * 1024);
        assert_eq!(identities[0].index, 0);
        for text in [
            "0, GPU-a, N/A",
            "0, GPU-a, -1",
            "0, GPU-a, 1\n0, GPU-a, 2",
            "0, GPU-a, 1",
            "0, GPU-a, 18446744073709551615\n1, GPU-b, 0",
        ] {
            assert!(nvidia_memory_csv(text, &[0, 1]).is_err());
        }
    }
    #[test]
    fn missing_measurements_never_become_zero_or_complete() {
        let mut samples = SampledMemory::default();
        samples.record(None, 0.0);
        assert_eq!(samples.observed_peak_bytes, None);
        assert!(!samples.complete());
        samples.record(Some(0), 1.0);
        samples.record(Some(128), 2.0);
        samples.record(Some(64), 3.0);
        assert_eq!(samples.observed_peak_bytes, Some(128));
        assert_eq!(samples.peak_offset_ms, Some(2.0));
        assert!(!samples.complete());
    }
}
