use crate::MemorySnapshot;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessMemorySample {
    pub current_bytes: u64,
    pub high_water_bytes: u64,
    pub source: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessMemoryObservation {
    pub before_bytes: u64,
    pub after_bytes: u64,
    pub current_bytes: u64,
    pub high_water_bytes: u64,
    pub source: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessMemorySampler;

impl ProcessMemorySampler {
    pub fn sample(&self) -> Option<ProcessMemorySample> {
        sample_process_memory()
    }

    pub fn observe(&self, before: Option<ProcessMemorySample>) -> Option<ProcessMemoryObservation> {
        let after = self.sample()?;
        Some(ProcessMemoryObservation::from_samples(before, after))
    }
}

impl ProcessMemoryObservation {
    pub fn from_samples(before: Option<ProcessMemorySample>, after: ProcessMemorySample) -> Self {
        let before_bytes = before
            .as_ref()
            .map(|sample| sample.current_bytes)
            .unwrap_or(after.current_bytes);
        let high_water_bytes = before
            .as_ref()
            .map(|sample| sample.high_water_bytes)
            .unwrap_or(0)
            .max(after.high_water_bytes)
            .max(before_bytes)
            .max(after.current_bytes);
        Self {
            before_bytes,
            after_bytes: after.current_bytes,
            current_bytes: after.current_bytes,
            high_water_bytes,
            source: after.source,
        }
    }

    pub fn from_sample(sample: ProcessMemorySample) -> Self {
        Self::from_samples(Some(sample.clone()), sample)
    }

    pub fn to_snapshot(&self, scope: impl Into<String>, backend: Option<&str>) -> MemorySnapshot {
        MemorySnapshot {
            scope: scope.into(),
            backend: backend.map(str::to_string),
            before_bytes: Some(self.before_bytes),
            after_bytes: Some(self.after_bytes),
            current_bytes: Some(self.current_bytes),
            high_water_bytes: Some(self.high_water_bytes),
            available_bytes: None,
        }
    }
}

#[cfg(target_os = "linux")]
fn current_resident_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    resident_pages.checked_mul(page_size as u64)
}

#[cfg(all(not(windows), not(target_os = "linux")))]
fn current_resident_bytes() -> Option<u64> {
    None
}

#[cfg(unix)]
fn high_water_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let max_rss = unsafe { usage.assume_init() }.ru_maxrss;
    if max_rss <= 0 {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        Some(max_rss as u64)
    }
    #[cfg(not(target_os = "macos"))]
    {
        (max_rss as u64).checked_mul(1024)
    }
}

#[cfg(not(any(unix, windows)))]
fn high_water_bytes() -> Option<u64> {
    None
}

#[cfg(not(windows))]
pub fn sample_process_memory() -> Option<ProcessMemorySample> {
    let high_water = high_water_bytes()?;
    let current = current_resident_bytes().unwrap_or(high_water);
    Some(ProcessMemorySample {
        current_bytes: current,
        high_water_bytes: high_water.max(current),
        source: process_memory_source(),
    })
}

#[cfg(windows)]
pub fn sample_process_memory() -> Option<ProcessMemorySample> {
    use windows_sys::Win32::System::{
        ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
        Threading::GetCurrentProcess,
    };

    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: the current-process pseudo-handle remains valid without being closed;
    // counters points to a writable structure with the declared size.
    let succeeded =
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if succeeded == 0 {
        return None;
    }

    // Working-set counters are resident bytes, distinct from page-file commit charge.
    let current_bytes = counters.WorkingSetSize as u64;
    Some(ProcessMemorySample {
        current_bytes,
        high_water_bytes: (counters.PeakWorkingSetSize as u64).max(current_bytes),
        source: "windows_process_memory_counters",
    })
}

#[cfg(target_os = "linux")]
fn process_memory_source() -> &'static str {
    "procfs_statm_plus_getrusage"
}

#[cfg(all(unix, not(target_os = "linux")))]
fn process_memory_source() -> &'static str {
    "getrusage_maxrss"
}

#[cfg(not(any(unix, windows)))]
fn process_memory_source() -> &'static str {
    "unsupported"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_memory_observation_builds_valid_snapshot() {
        let before = ProcessMemorySample {
            current_bytes: 100,
            high_water_bytes: 110,
            source: "test",
        };
        let after = ProcessMemorySample {
            current_bytes: 150,
            high_water_bytes: 160,
            source: "test",
        };
        let observation = ProcessMemoryObservation::from_samples(Some(before), after);
        let snapshot = observation.to_snapshot("process", Some("actual"));
        assert_eq!(snapshot.before_bytes, Some(100));
        assert_eq!(snapshot.after_bytes, Some(150));
        assert_eq!(snapshot.current_bytes, Some(150));
        assert_eq!(snapshot.high_water_bytes, Some(160));
        snapshot.validate().unwrap();
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn process_memory_sampler_returns_valid_resident_bytes() {
        let sample = ProcessMemorySampler
            .sample()
            .expect("supported platform samples current-process memory");
        assert!(sample.current_bytes > 0);
        assert!(sample.high_water_bytes >= sample.current_bytes);
        ProcessMemoryObservation::from_sample(sample)
            .to_snapshot("process", None)
            .validate()
            .unwrap();
    }
}
