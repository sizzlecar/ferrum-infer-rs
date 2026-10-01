//! Private automatic cache storage. A clean manifest is only a replay candidate:
//! it never grants model or feedback authority without independent import.
use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;
use ferrum_types::SloAutomaticCalibrationReuseV1;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

mod codec;
mod coordinator;
mod journal;
mod manifest;
mod replay;
mod seed;
mod store;
pub(super) use coordinator::{AutomaticReuse, RestartOrigin, ReuseOpening};
pub(super) use journal::OriginalSourceJournal;
pub(super) use manifest::SourceKind;
use manifest::*;
pub(super) use replay::{ReplayContext, RestoredCostSeed};
pub(super) use store::{CacheOpen, CacheSession};

#[cfg(test)]
mod replay_tests;
#[cfg(test)]
mod tests;

/// Fixed-size diagnostics; cache failures leave automatic calibration enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CacheMiss {
    Disabled,
    UnsupportedClock,
    Location,
    Busy,
    Dirty,
    Missing,
    Identity,
    CrossBoot,
    Expired,
    Revoked,
    Capacity,
    Deadline,
    Corrupt,
    Io,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ReuseStage {
    ArchiveRestoredSources,
    TrackCatalog,
    PrepareSave,
    ExportSource,
    ReplayCatalog,
    VerifyFreshness,
    CleanShutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct ReuseFailure {
    pub stage: ReuseStage,
    pub reason: CacheMiss,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine) struct ReuseAudit {
    pub(super) first_failure: Option<ReuseFailure>,
    pub(super) prepared: bool,
    pub(super) clean_shutdown: bool,
}

impl ReuseAudit {
    fn failure(&mut self, stage: ReuseStage, reason: CacheMiss) {
        self.first_failure
            .get_or_insert(ReuseFailure { stage, reason });
    }
}
type Result<T> = std::result::Result<T, CacheMiss>;

#[test]
fn reuse_first_failure_survives_later_missing_clean_acknowledgement() {
    let mut audit = ReuseAudit::default();
    audit.failure(ReuseStage::VerifyFreshness, CacheMiss::Expired);
    audit.failure(ReuseStage::CleanShutdown, CacheMiss::Missing);
    assert_eq!(
        audit.first_failure,
        Some(ReuseFailure {
            stage: ReuseStage::VerifyFreshness,
            reason: CacheMiss::Expired,
        })
    );
    assert!(!audit.clean_shutdown);
    let wire = serde_json::to_value(audit).unwrap();
    assert_eq!(wire["first_failure"]["stage"], "verify_freshness");
    assert_eq!(wire["first_failure"]["reason"], "expired");
}
fn io(_: std::io::Error) -> CacheMiss {
    CacheMiss::Io
}

/// Derived from existing automatic and import bounds, never new per-source
/// grants. The same counters apply to every source in a cold transaction.
#[derive(Clone, Copy)]
pub(super) struct CacheLimits {
    pub maximum_bytes: u64,
    pub maximum_source_bytes: u64,
    pub maximum_sources: usize,
    pub maximum_retained_bytes: usize,
    pub maximum_samples: usize,
    pub maximum_rows: usize,
    pub maximum_duration: Duration,
}
impl CacheLimits {
    fn manifest_limit(&self) -> Result<usize> {
        self.maximum_sources
            .checked_mul(4096)
            .and_then(|v| v.checked_add(36 * 1024))
            .filter(|v| *v <= self.maximum_retained_bytes / 16)
            .ok_or(CacheMiss::Capacity)
    }
    fn validate(&self) -> Result<()> {
        if self.maximum_bytes == 0
            || self.maximum_source_bytes == 0
            || self.maximum_sources == 0
            || self.maximum_sources > 128
            || self.maximum_samples == 0
            || self.maximum_rows == 0
            || self.maximum_duration.is_zero()
        {
            return Err(CacheMiss::Capacity);
        }
        self.manifest_limit()?;
        Ok(())
    }
}

pub(super) struct ColdTransaction {
    deadline: Instant,
    maximum_samples: usize,
    maximum_rows: usize,
    samples: usize,
    rows: usize,
}
impl ColdTransaction {
    pub fn new(limits: CacheLimits) -> Result<Self> {
        Ok(Self {
            deadline: Instant::now()
                .checked_add(limits.maximum_duration)
                .ok_or(CacheMiss::Deadline)?,
            maximum_samples: limits.maximum_samples,
            maximum_rows: limits.maximum_rows,
            samples: 0,
            rows: 0,
        })
    }
    pub fn poll(&self) -> Result<()> {
        if Instant::now() >= self.deadline {
            Err(CacheMiss::Deadline)
        } else {
            Ok(())
        }
    }
    /// Importers receive these remaining limits before allocating/replaying;
    /// charge their verified totals afterward, without resetting the deadline.
    pub fn remaining_population(&self) -> (usize, usize) {
        (
            self.maximum_samples - self.samples,
            self.maximum_rows - self.rows,
        )
    }
    pub fn charge_population(&mut self, samples: usize, rows: usize) -> Result<()> {
        self.poll()?;
        let s = self
            .samples
            .checked_add(samples)
            .ok_or(CacheMiss::Capacity)?;
        let r = self.rows.checked_add(rows).ok_or(CacheMiss::Capacity)?;
        if s > self.maximum_samples || r > self.maximum_rows {
            return Err(CacheMiss::Capacity);
        }
        self.samples = s;
        self.rows = r;
        Ok(())
    }
}

fn regular_bytes(path: &Path) -> Result<u64> {
    let meta = fs::symlink_metadata(path).map_err(io)?;
    if !meta.file_type().is_file() {
        return Err(CacheMiss::Corrupt);
    }
    Ok(meta.len())
}
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path).and_then(|f| f.sync_all()).map_err(io)
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}
