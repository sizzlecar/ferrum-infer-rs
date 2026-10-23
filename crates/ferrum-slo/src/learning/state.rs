use super::{snapshot, LearningConfig};
use crate::{
    calibration::{bounded_sequence, OnlineCalibrator, PersistedCalibration},
    model::{ExecutionIdentity, FeatureDomain, FittedModel, ModelFamily, FEATURE_SCHEMA},
    EstimatorError,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const STATE_SCHEMA: u32 = 1;
const MAX_STATE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum LearningStateError {
    #[error(transparent)]
    Estimator(#[from] EstimatorError),
    #[error("learning state I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid learning state: {0}")]
    Invalid(String),
    #[error("learning state has no qualified unexpired domain")]
    Stale,
    #[error("wall clock moved backwards since learning state was saved")]
    ClockRollback,
    #[error("learning state exceeds the declared resource budget")]
    ResourceLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StateLoadStatus {
    Missing,
    Loaded,
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StateLoadReport {
    pub status: StateLoadStatus,
    pub loaded_domains: usize,
    pub expired_domains: usize,
    /// Entries in the global and conditional views, not distinct invocations.
    pub expired_residual_entries: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StateSaveReport {
    Saved { domains: usize },
    SkippedNoQualifiedState,
}

pub(super) struct RestoredDomain {
    pub model: Arc<FittedModel>,
    pub model_revision: u64,
    pub model_expires_at: Instant,
    pub calibration: OnlineCalibrator<FeatureDomain>,
}

pub(super) struct RestoredState {
    pub family: ModelFamily,
    pub revision: u64,
    pub domains: Vec<RestoredDomain>,
    pub report: StateLoadReport,
    pub loaded_at: Instant,
}

#[derive(Clone, Copy)]
pub(super) struct Clock {
    pub monotonic: Instant,
    pub wall: SystemTime,
}

impl Clock {
    pub fn now() -> Self {
        Self {
            monotonic: Instant::now(),
            wall: SystemTime::now(),
        }
    }
    fn for_save() -> Self {
        // Saving uses the later monotonic endpoint with the earlier wall
        // endpoint. Loading does the reverse in `now`. Clock-read skew can
        // therefore consume a little extra TTL, but cannot renew evidence.
        let wall = SystemTime::now();
        let monotonic = Instant::now();
        Self { monotonic, wall }
    }
    fn unix_ns(self) -> Result<u64, LearningStateError> {
        self.wall
            .duration_since(UNIX_EPOCH)
            .map_err(|_| LearningStateError::ClockRollback)?
            .as_nanos()
            .try_into()
            .map_err(|_| LearningStateError::Invalid("wall-clock range".into()))
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Semantics {
    // Files written before width history existed used Recent exclusively.
    // A missing field must never be inferred as the caller's selected policy.
    #[serde(default)]
    training_history: super::TrainingHistoryPolicy,
    calibration_samples: usize,
    minimum_samples: usize,
    calibration_ttl_ms: u64,
    model_ttl_ms: u64,
    risk_quantile: f64,
    #[serde(deserialize_with = "read_boundaries")]
    work_boundaries: Vec<usize>,
}

impl From<&LearningConfig> for Semantics {
    fn from(config: &LearningConfig) -> Self {
        Self {
            training_history: config.training_history,
            calibration_samples: config.calibration_samples,
            minimum_samples: config.min_calibration_samples,
            calibration_ttl_ms: config.calibration_ttl_ms,
            model_ttl_ms: config.model_ttl_ms,
            risk_quantile: config.training.risk_quantile,
            work_boundaries: config.work_bucket_boundaries.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedDomain {
    model: String,
    model_revision: u64,
    model_remaining_ns: u64,
    calibration: PersistedCalibration<FeatureDomain>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedState {
    schema: u32,
    feature_schema: u32,
    family: ModelFamily,
    semantics: Semantics,
    saved_unix_ns: u64,
    revision: u64,
    #[serde(deserialize_with = "read_domains")]
    domains: Vec<SavedDomain>,
}

fn read_domains<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<SavedDomain>, D::Error> {
    bounded_sequence::<D, SavedDomain, 9>(d)
}

fn read_boundaries<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<usize>, D::Error> {
    bounded_sequence::<D, usize, 8>(d)
}

fn invalid(error: impl std::fmt::Display) -> LearningStateError {
    LearningStateError::Invalid(error.to_string())
}

fn byte_limit(config: &LearningConfig) -> Result<usize, LearningStateError> {
    // The input buffer and typed decoded strings/records each fit this cap.
    // Restored models/calibration remain within the configured live-state cap;
    // reject a loader whose combined requested payload cannot fit that budget.
    let limit = MAX_STATE_BYTES.min(config.max_retained_bytes / 4);
    let per_model = config.training.owned_bytes_bound(0).map_err(invalid)?;
    let calibration =
        OnlineCalibrator::<FeatureDomain>::retained_bytes_bound(config.calibration_config())
            .ok_or(LearningStateError::ResourceLimit)?;
    let retained = (per_model + calibration)
        .checked_mul(config.max_domains)
        .ok_or(LearningStateError::ResourceLimit)?;
    if limit == 0
        || limit
            .checked_mul(2)
            .and_then(|n| n.checked_add(retained))
            .and_then(|n| n.checked_add(64 * 1024))
            .is_none_or(|n| n > config.max_retained_bytes)
    {
        return Err(LearningStateError::ResourceLimit);
    }
    Ok(limit)
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|size| size > self.limit)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "learning state byte limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn encode(
    snapshot: &snapshot::Snapshot,
    config: &LearningConfig,
    family: ModelFamily,
    clock: Clock,
) -> Result<(Vec<u8>, usize), LearningStateError> {
    let limit = byte_limit(config)?;
    let mut domains = Vec::with_capacity(snapshot.domains.len());
    for domain in &snapshot.domains {
        if clock.monotonic >= domain.model_expires_at
            || domain
                .calibration
                .correction(
                    domain.model_revision,
                    &domain.model.domain(),
                    u32::MAX,
                    clock.monotonic,
                )
                .is_none()
        {
            continue;
        }
        let model = String::from_utf8(domain.model.to_json(&config.training).map_err(invalid)?)
            .map_err(invalid)?;
        let model_remaining_ns = domain
            .model_expires_at
            .duration_since(clock.monotonic)
            .as_nanos()
            .try_into()
            .map_err(|_| invalid("model TTL range"))?;
        domains.push(SavedDomain {
            model,
            model_revision: domain.model_revision,
            model_remaining_ns,
            calibration: domain
                .calibration
                .persist(&domain.model.domain(), clock.monotonic)
                .map_err(invalid)?,
        });
    }
    if domains.is_empty() {
        return Err(LearningStateError::Stale);
    }
    let saved_domains = domains.len();
    let state = SavedState {
        schema: STATE_SCHEMA,
        feature_schema: FEATURE_SCHEMA,
        family,
        semantics: Semantics::from(config),
        saved_unix_ns: clock.unix_ns()?,
        revision: snapshot.revision,
        domains,
    };
    let mut writer = BoundedWriter {
        bytes: Vec::with_capacity(limit),
        limit,
    };
    serde_json::to_writer(&mut writer, &state).map_err(invalid)?;
    Ok((writer.bytes, saved_domains))
}

fn decode(
    bytes: &[u8],
    config: &LearningConfig,
    identity: &ExecutionIdentity,
    family: ModelFamily,
    clock: Clock,
) -> Result<RestoredState, LearningStateError> {
    decode_selected(bytes, config, identity, Some(family), clock)
}

fn decode_selected(
    bytes: &[u8],
    config: &LearningConfig,
    identity: &ExecutionIdentity,
    expected_family: Option<ModelFamily>,
    clock: Clock,
) -> Result<RestoredState, LearningStateError> {
    if bytes.len() > byte_limit(config)? {
        return Err(LearningStateError::ResourceLimit);
    }
    let state: SavedState = serde_json::from_slice(bytes).map_err(invalid)?;
    if state.schema != STATE_SCHEMA
        || state.feature_schema != FEATURE_SCHEMA
        || expected_family.is_some_and(|family| state.family != family)
        || state.semantics != Semantics::from(config)
        || state.domains.len() > config.max_domains
        || state.revision == 0
        || state.revision == u64::MAX
    {
        return Err(invalid(
            "state schema, family, configuration or domain bound mismatch",
        ));
    }
    let family = state.family;
    let elapsed = Duration::from_nanos(
        clock
            .unix_ns()?
            .checked_sub(state.saved_unix_ns)
            .ok_or(LearningStateError::ClockRollback)?,
    );
    let mut report = StateLoadReport {
        status: StateLoadStatus::Loaded,
        loaded_domains: 0,
        expired_domains: 0,
        expired_residual_entries: 0,
    };
    let mut domains: Vec<RestoredDomain> = Vec::with_capacity(state.domains.len());
    let mut seen = Vec::with_capacity(state.domains.len());
    for saved in state.domains {
        let model = FittedModel::from_json(saved.model.as_bytes(), identity, &config.training)
            .map_err(invalid)?;
        if model.family() != family
            || model
                .risk_quantile()
                .is_some_and(|q| q != config.training.risk_quantile)
            || saved.model_revision == 0
            || saved.model_revision > state.revision
            || seen.contains(&model.domain())
        {
            return Err(invalid("state model family, domain or revision mismatch"));
        }
        seen.push(model.domain());
        let (calibration, expired) = saved
            .calibration
            .restore(
                config.calibration_config(),
                &model.domain(),
                saved.model_revision,
                clock.monotonic,
                elapsed,
            )
            .map_err(invalid)?;
        report.expired_residual_entries += expired;
        let remaining = Duration::from_nanos(saved.model_remaining_ns);
        if remaining > Duration::from_millis(config.model_ttl_ms) {
            return Err(invalid("persisted model TTL exceeds configured TTL"));
        }
        let Some(remaining) = remaining.checked_sub(elapsed).filter(|d| !d.is_zero()) else {
            report.expired_domains += 1;
            continue;
        };
        if calibration
            .snapshot(clock.monotonic)
            .correction(
                saved.model_revision,
                &model.domain(),
                u32::MAX,
                clock.monotonic,
            )
            .is_none()
        {
            report.expired_domains += 1;
            continue;
        }
        let model_expires_at = clock
            .monotonic
            .checked_add(remaining)
            .ok_or_else(|| invalid("restored model TTL range"))?;
        domains.push(RestoredDomain {
            model: Arc::new(model),
            model_revision: saved.model_revision,
            model_expires_at,
            calibration,
        });
        report.loaded_domains += 1;
    }
    if domains.is_empty() {
        report.status = StateLoadStatus::Stale;
    }
    Ok(RestoredState {
        family,
        revision: state.revision,
        domains,
        report,
        loaded_at: clock.monotonic,
    })
}

pub(super) fn load(
    path: &Path,
    config: &LearningConfig,
    identity: &ExecutionIdentity,
    family: ModelFamily,
) -> Result<Option<RestoredState>, LearningStateError> {
    load_selected(path, config, identity, Some(family))
}

pub(super) fn load_selected(
    path: &Path,
    config: &LearningConfig,
    identity: &ExecutionIdentity,
    expected_family: Option<ModelFamily>,
) -> Result<Option<RestoredState>, LearningStateError> {
    let limit = byte_limit(config)?;
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if file.metadata()?.len() > limit as u64 {
        return Err(LearningStateError::ResourceLimit);
    }
    let mut bytes = Vec::with_capacity(limit + 1);
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    match expected_family {
        Some(family) => decode(&bytes, config, identity, family, Clock::now()),
        None => decode_selected(&bytes, config, identity, None, Clock::now()),
    }
    .map(Some)
}

pub(super) fn save(
    path: &Path,
    snapshot: &snapshot::Snapshot,
    config: &LearningConfig,
    family: ModelFamily,
) -> Result<StateSaveReport, LearningStateError> {
    let (bytes, domains) = match encode(snapshot, config, family, Clock::for_save()) {
        Ok(value) => value,
        Err(LearningStateError::Stale) => return Ok(StateSaveReport::SkippedNoQualifiedState),
        Err(error) => return Err(error),
    };
    atomic_write(path, &bytes)?;
    Ok(StateSaveReport::Saved { domains })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), LearningStateError> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| invalid("state path needs a filename"))?;
    for _ in 0..16 {
        let mut temporary_name = name.to_os_string();
        temporary_name.push(format!(
            ".ferrum-state-{}-{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let temporary = parent.join(temporary_name);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&temporary) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        };
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_file(&self.0);
            }
        }
        let _cleanup = Cleanup(temporary.clone());
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        return Ok(());
    }
    Err(invalid("could not reserve an atomic state temporary file"))
}

#[cfg(test)]
mod tests;
