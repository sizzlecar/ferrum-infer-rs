//! Shared run/serve validation and provenance for bounded diagnostic capture.

use ferrum_types::{
    ExecutionResourceAuthority, FerrumError, ObservabilityProfileDetail, Result,
    RuntimeConfigEntry, RuntimeConfigSnapshot, RuntimeConfigSource, RuntimeKnobs,
    PROFILE_MAX_FRAMES_PER_REQUEST_CONFIG_KEY,
};
use std::{
    num::NonZeroU32,
    path::{Path, PathBuf},
};

pub(super) const HELP: &str = "Capture at most N execution frames per request in resource/kernel/replay/verify/full profiles. Requires a native plan runtime and --profile-jsonl or --scheduler-trace-jsonl. Limits diagnostics only; keeps the full inference/output workload and existing first-frame modes unchanged. Config: runtime.profile_max_frames_per_request. No environment override.";

pub(super) fn push_cli_entry(entries: &mut Vec<RuntimeConfigEntry>, limit: Option<NonZeroU32>) {
    if let Some(limit) = limit {
        entries.push(RuntimeConfigEntry::new(
            PROFILE_MAX_FRAMES_PER_REQUEST_CONFIG_KEY,
            limit.to_string(),
            RuntimeConfigSource::Cli,
        ));
    }
}

pub(super) fn validate_requested(
    limit: Option<NonZeroU32>,
    detail: ObservabilityProfileDetail,
    profile_jsonl: Option<&Path>,
    scheduler_trace_jsonl: Option<&Path>,
    environment: &RuntimeConfigSnapshot,
) -> Result<()> {
    if limit.is_none() {
        return Ok(());
    }
    // Only the existing sink path aliases may come from the captured user
    // environment. The frame limit itself is exclusively typed CLI/config.
    let path = |cli: Option<&Path>, key| {
        cli.map(Path::to_owned).or_else(|| {
            crate::runtime_env::runtime_snapshot_value(environment, key).map(PathBuf::from)
        })
    };
    RuntimeKnobs {
        profile_detail: detail,
        profile_max_frames_per_request: limit,
        profile_jsonl: path(profile_jsonl, "FERRUM_PROFILE_JSONL"),
        scheduler_trace_jsonl: path(scheduler_trace_jsonl, "FERRUM_SCHEDULER_TRACE_JSONL"),
        ..Default::default()
    }
    .validate_profile_frame_limit()
    .map_err(FerrumError::config)
}

pub(super) fn validate_authority(
    limit: Option<NonZeroU32>,
    authority: ExecutionResourceAuthority,
) -> Result<()> {
    if limit.is_some() && authority != ExecutionResourceAuthority::PlanRuntime {
        return Err(FerrumError::unsupported(
            "--profile-max-frames-per-request requires a native plan runtime; legacy and synthetic execution do not support bounded frame capture",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
