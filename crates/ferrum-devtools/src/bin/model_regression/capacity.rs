//! Observe the executing admission policy and device-wide budget, not only the
//! requested command line. Capacity refusal remains a failed runner case: the
//! raw reason must be inspected before accepting an expected boundary result.
use super::Args;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum SequenceFitPolicy {
    FullInputMustFit,
    ImmediateOnly,
}

impl SequenceFitPolicy {
    pub(super) fn cli_name(self) -> &'static str {
        match self {
            Self::FullInputMustFit => "full-input-must-fit",
            Self::ImmediateOnly => "immediate-only",
        }
    }
    fn runtime_name(self) -> &'static str {
        match self {
            Self::FullInputMustFit => "full_input_must_fit",
            Self::ImmediateOnly => "immediate_only",
        }
    }
}

pub(super) fn validate_effective_policy(args: &Args, config: &Value) -> Result<()> {
    for (key, expected) in [
        (
            "FERRUM_SEQUENCE_FIT_POLICY",
            args.sequence_fit_policy
                .map(|policy| policy.cli_name().to_owned()),
        ),
        (
            "FERRUM_MAX_BATCHED_TOKENS",
            args.max_num_batched_tokens.map(|tokens| tokens.to_string()),
        ),
    ] {
        let Some(expected) = expected else { continue };
        let entries = config["entries"]
            .as_array()
            .context("effective config omitted runtime entries")?;
        let matches: Vec<_> = entries.iter().filter(|entry| entry["key"] == key).collect();
        ensure!(
            matches.len() == 1 && matches[0]["effective_value"] == expected,
            "effective {key} differs from explicit request"
        );
    }
    Ok(())
}

pub(super) fn validate_runtime_policy(args: &Args, health: &Value) -> Result<()> {
    if args.sequence_fit_policy.is_none() && args.max_num_batched_tokens.is_none() {
        // Existing release tasks can exercise other execution authorities. The
        // explicitly selected fit-policy probe requires vNext runtime evidence.
        return Ok(());
    }
    let executor = &health["cache"]["prefix_cache"];
    ensure!(
        executor["source"] == "vnext-native-sequence-checkpoint-cache",
        "admission policy evidence must come from the actual vNext executor"
    );
    if let Some(expected) = args.sequence_fit_policy {
        ensure!(
            executor["runtime_admission_policy"]["sequence_fit_policy"] == expected.runtime_name(),
            "executing sequence fit policy differs from explicit request"
        );
    }
    if let Some(expected) = args.max_num_batched_tokens {
        ensure!(
            executor["runtime_admission_policy"]["maximum_scheduled_tokens"].as_u64()
                == Some(u64::from(expected)),
            "executing maximum scheduled tokens differs from explicit request"
        );
    }
    validate_effective_policy(args, &health["auto_config"])?;
    if let Some(expected) = args.runtime_memory_budget_bytes {
        // This flag selects usable bytes; raw device capacity retains the reserve.
        let memory = &executor["runtime_memory_policy"];
        let raw = memory["capacity_bytes"]
            .as_u64()
            .context("executor omitted memory capacity")?;
        let reserve = memory["reserve_bytes"]
            .as_u64()
            .context("executor omitted memory reserve")?;
        ensure!(
            raw.checked_sub(reserve) == Some(expected),
            "executing usable memory budget differs from explicit request"
        );
        let pools = &executor["dynamic_pools"];
        for field in [
            "budget_device_wide_usable_ceiling_bytes",
            "effective_device_usable_ceiling_bytes",
        ] {
            ensure!(
                pools[field].as_u64() == Some(expected),
                "actual {field} differs from explicit memory budget"
            );
        }
    }
    Ok(())
}
