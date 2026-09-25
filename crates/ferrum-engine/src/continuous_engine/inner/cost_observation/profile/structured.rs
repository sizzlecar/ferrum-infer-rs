//! Startup replay validates the frozen source once. Candidate queries consume
//! cached typed inputs and retain the source clock; they perform no IO or fit.
use super::*;
use file::structured_v9::{
    load_structured_profile_v9, ImportedStructuredModelV1, StructuredProfilePhaseV9,
};
use model::structured::{StructuredUnknown, MODEL_REVISION, POPULATION_REVISION};

pub(super) fn load_seed(
    fingerprint: model::ExecutionFingerprint,
    config: &SloCostObservationConfig,
    path: Option<&Path>,
    clock: Option<file::ProfileLoadClock>,
) -> Result<TrainingSeed, FerrumError> {
    let Some(path) = path else {
        // Observe/calibration may collect a new source without an installed
        // predictor. It never starts the legacy online trainer in this mode.
        return Ok(TrainingSeed {
            trainer: None,
            snapshot: None,
            receipt: None,
        });
    };
    let clock =
        clock.ok_or_else(|| FerrumError::config("structured profile load clock missing"))?;
    let declared = config
        .profile_import
        .declared_local_clock_max_error_ns
        .ok_or_else(|| {
            FerrumError::config("structured profile requires declared local wall-clock accuracy")
        })?;
    if clock.wall_max_error_ns != Some(declared) {
        return Err(FerrumError::config(
            "structured profile clock differs from declared policy",
        ));
    }
    let imported = load_structured_profile_v9(
        path,
        &fingerprint,
        &load_limits(&config.profile_import),
        clock,
    )
    .map_err(profile_error)?;
    let receipt = import_receipt(&imported, declared)?;
    Ok(TrainingSeed {
        trainer: None,
        snapshot: Some(Arc::new(EngineCostSnapshot {
            inner: Snapshot::Structured(imported),
            fingerprint,
        })),
        receipt: Some(receipt),
    })
}

fn import_receipt(
    imported: &ImportedStructuredModelV1,
    declared: u64,
) -> Result<SloCostProfileReceipt, FerrumError> {
    let p = imported.provenance();
    let phases = p
        .phases
        .each_ref()
        .map(|phase| ferrum_types::SloStructuredPhaseReceiptV1 {
            phase: match phase.phase {
                StructuredProfilePhaseV9::Fit => ferrum_types::SloStructuredProfilePhaseV1::Fit,
                StructuredProfilePhaseV9::Residual => {
                    ferrum_types::SloStructuredProfilePhaseV1::Residual
                }
                StructuredProfilePhaseV9::Qualification => {
                    ferrum_types::SloStructuredProfilePhaseV1::Qualification
                }
            },
            members: phase.members,
            member_cutoff: phase.member_cutoff,
            accepted_fifo_cutoff: phase.accepted_fifo_cutoff,
            frozen_at_ns: phase.frozen_at_ns,
            source_prefix_bytes: phase.source_prefix_bytes,
            source_prefix_sha256: phase.source_prefix_sha256,
            parameters_sha256: phase.parameters_sha256,
        });
    let size = |n| {
        usize::try_from(n)
            .map_err(|_| FerrumError::config("structured profile receipt size overflow"))
    };
    Ok(SloCostProfileReceipt {
        selected_whole_wave: None,
        structured_whole_wave_v2: None,
        structured_whole_wave: Some(ferrum_types::SloStructuredWholeWaveReceiptV1 {
            model_revision: MODEL_REVISION.to_owned(),
            domain_signature: *imported.domain_signature(),
            capture_identity_sha256: p.capture_identity,
            protocol_sha256: p.protocol,
            rule_signature: p.rule_signature,
            parameters_sha256: p.parameters_sha256,
            source_path: p.source_path.clone(),
            source_bytes: p.source_bytes,
            phases,
        }),
        schema_version: p.schema_version,
        path: p.loaded_from.clone(),
        file_sha256: format!(
            "sha256:{}",
            p.file_sha256
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ),
        file_bytes: size(p.file_bytes)?,
        generated_unix_ns: p.generated_unix_ns,
        loaded_unix_ns: p.loaded_unix_ns,
        conservative_clock_error_ns: p.conservative_clock_error_ns,
        declared_local_clock_max_error_ns: declared,
        oldest_imported_age_ns: Some(p.oldest_imported_age_ns),
        newest_imported_age_ns: Some(p.newest_imported_age_ns),
        offered_samples: size(p.offered_attempts)?,
        recorded_samples: size(p.reserved_members)?,
        stale_samples: 0,
        skipped_samples: Default::default(),
        model_version: 1,
        bucket_count: 1,
        source_generator: p
            .producer
            .get("executable_path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unspecified")
            .to_owned(),
        source_generator_revision: p
            .producer
            .get("source_revision")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| {
                format!(
                    "package:{}",
                    p.producer
                        .get("package_version")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unspecified")
                )
            }),
        source_measurement_protocol: format!(
            "{POPULATION_REVISION}:sha256:{}",
            p.protocol
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ),
        source_observation_artifact_sha256: p.source_sha256,
    })
}

pub(super) fn predict(
    snapshot: &ImportedStructuredModelV1,
    fingerprint: &model::ExecutionFingerprint,
    shape: &model::WaveExecutionShape,
    evidence: Option<&PlanningCostEvidence>,
    now_ns: u64,
    version: u64,
) -> Option<PlanningCost> {
    let result = evidence
        .ok_or(StructuredUnknown::MissingEvidence)
        .and_then(|evidence| evidence.structured_input_for(shape))
        .and_then(|input| snapshot.predict_input(fingerprint, input, now_ns))
        .and_then(|value| {
            Ok(PlanningCost {
                typical_ns: value.fitted_ns,
                planning_ns: value.planning_ns,
                model_version: version,
                valid_for_ns: value
                    .valid_until_ns
                    .checked_sub(snapshot.model_now_ns(now_ns)?)
                    .ok_or(StructuredUnknown::Clock)?,
            })
        });
    super::super::query_metrics::record_structured(&result);
    match result {
        Ok(value) => Some(value),
        Err(reason) => {
            tracing::trace!(?reason, kind = ?shape.kind,
                decode_kv_tokens = ?shape.decode_kv_tokens,
                prefill_chunks = ?shape.prefill_chunks,
                "SLO candidate has no structured whole-wave cost prediction");
            None
        }
    }
}
