//! Explicit schema-6/7/8 startup import. Ordinary observation and heldout consumption
//! cannot change fit/residual/support/TTL. Explicit feedback has a separate epoch.
use super::*;
use file::statistical_v6::{load_whole_wave_profile_v6, ImportedWholeWaveModelV1};
use file::statistical_v7::load_whole_wave_profile_v7;
use file::statistical_v8::load_whole_wave_profile_v8;
use model::statistical::{
    model::{ModelUnknown, WholeWaveSettingsV1},
    SelectedStatisticalFamily,
};

pub(super) fn load_seed(
    fingerprint: model::ExecutionFingerprint,
    config: &SloCostObservationConfig,
    path: Option<&Path>,
    clock: Option<file::ProfileLoadClock>,
) -> Result<TrainingSeed, FerrumError> {
    // Never route new-mode capture through the old online trainer/export format.
    // The independent three-phase capture protocol owns explicitly versioned publication.
    if config.profile_export.is_some() {
        return Err(FerrumError::config(
            "selected whole-wave calibration requires its independent fit/residual capture protocol, not legacy profile_export",
        ));
    }
    let Some(path) = path else {
        return Ok(TrainingSeed {
            trainer: None,
            snapshot: None,
            receipt: None,
        });
    };
    let settings = WholeWaveSettingsV1::from_policy_limits(&model_settings(&config.model));
    settings
        .validate()
        .map_err(|e| FerrumError::config(format!("whole-wave settings: {e:?}")))?;
    let clock = clock.ok_or_else(|| FerrumError::config("cost profile load clock missing"))?;
    let declared = config
        .profile_import
        .declared_local_clock_max_error_ns
        .ok_or_else(|| {
            FerrumError::config("cost profile requires declared local wall-clock accuracy")
        })?;
    if clock.wall_max_error_ns != Some(declared) {
        return Err(FerrumError::config(
            "cost profile clock differs from declared policy",
        ));
    }
    type Loader = fn(
        &Path,
        &model::ExecutionFingerprint,
        &WholeWaveSettingsV1,
        &file::CostProfileLoadLimits,
        file::ProfileLoadClock,
    ) -> Result<ImportedWholeWaveModelV1, file::CostProfileError>;
    let (schema, loader): (u32, Loader) = match config.predictor {
        ferrum_types::SloCostPredictor::SelectedWholeWaveV1 => (6, load_whole_wave_profile_v6),
        ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2 => {
            (7, load_whole_wave_profile_v7)
        }
        ferrum_types::SloCostPredictor::SelectedWorkSupportV1 => (8, load_whole_wave_profile_v8),
        _ => return Err(FerrumError::config("not a selected predictor")),
    };
    let imported = loader(
        path,
        &fingerprint,
        &settings,
        &load_limits(&config.profile_import),
        clock,
    )
    .map_err(profile_error)?;
    let p = &imported.provenance;
    let receipt = SloCostProfileReceipt {
        structured_whole_wave: None,
        selected_whole_wave: Some(ferrum_types::SloSelectedWholeWaveReceiptV1 {
            capture_identity_sha256: imported.capture_identity_sha256,
            fit_parameters_sha256: imported.fit_parameters_sha256,
            model_revision: imported.model_revision().as_str().to_owned(),
            protocol_sha256: imported.protocol_sha256,
            fit_through_ordinal: p.fit_through_ordinal,
            residual_through_ordinal: p.residual_through_ordinal,
            fit_records: imported.fit_records,
            residual_records: imported.residual_records,
        }),
        schema_version: schema,
        path: p
            .loaded_from
            .clone()
            .ok_or_else(|| FerrumError::internal("file import has no source path"))?,
        file_sha256: format!(
            "sha256:{}",
            imported
                .file_sha256
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ),
        file_bytes: p.file_bytes,
        generated_unix_ns: p.generated_unix_ns,
        loaded_unix_ns: p.loaded_unix_ns,
        conservative_clock_error_ns: imported.conservative_clock_error_ns,
        declared_local_clock_max_error_ns: declared,
        oldest_imported_age_ns: Some(p.oldest_imported_age_ns),
        newest_imported_age_ns: Some(p.newest_imported_age_ns),
        offered_samples: imported.fit_records + imported.residual_records,
        recorded_samples: imported.fit_records + imported.residual_records,
        stale_samples: 0,
        skipped_samples: Default::default(),
        model_version: 1,
        bucket_count: imported.segment_count(),
        source_generator: p.source.generator.clone(),
        source_generator_revision: p.source.generator_revision.clone(),
        source_measurement_protocol: p.source.measurement_protocol.clone(),
        source_observation_artifact_sha256: imported.source_sha256,
    };
    Ok(TrainingSeed {
        trainer: None,
        snapshot: Some(Arc::new(EngineCostSnapshot {
            inner: Snapshot::Selected(SelectedSnapshot {
                model: Arc::new(imported),
                feedback: None,
            }),
            fingerprint,
        })),
        receipt: Some(receipt),
    })
}

pub(super) fn predict(
    snapshot: &SelectedSnapshot,
    fingerprint: &model::ExecutionFingerprint,
    shape: &model::WaveExecutionShape,
    evidence: Option<&PlanningCostEvidence>,
    now_ns: u64,
    version: u64,
) -> Option<PlanningCost> {
    let result = evidence
        .and_then(|e| e.input_for(shape))
        .ok_or(ModelUnknown::Evidence(
            ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown::MissingProducer,
        ))
        .and_then(|input| {
            let value = snapshot.model.predict_input(fingerprint, input, now_ns)?;
            snapshot.apply(
                value,
                input.family_signature_for(snapshot.model.selected_family())?,
            )
        });
    let result = result.and_then(|value| {
        Ok(PlanningCost {
            typical_ns: value.fitted_ns,
            planning_ns: value.planning_ns,
            model_version: version,
            // Imported model timestamps have an anchored epoch, never local now.
            valid_for_ns: value
                .valid_until_ns
                .checked_sub(
                    snapshot
                        .model
                        .clock
                        .model_now_ns(now_ns)
                        .map_err(|_| ModelUnknown::Clock)?,
                )
                .ok_or(ModelUnknown::Clock)?,
        })
    });
    super::super::query_metrics::record(
        super::super::query_metrics::QueryScope::Candidate,
        &result,
    );
    match result {
        Ok(value) => Some(value),
        Err(reason) => {
            tracing::trace!(?reason, kind = ?shape.kind, decode_kv_tokens = ?shape.decode_kv_tokens,
                prefill_chunks = ?shape.prefill_chunks, "SLO candidate has no selected whole-wave cost prediction");
            None
        }
    }
}

/// Base artifact remains shared and immutable across all feedback epochs.
pub(super) struct SelectedSnapshot {
    pub model: Arc<ImportedWholeWaveModelV1>,
    pub feedback: Option<Arc<super::super::selected_feedback::View>>,
}
impl SelectedSnapshot {
    pub fn family_signature<'a>(
        &self,
        evidence: &'a ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1,
    ) -> Result<&'a [u8; 32], ModelUnknown> {
        match self.model.selected_family() {
            SelectedStatisticalFamily::OrderedV1 => Ok(evidence.family_signature()),
            SelectedStatisticalFamily::IndependentAttentionV2 => evidence
                .independent_attention_v2()
                .map(|v| v.family_signature())
                .ok_or(ModelUnknown::Evidence(
                    ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown::MissingProducer,
                )),
        }
    }
    pub fn apply(
        &self,
        mut value: model::statistical::model::WholeWavePredictionV1,
        family: &[u8; 32],
    ) -> Result<model::statistical::model::WholeWavePredictionV1, ModelUnknown> {
        if let Some(view) = &self.feedback {
            if !view.current() {
                return Err(ModelUnknown::RuntimeValidity);
            }
            value.planning_ns = value
                .planning_ns
                .checked_add(view.margin(family))
                .ok_or(ModelUnknown::Numerical)?;
        }
        Ok(value)
    }
}
