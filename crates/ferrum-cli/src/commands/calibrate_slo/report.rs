use super::*;

mod structured_discovery;
mod structured_discovery_v2;
use ferrum_engine::continuous_engine::{
    CalibrationCommittedWork, CalibrationObservation, CalibrationWaveReport, HostStageCompleteness,
    HostStageEvidenceV1,
};
use ferrum_scheduler::implementations::continuous::{
    cost_model::{CostBoundary, CostPrediction, WaveExecutionShape},
    cost_profile::{v2::ProfileWaveShapeV2, ProfileFingerprint},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::Write,
};
pub(super) use structured_discovery_v2::DiscoverySummaryV2;

#[derive(Default, Serialize)]
pub(super) struct Summary {
    pub token_policy_residency_attempts: u64,
    pub token_policy_residency_invalidations: u64,
    pub token_policy_residency_cleared_entries: u64,
    pub wave_attempts: u64,
    pub observed_waves: u64,
    pub unavailable_waves: u64,
    pub training_requests: u64,
    pub residual_requests: u64,
    pub validation_requests: u64,
    pub validation_known: u64,
    pub validation_unknown: u64,
    pub validation_underestimates: u64,
    pub host_content_validation_offered: u64,
    pub host_content_validation_known: u64,
    pub host_content_validation_unknown: u64,
    pub host_content_validation_underestimates: u64,
    pub selected_validation_offered: u64,
    pub selected_validation_known: u64,
    pub selected_validation_unknown: u64,
    pub selected_validation_underestimates: u64,
    pub selected_validation_unknown_reasons: std::collections::BTreeMap<String, u64>,
    pub selected_fit_freeze: Option<serde_json::Value>,
    pub validation_model: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_calibration: Option<structured::StructuredReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_calibration_v2: Option<structured_v2::StructuredReportV2>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_calibration_group_v2: Option<structured_v2::GroupReportV2>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_discovery_v2: Option<DiscoverySummaryV2>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_future_audit_v2: Option<required_audit::AuditSummaryV2>,
    pub phases: PhaseCounts,
    pub reference_frozen_plan: Option<serde_json::Value>,
    pub reference: Option<reference::ReferenceReceipt>,
    /// Actual engine tokenization evidence, bounded by recovered input count.
    pub reference_input_identities: Option<reference::InputIdentityLedger>,
    pub collection_error: Option<String>,
    pub shutdown: Option<ShutdownEvidence>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    Warmup,
    Discovery,
    Reference,
    Training,
    Residual,
    Qualification,
    #[serde(rename = "validation")]
    Heldout,
}

#[derive(Default, Serialize)]
pub(super) struct PhaseCounts {
    pub warmup: PhaseCount,
    pub discovery: PhaseCount,
    pub reference: PhaseCount,
    pub training: PhaseCount,
    pub residual: PhaseCount,
    pub heldout: PhaseCount,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qualification: Option<PhaseCount>,
}

#[derive(Default, Serialize)]
pub(super) struct PhaseCount {
    pub requests: u64,
    pub wave_attempts: u64,
    pub observed_waves: u64,
    pub unavailable_waves: u64,
    /// Selected measured targets only, excluding preparatory and post-target waves.
    pub target_waves: u64,
    pub preparation_waves: u64,
    pub after_target_waves: u64,
}
impl PhaseCounts {
    pub(super) fn get_mut(&mut self, phase: Phase) -> &mut PhaseCount {
        match phase {
            Phase::Warmup => &mut self.warmup,
            Phase::Discovery => &mut self.discovery,
            Phase::Reference => &mut self.reference,
            Phase::Training => &mut self.training,
            Phase::Residual => &mut self.residual,
            Phase::Heldout => &mut self.heldout,
            Phase::Qualification => self.qualification.get_or_insert_with(PhaseCount::default),
        }
    }
}

impl Summary {
    pub(super) fn request(&mut self, phase: Phase) {
        self.phases.get_mut(phase).requests += 1;
        match phase {
            Phase::Training => self.training_requests += 1,
            Phase::Residual => self.residual_requests += 1,
            Phase::Heldout => self.validation_requests += 1,
            _ => {}
        }
    }
    pub(super) fn reference_progress(
        &mut self,
        phase: Phase,
        progress: reference::ObservationProgress,
    ) -> Result<()> {
        if !matches!(phase, Phase::Discovery | Phase::Reference) {
            return Err(FerrumError::internal(
                "reference observer used outside its declared phase",
            ));
        }
        let counts = self.phases.get_mut(phase);
        match progress {
            reference::ObservationProgress::Recorded
            | reference::ObservationProgress::TargetComplete => counts.target_waves += 1,
            reference::ObservationProgress::PreparingTarget => counts.preparation_waves += 1,
            reference::ObservationProgress::AfterTarget => counts.after_target_waves += 1,
            reference::ObservationProgress::NotSubmitted => {}
        }
        Ok(())
    }
}

#[derive(Serialize)]
pub(super) struct ShutdownEvidence {
    pub completed: bool,
    pub error: Option<String>,
}

pub(super) struct Artifacts {
    raw: File,
    report: File,
    bytes: u64,
    limit: u64,
    hash: Sha256,
    manifest: serde_json::Value,
    structured_capture: ferrum_types::SloStructuredCostCapture,
}

impl Artifacts {
    pub(super) fn create(
        raw: &std::path::Path,
        report: &std::path::Path,
        manifest: &manifest::Manifest,
    ) -> Result<Self> {
        // create_new enforces non-aliasing even for two paths to the same leaf.
        let raw = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(raw)
            .map_err(|error| {
                FerrumError::config(format!("create calibration raw output: {error}"))
            })?;
        let report = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(report)
            .map_err(|error| FerrumError::config(format!("create calibration report: {error}")))?;
        Ok(Self {
            raw,
            report,
            bytes: 0,
            limit: manifest.protocol.maximum_raw_bytes.get(),
            hash: Sha256::new(),
            manifest: serde_json::to_value(manifest).map_err(json_error)?,
            structured_capture: ferrum_types::SloStructuredCostCapture::Disabled,
        })
    }

    /// Read once from the actual created session, never from raw JSON or the
    /// requested manifest. Disabled capture must not evaluate discovery.
    pub(super) fn set_structured_capture(&mut self, mode: ferrum_types::SloStructuredCostCapture) {
        self.structured_capture = mode;
    }

    pub(super) fn record(&mut self, value: &serde_json::Value) -> Result<()> {
        let mut bytes = serde_json::to_vec(value).map_err(json_error)?;
        bytes.push(b'\n');
        let total = self
            .bytes
            .checked_add(bytes.len() as u64)
            .filter(|total| *total <= self.limit)
            .ok_or_else(|| {
                FerrumError::resource_exhausted("calibration raw evidence byte limit exceeded")
            })?;
        self.raw.write_all(&bytes).map_err(io_error)?;
        self.hash.update(&bytes);
        self.bytes = total;
        Ok(())
    }

    pub(super) fn wave(
        &mut self,
        phase: Phase,
        case: usize,
        repetition: usize,
        report: &CalibrationWaveReport,
        frozen: Option<&validation::ValidationModel>,
        totals: &mut Summary,
    ) -> Result<()> {
        let selected_model = frozen.and_then(validation::ValidationModel::selected);
        let host_model = frozen.filter(|model| {
            selected_model.is_none()
                && model.planning_boundary() == Some(CostBoundary::PreparationToHostSettledV1)
        });
        let legacy_model = frozen.filter(|_| host_model.is_none() && selected_model.is_none());
        let evidence = match &report.observation {
            CalibrationObservation::Observed {
                sample,
                actual_rows,
                commits,
                host_features,
                accepted_ordinal,
                disposition,
            } => {
                if actual_rows.len() != commits.len() || actual_rows.len() != host_features.len() {
                    return Err(FerrumError::internal(
                        "calibration observation row correlation is incomplete",
                    ));
                }
                totals.observed_waves += 1;
                totals.phases.get_mut(phase).observed_waves += 1;
                let prediction = if let Some(frozen) = legacy_model {
                    match frozen.predict(&sample.actual_shape)? {
                        Some(CostPrediction::Known(value)) => {
                            totals.validation_known += 1;
                            totals.validation_underestimates +=
                                u64::from(sample.timing.wall_total_ns > value.planning_ns);
                            serde_json::json!({"kind":"known", "model_source":frozen.kind(), "typical_ns":value.typical_ns,
                                "planning_ns":value.planning_ns, "sample_count":value.sample_count,
                                "model_version":value.model_version, "valid_for_ns":value.valid_for_ns})
                        }
                        other => {
                            totals.validation_unknown += 1;
                            serde_json::json!({"kind":"unknown", "model_source":frozen.kind(), "reason":format!("{other:?}")})
                        }
                    }
                } else {
                    serde_json::Value::Null
                };
                let rows = actual_rows.iter().zip(commits).zip(host_features).map(|((actual, commit), host)| {
                    let work = match commit.work {
                        CalibrationCommittedWork::Prefill { start, end, total_prompt_tokens, generated_before, generated_after } =>
                            serde_json::json!({"kind":"prefill", "start":start,"end":end,"total_prompt_tokens":total_prompt_tokens,"generated_before":generated_before,"generated_after":generated_after}),
                        CalibrationCommittedWork::Decode { kv_before, kv_after, generated_before, generated_after } =>
                            serde_json::json!({"kind":"decode","kv_before":kv_before,"kv_after":kv_after,"generated_before":generated_before,"generated_after":generated_after}),
                    };
                    serde_json::json!({"request_id":actual.request_id,"owner_incarnation":actual.owner_incarnation,
                        "work_generation":actual.work_generation,"input_index":actual.input_index,
                        "actual_work":format!("{:?}",actual.work),"host_features":host,
                        "committed_at_ns":commit.committed_at_ns,"commit":work})
                }).collect::<Vec<_>>();
                serde_json::json!({"kind":"observed", "fingerprint":ProfileFingerprint::from(&sample.fingerprint),
                    "shape":ProfileWaveShapeV2::from(&sample.actual_shape),"wall_total_ns":sample.timing.wall_total_ns,
                    "device_elapsed_ns":sample.timing.device_elapsed_ns,"observed_at_ns":sample.observed_at_ns,
                    "boundary":format!("{:?}",sample.boundary),"queue":format!("{disposition:?}"),"rows":rows,
                    "accepted_ordinal":accepted_ordinal,
                    "retrospective_frozen_prediction":prediction})
            }
            other => {
                totals.unavailable_waves += 1;
                totals.phases.get_mut(phase).unavailable_waves += 1;
                if legacy_model.is_some() {
                    totals.validation_unknown += 1;
                }
                serde_json::json!({"kind":"unavailable", "reason":format!("{other:?}")})
            }
        };
        let host_prediction = if let Some(model) = host_model {
            host_content_prediction(
                report.host_stages.as_deref(),
                model.kind(),
                totals,
                |shape| model.predict(shape),
            )?
        } else {
            serde_json::Value::Null
        };
        let selected_prediction = selected_model.map_or(serde_json::Value::Null, |model| {
            selected_prediction(model.evaluate_selected_wave(report), totals)
        });
        let mut record = serde_json::json!({"schema_version":1,"event":"wave", "phase":phase,
            "case":case,"repetition":repetition,"submission":format!("{:?}",report.submission),
            "error":report.error.as_ref().map(ToString::to_string),"evidence":evidence,
            "host_stages":report.host_stages.as_deref().map(|stages| stages.structured_diagnostic_view()),"host_stage_queue":report.host_stage_queue,
            "host_content_frozen_prediction":host_prediction,
            "selected_whole_wave_frozen_prediction":selected_prediction});
        if let Some(diagnostic) = &report.actual_evidence_diagnostic {
            record["actual_evidence_diagnostic"] =
                serde_json::to_value(diagnostic.as_ref()).map_err(json_error)?;
        }
        if let Some(discovery) = structured_discovery::inspect(self.structured_capture, || {
            report.structured_cost_input()
        }) {
            record["structured_cost_discovery"] =
                serde_json::to_value(discovery).map_err(json_error)?;
        }
        if let Some(discovery) = structured_discovery_v2::inspect(self.structured_capture, || {
            report.structured_cost_input_v2()
        }) {
            if let Some(inventory) = &mut totals.structured_discovery_v2 {
                inventory.observe(phase, &discovery);
            }
            record["structured_cost_discovery_v2"] =
                serde_json::to_value(discovery).map_err(json_error)?;
        }
        self.record(&record)
    }

    pub(super) fn finish(
        &mut self,
        provenance: Option<serde_json::Value>,
        summary: Option<Summary>,
        error: Option<&FerrumError>,
    ) -> Result<()> {
        self.raw.flush().map_err(io_error)?;
        let validation_scope = if self.manifest["validation_model"]["kind"]
            == "structured_discovery_v2"
        {
            "independent complete-cohort discovery only; no source membership, trained model, qualification or future horizon authorization"
        } else if self.manifest["validation_model"]["kind"] == "structured_whole_wave_v1"
            || self.manifest["validation_model"]["kind"] == "structured_whole_wave_v2"
        {
            "manifest validation cohorts are the third independent live qualification population; no fourth heldout, p99 guarantee, complete future horizon or serving SLO compliance is established"
        } else {
            "held-out actual-shape retrospective cost check; not pre-submission route validation or serving SLO compliance"
        };
        let report = serde_json::json!({"schema_version":1,"kind":"real_manual_calibration",
            "status":if error.is_some(){"failed"}else{"collected"},"error":error.map(ToString::to_string),
            "manifest":self.manifest,"provenance":provenance,"summary":summary,
            "raw_bytes":self.bytes,"raw_sha256":format!("{:x}",self.hash.clone().finalize()),
            "validation_scope":validation_scope,
            "profile_scope":"exported_profile identifies the imported training artifact; selected_whole_wave_v1/profile6/source1, selected_independent_attention_v2/profile7/source2 and selected_work_support_v1/profile8/source3 freeze fit before independent residual capture and reload their explicit version before heldout; work-support retains family schema2 with a distinct model revision and excludes only output_budget_sum from statistical support, preserving request authority and terminal categories; fresh completed phases are required, old headers cannot be relabeled, defaults/min_samples/residual quantile/TTL are unchanged; live_frozen has no deployable artifact; legacy shutdown export may include validation observations; structured_whole_wave_v1/profile9 uses three complete live fit/residual/qualification populations and only exports after qualification plus original source replay, with no additional heldout or serving SLO claim",
            "reference_scope":"optional singleton discovery -> persisted plan -> fresh trials -> original training-cut source join; target completion never shortens the request; only target_waves enter reference scoring, preparation proves the chain and after_target waves are not reference targets"});
        serde_json::to_writer_pretty(&mut self.report, &report).map_err(json_error)?;
        self.report.write_all(b"\n").map_err(io_error)?;
        self.report.flush().map_err(io_error)
    }
}

fn selected_prediction(
    result: std::result::Result<
        ferrum_scheduler::implementations::continuous::cost_model::statistical::model::HeldoutEvaluationV1,
        ferrum_scheduler::implementations::continuous::cost_model::statistical::model::ModelUnknown,
    >,
    totals: &mut Summary,
) -> serde_json::Value {
    totals.selected_validation_offered += 1;
    let (reason, actual_ns, query_identity) = match result {
        Ok(evaluation) => match evaluation.prediction {
            Ok(prediction) => {
                totals.selected_validation_known += 1;
                let underestimate = evaluation.actual_ns.saturating_sub(prediction.planning_ns);
                totals.selected_validation_underestimates += u64::from(underestimate > 0);
                return serde_json::json!({
                    "kind":"known", "query_identity":evaluation.query_identity, "boundary":"preparation_to_host_settled_v1",
                    "fitted_ns":prediction.fitted_ns, "residual_ns":prediction.residual_ns,
                    "static_margin_ns":prediction.static_margin_ns,
                    "planning_ns":prediction.planning_ns, "valid_until_ns":prediction.valid_until_ns,
                    "fit_samples":prediction.fit_samples, "residual_samples":prediction.residual_samples,
                    "actual_ns":evaluation.actual_ns, "underestimate_ns":underestimate,
                    "scope":"independent complete host-settled wave retrospective check; not future-route or serving SLO qualification"
                });
            }
            Err(reason) => (
                reason,
                Some(evaluation.actual_ns),
                evaluation.query_identity,
            ),
        },
        Err(reason) => (reason, None, None),
    };
    totals.selected_validation_unknown += 1;
    let reason = format!("{reason:?}");
    *totals
        .selected_validation_unknown_reasons
        .entry(reason.clone())
        .or_default() += 1;
    serde_json::json!({"kind":"unknown", "reason":reason, "actual_ns":actual_ns, "query_identity":query_identity})
}

fn host_content_prediction(
    stages: Option<&HostStageEvidenceV1>,
    model_source: &str,
    totals: &mut Summary,
    predict: impl FnOnce(&WaveExecutionShape) -> Result<Option<CostPrediction>>,
) -> Result<serde_json::Value> {
    totals.host_content_validation_offered += 1;
    let eligible = stages.filter(|stages| {
        stages.schema_version == 1
            && stages.completeness == HostStageCompleteness::CompleteSingleWave
            && !stages.rows.is_empty()
            && stages.full_wall_ns.is_some_and(|wall| wall > 0)
            && stages.actual_shape.as_ref().is_some_and(|shape| {
                shape
                    .host_content_features
                    .as_ref()
                    .is_some_and(|features| features.validate().is_ok())
                    || shape
                        .row_multiset_features
                        .as_ref()
                        .is_some_and(|features| {
                            features
                                .validate(shape.decode_kv_tokens.len() + shape.prefill_chunks.len())
                                .is_ok()
                        })
            })
    });
    if let Some(stages) = eligible {
        let shape = stages.actual_shape.as_ref().expect("eligible actual shape");
        match predict(shape)? {
            Some(CostPrediction::Known(value))
                if value.boundary == CostBoundary::PreparationToHostSettledV1 =>
            {
                totals.host_content_validation_known += 1;
                let actual = stages.full_wall_ns.expect("eligible full wall");
                totals.host_content_validation_underestimates +=
                    u64::from(actual > value.planning_ns);
                return Ok(
                    serde_json::json!({"kind":"known", "boundary":"preparation_to_host_settled_v1",
                    "model_source":model_source,"typical_ns":value.typical_ns,"planning_ns":value.planning_ns,
                    "model_version":value.model_version,"valid_for_ns":value.valid_for_ns,"sample_count":value.sample_count,
                    "observed_row_multiset_schema":shape.row_multiset_features.as_ref().map(|features|features.schema_version),
                    "actual_wall_ns":actual,"underestimate_ns":actual.saturating_sub(value.planning_ns),
                    "scope":"retrospective complete host-settled wave, empirical content residual; not future-query coverage"}),
                );
            }
            other => {
                totals.host_content_validation_unknown += 1;
                return Ok(
                    serde_json::json!({"kind":"unknown","reason":format!("{other:?}"),"model_source":model_source,
                        "observed_row_multiset_schema":shape.row_multiset_features.as_ref().map(|features|features.schema_version)}),
                );
            }
        }
    }
    totals.host_content_validation_unknown += 1;
    Ok(
        serde_json::json!({"kind":"unknown","reason":"complete actual host-content evidence unavailable","model_source":model_source}),
    )
}

#[cfg(test)]
mod host_content_tests;
#[cfg(test)]
mod selected_tests;

fn json_error(error: serde_json::Error) -> FerrumError {
    FerrumError::internal(format!("encode calibration evidence: {error}"))
}
fn io_error(error: std::io::Error) -> FerrumError {
    FerrumError::backend(format!("write calibration evidence: {error}"))
}
