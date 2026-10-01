use super::*;

pub struct CalibrationReferenceDiscoverySample {
    pub(super) session: Arc<()>,
    pub(super) input: CalibrationRequestEvidence,
    pub(super) witness: Witness,
}
impl CalibrationReferenceDiscoverySample {
    pub fn shape(&self) -> ProfileWaveShapeV2 {
        ProfileWaveShapeV2::from(&self.witness.sample.actual_shape)
    }
    pub fn host_features(&self) -> HostCostFeaturesV1 {
        self.witness.host
    }
    pub fn request_evidence(&self) -> &CalibrationRequestEvidence {
        &self.input
    }
    pub fn accepted_ordinal(&self) -> u64 {
        self.witness.accepted
    }
}

pub(super) struct Witness {
    pub accepted: u64,
    pub sample: WaveCostObservation,
    pub host: HostCostFeaturesV1,
    pub commit: CalibrationCommittedRow,
}
impl Witness {
    pub fn capture(report: &CalibrationWaveReport) -> Result<Self> {
        let CalibrationObservation::Observed {
            sample,
            actual_rows,
            commits,
            host_features,
            accepted_ordinal: Some(accepted),
            disposition: CalibrationQueueDisposition::Published,
        } = &report.observation
        else {
            return Err(invalid(format!(
                "reference wave has no accepted actual observation; diagnostic={}",
                unavailable_summary(report)
            )));
        };
        if report.submission != CalibrationSubmissionState::HostReconciled
            || report.error.is_some()
            || *accepted == 0
            || actual_rows.len() != 1
            || commits.len() != 1
            || host_features.len() != 1
            || sample.outcome != WaveObservationOutcome::Completed
            || sample.boundary != CostBoundary::PreparationToCommit
            || sample.timing.wall_total_ns == 0
            || sample.timing.stages.restore.is_some()
            || sample.timing.stages.maintenance.is_some()
        {
            return Err(invalid(
                "reference requires one successfully committed isolated actual row",
            ));
        }
        let actual = &actual_rows[0];
        let commit = &commits[0];
        let host = host_features[0].ok_or_else(|| invalid("reference host evidence is missing"))?;
        if actual.request_id != commit.request_id
            || actual.owner_incarnation != commit.owner_incarnation
            || actual.work_generation != commit.work_generation
            || actual.input_index != commit.input_index
            || commit.owner_incarnation == 0
            || commit.work_generation == 0
            || commit.committed_at_ns > sample.observed_at_ns
        {
            return Err(invalid(
                "reference actual/commit identity or clock mismatch",
            ));
        }
        let work = match commit.work {
            CalibrationCommittedWork::Prefill {
                start,
                end,
                total_prompt_tokens,
                generated_before,
                generated_after,
            } if start < end
                && end <= total_prompt_tokens
                && generated_before == 0
                && generated_after == u64::from(end == total_prompt_tokens) =>
            {
                ActualRowWork::Prefill {
                    offset: start,
                    count: end - start,
                    total_prompt_tokens,
                }
            }
            CalibrationCommittedWork::Decode {
                kv_before,
                kv_after,
                generated_before,
                generated_after,
            } if kv_before.checked_add(1) == Some(kv_after)
                && generated_before.checked_add(1) == Some(generated_after) =>
            {
                ActualRowWork::Decode {
                    kv_tokens: kv_before,
                }
            }
            _ => return Err(invalid("reference commit lacks exact successful progress")),
        };
        if actual.work != work
            || host.state.generated_tokens_before
                != match commit.work {
                    CalibrationCommittedWork::Prefill {
                        generated_before, ..
                    }
                    | CalibrationCommittedWork::Decode {
                        generated_before, ..
                    } => generated_before,
                }
        {
            return Err(invalid("reference work/host differs from actual commit"));
        }
        let shape = &sample.actual_shape;
        let numeric = shape
            .numeric_features
            .as_ref()
            .ok_or_else(|| invalid("reference requires actual numeric host evidence"))?;
        numeric
            .validate(1)
            .map_err(|_| invalid("reference numeric rows are invalid"))?;
        if shape.decode_kv_tokens.len() + shape.prefill_chunks.len() != 1 {
            return Err(invalid("reference requires a singleton shape"));
        }
        Ok(Self {
            accepted: *accepted,
            sample: (**sample).clone(),
            host,
            commit: commit.clone(),
        })
    }
    pub fn matches(&self, shape: &ProfileWaveShapeV2, host: HostCostFeaturesV1) -> bool {
        self.host == host && ProfileWaveShapeV2::from(&self.sample.actual_shape) == *shape
    }
}

/// Cold failure diagnostics only. Never serialize the full report, actual
/// shape, request identity, input text, token values, or a backend error body.
/// Retained detail is bounded independently of recorder and request capacity.
pub(in crate::continuous_engine::inner::calibration) fn unavailable_summary(
    report: &CalibrationWaveReport,
) -> serde_json::Value {
    use serde_json::json;
    const DETAIL_LIMIT: usize = 4;
    let reason = |value: &str| -> String {
        value
            .chars()
            .take(128)
            .map(|c| {
                if c.is_ascii_graphic() || c == ' ' {
                    c
                } else {
                    '?'
                }
            })
            .collect()
    };
    let observation = match &report.observation {
        CalibrationObservation::PendingOrUnavailable => json!({"kind":"pending_or_unavailable"}),
        CalibrationObservation::ConflictingCalls => json!({"kind":"conflicting_calls"}),
        CalibrationObservation::Rejected { reason: rejection } => {
            json!({"kind":"rejected", "reason":reason(rejection)})
        }
        CalibrationObservation::InvalidIdentityJoin => json!({"kind":"invalid_identity_join"}),
        CalibrationObservation::Observed {
            sample,
            actual_rows,
            commits,
            host_features,
            accepted_ordinal,
            disposition,
        } => {
            let queue = match disposition {
                CalibrationQueueDisposition::Published => json!({"kind":"published"}),
                CalibrationQueueDisposition::Dropped { reason: rejection } => {
                    json!({"kind":"dropped", "reason":reason(rejection)})
                }
            };
            json!({
                "kind":"observed", "queue":queue, "accepted_ordinal":accepted_ordinal,
                "actual_rows":actual_rows.len(), "commits":commits.len(),
                "host_features":host_features.len(), "boundary":format!("{:?}", sample.boundary),
                "outcome":format!("{:?}", sample.outcome), "wall_ns":sample.timing.wall_total_ns,
            })
        }
    };
    let stages = report.host_stages.as_ref().map(|stages| {
        let structured = match stages.structured_evidence.as_ref() {
            None => json!({"kind":"absent"}),
            Some(Ok(_)) => json!({"kind":"qualified"}),
            Some(Err(reason)) => json!({"kind":"rejected", "reason":format!("{reason:?}")}),
        };
        let rows = stages
            .rows
            .iter()
            .take(DETAIL_LIMIT)
            .map(|row| {
                json!({
                    "input_index":row.input_index, "completeness":row.completeness,
                    "host_processing_ordinal":row.host_processing_ordinal,
                    "host_started_at_ns":row.host_started_at_ns,
                    "token_committed_at_ns":row.token_committed_at_ns,
                    "output_published_at_ns":row.output_published_at_ns,
                    "completion_started_at_ns":row.completion_started_at_ns,
                    "settled_at_ns":row.settled_at_ns,
                    "terminal_present":row.terminal.is_some(),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema_version":stages.schema_version, "call_id":stages.call_id,
            "completeness":stages.completeness, "row_count":stages.rows.len(),
            "rows":rows, "rows_truncated":stages.rows.len()>DETAIL_LIMIT,
            "actual_shape_present":stages.actual_shape.is_some(),
            "fingerprint_present":stages.fingerprint.is_some(),
            "statistical_evidence_present":stages.statistical_evidence.is_some(),
            "structured_evidence":structured,
            "prepare_started_at_ns":stages.prepare_started_at_ns,
            "executor_returned_at_ns":stages.executor_returned_at_ns,
            "finalized_at_ns":stages.finalized_at_ns, "full_wall_ns":stages.full_wall_ns,
        })
    });
    let actual = report.actual_evidence_diagnostic.as_ref().map(|actual| {
        let waves = actual
            .waves
            .iter()
            .take(DETAIL_LIMIT)
            .map(|wave| {
                json!({
                    "physical_wave_ordinal":wave.physical_wave_ordinal,
                    "reason":format!("{:?}", wave.reason),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "call_id":actual.call_id, "physical_waves":actual.physical_waves,
            "retained_waves":actual.retained_waves, "lost_observations":actual.lost_observations,
            "dispatch_unknown":actual.dispatch_unknown.map(|reason|format!("{reason:?}")),
            "unknown_wave_count":actual.waves.len(), "waves":waves,
            "waves_truncated":actual.waves.len()>DETAIL_LIMIT,
            "retained_wave_details_complete":actual.retained_wave_details_complete,
        })
    });
    json!({
        "submission":format!("{:?}",report.submission), "execution_error_present":report.error.is_some(),
        "observation":observation, "host_stage_queue":report.host_stage_queue,
        "host_stages":stages, "actual_evidence":actual,
    })
}
