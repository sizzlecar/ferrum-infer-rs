use super::types::*;
use crate::{
    arrivals::poisson_arrival_times,
    slo::{SloStatus, TpotBoundary},
    BenchmarkPhase,
};
use rand::{rngs::StdRng, SeedableRng};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchProgress {
    Awaiting {
        phase: CapacityPhase,
        runs: Vec<CapacityRunKey>,
    },
    Complete,
}

/// Owns its immutable frozen contract and accepts each original run exactly
/// once. No public mutable fields or deserialization can manufacture progress.
pub struct CapacitySearch {
    session_history: super::session::SessionHistory,
    contract: CapacityContract,
    digest: String,
    records: BTreeMap<CapacityRunKey, CapacityRunAssessment>,
    instances: BTreeSet<String>,
    last_ended_unix_ns: u64,
}

impl CapacityContract {
    pub fn validate(&self) -> Result<(), CapacityError> {
        if self.schema_version != 1 || self.frozen_unix_ns == 0 {
            return Err(error(
                "schema version 1 and a freeze timestamp are required",
            ));
        }
        if self.rates_rps.is_empty()
            || self.rates_rps.iter().any(|r| !r.is_finite() || *r <= 0.0)
            || self.rates_rps.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(error(
                "rates must be explicit, finite, positive and strictly increasing",
            ));
        }
        for indices in [&self.coarse_indices, &self.aa_indices] {
            if indices.is_empty()
                || indices.iter().any(|i| *i >= self.rates_rps.len())
                || indices.windows(2).any(|p| p[0] >= p[1])
            {
                return Err(error(
                    "phase indices must be explicit, in range and strictly increasing",
                ));
            }
        }
        if self.coarse_indices.first() != Some(&0)
            || self.coarse_indices.last() != Some(&(self.rates_rps.len() - 1))
        {
            return Err(error(
                "coarse scan must include both ends of the declared rate range",
            ));
        }
        let n = &self.repetitions;
        if [n.aa_pairs, n.coarse, n.neighborhood, n.confirmation].contains(&0) {
            return Err(error("every phase requires explicit nonzero repetitions"));
        }
        let maximum = [
            self.aa_indices
                .len()
                .checked_mul(n.aa_pairs as usize)
                .and_then(|n| n.checked_mul(2)),
            self.coarse_indices.len().checked_mul(n.coarse as usize),
            (self.rates_rps.len() - self.coarse_indices.len()).checked_mul(n.neighborhood as usize),
            self.rates_rps.len().checked_mul(n.confirmation as usize),
        ]
        .into_iter()
        .try_fold(0_usize, |sum, count| sum.checked_add(count?))
        .ok_or_else(|| error("planned run count overflow"))?;
        if maximum > self.maximum_planned_runs
            || self.maximum_requests_per_run == 0
            || self.maximum_requests_per_run > u32::MAX as usize
            || self.maximum_requests_per_run.checked_add(1).is_none()
        {
            return Err(error(
                "explicit acquisition resource bounds are insufficient",
            ));
        }
        let w = &self.window;
        for v in [
            w.send_seconds,
            w.maximum_drain_seconds,
            w.maximum_queue_sample_gap_seconds,
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(error("windows must be finite and positive"));
            }
        }
        for v in [
            w.observe_from_seconds,
            w.maximum_request_start_lag_ms,
            w.maximum_unfinished_requests_slope_per_second,
            w.maximum_oldest_age_slope_ms_per_second,
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(error(
                    "observation and tolerance values must be finite and nonnegative",
                ));
            }
        }
        if w.observe_from_seconds >= w.send_seconds
            || w.maximum_queue_sample_gap_seconds > w.send_seconds - w.observe_from_seconds
        {
            return Err(error(
                "queue observation window must support at least two samples",
            ));
        }
        if !((w.send_seconds + w.maximum_drain_seconds + w.maximum_queue_sample_gap_seconds)
            * 1000.0
            + w.maximum_request_start_lag_ms)
            .is_finite()
        {
            return Err(error("derived measurement window overflows milliseconds"));
        }
        let workload_digest = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&self.workload.samples).map_err(|e| error(e.to_string()))?
            )
        );
        if workload_digest != self.workload.selection_sha256
            || workload_digest != self.identity.ordered_workload_sha256
        {
            return Err(error(
                "frozen ordered workload digest does not match actual sample rows",
            ));
        }
        let mut warmup = 0_u32;
        let mut measured = 0_u32;
        for sample in &self.workload.samples {
            if sample.input_tokens == 0
                || sample.requested_output_tokens == 0
                || sample.prompt_sha256.len() != 64
                || !sample.prompt_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return Err(error(
                    "workload needs prompt digests and positive input/output budgets",
                ));
            }
            let next = match sample.phase {
                BenchmarkPhase::Warmup if measured == 0 => &mut warmup,
                BenchmarkPhase::Measured => &mut measured,
                _ => return Err(error("warmup must precede measured workload")),
            };
            if sample.request_index != *next {
                return Err(error(
                    "workload request indices must be contiguous within each phase",
                ));
            }
            *next = next
                .checked_add(1)
                .ok_or_else(|| error("workload count overflow"))?;
        }
        if measured == 0 {
            return Err(error("a measured workload is required"));
        }
        if self.identity.capacity.slots == 0
            || self.identity.capacity.context_tokens_per_request == 0
            || self.identity.capacity.batch_tokens == 0
        {
            return Err(error("fixed server capacity must be positive"));
        }
        for hash in [
            &self.identity.ordered_workload_sha256,
            &self.identity.dataset_source_sha256,
            &self.identity.shared.hardware_fingerprint_sha256,
            &self.identity.shared.model_content_sha256,
            &self.identity.shared.tokenizer_sha256,
            &self.identity.shared.chat_template_sha256,
            &self.identity.shared.client_binary_sha256,
            &self.identity.shared.client_slo_config_sha256,
            &self.identity.server.binary_sha256,
            &self.identity.server.effective_configuration_sha256,
        ] {
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(error("execution/workload identities require SHA256 values"));
            }
        }
        let samples_per_gap =
            if self.queue_observation_source == QueueObservationSource::ServerAdmissionV1 {
                4.0
            } else {
                2.0
            };
        let required_queue_samples = (samples_per_gap * (w.send_seconds + w.maximum_drain_seconds)
            / w.maximum_queue_sample_gap_seconds)
            .ceil()
            + 2.0;
        if !required_queue_samples.is_finite()
            || required_queue_samples > self.maximum_queue_samples_per_run as f64
            || self.maximum_queue_samples_per_run < 2
        {
            return Err(error(
                "explicit queue sample budget cannot cover send and drain windows",
            ));
        }
        self.slo.validate().map_err(|e| error(e.to_string()))?;
        self.sampling.validate().map_err(|e| error(e.to_string()))?;
        if !matches!(self.http_connection_mode.as_str(), "fresh" | "pooled") {
            return Err(error("HTTP connection mode must be explicit"));
        }
        if self.slo.tpot_boundary != TpotBoundary::LastVisibleOutput {
            return Err(error(
                "capacity uses the declared last-visible-output TPOT boundary",
            ));
        }
        Ok(())
    }
}

impl CapacitySearch {
    pub fn new(contract: CapacityContract) -> Result<Self, CapacityError> {
        contract.validate()?;
        let digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&contract).map_err(|e| error(e.to_string()))?)
        );
        Ok(Self {
            session_history: Default::default(),
            contract,
            digest,
            records: BTreeMap::new(),
            instances: BTreeSet::new(),
            last_ended_unix_ns: 0,
        })
    }

    pub fn authorize_session_block(
        &self,
        block: &super::session::CapacitySessionBlock,
        origin_evidence_sha256: Option<&str>,
    ) -> Result<super::session::AuthorizedCapacitySessionBlock, CapacityError> {
        let SearchProgress::Awaiting { runs, .. } = self.progress() else {
            return Err(error("search is complete"));
        };
        if runs.first() != Some(&block.key) || block.checked_live_unix_ns < self.last_ended_unix_ns
        {
            return Err(error(
                "session authorization does not match next acquisition or its clock",
            ));
        }
        self.session_history
            .authorize(&self.contract, block, origin_evidence_sha256)
    }

    pub fn contract(&self) -> &CapacityContract {
        &self.contract
    }
    pub fn contract_sha256(&self) -> &str {
        &self.digest
    }

    fn keys(
        &self,
        phase: CapacityPhase,
        indices: &[usize],
        repetitions: u32,
    ) -> Vec<CapacityRunKey> {
        let mut result = Vec::new();
        for repetition in 0..repetitions {
            // Alternate traversal without adapting order to candidate speed.
            let rates: Vec<_> = if repetition % 2 == 0 {
                indices.to_vec()
            } else {
                indices.iter().rev().copied().collect()
            };
            for rate_index in rates {
                let replicas: &[u8] = if phase == CapacityPhase::Aa {
                    if repetition % 2 == 0 {
                        &[0, 1]
                    } else {
                        &[1, 0]
                    }
                } else {
                    &[0]
                };
                for &replica in replicas {
                    result.push(CapacityRunKey {
                        phase,
                        rate_index,
                        repetition,
                        replica,
                    });
                }
            }
        }
        result
    }

    fn rate_status(&self, phase: CapacityPhase, index: usize) -> SloStatus {
        let rows: Vec<_> = self
            .records
            .values()
            .filter(|r| r.key.phase == phase && r.key.rate_index == index)
            .collect();
        if rows.is_empty() {
            return SloStatus::Unknown;
        }
        if rows.iter().any(|r| r.status == SloStatus::Fail) {
            return SloStatus::Fail;
        }
        if rows.iter().all(|r| r.status == SloStatus::Pass) {
            SloStatus::Pass
        } else {
            SloStatus::Unknown
        }
    }

    fn neighborhood_indices(&self) -> Vec<usize> {
        (0..self.contract.rates_rps.len())
            .filter(|i| !self.contract.coarse_indices.contains(i))
            .collect()
    }

    fn measured_indices(&self) -> Vec<usize> {
        self.contract
            .coarse_indices
            .iter()
            .copied()
            .chain(self.neighborhood_indices())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn progress(&self) -> SearchProgress {
        for (phase, indices, repeats) in [
            (
                CapacityPhase::Aa,
                self.contract.aa_indices.clone(),
                self.contract.repetitions.aa_pairs,
            ),
            (
                CapacityPhase::Coarse,
                self.contract.coarse_indices.clone(),
                self.contract.repetitions.coarse,
            ),
            (
                CapacityPhase::Neighborhood,
                self.neighborhood_indices(),
                self.contract.repetitions.neighborhood,
            ),
            (
                CapacityPhase::Confirmation,
                self.measured_indices(),
                self.contract.repetitions.confirmation,
            ),
        ] {
            let runs: Vec<_> = self
                .keys(phase, &indices, repeats)
                .into_iter()
                .filter(|k| !self.records.contains_key(k))
                .collect();
            if !runs.is_empty() {
                return SearchProgress::Awaiting { phase, runs };
            }
        }
        SearchProgress::Complete
    }

    pub fn planned_run(&self, key: &CapacityRunKey) -> Result<PlannedCapacityRun, CapacityError> {
        let SearchProgress::Awaiting { runs, .. } = self.progress() else {
            return Err(error("search is complete"));
        };
        if !runs.contains(key) {
            return Err(error("run is not pending in the current phase"));
        }
        generate_run(&self.contract, &self.digest, key)
    }

    pub fn record(
        &mut self,
        evidence: CapacityRunEvidence,
    ) -> Result<&CapacityRunAssessment, CapacityError> {
        let SearchProgress::Awaiting { runs, .. } = self.progress() else {
            return Err(error("search is complete"));
        };
        if runs.first() != Some(&evidence.key) {
            return Err(error(
                "run does not follow the predeclared independent order",
            ));
        }
        if evidence.independent_run_id.trim().is_empty()
            || self.instances.contains(&evidence.independent_run_id)
        {
            return Err(error(
                "independent runs require distinct acquisition blocks",
            ));
        }
        if evidence.run_started_unix_ns < self.last_ended_unix_ns {
            return Err(error(
                "independent serving runs overlap or are out of order",
            ));
        }
        if evidence
            .session
            .as_ref()
            .is_some_and(|block| block.checked_live_unix_ns < self.last_ended_unix_ns)
        {
            return Err(error(
                "session checkpoint predates the preceding acquisition",
            ));
        }
        let planned = self.planned_run(&evidence.key)?;
        let reused = self
            .session_history
            .validate_run(&self.contract, &evidence)?;
        let assessed =
            super::evaluate::evaluate_run_verified(&self.contract, &planned, &evidence, reused)?;
        self.session_history
            .record(&self.contract, &evidence, &assessed)?;
        self.last_ended_unix_ns = evidence.run_ended_unix_ns;
        self.instances.insert(evidence.independent_run_id);
        let key = evidence.key;
        self.records.insert(key.clone(), assessed);
        Ok(&self.records[&key])
    }

    pub fn report(&self) -> CapacitySearchReport {
        let complete = self.progress() == SearchProgress::Complete;
        let measured: Vec<_> = self
            .records
            .keys()
            .filter(|k| k.phase != CapacityPhase::Aa)
            .map(|k| k.rate_index)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let confirmed: Vec<_> = measured
            .iter()
            .copied()
            .filter(|i| {
                let rows: Vec<_> = self
                    .records
                    .values()
                    .filter(|r| r.key.phase != CapacityPhase::Aa && r.key.rate_index == *i)
                    .collect();
                rows.iter()
                    .filter(|r| r.key.phase == CapacityPhase::Confirmation)
                    .count()
                    == self.contract.repetitions.confirmation as usize
                    && rows.iter().all(|r| r.status == SloStatus::Pass)
            })
            .collect();
        let mut aa_noise = Vec::new();
        for &rate_index in &self.contract.aa_indices {
            for pair in 0..self.contract.repetitions.aa_pairs {
                let a = self.records.get(&CapacityRunKey {
                    phase: CapacityPhase::Aa,
                    rate_index,
                    repetition: pair,
                    replica: 0,
                });
                let b = self.records.get(&CapacityRunKey {
                    phase: CapacityPhase::Aa,
                    rate_index,
                    repetition: pair,
                    replica: 1,
                });
                if let Some((a, b)) = a
                    .zip(b)
                    .filter(|(a, b)| {
                        a.evidence_complete
                            && b.evidence_complete
                            && a.workload_completed
                            && b.workload_completed
                            && a.arrival_schedule_delivered
                            && b.arrival_schedule_delivered
                    })
                    .and_then(|(a, b)| a.evaluation.as_ref().zip(b.evaluation.as_ref()))
                {
                    aa_noise.push(AaPairObservation {
                        rate_index,
                        pair,
                        successful_output_tps_delta: a
                            .successful_output
                            .tokens_per_second
                            .zip(b.successful_output.tokens_per_second)
                            .map(|(a, b)| b - a),
                        ttft_p99_ms_delta: a
                            .ttft
                            .observed_percentiles_ms
                            .zip(b.ttft.observed_percentiles_ms)
                            .map(|(a, b)| b.p99 - a.p99),
                        tpot_p99_ms_delta: a
                            .tpot
                            .observed_percentiles_ms
                            .zip(b.tpot.observed_percentiles_ms)
                            .map(|(a, b)| b.p99 - a.p99),
                        visible_itl_p99_ms_delta: a
                            .pooled_visible_itl
                            .observed_percentiles_ms
                            .zip(b.pooled_visible_itl.observed_percentiles_ms)
                            .map(|(a, b)| b.p99 - a.p99),
                    });
                }
            }
        }
        let aa_complete = aa_noise.len()
            == self.contract.aa_indices.len() * self.contract.repetitions.aa_pairs as usize;
        let transitions = self
            .contract
            .coarse_indices
            .windows(2)
            .filter(|p| {
                self.rate_status(CapacityPhase::Coarse, p[0])
                    != self.rate_status(CapacityPhase::Coarse, p[1])
            })
            .map(|p| [p[0], p[1]])
            .collect();
        CapacitySearchReport {contract_sha256:self.digest.clone(),complete,runs:self.records.values().cloned().collect(),aa_noise,unmeasured_rate_indices:(0..self.contract.rates_rps.len()).filter(|i|!measured.contains(i)).collect(),measured_rate_indices:measured,observed_coarse_transition_intervals:transitions,maximum_confirmed_tested_rate_rps:(complete&&aa_complete).then(||confirmed.last().map(|i|self.contract.rates_rps[*i])).flatten(),independently_confirmed_rate_indices:confirmed,scope:"Observed finite send-window plus drain, explicit queue-window slope limits, independent repeats, fixed workload and tested grid only. Unmeasured intervals and future steady-state capacity remain unknown; A/A differences are descriptive, not a confidence bound.".into()}
    }
}

pub(super) fn generate_run(
    contract: &CapacityContract,
    digest: &str,
    key: &CapacityRunKey,
) -> Result<PlannedCapacityRun, CapacityError> {
    let repeats = match key.phase {
        CapacityPhase::Aa => contract.repetitions.aa_pairs,
        CapacityPhase::Coarse => contract.repetitions.coarse,
        CapacityPhase::Neighborhood => contract.repetitions.neighborhood,
        CapacityPhase::Confirmation => contract.repetitions.confirmation,
    };
    if key.rate_index >= contract.rates_rps.len()
        || key.repetition >= repeats
        || key.replica > u8::from(key.phase == CapacityPhase::Aa)
    {
        return Err(error("planned run key is outside the frozen contract"));
    }
    let seed_bytes = Sha256::digest(
        serde_json::to_vec(&(contract.seed, key.phase, key.rate_index, key.repetition))
            .map_err(|e| error(e.to_string()))?,
    );
    // Pair members have identical offered arrivals; phases/repeats differ.
    let seed = u64::from_le_bytes(seed_bytes[..8].try_into().unwrap());
    let mut rng = StdRng::seed_from_u64(seed);
    let times = poisson_arrival_times(
        contract.rates_rps[key.rate_index],
        contract.maximum_requests_per_run + 1,
        &mut rng,
    );
    let count = times.partition_point(|t| *t < contract.window.send_seconds);
    if count == 0 {
        return Err(error(
            "declared seed/rate/window produces no arrivals; cannot claim capacity",
        ));
    }
    if count > contract.maximum_requests_per_run {
        return Err(error(
            "Poisson arrivals exceed explicit run bound; refusing to truncate",
        ));
    }
    let measured_indices: Vec<_> = contract
        .workload
        .samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.phase == BenchmarkPhase::Measured)
        .map(|(i, _)| i)
        .collect();
    let workload_sample_indices: Vec<_> = (0..count)
        .map(|i| measured_indices[i % measured_indices.len()])
        .collect();
    Ok(PlannedCapacityRun {
        contract_sha256: digest.into(),
        key: key.clone(),
        cell_id: format!(
            "capacity.{:?}.{}.{}.{}",
            key.phase, key.rate_index, key.repetition, key.replica
        ),
        rate_rps: contract.rates_rps[key.rate_index],
        seed,
        send_seconds: contract.window.send_seconds,
        scheduled_arrival_ms: times[..count].iter().map(|s| s * 1000.0).collect(),
        output_token_budgets: workload_sample_indices
            .iter()
            .map(|i| contract.workload.samples[*i].requested_output_tokens)
            .collect(),
        workload_sample_indices,
    })
}
