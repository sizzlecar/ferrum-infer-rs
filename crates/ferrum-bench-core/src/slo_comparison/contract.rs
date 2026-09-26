use super::*;
use crate::{slo::TpotBoundary, BenchmarkPhase};

impl FrozenComparisonContract {
    pub fn validate(&self) -> Result<(), ComparisonError> {
        let invalid = |message: &str| ComparisonError(message.to_owned());
        if self.schema_version != 1
            || self.frozen_unix_ns == 0
            || self.pairs.is_empty()
            || !self
                .cells
                .iter()
                .any(|cell| cell.scope == CellScope::Primary)
        {
            return Err(invalid(
                "contract needs version 1, freeze time, pairs and primary scope",
            ));
        }
        if self.capacity.slots == 0
            || self.capacity.context_tokens_per_request == 0
            || self.capacity.batch_tokens == 0
        {
            return Err(invalid("fixed server capacity must be positive"));
        }
        let mut cells = BTreeSet::new();
        for cell in &self.cells {
            if cell.concurrency == 0 || !cells.insert(cell.concurrency) {
                return Err(invalid("cell scope contains zero or duplicate concurrency"));
            }
        }
        for hash in [
            &self.shared.hardware_fingerprint_sha256,
            &self.shared.model_content_sha256,
            &self.shared.tokenizer_sha256,
            &self.shared.chat_template_sha256,
            &self.shared.client_binary_sha256,
            &self.shared.client_slo_config_sha256,
            &self.dataset.source_sha256,
            &self.dataset.tokenizer_sha256,
        ] {
            if !valid_digest(hash) {
                return Err(invalid("invalid frozen SHA-256 identity"));
            }
        }
        for text in [
            &self.shared.hardware_label,
            &self.shared.weight_precision,
            &self.shared.kv_precision,
            &self.http_connection_mode,
        ] {
            if text.trim().is_empty() {
                return Err(invalid("missing frozen execution or client boundary"));
            }
        }
        if !same_digest(
            &self.shared.tokenizer_sha256,
            &self.dataset.tokenizer_sha256,
        ) || self.dataset.dataset != "sharegpt"
        {
            return Err(invalid(
                "frozen dataset must be ShareGPT using the declared tokenizer",
            ));
        }
        for arm in [&self.baseline, &self.candidate] {
            if [
                &arm.implementation,
                &arm.backend,
                &arm.request_model_alias,
                &arm.numerical_policy,
            ]
            .iter()
            .any(|s| s.trim().is_empty())
                || !valid_digest(&arm.binary_sha256)
                || !valid_digest(&arm.effective_configuration_sha256)
            {
                return Err(invalid("missing frozen server identity or configuration"));
            }
        }
        self.sampling
            .validate()
            .map_err(|e| invalid(&e.to_string()))?;
        for policy in std::iter::once(&self.memory.device_allocation)
            .chain(self.memory.os_footprint.sampled())
        {
            if policy.window.trim().is_empty()
                || policy.interval_ms == 0
                || policy.max_sample_gap_ns == 0
            {
                return Err(invalid(
                    "sampled memory boundary, cadence and tolerated gap must be explicit",
                ));
            }
        }
        if self.memory.maximum_rss_window.trim().is_empty() {
            return Err(invalid("maximum RSS boundary must be explicit"));
        }
        if matches!(
            &self.memory.os_footprint,
            OsFootprintPolicy::ProcessLifetime(_)
        ) && self.memory.maximum_rss_window != "process_lifetime"
        {
            return Err(invalid(
                "native time footprint and maximum RSS both require process_lifetime",
            ));
        }
        let filter = &self.dataset.filter;
        if self.dataset.source_format.is_empty()
            || self.dataset.sampling.is_empty()
            || filter.min_input_tokens == 0
            || filter.min_output_tokens == 0
            || filter
                .max_input_tokens
                .is_some_and(|n| n < filter.min_input_tokens)
            || filter
                .max_output_tokens
                .is_some_and(|n| n < filter.min_output_tokens)
            || filter.fixed_output_tokens == Some(0)
            || filter.max_total_tokens == Some(0)
        {
            return Err(invalid("invalid frozen ShareGPT length policy"));
        }
        self.slo.validate().map_err(|e| invalid(&e.to_string()))?;
        if self.slo.tpot_boundary != TpotBoundary::LastVisibleOutput {
            return Err(invalid(
                "primary comparison requires explicitly last-visible TPOT",
            ));
        }
        if self.ratio_limits.len() != ComparisonMetric::ALL.len()
            || ComparisonMetric::ALL.iter().any(|metric| {
                self.ratio_limits
                    .get(metric)
                    .is_none_or(|value| !value.is_finite() || *value <= 0.0)
            })
        {
            return Err(invalid(
                "all seven positive finite ratio limits must be explicit",
            ));
        }
        let mut pair_ids = BTreeSet::new();
        for pair in &self.pairs {
            if pair.pair_id.trim().is_empty() || !pair_ids.insert(&pair.pair_id) {
                return Err(invalid("pair IDs must be nonempty and unique"));
            }
            let selection = &pair.selection;
            if !same_digest(
                &selection.selection_sha256,
                &json_digest(&selection.samples)?,
            ) {
                return Err(invalid(
                    "frozen selection hash does not match ordered samples",
                ));
            }
            let mut phase_counts = [0_u32; 2];
            let mut records = BTreeSet::new();
            for sample in &selection.samples {
                let phase = match sample.phase {
                    BenchmarkPhase::Warmup => 0,
                    BenchmarkPhase::Measured => 1,
                };
                if (phase == 0 && phase_counts[1] != 0)
                    || sample.request_index != phase_counts[phase]
                    || !records.insert(sample.source_record_index)
                    || sample.input_tokens == 0
                    || sample.requested_output_tokens == 0
                    || !valid_digest(&sample.prompt_sha256)
                    || !valid_digest(&sample.assistant_sha256)
                {
                    return Err(invalid("frozen selection has invalid order, counts, hash or duplicate source record"));
                }
                let total = u64::from(sample.input_tokens)
                    + u64::from(sample.requested_output_tokens)
                    + u64::from(filter.chat_template_reserve_tokens);
                if sample.input_tokens < filter.min_input_tokens
                    || filter
                        .max_input_tokens
                        .is_some_and(|n| sample.input_tokens > n)
                    || sample.reference_output_tokens < filter.min_output_tokens
                    || filter
                        .max_output_tokens
                        .is_some_and(|n| sample.reference_output_tokens > n)
                    || sample.requested_output_tokens
                        != filter
                            .fixed_output_tokens
                            .unwrap_or(sample.reference_output_tokens)
                    || filter
                        .max_total_tokens
                        .is_some_and(|n| total > u64::from(n))
                    || total > u64::from(self.capacity.context_tokens_per_request)
                {
                    return Err(invalid(
                        "selected request violates frozen length or context policy",
                    ));
                }
                phase_counts[phase] += 1;
            }
            if phase_counts[1] == 0 {
                return Err(invalid("frozen selection has no measured requests"));
            }
        }
        if let Some(method) = &self.uncertainty {
            if method.method_id.trim().is_empty()
                || method.analysis_unit.trim().is_empty()
                || !valid_digest(&method.configuration_sha256)
            {
                return Err(invalid("statistical method must be explicitly identified"));
            }
            statistics::validate_method(self, method)?;
        }
        Ok(())
    }
}
