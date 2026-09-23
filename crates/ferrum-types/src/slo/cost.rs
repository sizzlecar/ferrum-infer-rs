use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    path::PathBuf,
};

/// Bounded observation storage and CPU training work. These limits never
/// authorize model work or turn missing cost evidence into a prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloCostObservationConfig {
    pub max_queued_samples: NonZeroUsize,
    /// Sum of allocated row capacities in the pending sample queue.
    pub max_queued_shape_rows: NonZeroUsize,
    /// A worker publishes at most one snapshot for each bounded drain batch.
    pub max_samples_per_update: NonZeroUsize,
    pub max_waves_per_call: NonZeroUsize,
    pub max_rows_per_wave: NonZeroUsize,
    pub max_retained_rows_per_call: NonZeroUsize,
    pub model: SloCostModelConfig,
    pub profile_import: SloCostProfileImportConfig,
    /// Opt-in worker-owned capture and atomic publication at engine shutdown.
    pub profile_export: Option<SloCostProfileExportConfig>,
}

impl Default for SloCostObservationConfig {
    fn default() -> Self {
        Self {
            max_queued_samples: NonZeroUsize::new(256).unwrap(),
            max_queued_shape_rows: NonZeroUsize::new(8192).unwrap(),
            max_samples_per_update: NonZeroUsize::new(256).unwrap(),
            max_waves_per_call: NonZeroUsize::new(4).unwrap(),
            max_rows_per_wave: NonZeroUsize::new(1024).unwrap(),
            max_retained_rows_per_call: NonZeroUsize::new(4096).unwrap(),
            model: SloCostModelConfig::default(),
            profile_import: SloCostProfileImportConfig::default(),
            profile_export: None,
        }
    }
}

impl SloCostObservationConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.model.validate()?;
        self.profile_import.validate()?;
        if let Some(export) = &self.profile_export {
            export.validate()?;
        }
        if self.max_queued_samples.get() > 4096
            || self.max_queued_shape_rows.get() > 65_536
            || self.max_samples_per_update.get() > 4096
            || self.max_waves_per_call.get() > 4096
            || self.max_rows_per_wave.get() > 1024
            || self.max_retained_rows_per_call.get() > 65_536
        {
            return Err("cost_observation exceeds bounded recorder/queue capacity".to_owned());
        }
        if self.max_retained_rows_per_call < self.max_rows_per_wave {
            return Err("cost_observation retained rows cannot hold one maximum wave".to_owned());
        }
        if self.max_samples_per_update > self.max_queued_samples {
            return Err("cost_observation update batch exceeds pending sample capacity".to_owned());
        }
        Ok(())
    }
}

/// Explicit empirical calibration settings shared by live learning and import.
/// These defaults are bounded starting values, not measured capacity promises.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloCostFeatureModel {
    /// Original exact identities and explicitly configured context buckets.
    ExactV1 {},
    /// Empirical complete-wave envelope within measured joint numeric support.
    /// Work-axis monotonicity is an explicit modeling assumption, not a device
    /// latency guarantee. Provider and host branch identities remain exact.
    BoundedNumericV1 {
        host_history_bucket_tokens: NonZeroU32,
    },
    /// Complete host-settled wave costs conditioned on a verified host-content
    /// domain. Ungenerated text and sampling retries are empirical residuals,
    /// not measured worst-case work or a deterministic latency guarantee.
    /// Only the explicitly versioned host-settled boundary may train this mode.
    EmpiricalHostContentV1 {
        host_history_bucket_tokens: NonZeroU32,
    },
    /// Empirical host-settled costs with joint work/numeric/static row tuples
    /// pooled within unchanged physical role segments. This does not reorder
    /// execution or make a deterministic latency guarantee.
    EmpiricalRowMultisetV2 {
        host_history_bucket_tokens: NonZeroU32,
    },
    /// V3 keeps exact executable work and provider identities, but treats the
    /// actual full prefill context as a jointly observed support coordinate rather
    /// than a statistical equality key. Chunk count and final/output branches
    /// remain exact. This is an empirical model, not a latency guarantee.
    /// Persisted training requires profile schema 5; V4 is not relabeled.
    EmpiricalPromptRangeV3 {
        host_history_bucket_tokens: NonZeroU32,
    },
}

impl Default for SloCostFeatureModel {
    fn default() -> Self {
        Self::ExactV1 {}
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloCostModelConfig {
    pub feature_model: SloCostFeatureModel,
    pub max_buckets: NonZeroUsize,
    pub max_samples_per_bucket: NonZeroUsize,
    pub min_samples: NonZeroUsize,
    pub max_retained_samples: NonZeroUsize,
    pub max_retained_shape_rows: NonZeroUsize,
    pub residual_quantile: f64,
    pub drift_margin_ns: u64,
    pub max_wave_ns: NonZeroU64,
    pub max_sample_age_ns: NonZeroU64,
    pub context_bucket_tokens: NonZeroU32,
    pub prefill_offset_bucket_tokens: NonZeroU32,
    pub shape_limits: SloCostShapeLimits,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloCostShapeLimits {
    pub max_rows: NonZeroUsize,
    pub max_context_tokens: NonZeroU32,
    pub max_prefill_tokens_per_wave: NonZeroU64,
    pub max_state_bytes: NonZeroU64,
    pub max_maintenance_units: NonZeroU32,
}

impl Default for SloCostShapeLimits {
    fn default() -> Self {
        Self {
            max_rows: NonZeroUsize::new(128).unwrap(),
            max_context_tokens: NonZeroU32::new(262_144).unwrap(),
            max_prefill_tokens_per_wave: NonZeroU64::new(65_536).unwrap(),
            max_state_bytes: NonZeroU64::new(1 << 40).unwrap(),
            max_maintenance_units: NonZeroU32::new(1_048_576).unwrap(),
        }
    }
}

impl Default for SloCostModelConfig {
    fn default() -> Self {
        Self {
            feature_model: SloCostFeatureModel::default(),
            max_buckets: NonZeroUsize::new(1024).unwrap(),
            max_samples_per_bucket: NonZeroUsize::new(128).unwrap(),
            min_samples: NonZeroUsize::new(8).unwrap(),
            max_retained_samples: NonZeroUsize::new(16_384).unwrap(),
            max_retained_shape_rows: NonZeroUsize::new(262_144).unwrap(),
            residual_quantile: 0.99,
            drift_margin_ns: 100_000,
            max_wave_ns: NonZeroU64::new(60_000_000_000).unwrap(),
            max_sample_age_ns: NonZeroU64::new(300_000_000_000).unwrap(),
            context_bucket_tokens: NonZeroU32::new(1).unwrap(),
            prefill_offset_bucket_tokens: NonZeroU32::new(1).unwrap(),
            shape_limits: SloCostShapeLimits::default(),
        }
    }
}

impl SloCostModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.residual_quantile.is_finite()
            || self.residual_quantile <= 0.0
            || self.residual_quantile > 1.0
        {
            return Err("cost model residual_quantile must be finite in (0, 1]".into());
        }
        if self.max_buckets.get() > 65_536
            || self.max_samples_per_bucket.get() > 4096
            || self.max_retained_samples.get() > 131_072
            || self.max_retained_shape_rows.get() > 1_048_576
            || self.shape_limits.max_rows.get() > 1024
        {
            return Err("cost model exceeds hard retention limits".into());
        }
        if self.min_samples > self.max_samples_per_bucket
            || self.min_samples > self.max_retained_samples
            || self.shape_limits.max_rows > self.max_retained_shape_rows
        {
            return Err("cost model minimum coverage exceeds retention capacity".into());
        }
        self.max_wave_ns
            .get()
            .checked_add(self.drift_margin_ns)
            .ok_or_else(|| "cost model wave cost plus drift margin overflows".to_owned())?;
        Ok(())
    }
}

/// Bounded import policy. Clock accuracy is an operator declaration; a system
/// wall-clock reading alone does not establish it. None disables profile import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloCostProfileImportConfig {
    /// Per-file limit; the default stays 16 MiB. Larger evidence needs an
    /// explicit configuration, up to `Self::MAX_FILE_BYTES`.
    pub max_file_bytes: NonZeroUsize,
    pub max_samples: NonZeroUsize,
    pub max_total_shape_rows: NonZeroUsize,
    pub max_source_field_bytes: NonZeroUsize,
    pub max_profile_age_ns: NonZeroU64,
    pub max_clock_error_ns: u64,
    pub declared_local_clock_max_error_ns: Option<u64>,
}

impl Default for SloCostProfileImportConfig {
    fn default() -> Self {
        Self {
            max_file_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
            max_samples: NonZeroUsize::new(16_384).unwrap(),
            max_total_shape_rows: NonZeroUsize::new(262_144).unwrap(),
            max_source_field_bytes: NonZeroUsize::new(4096).unwrap(),
            max_profile_age_ns: NonZeroU64::new(86_400_000_000_000).unwrap(),
            max_clock_error_ns: 1_000_000_000,
            declared_local_clock_max_error_ns: None,
        }
    }
}

impl SloCostProfileImportConfig {
    /// Shared offline import/export ceiling, independent of retained RAM,
    /// sample count, shape rows, model precision and freshness policy.
    pub const MAX_FILE_BYTES: usize = 256 * 1024 * 1024;

    pub fn validate(&self) -> Result<(), String> {
        if self.max_file_bytes.get() > Self::MAX_FILE_BYTES
            || self.max_samples.get() > 131_072
            || self.max_total_shape_rows.get() > 1_048_576
            || self.max_source_field_bytes.get() > 16_384
        {
            return Err("cost profile import exceeds hard resource limits".into());
        }
        if self
            .declared_local_clock_max_error_ns
            .is_some_and(|error| error > self.max_clock_error_ns)
        {
            return Err("declared local clock error exceeds cost profile policy".into());
        }
        Ok(())
    }
}

/// Actual import receipt, separate from user policy and its original hash.
/// File integrity and declarations are not hardware or clock attestations;
/// recorded samples do not imply coverage of a subsequent execution shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SloCostProfileReceipt {
    pub schema_version: u32,
    pub path: PathBuf,
    pub file_sha256: String,
    pub file_bytes: usize,
    pub generated_unix_ns: u64,
    pub loaded_unix_ns: u64,
    pub conservative_clock_error_ns: u64,
    pub declared_local_clock_max_error_ns: u64,
    pub oldest_imported_age_ns: Option<u64>,
    pub newest_imported_age_ns: Option<u64>,
    pub offered_samples: usize,
    pub recorded_samples: usize,
    pub stale_samples: usize,
    pub skipped_samples: BTreeMap<String, usize>,
    pub model_version: u64,
    pub bucket_count: usize,
    pub source_generator: String,
    pub source_generator_revision: String,
    pub source_measurement_protocol: String,
    pub source_observation_artifact_sha256: [u8; 32],
}

pub const SLO_COST_PROFILE_RECEIPT_RUNTIME_KEY: &str = "slo_cost_profile_receipt";

/// Bounded raw observations and an importable profile, written by the CPU
/// training worker. Each destination must be absent; parent directories must
/// exist. Publication never replaces an existing artifact. Paths are relative
/// to the policy file in the CLI, and to the working directory for direct API
/// callers. The clock declaration applies over the entire capture interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloCostProfileExportConfig {
    pub path: PathBuf,
    pub observations_path: PathBuf,
    /// Per-file byte ceiling, independently applied to the JSON and JSONL.
    pub max_file_bytes: NonZeroUsize,
    pub max_samples: NonZeroUsize,
    pub max_total_shape_rows: NonZeroUsize,
    /// None is unknown clock accuracy and cannot enable profile export.
    pub declared_clock_max_error_ns: Option<u64>,
}

impl Default for SloCostProfileExportConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            observations_path: PathBuf::new(),
            max_file_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
            max_samples: NonZeroUsize::new(16_384).unwrap(),
            max_total_shape_rows: NonZeroUsize::new(262_144).unwrap(),
            declared_clock_max_error_ns: None,
        }
    }
}

impl SloCostProfileExportConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.path.as_os_str().is_empty()
            || self.observations_path.as_os_str().is_empty()
            || self.path == self.observations_path
        {
            return Err("cost profile export requires two distinct nonempty file paths".into());
        }
        if self.max_file_bytes.get() > SloCostProfileImportConfig::MAX_FILE_BYTES
            || self.max_samples.get() > 131_072
            || self.max_total_shape_rows.get() > 1_048_576
        {
            return Err("cost profile export exceeds hard resource limits".into());
        }
        if self.declared_clock_max_error_ns.is_none() {
            return Err("cost profile export requires declared clock accuracy".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_export_requires_explicit_paths_clock_and_bounded_capacity() {
        let value = serde_json::json!({"path":"cost.json", "observations_path":"raw.jsonl", "declared_clock_max_error_ns":0});
        let valid: SloCostProfileExportConfig = serde_json::from_value(value.clone()).unwrap();
        valid.validate().unwrap();
        assert!(SloCostProfileExportConfig::default().validate().is_err());
        for (field, invalid) in [
            ("path", serde_json::json!("")),
            ("observations_path", serde_json::json!("cost.json")),
            ("declared_clock_max_error_ns", serde_json::Value::Null),
            (
                "max_file_bytes",
                serde_json::json!(SloCostProfileImportConfig::MAX_FILE_BYTES + 1),
            ),
            ("max_samples", serde_json::json!(131_073)),
            ("max_total_shape_rows", serde_json::json!(1_048_577)),
        ] {
            let mut invalid_value = value.clone();
            invalid_value[field] = invalid;
            assert!(
                serde_json::from_value::<SloCostProfileExportConfig>(invalid_value)
                    .unwrap()
                    .validate()
                    .is_err(),
                "{field}"
            );
        }
        for (field, invalid) in [
            ("max_samples", serde_json::json!(0)),
            ("assume_complete", serde_json::json!(true)),
        ] {
            let mut invalid_value = value.clone();
            invalid_value[field] = invalid;
            assert!(serde_json::from_value::<SloCostProfileExportConfig>(invalid_value).is_err());
        }
        let mut policy = super::super::SloConfig::default();
        policy.cost_observation.profile_export = Some(valid);
        assert!(
            policy.validate().is_err(),
            "Off must not silently accept an enabled export"
        );
        let encoded = serde_json::to_string(&policy.cost_observation).unwrap();
        assert_eq!(
            serde_json::from_str::<SloCostObservationConfig>(&encoded).unwrap(),
            policy.cost_observation
        );
    }

    #[test]
    fn cost_model_policy_rejects_invalid_precision_and_impossible_capacity() {
        for quantile in [f64::NAN, f64::INFINITY, -0.1, 0.0, 1.1] {
            let config = SloCostModelConfig {
                residual_quantile: quantile,
                ..Default::default()
            };
            assert!(config.validate().is_err());
        }
        for source in [
            r#"{"model":{"max_buckets":65537}}"#,
            r#"{"model":{"min_samples":129}}"#,
            r#"{"model":{"max_retained_shape_rows":127}}"#,
            r#"{"model":{"max_wave_ns":18446744073709551615}}"#,
            r#"{"profile_import":{"max_file_bytes":268435457}}"#,
            r#"{"profile_import":{"max_samples":131073}}"#,
            r#"{"profile_import":{"max_total_shape_rows":1048577}}"#,
            r#"{"profile_import":{"declared_local_clock_max_error_ns":1000000001}}"#,
        ] {
            let config: SloCostObservationConfig = serde_json::from_str(source).unwrap();
            assert!(config.validate().is_err(), "{source}");
        }
        for source in [
            r#"{"model":{"min_samples":0}}"#,
            r#"{"model":{"shape_limits":{"max_rows":0}}}"#,
            r#"{"profile_import":{"max_profile_age_ns":0}}"#,
            r#"{"profile_import":{"assume_clock_trusted":true}}"#,
        ] {
            assert!(
                serde_json::from_str::<SloCostObservationConfig>(source).is_err(),
                "{source}"
            );
        }
        assert_eq!(
            SloCostProfileImportConfig::default().declared_local_clock_max_error_ns,
            None
        );
    }

    #[test]
    fn profile_receipt_cannot_be_supplied_by_serialized_engine_input() {
        let mut input = serde_json::to_value(crate::EngineConfig::default()).unwrap();
        input["slo_cost_profile_receipt"] = serde_json::json!({"file_sha256":"invented receipt"});
        let config: crate::EngineConfig = serde_json::from_value(input).unwrap();
        assert!(config.slo_cost_profile_receipt.is_none());
        assert!(crate::is_typed_only_runtime_config_key(
            SLO_COST_PROFILE_RECEIPT_RUNTIME_KEY
        ));
    }

    #[test]
    fn cost_observation_config_preserves_defaults_and_rejects_unbounded_storage() {
        let defaults = SloCostObservationConfig::default();
        defaults.validate().unwrap();
        assert_eq!(
            serde_json::from_str::<SloCostObservationConfig>("{}").unwrap(),
            defaults
        );
        for source in [
            r#"{"max_queued_samples":0}"#,
            r#"{"max_rows_per_wave":0}"#,
            r#"{"unknown_capacity":1}"#,
        ] {
            assert!(serde_json::from_str::<SloCostObservationConfig>(source).is_err());
        }
        for source in [
            r#"{"max_queued_samples":4097}"#,
            r#"{"max_queued_shape_rows":65537}"#,
            r#"{"max_rows_per_wave":1025}"#,
            r#"{"max_waves_per_call":4097}"#,
            r#"{"max_retained_rows_per_call":65537}"#,
            r#"{"max_retained_rows_per_call":8,"max_rows_per_wave":9}"#,
            r#"{"max_queued_samples":4,"max_samples_per_update":5}"#,
        ] {
            let config: SloCostObservationConfig = serde_json::from_str(source).unwrap();
            assert!(config.validate().is_err(), "{source}");
        }
    }

    #[test]
    fn cost_observation_capacity_roundtrips_in_the_effective_policy() {
        let mut policy = super::super::SloConfig::default();
        policy.cost_observation.max_samples_per_update = NonZeroUsize::new(17).unwrap();
        policy.cost_observation.model.min_samples = NonZeroUsize::new(11).unwrap();
        policy.cost_observation.model.feature_model = SloCostFeatureModel::BoundedNumericV1 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        };
        policy
            .cost_observation
            .profile_import
            .declared_local_clock_max_error_ns = Some(250);
        let snapshot =
            crate::RuntimeConfigSnapshot::from_entries([crate::RuntimeConfigEntry::new(
                crate::SLO_CONFIG_RUNTIME_KEY,
                serde_json::to_string(&policy).unwrap(),
                crate::RuntimeConfigSource::ConfigFile,
            )]);
        let mut config = crate::EngineConfig::default();
        config.apply_runtime_config_snapshot(&snapshot).unwrap();
        assert_eq!(
            config.scheduler.slo.cost_observation,
            policy.cost_observation
        );
    }

    #[test]
    fn numeric_model_selection_is_explicit_and_strict() {
        assert!(matches!(
            SloCostModelConfig::default().feature_model,
            SloCostFeatureModel::ExactV1 {}
        ));
        for source in [
            r#"{"kind":"bounded_numeric_v1"}"#,
            r#"{"kind":"bounded_numeric_v1","host_history_bucket_tokens":0}"#,
            r#"{"kind":"exact_v1","ignore_features":true}"#,
            r#"{"kind":"bounded_numeric_v2","host_history_bucket_tokens":64}"#,
        ] {
            assert!(
                serde_json::from_str::<SloCostFeatureModel>(source).is_err(),
                "{source}"
            );
        }
        let mode: SloCostFeatureModel = serde_json::from_str(
            r#"{"kind":"bounded_numeric_v1","host_history_bucket_tokens":64}"#,
        )
        .unwrap();
        assert_eq!(
            serde_json::from_str::<SloCostFeatureModel>(&serde_json::to_string(&mode).unwrap())
                .unwrap(),
            mode
        );
    }
}
