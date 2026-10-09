//! Accounting is scoped to invocation preparation through dispatch return, including
//! failed dispatch attempts. Later observation can materialize more owned parts.
use super::*;

#[derive(Default)]
pub(super) struct PreparationMetrics {
    dispatch_attempts: AtomicU64,
    projected_identities: AtomicU64,
    parts_materialized: AtomicU64,
    compact_binding_nodes: AtomicU64,
}

impl InvocationPreparationSink for PreparationMetrics {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.dispatch_attempts.fetch_add(1, Ordering::Relaxed);
        self.projected_identities
            .fetch_add(stats.projected_identities, Ordering::Relaxed);
        self.parts_materialized
            .fetch_add(stats.parts_materialized, Ordering::Relaxed);
        self.compact_binding_nodes
            .fetch_add(stats.compact_binding_nodes, Ordering::Relaxed);
    }
}

impl PreparationMetrics {
    pub(super) fn reset(&self) {
        for counter in [
            &self.dispatch_attempts,
            &self.projected_identities,
            &self.parts_materialized,
            &self.compact_binding_nodes,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    pub(super) fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "scope": "invocation_preparation_through_dispatch_return_including_failures",
            "full_mode": "projection_counters_zero_do_not_measure_full_preparation_work",
            "later_observation": "owned_parts_materialized_after_dispatch_return_are_excluded",
            "compact_binding_scope": "provider_successfully_produced_compact_binding_before_dispatch_return;does_not_prove_submit_or_gpu_completion",
            "dispatch_attempts": self.dispatch_attempts.load(Ordering::Relaxed),
            "projected_identities": self.projected_identities.load(Ordering::Relaxed),
            "parts_materialized": self.parts_materialized.load(Ordering::Relaxed),
            "compact_binding_nodes": self.compact_binding_nodes.load(Ordering::Relaxed),
        })
    }
}

/// Off avoids timing calls while retaining independent preparation accounting.
pub(super) struct DisabledTiming;
impl DeviceSubmissionTimingSink for DisabledTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _stage: DeviceSubmissionStage, _elapsed: Duration) {}
}
impl SubmissionWaveDispatchTimingSink for DisabledTiming {
    fn record(&self, _stage: SubmissionWaveDispatchStage, _elapsed: Duration) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invocation_preparation_metrics_aggregate_dispatch_snapshots_and_reset_after_startup() {
        let executor_metrics = VNextExecutorMetrics::default();
        let metrics = &executor_metrics.invocation_preparation;
        metrics.record_preparation(InvocationPreparationStats {
            projected_identities: 5,
            parts_materialized: 2,
            compact_binding_nodes: 4,
        });
        metrics.record_preparation(InvocationPreparationStats {
            projected_identities: 3,
            parts_materialized: 0,
            compact_binding_nodes: 2,
        });
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["dispatch_attempts"], 2);
        assert_eq!(snapshot["projected_identities"], 8);
        assert_eq!(snapshot["parts_materialized"], 2);
        assert_eq!(snapshot["compact_binding_nodes"], 6);
        executor_metrics.reset_after_startup();
        let reset = metrics.snapshot();
        for field in [
            "dispatch_attempts",
            "projected_identities",
            "parts_materialized",
            "compact_binding_nodes",
        ] {
            assert_eq!(reset[field], 0);
        }
        metrics.record_preparation(InvocationPreparationStats {
            projected_identities: 1,
            parts_materialized: 1,
            compact_binding_nodes: 0,
        });
        let next = metrics.snapshot();
        assert_eq!(next["dispatch_attempts"], 1);
        assert_eq!(next["projected_identities"], 1);
        assert_eq!(next["parts_materialized"], 1);
        assert_eq!(next["compact_binding_nodes"], 0);
    }

    #[test]
    fn invocation_preparation_reaches_shared_executor_without_changing_device_policy() {
        use ferrum_kernels::backend::reference::ReferenceVNextComposition;
        use ferrum_types::{DataType, ModelType, RuntimeConfigSnapshot};
        let composition = ReferenceVNextComposition::create(
            DeviceId::new("device.invocation-preparation-policy").unwrap(),
        )
        .unwrap();
        let info = ModelInfo {
            model_id: "preparation-policy-fixture".into(),
            model_type: ModelType::Custom("preparation-policy-fixture".into()),
            num_parameters: 0,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 16,
            max_sequence_length: 64,
            dtype: DataType::FP16,
            device: Device::CPU,
            version: None,
            license: None,
            metadata: Default::default(),
        };
        let mut engine = EngineConfig::default();
        let default =
            VNextExecutorConfig::from_engine_config(&engine, &info, composition.runtime().as_ref())
                .unwrap();
        assert_eq!(
            default.invocation_preparation_strategy,
            InvocationPreparationStrategy::Full
        );
        for strategy in [
            InvocationPreparationStrategy::IdentityProjection,
            InvocationPreparationStrategy::CompactBindings,
            InvocationPreparationStrategy::Full,
        ] {
            engine
                .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                    "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                    strategy.as_runtime_value(),
                )]))
                .unwrap();
            let config = VNextExecutorConfig::from_engine_config(
                &engine,
                &info,
                composition.runtime().as_ref(),
            )
            .unwrap();
            assert_eq!(config.invocation_preparation_strategy, strategy);
            assert_eq!(
                config.runtime_policy.fingerprint_str(),
                default.runtime_policy.fingerprint_str()
            );
            assert_eq!(config.maximum_model_tokens, default.maximum_model_tokens);
            assert_eq!(
                config.device_reusable_execution_enabled,
                default.device_reusable_execution_enabled
            );
        }
    }
}
