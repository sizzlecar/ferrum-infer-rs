//! Accounting is scoped to identity preparation through dispatch return, including
//! failed dispatch attempts. Later observation can materialize more owned parts.
use super::*;

#[derive(Default)]
pub(super) struct PreparationMetrics {
    dispatch_attempts: AtomicU64,
    projected_identities: AtomicU64,
    parts_materialized: AtomicU64,
    segment_hits: AtomicU64,
    segment_misses: AtomicU64,
    segment_encoded_nodes: AtomicU64,
    segment_dynamic_resource_requests: AtomicU64,
    segment_unique_physical_buffers: AtomicU64,
    segment_no_resident_program: AtomicU64,
    segment_no_cached_recipe: AtomicU64,
    segment_immutable_capability_unavailable: AtomicU64,
    segment_unsupported_encoder: AtomicU64,
    segment_incomplete_declarations: AtomicU64,
}

impl InvocationPreparationSink for PreparationMetrics {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.dispatch_attempts.fetch_add(1, Ordering::Relaxed);
        self.projected_identities
            .fetch_add(stats.projected_identities, Ordering::Relaxed);
        self.parts_materialized
            .fetch_add(stats.parts_materialized, Ordering::Relaxed);
        for (counter, value) in [
            (&self.segment_hits, stats.segment_hits),
            (&self.segment_misses, stats.segment_misses),
            (&self.segment_encoded_nodes, stats.segment_encoded_nodes),
            (
                &self.segment_dynamic_resource_requests,
                stats.segment_dynamic_resource_requests,
            ),
            (
                &self.segment_unique_physical_buffers,
                stats.segment_unique_physical_buffers,
            ),
            (
                &self.segment_no_resident_program,
                stats.segment_no_resident_program,
            ),
            (
                &self.segment_no_cached_recipe,
                stats.segment_no_cached_recipe,
            ),
            (
                &self.segment_immutable_capability_unavailable,
                stats.segment_immutable_capability_unavailable,
            ),
            (
                &self.segment_unsupported_encoder,
                stats.segment_unsupported_encoder,
            ),
            (
                &self.segment_incomplete_declarations,
                stats.segment_incomplete_declarations,
            ),
        ] {
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(value))
            });
        }
    }
}

impl PreparationMetrics {
    pub(super) fn reset(&self) {
        for counter in [
            &self.dispatch_attempts,
            &self.projected_identities,
            &self.parts_materialized,
            &self.segment_hits,
            &self.segment_misses,
            &self.segment_encoded_nodes,
            &self.segment_dynamic_resource_requests,
            &self.segment_unique_physical_buffers,
            &self.segment_no_resident_program,
            &self.segment_no_cached_recipe,
            &self.segment_immutable_capability_unavailable,
            &self.segment_unsupported_encoder,
            &self.segment_incomplete_declarations,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    pub(super) fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "scope": "identity_projection_through_dispatch_return_including_failures",
            "full_mode": "projection_counters_zero_do_not_measure_full_preparation_work",
            "later_observation": "owned_parts_materialized_after_dispatch_return_are_excluded",
            "dispatch_attempts": self.dispatch_attempts.load(Ordering::Relaxed),
            "projected_identities": self.projected_identities.load(Ordering::Relaxed),
            "parts_materialized": self.parts_materialized.load(Ordering::Relaxed),
            "decode_segment": {
                "scope": "preparation_through_dispatch_return_including_later_failures",
                "accounting": "hits_are_complete_segment_encode_successes_misses_are_cold_or_unsupported_fallbacks_neither_is_gpu_completion",
                "limitations": "failed_preparation_before_an_outcome_is_not_a_miss_nonatomic_snapshot_saturating_counters",
                "hits": self.segment_hits.load(Ordering::Relaxed),
                "misses": self.segment_misses.load(Ordering::Relaxed),
                "encoded_nodes": self.segment_encoded_nodes.load(Ordering::Relaxed),
                "dynamic_resource_requests": self.segment_dynamic_resource_requests.load(Ordering::Relaxed),
                "unique_physical_buffers": self.segment_unique_physical_buffers.load(Ordering::Relaxed),
                "miss_reason_scope": "observed_reason_events_may_overlap_one_fallback_cache_absence_combines_entry_epoch_owner_and_missing_cache",
                "miss_reasons": {
                    "no_resident_program": self.segment_no_resident_program.load(Ordering::Relaxed),
                    "no_cached_recipe": self.segment_no_cached_recipe.load(Ordering::Relaxed),
                    "immutable_capability_unavailable": self.segment_immutable_capability_unavailable.load(Ordering::Relaxed),
                    "unsupported_encoder": self.segment_unsupported_encoder.load(Ordering::Relaxed),
                    "incomplete_declarations": self.segment_incomplete_declarations.load(Ordering::Relaxed),
                },
            },
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
            segment_hits: 1,
            segment_encoded_nodes: 3,
            segment_dynamic_resource_requests: 4,
            segment_unique_physical_buffers: 5,
            ..Default::default()
        });
        metrics.record_preparation(InvocationPreparationStats {
            projected_identities: 3,
            parts_materialized: 0,
            segment_misses: 1,
            segment_no_resident_program: 1,
            segment_no_cached_recipe: 2,
            segment_immutable_capability_unavailable: 3,
            segment_unsupported_encoder: 4,
            segment_incomplete_declarations: 5,
            ..Default::default()
        });
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["dispatch_attempts"], 2);
        assert_eq!(snapshot["projected_identities"], 8);
        assert_eq!(snapshot["parts_materialized"], 2);
        assert_eq!(snapshot["decode_segment"]["hits"], 1);
        assert_eq!(snapshot["decode_segment"]["misses"], 1);
        assert_eq!(snapshot["decode_segment"]["encoded_nodes"], 3);
        assert_eq!(snapshot["decode_segment"]["dynamic_resource_requests"], 4);
        assert_eq!(snapshot["decode_segment"]["unique_physical_buffers"], 5);
        for (field, value) in [
            ("no_resident_program", 1),
            ("no_cached_recipe", 2),
            ("immutable_capability_unavailable", 3),
            ("unsupported_encoder", 4),
            ("incomplete_declarations", 5),
        ] {
            assert_eq!(snapshot["decode_segment"]["miss_reasons"][field], value);
        }
        executor_metrics.reset_after_startup();
        let reset = metrics.snapshot();
        for field in [
            "dispatch_attempts",
            "projected_identities",
            "parts_materialized",
        ] {
            assert_eq!(reset[field], 0);
        }
        for field in [
            "hits",
            "misses",
            "encoded_nodes",
            "dynamic_resource_requests",
            "unique_physical_buffers",
        ] {
            assert_eq!(reset["decode_segment"][field], 0);
        }
        for field in [
            "no_resident_program",
            "no_cached_recipe",
            "immutable_capability_unavailable",
            "unsupported_encoder",
            "incomplete_declarations",
        ] {
            assert_eq!(reset["decode_segment"]["miss_reasons"][field], 0);
        }
        metrics.record_preparation(InvocationPreparationStats {
            projected_identities: 1,
            parts_materialized: 1,
            ..Default::default()
        });
        let next = metrics.snapshot();
        assert_eq!(next["dispatch_attempts"], 1);
        assert_eq!(next["projected_identities"], 1);
        assert_eq!(next["parts_materialized"], 1);
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
            InvocationPreparationStrategy::DecodeSegment,
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
