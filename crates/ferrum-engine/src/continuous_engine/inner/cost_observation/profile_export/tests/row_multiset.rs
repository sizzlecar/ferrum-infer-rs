//! Actual evidence order is preserved; only the model normalizes row tuples.
use super::host_content::{actual, evaluation, only};
use super::*;
use std::{
    num::NonZeroU32,
    sync::atomic::{AtomicU64, Ordering},
};

fn mode() -> model::CostFeatureModel {
    model::CostFeatureModel::EmpiricalRowMultisetV2 {
        host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
    }
}
fn stages(start: u64, reversed: bool) -> Arc<HostStageEvidenceV1> {
    let mut stages = (*host_content::stages(start, 50, true)).clone();
    let mut second = stages.rows[0].clone();
    second.request_id = ferrum_types::RequestId::new();
    second.owner_incarnation = 2;
    second.input_index = 1;
    second.host_processing_ordinal = Some(1);
    second.actual_work = HostStageWork::Decode { kv_tokens: 256 };
    stages.rows.push(second);
    let shape = stages.actual_shape.as_mut().unwrap();
    // V2 does not require, reconstruct, or hash an old whole-wave V1 field.
    shape.host_content_features = None;
    shape.decode_kv_tokens.push(256);
    let numeric = shape.numeric_features.as_mut().unwrap();
    numeric.rows.push(numeric.rows[0]);
    shape.row_multiset_features = Some(HostRowMultisetCostFeaturesV2 {
        schema_version: HOST_ROW_MULTISET_FEATURE_SCHEMA_V2,
        wave_policy_signature: [30; 32],
        rows: vec![
            HostRowStaticCostFeaturesV2 {
                role: HostRowRoleV2::Decode,
                categorical_signature: [31; 32],
            },
            HostRowStaticCostFeaturesV2 {
                role: HostRowRoleV2::Decode,
                categorical_signature: [32; 32],
            },
        ],
    });
    if reversed {
        stages.rows.reverse();
        shape.decode_kv_tokens.reverse();
        numeric.rows.reverse();
        shape.row_multiset_features.as_mut().unwrap().rows.reverse();
    }
    Arc::new(stages)
}

#[tokio::test]
async fn row_multiset_worker_cut_preserves_physical_order_and_product_loads_same_entry_evidence() {
    worker_cut_roundtrip(mode(), 4).await;
}

#[tokio::test]
async fn prompt_range_worker_cut_exports_v5_and_loads_original_entry_evidence() {
    worker_cut_roundtrip(
        model::CostFeatureModel::EmpiricalPromptRangeV3 {
            host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
        },
        5,
    )
    .await;
}

async fn worker_cut_roundtrip(feature_model: model::CostFeatureModel, schema_version: u32) {
    struct Clock(AtomicU64);
    impl CostObservationClock for Clock {
        fn now_ns(&self) -> Option<u64> {
            Some(self.0.load(Ordering::Acquire))
        }
    }
    let f = Fixture::new();
    let mut config = SloCostObservationConfig::default();
    config.model.feature_model = feature_model.clone();
    config.model.min_samples = NonZeroUsize::new(2).unwrap();
    config.model.drift_margin_ns = 0;
    config.model.max_sample_age_ns = NonZeroU64::new(1_000_000_000_000).unwrap();
    config.profile_export = Some(f.options.clone());
    let clock = Arc::new(Clock(AtomicU64::new(100)));
    let runtime = EngineCostRuntime::build(identity(), clock.clone(), &config, false).unwrap();
    let first = stages(100, false);
    let second = stages(300, true);
    runtime
        .sink
        .offer_evidence_numbered(only(first.clone()))
        .unwrap();
    // The middle entry has real old evidence, but no new row classes. Its
    // ordinal remains visible and cannot be reconstructed from the V1 hash.
    runtime
        .sink
        .offer_evidence_numbered(only(host_content::stages(200, 50, true)))
        .unwrap();
    runtime
        .sink
        .offer_evidence_numbered(only(second.clone()))
        .unwrap();
    let waiter = runtime
        .request_profile_cut(CostProfileCutPaths {
            profile: f.dir.join("row-cut.json"),
            source: f.dir.join("row-cut.jsonl"),
        })
        .unwrap();
    clock.0.store(351, Ordering::Release);
    runtime.consume_samples();
    let frozen = waiter.wait().await.unwrap();
    let cut = frozen.profile_cut.unwrap().unwrap();
    assert_eq!(cut.accepted_ordinal, 3);
    assert_eq!(cut.retained_samples, 2);
    assert_eq!(cut.raw_retained_observations, 0);
    let audit = runtime.audit_snapshot();
    assert_eq!(audit.training.host_content.offered_entries, 3);
    assert_eq!(audit.training.host_content.outcomes.recorded, 2);
    assert_eq!(audit.training.outcomes.recorded, 0);
    let raw: Vec<serde_json::Value> = fs::read_to_string(&cut.source)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(raw[0]["schema_version"], 4);
    for (index, ordinal) in [(1, 1), (3, 2), (5, 3)] {
        assert_eq!(raw[index]["accepted_ordinal"], ordinal);
        assert!(raw[index]["source_record"].is_null());
        assert_eq!(raw[index + 1]["kind"], "host_row_multiset_training_v2");
        assert_eq!(raw[index + 1]["accepted_ordinal"], ordinal);
        assert_eq!(raw[index + 1]["host_source_record"], ordinal - 1);
    }
    assert!(raw[4]["sample"].is_null());
    let bytes = fs::read(&cut.profile).unwrap();
    let file: profile_v4::CostProfileFileV4 = serde_json::from_slice(&bytes).unwrap();
    // V5 deliberately preserves the actual-evidence payload. The product
    // loader below validates the explicit schema/model pair, not this serde read.
    assert_eq!(file.schema_version, schema_version);
    assert_eq!(file.settings.feature_model, feature_model);
    assert_eq!(
        file.samples
            .iter()
            .map(|s| (s.accepted_ordinal, s.source_record))
            .collect::<Vec<_>>(),
        [(1, 0), (3, 2)]
    );
    for (sample, source) in file.samples.iter().zip([&first, &second]) {
        let shape = source.actual_shape.as_ref().unwrap();
        assert_eq!(
            sample.shape.row_multiset_features,
            *shape.row_multiset_features.as_ref().unwrap()
        );
        assert_eq!(
            sample.shape.exact.exact.decode_kv_tokens,
            shape.decode_kv_tokens
        );
        assert_eq!(sample.shape.exact.numeric_features, shape.numeric_features);
        assert!(sample.shape.host_content_features.is_none());
        assert_eq!(
            sample.measured_unix_ns,
            raw[0]["opening"]["wall_unix_ns"].as_u64().unwrap() + source.finalized_at_ns.unwrap()
                - 100
        );
    }
    let settings = super::super::super::profile::model_settings(&config.model);
    let loaded = profile::load_cost_profile_bytes(
        &bytes,
        &fingerprint(),
        &settings,
        &Default::default(),
        profile::ProfileLoadClock {
            wall_unix_ns: Some(file.generated_unix_ns),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        },
    )
    .unwrap();
    for shape in [&first, &second].map(|s| s.actual_shape.as_ref().unwrap()) {
        assert!(matches!(
            loaded.snapshot.predict(
                &fingerprint(),
                shape,
                model::CostBoundary::PreparationToHostSettledV1,
                0
            ),
            model::CostPrediction::Known(_)
        ));
        let mut legacy = shape.clone();
        legacy.row_multiset_features = None;
        assert!(matches!(
            loaded.snapshot.predict(
                &fingerprint(),
                &legacy,
                model::CostBoundary::PreparationToHostSettledV1,
                0
            ),
            model::CostPrediction::Unknown(_)
        ));
    }
    runtime.shutdown().await.unwrap();
    assert_eq!(f.records()[0]["schema_version"], 6);
}

#[test]
fn row_multiset_profile_keeps_original_age_and_charges_new_retained_rows() {
    let mut f = Fixture::new();
    f.settings.feature_model = mode();
    f.settings.drift_margin_ns = 0;
    let stages = stages(100, false);
    let sample = actual(&stages);
    // Two stage rows + two work + two numeric + two static, then another
    // two work + two numeric + two static for the immutable exported sample.
    let required = stages.retained_rows().unwrap() + 6;
    f.options.max_total_shape_rows = NonZeroUsize::new(required - 1).unwrap();
    let mut rejected = f.exporter().unwrap();
    rejected
        .record_entry_with_host(
            1,
            None,
            Some(stages.clone()),
            Some(CostCallRejection::Composite),
            Some((evaluation(), Some(&sample))),
        )
        .unwrap();
    assert_eq!(rejected.counts.raw_retained_entries, 0);
    assert_eq!(rejected.counts.host_stages_dropped_shape_row_limit, 1);
    f.options.max_total_shape_rows = NonZeroUsize::new(required).unwrap();
    let mut export = f.exporter().unwrap();
    export
        .record_entry_with_host(
            1,
            None,
            Some(stages),
            Some(CostCallRejection::Composite),
            Some((evaluation(), Some(&sample))),
        )
        .unwrap();
    let cut = export
        .write_cut(
            CostProfileCutPaths {
                profile: f.dir.join("ttl.json"),
                source: f.dir.join("ttl.jsonl"),
            },
            1,
            ExportClockReading {
                wall_unix_ns: 1300,
                monotonic_ns: 400,
            },
            stats(),
            &training_stats(),
        )
        .unwrap();
    let loaded = profile::load_cost_profile(
        &cut.profile,
        &fingerprint(),
        &f.settings,
        &Default::default(),
        profile::ProfileLoadClock {
            wall_unix_ns: Some(1400),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        },
    )
    .unwrap();
    let model::CostPrediction::Known(prediction) = loaded.snapshot.predict(
        &fingerprint(),
        &sample.actual_shape,
        model::CostBoundary::PreparationToHostSettledV1,
        0,
    ) else {
        panic!("new profile is independently importable")
    };
    assert_eq!(prediction.valid_for_ns, 651);
    assert_eq!(prediction.planning_ns, 50);
    assert!(matches!(
        loaded.snapshot.predict(
            &fingerprint(),
            &sample.actual_shape,
            model::CostBoundary::PreparationToHostSettledV1,
            652
        ),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
}

#[tokio::test]
async fn row_multiset_missing_numeric_or_wrong_physical_role_never_trains() {
    let mut config = SloCostObservationConfig::default();
    config.model.feature_model = mode();
    config.model.min_samples = NonZeroUsize::MIN;
    let runtime = runtime_with_fixed_clock(&config, false);
    for malformed in 0..3 {
        let mut row = stages(100 + malformed * 100, false);
        let evidence = Arc::make_mut(&mut row);
        let shape = evidence.actual_shape.as_mut().unwrap();
        match malformed {
            0 => shape.numeric_features = None,
            1 => {
                shape.row_multiset_features.as_mut().unwrap().rows[0].role = HostRowRoleV2::Prefill
            }
            _ => shape
                .row_multiset_features
                .as_mut()
                .unwrap()
                .rows
                .pop()
                .map(|_| ())
                .unwrap(),
        }
        runtime.sink.offer_evidence_numbered(only(row)).unwrap();
    }
    runtime.consume_samples();
    assert_eq!(runtime.trained_samples(), 0);
    assert_eq!(
        runtime
            .audit_snapshot()
            .training
            .host_content
            .offered_entries,
        3
    );
    runtime.shutdown().await.unwrap();
}

#[test]
fn old_host_profile_and_raw_wire_ignore_new_extension_without_relabeling() {
    let mut f = Fixture::new();
    f.settings.feature_model = model::CostFeatureModel::EmpiricalHostContentV1 {
        host_history_bucket_tokens: NonZeroU32::new(64).unwrap(),
    };
    let mut stages = stages(100, false);
    Arc::make_mut(&mut stages)
        .actual_shape
        .as_mut()
        .unwrap()
        .host_content_features = Some(HostContentCostFeaturesV1 {
        schema_version: 1,
        output_policy_signature: [40; 32],
    });
    let sample = actual(&stages);
    let mut export = f.exporter().unwrap();
    export
        .record_entry_with_host(
            1,
            None,
            Some(stages.clone()),
            Some(CostCallRejection::Composite),
            Some((evaluation(), Some(&sample))),
        )
        .unwrap();
    f.finish(export).unwrap();
    let bytes = fs::read(&f.options.path).unwrap();
    let old: profile_v3::CostProfileFileV3 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(old.schema_version, 3);
    assert_eq!(old.samples.len(), 1);
    assert!(serde_json::from_slice::<profile_v4::CostProfileFileV4>(&bytes).is_err());
    let raw = f.records();
    assert_eq!(raw[0]["schema_version"], 5);
    assert!(raw[1]["evidence"]["actual_shape"]
        .get("row_multiset_features")
        .is_none());
    assert!(raw[2]["sample"]["shape"]
        .get("row_multiset_features")
        .is_none());
    // Live/calibration evidence still contains the actual extension; the
    // versioned exporter alone omits it from old wire contracts.
    let live = serde_json::to_value(stages.as_ref()).unwrap();
    assert_eq!(
        live["actual_shape"]["row_multiset_features"]["schema_version"],
        2
    );
}

#[test]
fn row_multiset_queue_capacity_counts_actual_new_rows_before_accepting_ordinal() {
    let evidence = stages(100, false);
    let rows = evidence.retained_rows().unwrap();
    let sink = BoundedCostSampleSink::new(CostSampleSinkLimits {
        max_samples: 2,
        max_shape_rows: rows - 1,
    })
    .unwrap();
    assert_eq!(
        sink.offer_evidence_numbered(only(evidence.clone())),
        Err(CostSampleDrop::Capacity)
    );
    assert_eq!(sink.stats().entries_published, 0);
    let sink = BoundedCostSampleSink::new(CostSampleSinkLimits {
        max_samples: 2,
        max_shape_rows: rows,
    })
    .unwrap();
    assert_eq!(sink.offer_evidence_numbered(only(evidence)), Ok(1));
}
