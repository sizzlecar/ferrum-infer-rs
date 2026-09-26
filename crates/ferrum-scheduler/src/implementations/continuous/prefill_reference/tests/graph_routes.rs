use super::*;

fn graph_artifact(decode_graph: ProfileGraphState) -> ReferenceCalibrationV1 {
    let mut value = artifact();
    value.protocol.graph_routes = ReferenceGraphRoutes::ExactObserved;
    value.protocol.decode_shape.exact.graph_state = decode_graph;
    for sample in &mut value.decode_samples {
        sample.observation.shape = value.protocol.decode_shape.clone();
    }
    for curve in &mut value.curves {
        for (shape, graph) in curve.partition.iter_mut().zip([
            ProfileGraphState::Cold,
            ProfileGraphState::ConfiguredEager,
            ProfileGraphState::Warm,
        ]) {
            shape.exact.graph_state = graph;
        }
        for trial in &mut curve.trials {
            for (sample, shape) in trial.samples.iter_mut().zip(&curve.partition) {
                sample.observation.shape = shape.clone();
            }
        }
    }
    value
}

#[test]
fn exact_observed_routes_preserve_declared_shapes_and_measured_work() {
    for graph in [
        ProfileGraphState::Disabled,
        ProfileGraphState::Cold,
        ProfileGraphState::Warm,
        ProfileGraphState::ConfiguredEager,
    ] {
        let value = graph_artifact(graph);
        let loaded = load(&value).unwrap();
        assert_eq!(loaded.protocol(), &value.protocol);
        assert_eq!(loaded.tau_ref_ns().get(), 7);
        assert_eq!(loaded.curve(n32(10)).unwrap().work_at(10), Some(15));
        assert_eq!(loaded.sample_count(), 12);
        let mut disabled = value;
        disabled.protocol.graph_routes = ReferenceGraphRoutes::DisabledOnly;
        assert!(matches!(
            load(&disabled),
            Err(ReferenceError::Evidence("unsupported reference route"))
        ));
    }
}

#[test]
fn graph_policy_preserves_legacy_wire_and_binds_explicit_protocol_identity() {
    let old = artifact().protocol;
    let bytes = serde_json::to_vec(&old).unwrap();
    let wire = serde_json::to_value(&old).unwrap();
    assert!(wire.get("graph_routes").is_none());
    let parsed: ReferenceProtocolV1 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed.graph_routes, ReferenceGraphRoutes::DisabledOnly);
    assert_eq!(serde_json::to_vec(&parsed).unwrap(), bytes);
    assert_eq!(parsed.sha256().unwrap(), old.sha256().unwrap());
    let mut explicit = old.clone();
    explicit.graph_routes = ReferenceGraphRoutes::ExactObserved;
    assert_ne!(explicit.sha256().unwrap(), old.sha256().unwrap());
    let mut bad = wire;
    bad["graph_routes"] = serde_json::json!("assume_warm");
    assert!(serde_json::from_value::<ReferenceProtocolV1>(bad).is_err());
}

#[test]
fn exact_observed_routes_reject_each_trial_route_drift_and_wrong_fingerprint() {
    let value = graph_artifact(ProfileGraphState::Warm);
    for graph in [
        ProfileGraphState::Disabled,
        ProfileGraphState::Cold,
        ProfileGraphState::ConfiguredEager,
    ] {
        let mut changed = value.clone();
        changed.decode_samples[1]
            .observation
            .shape
            .exact
            .graph_state = graph;
        assert!(load(&changed).is_err());
    }
    let mut changed = value.clone();
    changed.curves[0].trials[1].samples[0]
        .observation
        .shape
        .exact
        .graph_state = ProfileGraphState::Warm;
    assert!(load(&changed).is_err());
    let bytes = serde_json::to_vec(&value).unwrap();
    assert!(matches!(
        load_prefill_reference_bytes(
            &bytes,
            &fingerprint(),
            artifact().protocol.sha256().unwrap(),
            &Default::default()
        ),
        Err(ReferenceError::Incompatible)
    ));
    let mut other_execution = fingerprint();
    other_execution.execution_config = [99; 32];
    assert!(matches!(
        load_prefill_reference_bytes(
            &bytes,
            &other_execution,
            value.protocol.sha256().unwrap(),
            &Default::default()
        ),
        Err(ReferenceError::Incompatible)
    ));
}

#[test]
fn exact_observed_routes_keep_host_output_source_and_commit_requirements() {
    for case in 0..9 {
        let mut value = graph_artifact(ProfileGraphState::Warm);
        match case {
            0 => {
                value.decode_samples[0]
                    .observation
                    .shape
                    .exact
                    .provider_signature = [42; 32]
            }
            1 => {
                value.decode_samples[0]
                    .observation
                    .shape
                    .exact
                    .output_policy_signature = [42; 32]
            }
            2 => value.decode_samples[0].observation.shape.numeric_features = None,
            3 => value.decode_samples[0].record.source_sha256 = [0; 32],
            4 => value.decode_samples[0].observation.boundary = ProfileCostBoundary::DeviceOnly,
            5 => {
                value.decode_samples[0].observation.outcome =
                    ProfileObservationOutcome::NotSubmitted {}
            }
            6 => value.decode_samples[0].commit.origin = ReferenceStateOrigin::Restored,
            7 => value.protocol.decode_host.state.pending_decoded_utf8 = true,
            8 => value.curves[0].trials[0].samples[1].commit.previous_record = None,
            _ => unreachable!(),
        }
        assert!(load(&value).is_err(), "reference qualification case {case}");
    }
    // Agreement between declared and observed shapes must not authorize a
    // fallback route or extra physical work outside this reference protocol.
    for case in 0..5 {
        let mut value = graph_artifact(ProfileGraphState::Warm);
        let shape = &mut value.protocol.decode_shape.exact;
        match case {
            0 => shape.path = ProfileExecutionPath::CapacityFallback,
            1 => shape.order = ProfileBatchOrder::IndependentRows,
            2 => shape.restore_bytes = 1,
            3 => shape.maintenance_bytes = 1,
            4 => shape.maintenance_units = 1,
            _ => unreachable!(),
        }
        for sample in &mut value.decode_samples {
            sample.observation.shape = value.protocol.decode_shape.clone();
        }
        assert!(load(&value).is_err(), "unsupported route case {case}");
    }
}
