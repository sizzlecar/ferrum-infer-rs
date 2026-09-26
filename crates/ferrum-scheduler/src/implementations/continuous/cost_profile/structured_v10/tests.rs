use super::*;
mod fit_floor;
mod fixture;
use fixture::*;

#[test]
fn configured_eager_profile10_preserves_distinct_graph_identity() {
    use ferrum_interfaces::execution_cost::ActualWaveGraphState;
    for generated in 0..3 {
        let (p, rows, native) = prepared_route(
            "eager",
            generated + 1,
            generated,
            None,
            ActualWaveGraphState::ConfiguredEager,
        );
        assert_eq!(super::prepared::project(&p, &rows).unwrap(), native);
        let (_, _, disabled) = prepared("eager", generated + 1, generated);
        assert_ne!(native.owner(), disabled.owner());
        for graph in [
            ProfileGraphState::Disabled,
            ProfileGraphState::Cold,
            ProfileGraphState::Warm,
        ] {
            let mut mislabeled = p.clone();
            mislabeled.exact.exact.graph_state = graph;
            assert!(super::prepared::project(&mislabeled, &rows).is_err());
        }
    }
}

struct Files {
    dir: PathBuf,
    source: PathBuf,
    profile: PathBuf,
}
impl Files {
    fn new(bytes: &[u8]) -> Self {
        let dir =
            std::env::temp_dir().join(format!("ferrum-structured-v10-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let source = dir.join("source.jsonl");
        let profile = dir.join("profile.json");
        std::fs::write(&source, bytes).unwrap();
        Self {
            dir,
            source,
            profile,
        }
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
fn limits() -> CostProfileLoadLimits {
    CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
        ..Default::default()
    }
}
fn mutate(bytes: &[u8], mut f: impl FnMut(&mut serde_json::Value)) -> Vec<u8> {
    let mut out = Vec::new();
    for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
        f(&mut value["record"]);
        serde_json::to_writer(&mut out, &value).unwrap();
        out.push(b'\n');
    }
    out
}
fn rejected(bytes: &[u8]) {
    let f = Files::new(bytes);
    assert!(export_structured_profile_v10(
        &f.source,
        Sha256::digest(bytes).into(),
        &f.profile,
        0,
        &limits()
    )
    .is_err());
    assert!(!f.profile.exists());
}
#[test]
fn structured_v10_original_producer_and_private_replay_projection_agree() {
    for generated in 0..3 {
        let (p, rows, native) = prepared("request", generated + 1, generated);
        let replay = super::prepared::project(&p, &rows).unwrap();
        assert_eq!(replay, native);
        let mut exchanged = p.clone();
        exchanged.owner_facts["algorithms"][0]["signature"][0] = 99.into();
        assert!(super::prepared::project(&exchanged, &rows).is_err());
        let mut missing = p;
        missing.exact.row_multiset_features = None;
        assert!(super::prepared::project(&missing, &rows).is_err());
    }
}
#[test]
fn structured_v10_replays_full_cohorts_outside_fifo_and_three_phases() {
    let (bytes, input) = source();
    assert_full_source(&bytes, input);
}

fn assert_full_source(bytes: &[u8], input: StructuredInputV2) {
    let f = Files::new(&bytes);
    let receipt = export_structured_profile_v10(
        &f.source,
        Sha256::digest(&bytes).into(),
        &f.profile,
        0,
        &limits(),
    )
    .unwrap();
    let clock = ProfileLoadClock {
        wall_unix_ns: Some(1_000_000 + 72 * 2000 + 1299 + 100),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 7,
    };
    let model = load_structured_profile_v10(&f.profile, &fingerprint(), &limits(), clock).unwrap();
    assert_eq!(model.provenance().schema_version, 10);
    assert_eq!(model.provenance().offered_attempts, 72);
    assert_eq!(model.provenance().reserved_members, 24);
    assert_eq!(model.provenance().total_shape_rows, 72);
    let (same, epoch) = model
        .predict_query_local_with_clock(
            &fingerprint(),
            &StructuredQueryV2::exact(input.clone()),
            17,
        )
        .unwrap();
    assert_eq!(epoch, model.model_now_ns(17).unwrap());
    assert!(same.valid_until_ns >= epoch);
    assert_eq!(
        model
            .provenance()
            .phases
            .iter()
            .map(|p| p.members)
            .collect::<Vec<_>>(),
        [8, 8, 8]
    );
    assert_eq!(model.provenance().phases[2].accepted_fifo_cutoff, 72);
    assert_eq!(model.parameters_signature(), receipt.parameters_sha256);
    assert_eq!(model.owner(), input.owner());
    assert_eq!(model.model_now_ns(7).unwrap(), 72 * 2000 + 1400);
    assert_eq!(model.model_now_ns(17).unwrap(), 72 * 2000 + 1410);
    let prediction = model
        .predict_query_local(&fingerprint(), &StructuredQueryV2::exact(input.clone()), 17)
        .unwrap();
    assert!((1000..=1001).contains(&prediction.fitted_upper_ns));
    assert!(prediction.valid_until_ns > model.model_now_ns(17).unwrap());
    assert!(matches!(
        model.model_now_ns(6),
        Err(StructuredUnknownV2::Clock)
    ));
    assert!(matches!(
        model.predict_query_local(
            &fingerprint(),
            &StructuredQueryV2::exact(input),
            2_000_000_000
        ),
        Err(StructuredUnknownV2::Stale)
    ));
    assert!(export_structured_profile_v10(
        &f.source,
        Sha256::digest(&bytes).into(),
        &f.profile,
        0,
        &limits()
    )
    .is_err()); // no overwrite
}

#[test]
fn structured_v10_warm_source_replay_preserves_shared_domain_not_resident_binding() {
    let (a, rows, input_a) = prepared_graph("request", 2, 1, Some("resident-A"));
    let (b, _, input_b) = prepared_graph("request", 2, 1, Some("resident-B"));
    assert_ne!(a.recipe.exact_binding, b.recipe.exact_binding);
    assert_ne!(
        a.recipe
            .device
            .replay_work
            .as_ref()
            .unwrap()
            .resident_binding,
        b.recipe
            .device
            .replay_work
            .as_ref()
            .unwrap()
            .resident_binding
    );
    assert_eq!(input_a, input_b);
    assert_eq!(super::prepared::project(&a, &rows).unwrap(), input_a);
    assert_eq!(super::prepared::project(&b, &rows).unwrap(), input_b);
    let (_, _, eager) = prepared("request", 2, 1);
    assert_ne!(input_a.domain_signature(), eager.domain_signature());
    assert_eq!(
        input_a.regression_axes().len(),
        eager.regression_axes().len() + 3
    );
    assert_eq!(
        input_a.joint_support_coordinates().len(),
        eager.joint_support_coordinates().len() + 3
    );

    // Fit/residual/qualification replay consumes the complete original FIFO,
    // including outside terminal waves. A distinct legitimate resident instance
    // may query the same numeric model without obtaining A's live authority.
    let (bytes, _) = source_graph(Some("resident-A"));
    assert_full_source(&bytes, input_b);
}

#[test]
fn structured_v10_warm_projection_rejects_missing_exchanged_or_cold_replay_facts() {
    let (original, rows, _) = prepared_graph("request", 2, 1, Some("resident-A"));
    for failure in 0..8 {
        let mut changed = original.clone();
        match failure {
            0 => changed.recipe.device.replay_work = None,
            1 => {
                changed
                    .recipe
                    .device
                    .replay_work
                    .as_mut()
                    .unwrap()
                    .exact_binding = [9; 32]
            }
            2 => {
                changed
                    .recipe
                    .device
                    .replay_work
                    .as_mut()
                    .unwrap()
                    .resident_binding = [0; 32]
            }
            3 => {
                changed
                    .recipe
                    .device
                    .replay_work
                    .as_mut()
                    .unwrap()
                    .replayed_segments = 0
            }
            4 => {
                changed
                    .recipe
                    .device
                    .replay_work
                    .as_mut()
                    .unwrap()
                    .replayed_segments = 2
            }
            5 => {
                changed
                    .recipe
                    .device
                    .replay_work
                    .as_mut()
                    .unwrap()
                    .logical_commands = 2
            }
            6 => {
                changed
                    .recipe
                    .device
                    .replay_work
                    .as_mut()
                    .unwrap()
                    .native_graph_nodes = 0
            }
            _ => changed.exact.exact.graph_state = ProfileGraphState::Cold,
        }
        assert!(
            super::prepared::project(&changed, &rows).is_err(),
            "case {failure}"
        );
    }
    let (mut eager, eager_rows, _) = prepared("request", 2, 1);
    eager.recipe.device.replay_work = original.recipe.device.replay_work;
    assert!(super::prepared::project(&eager, &eager_rows).is_err());
}

#[test]
fn structured_v10_warm_source_cannot_drop_prepared_or_settled_replay_bindings() {
    let (bytes, _) = source_graph(Some("resident-A"));
    for field in ["prepared", "settled"] {
        let changed = mutate(&bytes, |record| {
            if field == "prepared" && record["kind"] == "reserved" && record["offered"] == 2 {
                record["prepared"]["recipe"]["device"]
                    .as_object_mut()
                    .unwrap()
                    .remove("replay_work");
            }
            if field == "settled" && record["kind"] == "completed" && record["offered"] == 2 {
                record["host_stages"]["structured_evidence"]["Ok"]["recipe"]["device"]
                    .as_object_mut()
                    .unwrap()
                    .remove("replay_work");
            }
        });
        rejected(&changed);
    }
}
#[test]
fn structured_v10_footer_cannot_hide_failed_or_filtered_population() {
    let (bytes, _) = source();
    for changed in [
        "member", "fifo", "failure", "budget", "coverage", "freeze", "cohorts",
    ] {
        let edited = mutate(&bytes, |r| {
            if r["kind"] == "reserved" && r["offered"] == 2 && changed == "member" {
                r["member"] = serde_json::Value::Null;
            }
            if r["kind"] == "completed" && r["offered"] == 2 {
                if changed == "fifo" {
                    r["queue"]["accepted_ordinal"] = 99.into();
                }
                if changed == "failure" {
                    r["conversion_error"] = "unmeasured".into();
                }
            }
            if r["kind"] == "request_admitted" && changed == "budget" {
                r["maximum_output"] = 2.into();
            }
            if r["kind"] == "coverage" && changed == "coverage" {
                r["report"]["missing"]["pending_counts"] = serde_json::json!([0]);
            }
            if r["kind"] == "phase_freeze" && changed == "freeze" {
                r["receipt"]["parameters_sha256"][0] = 77.into();
            }
            if r["kind"] == "cohort_end" && changed == "cohorts" {
                r["completed_count"] = 0.into();
            }
        });
        rejected(&edited);
    }
}
#[test]
fn structured_v10_outside_terminal_is_bound_to_original_prepared_and_full_lifecycle() {
    let (bytes, _) = source();
    for changed in [
        "work",
        "owner",
        "terminal",
        "hash",
        "missing_terminal_event",
    ] {
        let edited = mutate(&bytes, |r| {
            if r["kind"] == "completed" && r["offered"] == 3 {
                let s = &mut r["outside_settlement"];
                if changed == "work" {
                    s["rows"][0]["actual_work"]["kv_tokens"] = 63.into();
                }
                if changed == "owner" {
                    s["rows"][0]["owner_incarnation"] = 2.into();
                }
                if changed == "terminal" {
                    s["rows"][0]["terminal"]["finish_reason"] = "EOS".into();
                }
                if changed == "hash" {
                    s["stage_binding"][0] = 55.into();
                }
            }
            if r["kind"] == "request_completed" && changed == "missing_terminal_event" {
                r["request"]["generated_tokens"] = 2.into();
            }
        });
        rejected(&edited);
    }
}
#[test]
fn structured_v10_immutable_manifest_and_independent_sidecar_are_required() {
    let (bytes, _) = source();
    for changed in ["manifest", "window", "independent"] {
        let edited = mutate(&bytes, |r| {
            if r["schema_version"] == 3 && changed == "manifest" {
                r["cohort_manifest_payload"]["output"] = 2.into();
            }
            if r["schema_version"] == 3 && changed == "window" {
                r["membership_rule"]["windows"][0]["rows"][0]["generated_before"]["minimum"] =
                    0.into();
            }
            if r["kind"] == "reserved" && changed == "independent" {
                r["prepared"]
                    .as_object_mut()
                    .unwrap()
                    .remove("selected_independent_attention_v2");
            }
        });
        rejected(&edited);
    }
}
#[test]
fn structured_v10_source_record_bound_applies_before_header_json_allocation() {
    let bytes = vec![b' '; 8 * 1024 * 1024 + 1];
    assert!(matches!(
        replay::replay_source(&bytes, &limits()),
        Err(CostProfileError::Metadata(_))
    ));
    let mut line = bytes;
    line.push(b'\n');
    assert!(matches!(
        replay::replay_source(&line, &limits()),
        Err(CostProfileError::Limit(_))
    ));
}
