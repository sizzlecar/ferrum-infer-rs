use super::*;

#[test]
fn noise_policies_preserve_declared_profiles_and_legacy_absence() {
    let original = events();
    let legacy = analyze(&artifact(&original, |_| {})).unwrap();
    let identity = legacy.identity.as_ref().unwrap();
    assert_eq!(identity.initial_noise_policy, None);
    assert_eq!(identity.edit_noise_policy, None);
    let serialized = serde_json::to_value(identity).unwrap();
    assert_eq!(serialized.get("initial_noise_policy"), Some(&Value::Null));
    assert_eq!(serialized.get("edit_noise_policy"), Some(&Value::Null));

    let edit = "p_h3_image.edit_noise.native_cuda_philox.posterior42.patch_reference_seed.v1";
    for initial in [
        "p_h3_image.noise.native_cuda_philox.joint_image_audio.v1",
        "p_h3_image.noise.cpu_mt19937.image_audio.scalar_f32_platform_libm.v1",
    ] {
        let mut current = original.clone();
        current[0]["initial_noise_policy"] = json!(initial);
        current[0]["edit_noise_policy"] = json!(edit);
        let report = analyze(&artifact(&current, |_| {})).unwrap();
        assert_eq!(report.status, ImageStepEvidenceStatus::Observed);
        assert_eq!(report.facts.committed_waves, legacy.facts.committed_waves);
        let identity = report.identity.unwrap();
        assert_eq!(identity.initial_noise_policy.as_deref(), Some(initial));
        assert_eq!(identity.edit_noise_policy.as_deref(), Some(edit));
        assert_eq!(identity.noise_profile, original[0]["noise_profile"]);
        let serialized = serde_json::to_value(identity).unwrap();
        assert_eq!(serialized["initial_noise_policy"], initial);
        assert_eq!(serialized["edit_noise_policy"], edit);

        for absent in ["initial_noise_policy", "edit_noise_policy"] {
            let mut partial = current.clone();
            partial[0].as_object_mut().unwrap().remove(absent);
            let identity = analyze(&artifact(&partial, |_| {}))
                .unwrap()
                .identity
                .unwrap();
            assert_eq!(
                identity.initial_noise_policy.as_deref(),
                (absent != "initial_noise_policy").then_some(initial)
            );
            assert_eq!(
                identity.edit_noise_policy.as_deref(),
                (absent != "edit_noise_policy").then_some(edit)
            );
        }
    }
}

#[test]
fn noise_policies_reject_empty_wrong_types_and_unknown_header_fields() {
    for field in ["initial_noise_policy", "edit_noise_policy"] {
        for value in [
            json!(""),
            json!(" \t\n"),
            Value::Null,
            json!(7),
            json!(1.5),
            json!(true),
            json!([]),
            json!({"profile":"noise.actual"}),
        ] {
            let mut bad = events();
            bad[0][field] = value;
            let error = analyze(&artifact(&bad, |_| {})).unwrap_err();
            assert_eq!(error.line, 1);
            assert_eq!(error.kind, ImageContinuousTraceErrorKind::Invalid);
        }
    }
    let mut bad = events();
    bad[0]["initial_noise_policy"] = json!("noise.initial.actual");
    bad[0]["edit_noise_policy"] = json!("noise.edit.actual");
    bad[0]["unknown_noise_policy"] = json!("noise.other");
    rejects(&bad, "declared trace schema");
}

const OWNER: &str = "cced078a-1145-4ad4-a471-f42b8d549aaa";
fn wave(job: u64, index: usize, ordinal: u64, before: u64) -> Value {
    json!({"kind":"prompt_expansion_token_wave","shared_owner":{"id":OWNER,"participants":2},
        "wave":ordinal,"participants":2,"input_index":index,"job":job,
        "request_id":format!("00000000-0000-0000-0000-{job:012}"),
        "before_consumed_tokens":before,"after_consumed_tokens":before+1,
        "outcome":"completed","native_outcome":"completed"})
}
#[test]
fn actual_token_wave_wire_is_typed_bounded_and_does_not_establish_step_timing() {
    let rows = [
        wave(10, 0, 0, 8),
        wave(11, 1, 0, 18),
        wave(11, 0, 1, 19),
        wave(10, 1, 1, 9),
    ];
    let report = analyze(&artifact(&rows, |_| {})).unwrap();
    assert_eq!(report.status, ImageStepEvidenceStatus::NotEvaluated);
    assert_eq!(report.timing.status, ImageStepEvidenceStatus::NotEvaluated);
    assert_eq!(report.facts.committed_waves, 0);
    let summary = report.prompt_expansion_token_waves.unwrap();
    assert_eq!(
        (
            summary.observed_records,
            summary.observed_wave_groups,
            summary.completed_rows,
            summary.failed_rows
        ),
        (4, 2, 4, 0)
    );
    assert_eq!(summary.maximum_observed_participants, 2);
    let mut original = events();
    original.extend(rows);
    let report = analyze(&artifact(&original, |_| {})).unwrap();
    assert_eq!(report.facts.committed_waves, 3);
    let mut limits = ImageContinuousTraceReadLimits::default();
    limits.maximum_requests = 1;
    let error = analyze_image_continuous_trace_bytes(
        &artifact(&[wave(10, 0, 0, 8), wave(11, 1, 0, 18)], |_| {}),
        &limits,
    )
    .unwrap_err();
    assert_eq!(error.kind, ImageContinuousTraceErrorKind::Capacity);
}

#[test]
fn token_wave_rejects_missing_unknown_duplicate_regressing_and_foreign_fields() {
    for field in [
        "shared_owner",
        "wave",
        "participants",
        "input_index",
        "job",
        "request_id",
        "before_consumed_tokens",
        "after_consumed_tokens",
        "outcome",
        "native_outcome",
    ] {
        let mut bad = wave(10, 0, 0, 8);
        bad.as_object_mut().unwrap().remove(field);
        assert!(
            analyze(&artifact(&[bad], |_| {})).is_err(),
            "missing {field}"
        );
    }
    for (field, value) in [
        ("participants", json!(0)),
        ("input_index", json!(2)),
        ("native_outcome", json!("guess")),
        ("native_outcome", json!("not_submitted")),
        ("after_consumed_tokens", json!(7)),
        ("after_consumed_tokens", Value::Null),
        ("outcome", json!("cancelled")),
        ("extra", json!(true)),
    ] {
        let mut bad = wave(10, 0, 0, 8);
        bad[field] = value;
        assert!(
            analyze(&artifact(&[bad], |_| {})).is_err(),
            "invalid {field}"
        );
    }
    assert!(analyze(&artifact(&[wave(10, 0, 0, 8), wave(11, 0, 0, 18)], |_| {})).is_err());
    assert!(analyze(&artifact(&[wave(10, 0, 0, 8), wave(10, 1, 0, 9)], |_| {})).is_err());
    assert!(analyze(&artifact(&[wave(10, 0, 1, 8), wave(10, 0, 0, 9)], |_| {})).is_err());
    assert!(analyze(&artifact(&[wave(10, 0, 0, 8), wave(10, 0, 1, 7)], |_| {})).is_err());
    for mutation in ["owner", "request", "native", "extent"] {
        let mut second = wave(11, 1, 0, 18);
        match mutation {
            "owner" => {
                second["job"] = json!(10);
                second["shared_owner"]["id"] = json!(SESSION);
            }
            "request" => {
                second["job"] = json!(10);
            }
            "native" => second["native_outcome"] = json!("interrupted"),
            "extent" => second["participants"] = json!(1),
            _ => unreachable!(),
        }
        assert!(
            analyze(&artifact(&[wave(10, 0, 0, 8), second], |_| {})).is_err(),
            "{mutation}"
        );
    }
    // A failed/unknown call may retain its unchanged frontier. It is not
    // promoted into a completed native wave or image timing evidence.
    let mut failed = wave(10, 0, 0, 8);
    failed["outcome"] = json!("failed");
    failed["native_outcome"] = json!("not_submitted");
    failed["after_consumed_tokens"] = json!(8);
    let report = analyze(&artifact(&[failed.clone()], |_| {})).unwrap();
    let summary = report.prompt_expansion_token_waves.unwrap();
    assert_eq!((summary.completed_rows, summary.failed_rows), (0, 1));
    assert!(analyze(&artifact(&[failed, wave(10, 0, 1, 8)], |_| {})).is_err());
}

#[test]
fn expansion_summary_keeps_shared_owner_and_must_match_prior_wave_identity() {
    let mut observation = super::expansion::observation(false);
    observation["job"] = json!(10);
    observation["request_id"] = wave(10, 0, 0, 8)["request_id"].clone();
    observation["shared_owner"] = json!({"id":OWNER,"participants":2});
    let report = analyze(&artifact(&[wave(10, 0, 0, 8), observation.clone()], |_| {})).unwrap();
    assert_eq!(
        report.prompt_expansion_observations[0]
            .shared_owner
            .unwrap()
            .participants,
        2
    );
    for field in ["id", "participants"] {
        let mut bad = observation.clone();
        bad["shared_owner"][field] = if field == "id" {
            json!(SESSION)
        } else {
            json!(3)
        };
        assert!(analyze(&artifact(&[wave(10, 0, 0, 8), bad], |_| {})).is_err());
    }
    assert!(analyze(&artifact(&[observation, wave(10, 0, 0, 8)], |_| {})).is_err());
}

#[test]
fn reference_latent_digest_is_checked_metadata_and_legacy_absence_stays_valid() {
    let original = events();
    let legacy = analyze(&artifact(&original, |_| {})).unwrap();
    assert!(legacy.reference_latents.is_empty());
    assert_eq!(legacy.identity.unwrap().vae_decoder_tile_batch, 1);
    let mut valid = original.clone();
    valid[0]["vae_decoder_tile_batch"] = json!(4);
    valid[1]["reference_latent"] = json!({"shape":[1,24,1,2,3],"sha256":"a".repeat(64)});
    let report = analyze(&artifact(&valid, |_| {})).unwrap();
    assert_eq!(report.reference_latents["a"].shape, [1, 24, 1, 2, 3]);
    assert_eq!(report.identity.unwrap().vae_decoder_tile_batch, 4);
    for shape in [
        json!([2, 24, 1, 2, 3]),
        json!([1, 24, 2, 2, 3]),
        json!([1, 24, 1, 0, 3]),
        json!([1, u64::MAX, 1, 2, 3]),
    ] {
        let mut bad = valid.clone();
        bad[1]["reference_latent"]["shape"] = shape;
        assert!(analyze(&artifact(&bad, |_| {})).is_err());
    }
    for mutation in ["hash", "unknown", "not_edit"] {
        let mut bad = valid.clone();
        match mutation {
            "hash" => bad[1]["reference_latent"]["sha256"] = json!("z".repeat(64)),
            "unknown" => bad[1]["reference_latent"]["other"] = json!(1),
            "not_edit" => bad[1]["editing"] = json!(false),
            _ => unreachable!(),
        }
        assert!(analyze(&artifact(&bad, |_| {})).is_err());
    }
    for value in [json!(0), json!(5), json!(-1), json!(1.5), json!("4")] {
        let mut bad = valid.clone();
        bad[0]["vae_decoder_tile_batch"] = value;
        assert!(analyze(&artifact(&bad, |_| {})).is_err());
    }
}
