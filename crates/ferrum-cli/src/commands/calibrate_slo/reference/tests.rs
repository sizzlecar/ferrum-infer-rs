use super::*;

#[test]
fn piecewise_partition_only_changes_reference_trials_and_preserves_service_output() {
    use super::super::report::Phase;
    let mut value = config();
    value.piecewise = Some(PiecewiseReferenceSpec {
        minimum_prompt_tokens: NonZeroU32::MIN,
        maximum_prompt_tokens: NonZeroU32::new(10).unwrap(),
        body_endpoints: vec![NonZeroU32::new(4).unwrap(), NonZeroU32::new(9).unwrap()],
    });
    let manifest = manifest();
    value
        .validate(&manifest, &inputs::PreparedInputs::Rendered)
        .unwrap();
    let fallback = NonZeroU32::new(10).unwrap();
    for phase in [Phase::Discovery, Phase::Reference] {
        assert_eq!(
            prefill_count(Some(&value), phase, 10, 0, fallback)
                .unwrap()
                .get(),
            4
        );
        assert_eq!(
            prefill_count(Some(&value), phase, 10, 4, fallback)
                .unwrap()
                .get(),
            5
        );
        assert_eq!(
            prefill_count(Some(&value), phase, 10, 9, fallback)
                .unwrap()
                .get(),
            1
        );
        assert_eq!(
            prefill_count(Some(&value), phase, 7, 4, fallback)
                .unwrap()
                .get(),
            2
        );
        assert!(prefill_count(Some(&value), phase, 11, 0, fallback).is_err());
    }
    for phase in [Phase::Training, Phase::Heldout, Phase::Warmup] {
        assert_eq!(
            prefill_count(Some(&value), phase, 10, 0, fallback)
                .unwrap()
                .get(),
            10
        );
    }
    assert_eq!(manifest.prompts[0].sampling.max_tokens, 73);
    let bytes = serde_json::to_vec(&value).unwrap();
    let parsed: ReferenceConfig = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed.piecewise, value.piecewise);
    let old = serde_json::to_value(config()).unwrap();
    assert!(old.get("piecewise").is_none());
    assert!(serde_json::from_value::<ReferenceConfig>(old)
        .unwrap()
        .piecewise
        .is_none());
}

pub(super) fn config() -> ReferenceConfig {
    ReferenceConfig {
        piecewise: None,
        request_policy: ReferenceRequestPolicy::OriginalInput {},
        revision: NonZeroU64::MIN,
        artifact_path: "reference.json".into(),
        frozen_plan_path: "frozen.json".into(),
        warmup: Vec::new(),
        curve_prompt_indices: vec![0],
        granule_tokens: NonZeroU32::new(4).unwrap(),
        repetitions: NonZeroUsize::new(2).unwrap(),
        decode_unit: DecodeUnit {
            prompt_index: 0,
            generated_before: NonZeroU32::new(2).unwrap(),
        },
        limits: Default::default(),
    }
}
pub(super) fn manifest() -> manifest::Manifest {
    let mut manifest = super::super::tests::manifest();
    manifest.validation_model = manifest::ValidationSource::ExportedProfile {
        profile: "training.json".into(),
        source: "training.jsonl".into(),
    };
    manifest
}

#[test]
fn reference_requires_cut_explicit_warmup_and_exact_unmodified_output_policy() {
    let mut manifest = manifest();
    let config = config();
    assert!(config
        .validate(&manifest, &inputs::PreparedInputs::Rendered)
        .is_ok());
    assert_eq!(manifest.prompts[0].sampling.max_tokens, 73);
    manifest.validation_model = manifest::ValidationSource::LiveFrozen;
    assert!(config
        .validate(&manifest, &inputs::PreparedInputs::Rendered)
        .is_err());
    let mut wire = serde_json::to_value(config).unwrap();
    wire.as_object_mut().unwrap().remove("warmup");
    assert!(serde_json::from_value::<ReferenceConfig>(wire).is_err());
}

#[test]
fn missing_inputs_duplicate_curves_unreachable_decode_and_incompatible_budgets_fail() {
    let mut manifest = manifest();
    let inputs = inputs::PreparedInputs::Rendered;
    let mut value = config();
    value.curve_prompt_indices.push(0);
    assert!(value.validate(&manifest, &inputs).is_err());
    value = config();
    value.decode_unit.prompt_index = 1;
    assert!(value.validate(&manifest, &inputs).is_err());
    value = config();
    value.decode_unit.generated_before = NonZeroU32::new(73).unwrap();
    assert!(value.validate(&manifest, &inputs).is_err());
    value = config();
    manifest.prompts.push(manifest.prompts[0].clone());
    manifest.prompts[1].sampling.max_tokens += 1;
    value.curve_prompt_indices.push(1);
    assert!(value.validate(&manifest, &inputs).is_err());
    assert_eq!(manifest.prompts[0].sampling.max_tokens, 73);
    assert_eq!(manifest.prompts[1].sampling.max_tokens, 74);
}

#[test]
fn reference_limits_apply_to_all_phases_and_strict_schema() {
    let mut value = config();
    let manifest = manifest();
    value.warmup.push(manifest::Cohort {
        prompts: vec![0, 0, 0],
        repetitions: NonZeroUsize::MIN,
        prefill_chunk_tokens: NonZeroU32::MIN,
        execution: manifest::Execution::Mixed,
        decode_route: ferrum_engine::continuous_engine::CalibrationDecodeRoute::Actual,
        token_policy_residency: manifest::TokenPolicyResidencyPolicy::Preserve,
    });
    assert!(value
        .validate(&manifest, &inputs::PreparedInputs::Rendered)
        .is_err());
    value = config();
    value.limits.max_samples = NonZeroUsize::new(usize::MAX).unwrap();
    assert!(value
        .validate(&manifest, &inputs::PreparedInputs::Rendered)
        .is_err());
    let mut wire = serde_json::to_value(config()).unwrap();
    wire["decode_unit"]["pick_fastest"] = true.into();
    assert!(serde_json::from_value::<ReferenceConfig>(wire).is_err());
}

#[test]
fn singleton_recipe_only_changes_cohort_work_and_preserves_requested_maximum() {
    let value = config();
    let manifest = manifest();
    for target in [
        DiscoveryTarget::Prefill { curve: 0 },
        DiscoveryTarget::Decode,
    ] {
        let case = value.singleton_case(target).unwrap();
        assert_eq!(case.prompts, vec![0]);
        assert_eq!(case.prefill_chunk_tokens, value.granule_tokens);
        assert!(matches!(case.execution, manifest::Execution::Split));
    }
    assert!(value
        .singleton_case(DiscoveryTarget::Prefill { curve: 1 })
        .is_err());
    assert_eq!(manifest.prompts[0].sampling.max_tokens, 73);
}

#[test]
fn no_submit_and_after_target_are_distinct_and_neither_hides_execution_failure() {
    use CalibrationSubmissionState as State;
    assert_eq!(
        observer::skip_submission(false, State::NotSubmitted, false).unwrap(),
        Some(ObservationProgress::NotSubmitted)
    );
    assert_eq!(
        observer::skip_submission(false, State::HostReconciled, false).unwrap(),
        None
    );
    assert_eq!(
        observer::skip_submission(true, State::HostReconciled, false).unwrap(),
        Some(ObservationProgress::AfterTarget)
    );
    assert_eq!(
        observer::skip_submission(true, State::NotSubmitted, false).unwrap(),
        Some(ObservationProgress::NotSubmitted)
    );
    for completed in [false, true] {
        assert!(observer::skip_submission(completed, State::Submitted, false).is_err());
        assert!(observer::skip_submission(completed, State::InFlightUnknown, false).is_err());
        assert!(observer::skip_submission(completed, State::HostReconciled, true).is_err());
    }
    assert!(DiscoveryObserver::new(&config(), DiscoveryTarget::Decode)
        .unwrap()
        .finish()
        .is_err());
}

#[test]
fn retention_includes_unselected_decode_preparation_not_just_targets() {
    let mut value = config();
    // S=3 prefill segments; decode preparation ceil(9/4)=3 plus two
    // generated decode steps. Four discoveries + 2*(3+3+2)=20 retained.
    value.limits.max_samples = NonZeroUsize::new(20).unwrap();
    assert!(discovery::retention_bound(&value, 3, 1, 9).is_ok());
    value.limits.max_samples = NonZeroUsize::new(19).unwrap();
    assert!(discovery::retention_bound(&value, 3, 1, 9).is_err());
    assert!(discovery::retention_bound(&value, usize::MAX, 1, 9).is_err());
}

#[test]
fn frozen_plan_is_bounded_atomic_no_replace_and_content_verified() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plan.json");
    let value = serde_json::json!({"protocol": "declared", "accepted_cut": 7, "ordinals": [1, 3]});
    let length = serde_json::to_vec(&value).unwrap().len();
    assert!(files::publish(&path, &value, length - 1).is_err());
    assert!(!path.exists());
    let receipt = files::publish(&path, &value, length).unwrap();
    files::verify(&receipt).unwrap();
    assert_eq!(receipt.bytes, length as u64);
    assert!(files::publish(&path, &serde_json::json!({}), length).is_err());
    assert_eq!(
        std::fs::read(&path).unwrap(),
        serde_json::to_vec(&value).unwrap()
    );
    let mut tampered = std::fs::read(&path).unwrap();
    tampered[0] = b'['; // same length: a size check alone is insufficient.
    std::fs::write(&path, tampered).unwrap();
    assert!(files::verify(&receipt).is_err());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[cfg(unix)]
#[test]
fn frozen_plan_publication_does_not_follow_or_remove_destination_alias() {
    let dir = tempfile::tempdir().unwrap();
    let original = dir.path().join("original");
    let alias = dir.path().join("alias");
    std::fs::write(&original, b"preserved").unwrap();
    std::os::unix::fs::symlink(&original, &alias).unwrap();
    assert!(files::publish(&alias, &serde_json::json!({}), 64).is_err());
    assert_eq!(std::fs::read(&original).unwrap(), b"preserved");
    assert!(std::fs::symlink_metadata(&alias)
        .unwrap()
        .file_type()
        .is_symlink());
}

#[test]
fn reference_paths_resolve_against_manifest_parent_without_rewriting_absolute_paths() {
    let dir = tempfile::tempdir().unwrap();
    let mut value = config();
    value.artifact_path = dir.path().join("already-absolute.json");
    let original = value.artifact_path.clone();
    value.resolve_paths(dir.path());
    assert_eq!(value.artifact_path, original);
    assert_eq!(value.frozen_plan_path, dir.path().join("frozen.json"));
}

#[test]
fn optional_reference_manifest_resolves_outputs_and_validates_full_declaration() {
    let dir = tempfile::tempdir().unwrap();
    let manifest_path = dir.path().join("manifest.json");
    let mut value = manifest();
    value.reference = Some(config());
    std::fs::write(&manifest_path, serde_json::to_vec(&value).unwrap()).unwrap();
    let loaded = manifest::load(&manifest_path).unwrap();
    let configured = loaded.reference.as_ref().unwrap();
    let base = dir.path().canonicalize().unwrap();
    assert_eq!(configured.artifact_path, base.join("reference.json"));
    assert_eq!(configured.frozen_plan_path, base.join("frozen.json"));
    configured
        .validate(&loaded, &inputs::PreparedInputs::Rendered)
        .unwrap();
    let mut invalid = loaded;
    invalid
        .reference
        .as_mut()
        .unwrap()
        .warmup
        .push(manifest::Cohort {
            prompts: vec![1],
            repetitions: NonZeroUsize::MIN,
            prefill_chunk_tokens: NonZeroU32::MIN,
            execution: manifest::Execution::Split,
            decode_route: ferrum_engine::continuous_engine::CalibrationDecodeRoute::Actual,
            token_policy_residency: manifest::TokenPolicyResidencyPolicy::Preserve,
        });
    assert!(invalid.validate().is_err());
}

#[test]
fn documented_reference_fragment_uses_the_real_schema() {
    let docs = include_str!("../../../../../../docs/slo-configuration.md");
    let start = docs
        .split("<!-- calibrate-slo-reference -->")
        .nth(1)
        .unwrap();
    let json = start
        .split("```json\n")
        .nth(1)
        .unwrap()
        .split("```")
        .next()
        .unwrap();
    let value: ReferenceConfig = serde_json::from_str(json).unwrap();
    value
        .validate(&manifest(), &inputs::PreparedInputs::Rendered)
        .unwrap();
}

#[test]
fn phase_denominators_and_target_roles_do_not_relabel_full_request_work() {
    use super::super::report::{Phase, Summary};
    let mut summary = Summary::default();
    for phase in [
        Phase::Warmup,
        Phase::Discovery,
        Phase::Reference,
        Phase::Training,
        Phase::Heldout,
    ] {
        summary.request(phase);
    }
    summary
        .reference_progress(Phase::Discovery, ObservationProgress::Recorded)
        .unwrap();
    for role in [
        ObservationProgress::PreparingTarget,
        ObservationProgress::TargetComplete,
        ObservationProgress::AfterTarget,
        ObservationProgress::NotSubmitted,
    ] {
        summary.reference_progress(Phase::Reference, role).unwrap();
    }
    assert!(summary
        .reference_progress(Phase::Heldout, ObservationProgress::TargetComplete)
        .is_err());
    assert_eq!(summary.training_requests, 1);
    assert_eq!(summary.validation_requests, 1);
    assert_eq!(summary.phases.warmup.requests, 1);
    assert_eq!(summary.phases.discovery.requests, 1);
    assert_eq!(summary.phases.reference.requests, 1);
    assert_eq!(summary.phases.discovery.target_waves, 1);
    assert_eq!(summary.phases.reference.target_waves, 1);
    assert_eq!(summary.phases.reference.preparation_waves, 1);
    assert_eq!(summary.phases.reference.after_target_waves, 1);
    assert_eq!(summary.phases.heldout.target_waves, 0);
}

#[test]
fn reference_destinations_participate_in_real_output_alias_validation() {
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        command: super::super::CalibrateSloCommand,
    }
    let dir = tempfile::tempdir().unwrap();
    let mut command = Args::parse_from([
        "ferrum",
        "model",
        "--manifest",
        "manifest.json",
        "--slo-config",
        "policy.toml",
        "--out",
        "report.json",
        "--observations",
        "raw.jsonl",
        "--startup-usage",
        "serve",
    ])
    .command;
    command.out = dir.path().join("report.json");
    command.observations = dir.path().join("raw.jsonl");
    let mut value = manifest();
    value.validation_model = manifest::ValidationSource::ExportedProfile {
        profile: dir.path().join("cut.json"),
        source: dir.path().join("cut.jsonl"),
    };
    let mut reference = config();
    reference.resolve_paths(dir.path());
    value.reference = Some(reference);
    super::super::paths::validate(super::super::paths::outputs(&command, &value)).unwrap();
    value.reference.as_mut().unwrap().frozen_plan_path = command.observations.clone();
    assert!(super::super::paths::validate(super::super::paths::outputs(&command, &value)).is_err());
    assert!(!command.observations.exists());
}
