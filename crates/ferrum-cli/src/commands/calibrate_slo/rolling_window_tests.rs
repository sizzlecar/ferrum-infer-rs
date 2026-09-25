//! Arrival policy tests. Model fitting and its worker barriers remain in the
//! existing selected-calibration session; these tests do not fabricate costs.
use super::*;
use std::collections::HashSet;

fn rolling() -> manifest::Manifest {
    let mut value = manifest();
    value.training[0].prompts = vec![0; 5];
    value.training[0].rolling_window = Some(manifest::RollingWindow {
        maximum_in_flight: NonZeroUsize::new(2).unwrap(),
    });
    value.validation = value.training.clone();
    value
}

#[test]
fn rolling_manifest_is_explicit_in_protocol_identity_and_preserves_legacy_barrier() {
    let legacy = manifest();
    let old = serde_json::to_value(&legacy).unwrap();
    assert!(old["training"][0].get("rolling_window").is_none());
    let decoded: manifest::Manifest = serde_json::from_value(old.clone()).unwrap();
    assert!(decoded.training[0].rolling_window.is_none());
    assert_eq!(decoded.training[0].maximum_in_flight(), 2);
    assert_eq!(serde_json::to_value(decoded).unwrap(), old);

    // Same inputs and output budgets, different explicit arrival protocol.
    let mut explicit = legacy.clone();
    explicit.training[0].rolling_window = Some(manifest::RollingWindow {
        maximum_in_flight: NonZeroUsize::new(2).unwrap(),
    });
    explicit.validate().unwrap();
    assert_ne!(
        Sha256::digest(serde_json::to_vec(&explicit).unwrap()),
        Sha256::digest(serde_json::to_vec(&legacy).unwrap())
    );
    assert_eq!(explicit.training[0].prompts, legacy.training[0].prompts);
    assert_eq!(
        serde_json::to_value(&explicit.prompts).unwrap(),
        serde_json::to_value(&legacy.prompts).unwrap()
    );
    let mut longer = rolling();
    longer.validate().unwrap();
    longer.training[0].rolling_window = None;
    assert!(
        longer.validate().is_err(),
        "legacy barrier cannot exceed its capacity"
    );
}

#[test]
fn rolling_manifest_keeps_capacity_source_and_total_owner_bounds() {
    let value = rolling();
    value.validate().unwrap();
    let mut wire = serde_json::to_value(&value).unwrap();
    wire["training"][0]["rolling_window"]["maximum_in_flight"] = 0.into();
    assert!(serde_json::from_value::<manifest::Manifest>(wire).is_err());
    let mut wire = serde_json::to_value(&value).unwrap();
    wire["training"][0]["rolling_window"]["unbounded_refill"] = true.into();
    assert!(serde_json::from_value::<manifest::Manifest>(wire).is_err());
    let mut invalid = value.clone();
    invalid.training[0]
        .rolling_window
        .as_mut()
        .unwrap()
        .maximum_in_flight = NonZeroUsize::new(3).unwrap();
    assert!(invalid.validate().is_err());
    let mut invalid = value.clone();
    invalid.training[0].prompts.push(value.prompts.len());
    assert!(invalid.validate().is_err());
    let mut invalid = value.clone();
    invalid.training[0].prompts.clear();
    assert!(invalid.validate().is_err());
    let mut invalid = value;
    invalid.training[0].prompts = vec![0; 65_536];
    assert!(
        invalid.validate().is_err(),
        "heldout owners still count toward the total"
    );
}

#[test]
fn rolling_three_phases_build_fresh_requests_with_fixed_full_budgets() {
    let mut value = rolling();
    let mut second = value.prompts[0].clone();
    second.source_id = "second-pinned-request".into();
    second.rendered_prompt.push_str(" Different actual input.");
    second.rendered_prompt_sha256 = Sha256::digest(second.rendered_prompt.as_bytes()).into();
    second.sampling.max_tokens = 101;
    value.prompts.push(second);
    value.training[0].prompts = vec![0, 1, 0, 1, 0];
    value.validation = value.training.clone();
    value.validation_model = manifest::ValidationSource::SelectedWorkSupportV1 {
        export: ferrum_types::SloCostProfileExportConfig {
            path: "profile8.json".into(),
            observations_path: "source3.jsonl".into(),
            declared_clock_max_error_ns: Some(0),
            ..Default::default()
        },
        residual: value.training.clone(),
    };
    value.validate().unwrap();
    let frozen_declaration = serde_json::to_value(&value).unwrap();
    let inputs = inputs::PreparedInputs::Rendered;
    let model = ferrum_types::ModelId::new("fixture.model");
    let mut identities = HashSet::new();
    let phases = [
        (report::Phase::Training, &value.training[0]),
        (
            report::Phase::Residual,
            &value.validation_model.residual()[0],
        ),
        (report::Phase::Heldout, &value.validation[0]),
    ];
    for (phase, case) in phases {
        let mut budgets = Vec::new();
        for &index in &case.prompts {
            let (request, _) = inputs
                .request_for_phase(&value, index, &model, Default::default(), phase)
                .unwrap();
            assert!(
                identities.insert(request.id),
                "no owner may cross a phase boundary"
            );
            assert_eq!(request.prompt, value.prompts[index].rendered_prompt);
            assert_eq!(
                request.sampling_params.max_tokens,
                value.prompts[index].sampling.max_tokens
            );
            budgets.push(request.sampling_params.max_tokens);
        }
        assert_eq!(budgets, [73, 101, 73, 101, 73]);
    }
    assert_eq!(identities.len(), 15);
    assert_eq!(serde_json::to_value(&value).unwrap(), frozen_declaration);
    let manifest::ValidationSource::SelectedWorkSupportV1 { residual, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    residual.clear();
    assert!(
        value.validate().is_err(),
        "rolling is not permission to reuse fit as residual"
    );
}
