use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::prefixes::*;

fn prefix_manifest() -> manifest::Manifest {
    let mut m = manifest();
    let manifest::ValidationSource::StructuredWholeWaveGroupV2 {
        mut capture,
        residual,
    } = m.validation_model
    else {
        unreachable!()
    };
    capture.shared_source = Some("source5.jsonl".into());
    for child in &mut capture.children {
        child.source = "source5.jsonl".into();
        child.profile = capture.catalog.clone();
    }
    let phases = [&m.training[..], &residual[..], &m.validation[..]].map(|cases| {
        cases
            .iter()
            .flat_map(|case| {
                std::iter::repeat_n(
                    Some(StructuredPrefixCohortV5 {
                        release_generated: 1,
                        slots: case
                            .prompts
                            .iter()
                            .map(|_| StructuredPrefixSlotV5 {
                                tokenizer_policy_sha256: [7; 32],
                                token_ids: vec![ferrum_types::TokenId::new(11)],
                                token_bytes: vec![vec![0xc3]],
                            })
                            .collect(),
                    }),
                    case.repetitions.get(),
                )
            })
            .collect()
    });
    m.validation_model = manifest::ValidationSource::StructuredPrefixWholeWaveGroupV5 {
        capture,
        prefixes: StructuredPrefixPlanV5 { phases },
        residual,
    };
    m
}

#[test]
fn source5_cli_explicit_wire_binds_exact_slots_and_keeps_old_source4_separate() {
    let m = prefix_manifest();
    m.validate().unwrap();
    let value = serde_json::to_value(&m).unwrap();
    assert_eq!(
        value["validation_model"]["kind"],
        "structured_prefix_whole_wave_group_v5"
    );
    let loaded: manifest::Manifest = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        loaded.validation_model.prefix_plan_v5(),
        m.validation_model.prefix_plan_v5()
    );
    let mut legacy = value;
    legacy["validation_model"]["kind"] = "structured_whole_wave_group_v2".into();
    assert!(
        serde_json::from_value::<manifest::Manifest>(legacy).is_err(),
        "old wire cannot accept prefix declarations as ignored metadata"
    );
    let mut wrong = m.clone();
    let manifest::ValidationSource::StructuredPrefixWholeWaveGroupV5 { prefixes, .. } =
        &mut wrong.validation_model
    else {
        unreachable!()
    };
    prefixes.phases[0][0].as_mut().unwrap().slots.pop();
    assert!(wrong.validate().is_err());
    let mut wrong = m.clone();
    wrong.training[0].rolling_window = Some(manifest::RollingWindow {
        maximum_in_flight: NonZeroUsize::MIN,
    });
    assert!(
        wrong.validate().is_err(),
        "common release requires every original slot installed before preparation"
    );
}

#[test]
fn source5_cli_declared_prefix_changes_full_manifest_binding_not_numerical_gates() {
    let m = prefix_manifest();
    let mut changed = m.clone();
    let manifest::ValidationSource::StructuredPrefixWholeWaveGroupV5 { prefixes, .. } =
        &mut changed.validation_model
    else {
        unreachable!()
    };
    prefixes.phases[0][0].as_mut().unwrap().slots[0].token_ids[0] = ferrum_types::TokenId::new(10);
    prefixes.phases[0][0].as_mut().unwrap().slots[0].token_bytes[0] = b"a".to_vec();
    changed.validate().unwrap();
    let plan = super::super::super::config::cohort_plan(&m, |_, _| Ok(73)).unwrap();
    let original = serde_json::to_value(&m).unwrap();
    let altered = serde_json::to_value(&changed).unwrap();
    assert_ne!(
        plan.signature(&original).unwrap(),
        plan.signature(&altered).unwrap()
    );
    assert_eq!(original["protocol"], altered["protocol"]);
    assert_eq!(
        original["validation_model"]["capture"],
        altered["validation_model"]["capture"]
    );
    assert_eq!(original["training"], altered["training"]);
    assert_eq!(original["validation"], altered["validation"]);
}
