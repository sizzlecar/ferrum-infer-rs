use super::*;

fn case(width: usize, native: bool) -> Case {
    let mut case = Case {
        product: OpportunityProduct::Full,
        template: 0,
        width,
        maximum_output: NonZeroUsize::new(4).unwrap(),
        release_generated: 2,
        suffix_tokens: 2,
        preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
        prefix: PrefixKind::Pending,
        route: CalibrationDecodeRoute::Actual,
        reset: false,
        acquisition: None,
    };
    if native {
        case.acquisition = Some(PreparedProbeAcquisition {
            template: 0,
            maximum_output: case.maximum_output,
            preset: case.preset,
            plan: ProbePrefixAcquisitionPlan::new(7, 6, NonZeroU32::new(2).unwrap()).unwrap(),
            input_tokens_sha256: [7; 32],
        });
    }
    case
}

#[test]
fn native_probe_work_keeps_offered_clock_distinct_from_actions() {
    let cold = case_work(&case(2, false), 7, 4, None).unwrap();
    assert_eq!(
        cold,
        CaseWork {
            declared_offers_upper: 11,
            declared_offers_minimum: 10,
            requests: 2,
            serial_declared_offer_rows: 14,
            execution_actions: 14,
            serial_token_work: 20,
        }
    );
    let native = case_work(&case(2, true), 7, 4, None).unwrap();
    assert_eq!(
        native,
        CaseWork {
            declared_offers_upper: 5,
            declared_offers_minimum: 4,
            requests: 2,
            serial_declared_offer_rows: 8,
            execution_actions: 10,
            serial_token_work: 8,
        }
    );
    // Two fresh restore ACKs are execution actions, not offers or eligible rows.
    assert_eq!(
        native.execution_actions - native.serial_declared_offer_rows,
        2
    );
    let setup = setup_for_cases(&[case(2, true)]).unwrap();
    // One seed executes three two-token spans and one native capture. None is
    // part of the later source's numerical population.
    assert_eq!(
        setup,
        CaseWork {
            requests: 1,
            execution_actions: 4,
            serial_token_work: 6,
            ..CaseWork::default()
        }
    );
}

#[test]
fn native_probe_setup_is_once_per_source_and_exact_key_across_phases() {
    let repeated = vec![case(2, true); 3];
    let setup = setup_for_cases(&repeated).unwrap();
    assert_eq!(setup.requests, 1);
    assert_eq!(setup_for_indices(&repeated, &[0, 1, 2, 0]).unwrap(), setup);
    // A later source needs its own acquisition; no global deduplication or
    // extended lease lifetime is implied by equal source-local keys.
    assert_eq!(
        setup_for_cases(&repeated).unwrap().requests + setup.requests,
        2
    );
    let mut distinct = repeated.clone();
    distinct[1].template = 1;
    distinct[1].acquisition.as_mut().unwrap().template = 1;
    distinct[1]
        .acquisition
        .as_mut()
        .unwrap()
        .input_tokens_sha256 = [8; 32];
    let doubled = setup_for_cases(&distinct).unwrap();
    assert_eq!(
        (
            doubled.requests,
            doubled.execution_actions,
            doubled.serial_token_work
        ),
        (2, 8, 12)
    );
    assert_eq!(setup_for_indices(&distinct, &[2, 0]).unwrap(), setup);
    assert_eq!(
        setup_for_indices(&distinct, &[1, 0, 1, 2]).unwrap(),
        doubled
    );
    assert!(setup_for_indices(&distinct, &[0, distinct.len()]).is_err());
    assert_eq!(
        setup_for_indices(&distinct, &[]).unwrap(),
        CaseWork::default()
    );
}

#[test]
fn native_probe_shared_key_preserves_each_future_policy_binding() {
    let first = case(2, true);
    let mut second = first.clone();
    second.maximum_output = NonZeroUsize::new(5).unwrap();
    second.preset = SloAutomaticCostProbeSamplingPresetV1::GreedyLength;
    let key = second.acquisition.as_mut().unwrap();
    key.maximum_output = second.maximum_output;
    key.preset = second.preset;
    assert_eq!(first.acquisition, second.acquisition);
    let first_work = case_work(&first, 7, 4, None).unwrap();
    let second_work = case_work(&second, 7, 4, None).unwrap();
    assert_eq!(first_work.declared_offers_minimum, 4);
    assert_eq!(second_work.declared_offers_minimum, 6);
    assert_eq!(second_work.declared_offers_upper, 6);
    assert_eq!(
        second_work.serial_token_work - first_work.serial_token_work,
        2
    );
    assert_eq!(
        setup_for_cases(&[first.clone(), second.clone()])
            .unwrap()
            .requests,
        1
    );
    // Equality permits sharing state, not substituting one case's declared
    // output/sampling contract for another case's execution contract.
    let mut wrong_output = second.clone();
    wrong_output.acquisition.as_mut().unwrap().maximum_output = first.maximum_output;
    assert!(case_work(&wrong_output, 7, 4, None).is_err());
    assert!(setup_for_cases(&[first.clone(), wrong_output]).is_err());
    second.acquisition.as_mut().unwrap().preset = first.preset;
    assert!(case_work(&second, 7, 4, None).is_err());
    assert!(setup_for_cases(&[first, second]).is_err());
}

#[test]
fn native_probe_setup_keeps_template_digest_boundary_and_chunk_in_identity() {
    let first = case(2, true);
    let mut other_template = first.clone();
    other_template.template = 1;
    other_template.acquisition.as_mut().unwrap().template = 1;
    let mut other_tokens = first.clone();
    other_tokens
        .acquisition
        .as_mut()
        .unwrap()
        .input_tokens_sha256 = [8; 32];
    let mut other_boundary = first.clone();
    other_boundary.acquisition.as_mut().unwrap().plan =
        ProbePrefixAcquisitionPlan::new(7, 4, NonZeroU32::new(2).unwrap()).unwrap();
    let mut other_chunk = first.clone();
    other_chunk.product = OpportunityProduct::PrefillSpan {
        offset: 0,
        chunk: NonZeroU32::MIN,
    };
    other_chunk.acquisition.as_mut().unwrap().plan =
        ProbePrefixAcquisitionPlan::new(7, 6, NonZeroU32::MIN).unwrap();
    for other in [other_template, other_tokens, other_boundary, other_chunk] {
        assert_ne!(first.acquisition, other.acquisition);
        case_work(&other, 7, 4, None).unwrap();
        assert_eq!(
            setup_for_cases(&[first.clone(), other]).unwrap().requests,
            2
        );
    }
}

#[test]
fn native_probe_work_rejects_changed_shape_and_preserves_ordinary_prefill() {
    assert!(case_work(&case(2, true), 8, 4, None).is_err());
    assert!(case_work(&case(2, true), 7, 2, None).is_err());
    let mut ordinary = case(2, true);
    ordinary.prefix = PrefixKind::Ordinary;
    ordinary.release_generated = 0;
    assert!(case_work(&ordinary, 7, 4, None).is_err());
    assert!(setup_for_cases(&[ordinary.clone()]).is_err());
    ordinary.acquisition = None;
    let work = case_work(&ordinary, 7, 4, None).unwrap();
    // Four joint prefill waves, then up to three joint decode waves. The
    // conservative serial action bound still covers fourteen row executions.
    assert_eq!(
        (work.declared_offers_minimum, work.declared_offers_upper),
        (4, 7)
    );
    assert_eq!((work.execution_actions, work.serial_token_work), (14, 20));
    assert_eq!(setup_for_cases(&[ordinary]).unwrap(), CaseWork::default());
}

#[test]
fn native_probe_planning_preserves_actual_capture_span_and_proper_suffix() {
    use std::num::NonZeroU64;
    let mut prepared = case(2, false);
    let blueprint = PrefixBlueprint {
        prompt_tokens: 9,
        boundary: 8,
        span: CheckpointTokenSpanConstraint::new(
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(4).unwrap(),
        )
        .unwrap(),
        input_tokens_sha256: [9; 32],
    };
    let whole = NonZeroU32::new(8).unwrap();
    let key = declared_plan(&prepared, blueprint, whole, None)
        .unwrap()
        .unwrap();
    assert_eq!(key.plan().boundary(), 8);
    assert_eq!(key.template(), prepared.template);
    assert_eq!(key.prefill_chunk().get(), 4);
    assert!(declared_plan(
        &prepared,
        PrefixBlueprint {
            boundary: 6,
            ..blueprint
        },
        whole,
        None
    )
    .unwrap()
    .is_none());
    // A legal total boundary cannot authorize an unsupported final physical span.
    assert!(
        declared_plan(&prepared, blueprint, whole, NonZeroU32::new(2))
            .unwrap()
            .is_none()
    );
    for boundary in [0, blueprint.prompt_tokens, blueprint.prompt_tokens + 1] {
        assert!(declared_plan(
            &prepared,
            PrefixBlueprint {
                boundary,
                ..blueprint
            },
            whole,
            None
        )
        .is_err());
    }
    prepared.prefix = PrefixKind::Ordinary;
    assert!(declared_plan(&prepared, blueprint, whole, None)
        .unwrap()
        .is_none());
}

#[test]
fn native_probe_work_rejects_arithmetic_overflow_without_partial_setup_charge() {
    let mut single = case(1, false);
    single.maximum_output = NonZeroUsize::MIN;
    single.release_generated = 0;
    assert_eq!(
        case_work(&single, usize::MAX, 1, None)
            .unwrap()
            .serial_token_work,
        usize::MAX
    );
    single.maximum_output = NonZeroUsize::new(2).unwrap();
    assert!(case_work(&single, usize::MAX, 1, None).is_err());
    assert!(case_work(&single, 7, 0, None).is_err());
    assert!(case_work(&case(0, false), 7, 4, None).is_err());
    assert!(case_work(&case(usize::MAX, false), 7, 4, None).is_err());
    let key = case(2, true).acquisition.unwrap();
    for mut total in [
        CaseWork {
            requests: usize::MAX,
            ..CaseWork::default()
        },
        CaseWork {
            execution_actions: usize::MAX,
            ..CaseWork::default()
        },
        CaseWork {
            serial_token_work: usize::MAX,
            ..CaseWork::default()
        },
    ] {
        let before = total;
        assert!(add_setup(&mut total, key).is_err());
        assert_eq!(total, before);
    }
}
