//! Source-bound metadata contracts; native numerical coverage lives in parent.
use super::*;
use crate::image_generation::p_h3::expansion::visual_inputs::VisualExpansionInputs;
use ferrum_models::qwen35_config::Qwen35TextConfig;

fn rejected(fixture: &VisualFixture, first: PreparedTextExpansion, second: PreparedTextExpansion) {
    assert!(!fixture.assets.cohort_compatible(&first, &second).unwrap());
    assert!(fixture.assets.prepare_cohort(vec![first, second]).is_err());
}

#[test]
fn visual_cohort_metadata_accepts_distinct_geometry_and_preserves_bound_input_owners() {
    let fixture = VisualFixture::new();
    let mut first = fixture.prepare(MIXED);
    let mut second = fixture.prepare(OTHER);
    assert!(fixture.assets.cohort_compatible(&first, &second).unwrap());
    assert!(!fixture
        .assets
        .text_cohort_compatible(&first, &second)
        .unwrap());
    assert!(fixture
        .assets
        .prepare_text_cohort(vec![fixture.prepare(MIXED), fixture.prepare(OTHER)])
        .is_err());
    assert_ne!(first.tokens.len(), second.tokens.len());
    let tokens = [first.tokens.clone(), second.tokens.clone()];
    let context = first
        .engine
        .runtime
        .max_model_len
        .unwrap()
        .max(second.engine.runtime.max_model_len.unwrap());
    let host = host();
    bind(&mut first, &host, MIXED);
    bind(&mut second, &host, OTHER);
    let owners = [
        Arc::clone(first.causal_inputs.as_ref().unwrap()),
        Arc::clone(second.causal_inputs.as_ref().unwrap()),
    ];
    assert!(!Arc::ptr_eq(&owners[0], &owners[1]));
    let before = host.usage().unwrap().allocated_bytes();
    assert!(fixture.assets.cohort_compatible(&first, &second).unwrap());
    let group = fixture.assets.prepare_cohort(vec![first, second]).unwrap();
    assert_eq!(group.len(), 2);
    assert_eq!(group.engine.runtime.max_model_len, Some(context));
    assert_eq!(group.engine.scheduler.max_running_requests, 2);
    assert_eq!(group.engine.scheduler.max_waiting_requests, 2);
    assert_eq!(
        group.engine.batching.max_num_batched_tokens,
        MIXED.chunk + OTHER.chunk
    );
    assert_eq!(group.engine.sampling.default_params.max_tokens, 2);
    for (index, request) in group.requests.iter().enumerate() {
        assert_eq!(request.tokens, tokens[index]);
        assert!(Arc::ptr_eq(
            request.causal_inputs.as_ref().unwrap(),
            &owners[index]
        ));
    }
    assert_eq!(host.usage().unwrap().allocated_bytes(), before);
    drop(group);
    assert_eq!(host.usage().unwrap().allocated_bytes(), before);
    drop(owners);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn visual_cohort_metadata_keeps_text_entrypoints_and_rejects_text_visual_mixing() {
    let fixture = VisualFixture::new();
    let first = prepared(&fixture.assets, 2);
    let second = prepared(&fixture.assets, 3);
    assert!(fixture.assets.cohort_compatible(&first, &second).unwrap());
    assert!(fixture
        .assets
        .text_cohort_compatible(&first, &second)
        .unwrap());
    let group = fixture.assets.prepare_cohort(vec![first, second]).unwrap();
    assert!(group
        .requests
        .iter()
        .all(|r| r.visual.is_none() && r.causal_inputs.is_none()));
    assert_eq!(group.engine.sampling.default_params.max_tokens, 3);
    assert!(fixture.assets.prepare_cohort(vec![]).is_err());
    rejected(
        &fixture,
        prepared(&fixture.assets, 2),
        fixture.prepare(MIXED),
    );
    rejected(
        &fixture,
        fixture.prepare(MIXED),
        prepared(&fixture.assets, 2),
    );
}

#[test]
fn visual_cohort_metadata_rejects_foreign_sources_or_different_execution_profiles_and_policy() {
    let fixture = VisualFixture::new();
    let foreign = VisualFixture::new();
    rejected(&fixture, fixture.prepare(MIXED), foreign.prepare(OTHER));
    let mutations: [fn(&mut PreparedTextExpansion); 2] = [
        |r| r.engine.backend.device = ferrum_types::Device::CUDA(0),
        |r| r.engine.backend.enable_reusable_execution = true,
    ];
    for mutate in mutations {
        let mut second = fixture.prepare(OTHER);
        mutate(&mut second);
        rejected(&fixture, fixture.prepare(MIXED), second);
    }
    // Agreement on a different requested profile is insufficient: both must
    // still refer to the profile of their original prepared model family.
    let mut first = fixture.prepare(MIXED);
    let mut second = fixture.prepare(OTHER);
    for request in [&mut first, &mut second] {
        request.engine.numerical_execution = "fixture.unprepared-profile".parse().unwrap();
    }
    rejected(&fixture, first, second);
}

#[test]
fn visual_cohort_metadata_rejects_mismatched_schema_or_changed_original_token_plan() {
    let fixture = VisualFixture::new();
    let mut second = fixture.prepare(OTHER);
    let tokens = second.visual.as_ref().unwrap().tokens().clone();
    let raw = serde_json::from_slice(fixture.assets.defined.sources().config_json()).unwrap();
    let mut text = Qwen35TextConfig::from_hf_config_value(&raw).unwrap();
    text.hidden_size += 2;
    second.visual = Some(VisualExpansionInputs::new(tokens, &text).unwrap());
    rejected(&fixture, fixture.prepare(MIXED), second);

    let mut second = fixture.prepare(OTHER);
    second.tokens[0] = TokenId::new(ANSWER_ID);
    rejected(&fixture, fixture.prepare(MIXED), second);
    let mut second = fixture.prepare(OTHER);
    second.engine.runtime.max_model_len = Some(1);
    rejected(&fixture, fixture.prepare(MIXED), second);
}

#[test]
fn visual_cohort_metadata_rejects_partial_binding_wrong_extent_and_undeclared_owner() {
    let fixture = VisualFixture::new();
    let host = host();
    let mut first = fixture.prepare(MIXED);
    bind(&mut first, &host, MIXED);
    rejected(&fixture, first, fixture.prepare(OTHER));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);

    let mut first = fixture.prepare(MIXED);
    let mut second = fixture.prepare(OTHER);
    bind(&mut first, &host, MIXED);
    // Same schema does not authorize another request's longer continuation
    // table. The original immutable owner must match its own visual plan.
    second.causal_inputs = first.causal_inputs.as_ref().map(Arc::clone);
    rejected(&fixture, first, second);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);

    let mut visual = fixture.prepare(MIXED);
    bind(&mut visual, &host, MIXED);
    let mut text = prepared(&fixture.assets, 2);
    text.causal_inputs = visual.causal_inputs.take();
    rejected(&fixture, prepared(&fixture.assets, 2), text);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
