use super::*;
#[test]
fn refine_only_keeps_complete_schema_and_refs_but_executes_only_fresh_text() {
    let c = super::super::tests::config(2);
    let limits = super::super::tests::limits();
    let first = program::compose(&c, limits).unwrap();
    let steady =
        program::compose(&c.clone().with_phase(PH3ImageDiTPhase::RefinedText), limits).unwrap();
    let refine = c.with_phase(PH3ImageDiTPhase::RefineOnly);
    let only = program::compose(&refine, limits).unwrap();
    assert_eq!(only.schema, first.schema);
    assert_eq!(only.schema, steady.schema);
    assert_eq!(only.program.weights(), first.program.weights());
    assert_eq!(only.program.weights(), steady.program.weights());
    assert_eq!(
        only.program.inputs(),
        &[ProgramValueId::new(PH3ImageDiTFamily::RAW_TEXT_ID).unwrap()]
    );
    assert_eq!(
        only.program.outputs(),
        &[ProgramValueId::new(PH3ImageDiTFamily::REFINED_TEXT_ID).unwrap()]
    );
    assert_eq!(
        only.profile.primary_activation.as_str(),
        PH3ImageDiTFamily::REFINED_TEXT_ID
    );
    assert!(only
        .program
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .all(|n| n.id.as_str().starts_with("node.p_h3_image.token_refiner.")));
    assert_eq!(
        only.program.blocks()[0].nodes.len(),
        first.program.blocks()[0]
            .nodes
            .iter()
            .filter(|n| n.id.as_str().starts_with("node.p_h3_image.token_refiner."))
            .count()
    );
    assert!(only.program.states().is_empty());
    assert!(only.options.completion_retention().is_empty());
    assert!(only.profile.operations.iter().all(|o| !matches!(
        o.operation_id.as_str(),
        FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID
            | MODULATION_SCALE_BF16_OPERATION_ID
            | ADDCMUL_BF16_OPERATION_ID
            | ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID
    )));
    let expected = only
        .program
        .inputs()
        .iter()
        .chain(
            only.program
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|n| &n.outputs),
        )
        .collect::<BTreeSet<_>>();
    assert_eq!(expected, only.options.tensor_specs().keys().collect());
    only.profile.validate_program(&only.program).unwrap();
}
#[test]
fn every_pending_cache_uses_its_request_scope_and_late_scope_failure_refunds() {
    let c = super::super::tests::config(2).with_phase(PH3ImageDiTPhase::RefineOnly);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut raw = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            (c.text_rows().unwrap() * c.text_dim * 4) as usize,
        )
        .unwrap();
    for b in raw.as_mut().chunks_exact_mut(4) {
        b.copy_from_slice(&0.25_f32.to_le_bytes());
    }
    let raw = inputs::managed(
        "value.connector",
        config::tensor(vec![c.text_rows().unwrap(), c.text_dim], ElementType::F32),
        raw.freeze(),
    )
    .unwrap();
    let keys = cache::fixture_keys(&c, &raw);
    let bytes = (c.requests[0].text_rows * c.hidden * 2) as usize;
    let a = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes)])
        .unwrap();
    let b = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes - 1)])
        .unwrap();
    let destinations = [
        PH3ImageRefinementDestination::new(keys[0].request_identity(), &a),
        PH3ImageRefinementDestination::new(keys[1].request_identity(), &b),
    ];
    let before = host.usage().unwrap().committed_bytes();
    assert!(
        cache::PreparedRefinedTextCaches::new(&c, &host, keys.clone(), &destinations, 4096)
            .is_err()
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    drop(
        a.account()
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
            .unwrap(),
    );
    drop(destinations);
    drop(b);
    let b = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes)])
        .unwrap();
    let destinations = [
        PH3ImageRefinementDestination::new(keys[0].request_identity(), &a),
        PH3ImageRefinementDestination::new(keys[1].request_identity(), &b),
    ];
    let pending =
        cache::PreparedRefinedTextCaches::new(&c, &host, keys.clone(), &destinations, 4096)
            .unwrap();
    drop(pending);
    drop(
        a.account()
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
            .unwrap(),
    );
    drop(
        b.account()
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
            .unwrap(),
    );
    let swapped = [
        PH3ImageRefinementDestination::new(keys[1].request_identity(), &a),
        PH3ImageRefinementDestination::new(keys[0].request_identity(), &b),
    ];
    assert!(
        cache::PreparedRefinedTextCaches::new(&c, &host, keys.clone(), &swapped, 4096).is_err()
    );
    assert!(
        cache::PreparedRefinedTextCaches::new(&c, &host, keys.clone(), &destinations, 1).is_err()
    );
    drop((destinations, swapped));
    drop((a, b, raw, keys));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
