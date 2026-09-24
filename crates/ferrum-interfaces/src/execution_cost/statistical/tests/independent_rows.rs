use super::*;

fn work() -> KernelNumericWorkV1 {
    KernelNumericWorkV1 {
        logical_units: 1,
        padded_units: 1,
        inner_units_per_logical_unit: 4,
        grid: [1; 3],
        scratch_bytes: 64,
        staged_weight_bytes: 0,
    }
}
fn push(
    b: &mut SelectedCommandCostBuilderV1,
    name: &str,
) -> Result<(), StatisticalEvidenceUnknown> {
    b.kernel(algorithm(name), work())
}
fn row_chain(
    b: &mut SelectedCommandCostBuilderV1,
    grouped: bool,
    reverse: bool,
) -> Result<(), StatisticalEvidenceUnknown> {
    push(b, "prepare")?;
    if grouped {
        for entry in if reverse {
            ["reduce", "partial"]
        } else {
            ["partial", "reduce"]
        } {
            push(b, entry)?;
        }
        Ok(())
    } else {
        push(b, "direct")
    }
}
fn selected(
    rows: &[bool],
    group: bool,
    reverse: bool,
    suffix: &str,
) -> SelectedCommandCostEvidenceV1 {
    let mut b = SelectedCommandCostBuilderV1::new(rows.len() as u64);
    push(&mut b, "packed-projections").unwrap();
    if group {
        b.independent_attention_rows_v2(rows.iter().copied(), |b, g| row_chain(b, g, reverse))
            .unwrap();
    } else {
        for &g in rows {
            row_chain(&mut b, g, reverse).unwrap();
        }
    }
    push(&mut b, suffix).unwrap();
    b.finish().unwrap()
}

#[test]
fn independent_rows_v2_preserves_v1_bytes_and_only_forgets_whole_row_order() {
    let ordered = selected(&[false, true, false], false, false, "output");
    let dgd = selected(&[false, true, false], true, false, "output");
    let gdd = selected(&[true, false, false], true, false, "output");
    assert_eq!(
        serde_json::to_vec(&ordered).unwrap(),
        serde_json::to_vec(&dgd).unwrap()
    );
    assert_eq!(
        ordered, dgd,
        "the V1 equality contract does not include passive V2"
    );
    assert_ne!(dgd.family_signature(), gdd.family_signature());
    assert_eq!(dgd.work(), gdd.work());
    assert_eq!(
        dgd.independent_attention_family_v2(),
        gdd.independent_attention_family_v2()
    );
    // Independent old encoding, not a comparison of two V2 code paths.
    let mut old = Sha256::new();
    bytes(&mut old, b"ferrum.selected-command-cost.v1");
    for entry in [
        "packed-projections",
        "prepare",
        "direct",
        "prepare",
        "partial",
        "reduce",
        "prepare",
        "direct",
        "output",
    ] {
        number(&mut old, 0);
        old.update(algorithm(entry).signature());
    }
    let old: [u8; 32] = old.finalize().into();
    assert_eq!(*dgd.family_signature(), old);
    let reversed = selected(&[false, true, false], true, true, "output");
    let output_changed = selected(&[false, true, false], true, false, "other-output");
    assert_ne!(
        dgd.independent_attention_family_v2(),
        reversed.independent_attention_family_v2()
    );
    assert_ne!(
        dgd.independent_attention_family_v2(),
        output_changed.independent_attention_family_v2()
    );
    assert_ne!(
        dgd.independent_attention_family_v2(),
        ordered.independent_attention_family_v2(),
        "lack of independence proof cannot silently share a grouped family"
    );
}

#[test]
fn independent_rows_v2_rejects_partial_transfer_duplicate_and_over_capacity_groups() {
    for count in [0, 1, MAX_COST_ROWS + 1] {
        let mut b = SelectedCommandCostBuilderV1::new(1);
        b.independent_attention_rows_v2(0..count, |b, _| row_chain(b, false, false))
            .unwrap();
        push(&mut b, "suffix").unwrap();
        assert!(b
            .finish()
            .unwrap()
            .independent_attention_family_v2()
            .is_none());
    }
    let mut b = SelectedCommandCostBuilderV1::new(2);
    b.independent_attention_rows_v2(0..2, |b, _| push(b, "prepare-only"))
        .unwrap();
    assert!(b
        .finish()
        .unwrap()
        .independent_attention_family_v2()
        .is_none());
    let mut b = SelectedCommandCostBuilderV1::new(2);
    b.independent_attention_rows_v2(0..2, |b, _| {
        row_chain(b, false, false)?;
        b.transfer(
            algorithm("upload"),
            StatisticalTransferKindV1::HostToDevice,
            4,
        )
    })
    .unwrap();
    assert!(b
        .finish()
        .unwrap()
        .independent_attention_family_v2()
        .is_none());
    let mut b = SelectedCommandCostBuilderV1::new(2);
    for _ in 0..2 {
        b.independent_attention_rows_v2(0..2, |b, _| row_chain(b, false, false))
            .unwrap();
    }
    assert!(b
        .finish()
        .unwrap()
        .independent_attention_family_v2()
        .is_none());
    let mut b = SelectedCommandCostBuilderV1::new(2);
    assert_eq!(
        b.independent_attention_rows_v2(0..2, |b, _| {
            push(b, "prepare")?;
            Err(StatisticalEvidenceUnknown::MissingProducer)
        }),
        Err(StatisticalEvidenceUnknown::MissingProducer)
    );
    assert_eq!(b.finish(), Err(StatisticalEvidenceUnknown::MissingProducer));
}

fn decode_wave(order: &[bool], generated: u64, full: bool, mask: bool) -> CanonicalStatisticalWave {
    let e = selected(order, true, false, "output");
    let count = 2 + order.iter().map(|&g| if g { 3 } else { 2 }).sum::<u64>();
    let cmd = OperationCostCommand::new(
        "causal",
        DeviceCommandPhase::Compute,
        DeviceBatchingForm::Packed,
        0,
        order.len() as u32,
        order.len() as u64,
        count,
        0,
    )
    .unwrap()
    .with_statistical_evidence(e)
    .unwrap();
    let mut b = CanonicalWaveCostBuilder::new(
        0,
        if full {
            CostProductOutput::FullLogits
        } else {
            CostProductOutput::GreedyToken
        },
    );
    b.physical_command(cmd.canonical_command(0, 0, provider()).unwrap())
        .unwrap();
    b.core_readback_route(CoreReadbackRoute::NoReadback)
        .unwrap();
    for &g in order {
        let mut r = row(1, false);
        r.work = ActualRowWork::Decode {
            kv_tokens: if g { 300 } else { 100 },
        };
        r.output = CostRowOutput::Decode {
            requires_full_logits: full,
            repetition_tokens: 0,
            repetition_penalty_bits: 1.0_f32.to_bits(),
        };
        r.mask_upload_required = mask;
        let host = r.host_features.as_mut().unwrap();
        host.state.generated_tokens_before = generated;
        host.state.sampling_history_tokens = generated;
        host.state.maximum_output_tokens = 8;
        b.row(r).unwrap();
    }
    b.finish_with_statistics(
        ActualWaveKind::Decode,
        ActualWavePath::PlanRuntime,
        ActualWaveGraphState::Disabled,
        ActualWaveRowOrder::Ordered,
        0,
    )
    .unwrap()
}

#[test]
fn independent_rows_v2_keeps_exact_binding_terminal_full_and_mask_boundaries() {
    let dgd = decode_wave(&[false, true, false], 1, false, false);
    let gdd = decode_wave(&[true, false, false], 1, false, false);
    let a = dgd
        .statistical
        .as_ref()
        .unwrap()
        .independent_attention_v2()
        .unwrap();
    let b = gdd
        .statistical
        .as_ref()
        .unwrap()
        .independent_attention_v2()
        .unwrap();
    assert_eq!(a.family_signature(), b.family_signature());
    assert_ne!(dgd.exact, gdd.exact);
    a.validate_exact(&dgd.exact).unwrap();
    assert_eq!(
        a.validate_exact(&gdd.exact),
        Err(StatisticalEvidenceUnknown::ExactBindingMismatch)
    );
    for other in [
        decode_wave(&[false, true, false], 1, true, false),
        decode_wave(&[false, true, false], 1, false, true),
        decode_wave(&[false, true, false], 7, false, false),
    ] {
        assert_ne!(
            a.family_signature(),
            other
                .statistical
                .unwrap()
                .independent_attention_v2()
                .unwrap()
                .family_signature()
        );
    }
    let terminal_a = decode_wave(&[false, true, false], 7, false, false);
    let terminal_b = decode_wave(&[true, false, false], 7, false, false);
    assert_ne!(
        terminal_a
            .statistical
            .unwrap()
            .independent_attention_v2()
            .unwrap()
            .family_signature(),
        terminal_b
            .statistical
            .unwrap()
            .independent_attention_v2()
            .unwrap()
            .family_signature()
    );
    let v1 = dgd.statistical.unwrap();
    let wire = v1.to_wire_v1();
    let json = serde_json::to_value(&wire).unwrap();
    assert_eq!(json.as_object().unwrap().len(), 5);
    let imported = StatisticalWaveEvidenceV1::from_wire_v1(wire, &dgd.exact).unwrap();
    assert_eq!(v1, imported);
    assert!(
        imported.independent_attention_v2().is_none(),
        "old wire cannot invent V2 evidence"
    );
}

#[test]
fn independent_rows_v2_wire_is_explicit_and_bound_to_its_own_exact_wave() {
    let wave = decode_wave(&[false, true, false], 1, false, false);
    let other = decode_wave(&[true, false, false], 1, false, false);
    let e = wave
        .statistical
        .as_ref()
        .unwrap()
        .independent_attention_v2()
        .unwrap();
    let wire = e.to_wire_v2();
    let encoded = serde_json::to_vec(&wire).unwrap();
    let decoded = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(
        &IndependentAttentionWaveEvidenceV2::from_wire_v2(decoded, &wave.exact).unwrap(),
        e
    );
    assert!(IndependentAttentionWaveEvidenceV2::from_wire_v2(wire.clone(), &other.exact).is_err());
    let mut json = serde_json::to_value(&wire).unwrap();
    json["schema_version"] = 1.into();
    let old_label = serde_json::from_value(json.clone()).unwrap();
    assert!(IndependentAttentionWaveEvidenceV2::from_wire_v2(old_label, &wave.exact).is_err());
    json["invented_field"] = true.into();
    assert!(serde_json::from_value::<IndependentAttentionWaveEvidenceWireV2>(json).is_err());
    let old = wave.statistical.as_ref().unwrap().to_wire_v1();
    let invented = serde_json::from_value(serde_json::to_value(old).unwrap()).unwrap();
    assert!(
        IndependentAttentionWaveEvidenceV2::from_wire_v2(invented, &wave.exact).is_err(),
        "the old schema tag cannot be silently treated as new evidence"
    );
}
