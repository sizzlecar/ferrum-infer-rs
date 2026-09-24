use super::*;

pub(crate) fn independent_sample(ordinal: u64, rows: [bool; 3]) -> WholeWaveObservationV1 {
    let mut command = SelectedCommandCostBuilderV1::new(3);
    let emit = |b: &mut SelectedCommandCostBuilderV1, name: &str| {
        b.kernel(
            SelectedAlgorithmClassV1::new(name, 1, [2; 32], [3; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 1,
                padded_units: 1,
                inner_units_per_logical_unit: 4,
                grid: [1; 3],
                scratch_bytes: 64,
                staged_weight_bytes: 0,
            },
        )
    };
    command
        .independent_attention_rows_v2(rows.iter(), |b, &grouped| {
            emit(b, "prepare")?;
            if grouped {
                emit(b, "partial")?;
                emit(b, "reduce")
            } else {
                emit(b, "direct")
            }
        })
        .unwrap();
    let selected = command.finish().unwrap();
    let mut b = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "attention",
        command_index: 0,
        node_index: Some(0),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(CostProviderIdentity {
            provider_id: "fixture.independent-attention",
            implementation_fingerprint: "fixture.independent-attention.v1",
            operation_fingerprint: "fixture.attention.rows",
        }),
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 3,
        token_count: 3,
        batching_form: "packed",
        compute_dispatch_count: 7,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    b.core_readback_route(CoreReadbackRoute::NoReadback)
        .unwrap();
    for grouped in rows {
        b.row(CanonicalCostRow {
            work: ActualRowWork::Decode {
                kv_tokens: if grouped { 300 } else { 100 },
            },
            host_policy_signature: [1; 32],
            mask_upload_required: false,
            output: CostRowOutput::Decode {
                requires_full_logits: false,
                repetition_tokens: 0,
                repetition_penalty_bits: 1_f32.to_bits(),
            },
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                    categorical_signature: [2; 32],
                    decoder_text_bytes_per_token: 8,
                    decoder_scratch_bytes_per_token: 4,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 1,
                    maximum_output_tokens: 16,
                    sampling_history_tokens: 1,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: false,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    }
    let canonical = b
        .finish_with_statistics(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            64,
        )
        .unwrap();
    WholeWaveObservationV1 {
        source_sha256: partition().source_sha256,
        accepted_ordinal: ordinal,
        call_id: ordinal,
        fingerprint: fingerprint(),
        exact: canonical.exact,
        selected: canonical.statistical.unwrap(),
        boundary: CostBoundary::PreparationToHostSettledV1,
        outcome: WaveObservationOutcome::Completed,
        observed_at_ns: 100 + ordinal,
        wall_ns: 100,
    }
}

#[test]
fn independent_family_v2_fit_residual_heldout_are_explicit_and_keep_minimum_and_ttl() {
    let fit = (1..=8)
        .map(|n| independent_sample(n, [false, true, false]))
        .collect::<Vec<_>>();
    let residual = (9..=16)
        .map(|n| independent_sample(n, [true, false, false]))
        .collect::<Vec<_>>();
    let heldout = independent_sample(17, [false, false, true]);
    let legacy =
        FittedWholeWaveModelV1::fit(fingerprint(), settings(), partition(), &fit, 108).unwrap();
    assert!(matches!(
        legacy.calibrate(&residual, 116),
        Err(ModelUnknown::InsufficientResidual)
    ));
    assert!(matches!(
        FittedWholeWaveModelV1::fit_independent_attention_v2(
            fingerprint(),
            settings(),
            partition(),
            &fit[..7],
            108
        ),
        Err(ModelUnknown::InsufficientFit)
    ));
    let fitted = FittedWholeWaveModelV1::fit_independent_attention_v2(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    let short_residual = fitted.clone().calibrate(&residual[..7], 116);
    assert!(matches!(
        short_residual,
        Err(ModelUnknown::InsufficientResidual)
    ));
    let model = fitted.calibrate(&residual, 116).unwrap();
    let prediction = model
        .evaluate_heldout(&heldout, 117)
        .unwrap()
        .prediction
        .unwrap();
    assert_eq!(prediction.fit_samples, 8);
    assert_eq!(prediction.residual_samples, 8);
    assert_eq!(
        prediction.valid_until_ns,
        101 + settings().max_sample_age_ns.get()
    );
    assert!(matches!(
        model.predict(
            &heldout.fingerprint,
            &heldout.exact,
            &heldout.selected,
            prediction.valid_until_ns + 1
        ),
        Err(ModelUnknown::Stale)
    ));
    assert!(matches!(
        model.evaluate_heldout(&residual[0], 117),
        Err(ModelUnknown::PhaseLeakage)
    ));
}

#[test]
fn independent_family_v2_cannot_invent_old_import_evidence_or_change_ordered_fit() {
    let fit = (1..=8)
        .map(|n| independent_sample(n, [false, true, false]))
        .collect::<Vec<_>>();
    let old = fit
        .iter()
        .cloned()
        .map(|mut sample| {
            sample.selected = StatisticalWaveEvidenceV1::from_wire_v1(
                sample.selected.to_wire_v1(),
                &sample.exact,
            )
            .unwrap();
            sample
        })
        .collect::<Vec<_>>();
    let before =
        FittedWholeWaveModelV1::fit(fingerprint(), settings(), partition(), &fit, 108).unwrap();
    let after =
        FittedWholeWaveModelV1::fit(fingerprint(), settings(), partition(), &old, 108).unwrap();
    assert_eq!(before.parameter_signature(), after.parameter_signature());
    assert!(matches!(
        FittedWholeWaveModelV1::fit_independent_attention_v2(
            fingerprint(),
            settings(),
            partition(),
            &old,
            108
        ),
        Err(ModelUnknown::Evidence(
            StatisticalEvidenceUnknown::MissingProducer
        ))
    ));
    let new = FittedWholeWaveModelV1::fit_independent_attention_v2(
        fingerprint(),
        settings(),
        partition(),
        &fit,
        108,
    )
    .unwrap();
    assert_ne!(new.parameter_signature(), before.parameter_signature());
    let a = StatisticalModelInputV1::from_future(&fit[0].exact, &fit[0].selected).unwrap();
    let b = StatisticalModelInputV1::from_future(&old[0].exact, &old[0].selected).unwrap();
    assert_eq!(a.device(), b.device());
    assert_eq!(a.host_and_sequence(), b.host_and_sequence());
    assert_eq!(a.family_signature(), b.family_signature());
    assert!(b
        .family_signature_for(SelectedStatisticalFamily::IndependentAttentionV2)
        .is_err());
}
