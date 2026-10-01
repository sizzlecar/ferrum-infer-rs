//! Pure lifecycle fixtures built from the real canonical producer. Synthetic
//! clocks/walls and crate-private stamps grant no live/profile authority.
use super::*;
mod diagnostic;
mod owner_blocks;
use ferrum_types::FinishReason;
use std::num::{NonZeroU32, NonZeroU64};

fn prefill_body_input(final_chunk: bool) -> StructuredInputV2 {
    let mut command = SelectedCommandCostBuilderV1::new_with_algorithm_work(2);
    command
        .kernel(
            SelectedAlgorithmClassV1::new("fixture.prefill.body", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 2,
                padded_units: 2,
                inner_units_per_logical_unit: 32,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = command.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "fixture.prefill.body",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: Some(CostProviderIdentity {
                provider_id: "numerical-fixture",
                implementation_fingerprint: "v1",
                operation_fingerprint: "v1",
            }),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: 1,
            token_count: 2,
            batching_form: "serial",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&selected),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    builder
        .row(CanonicalCostRow {
            work: ActualRowWork::Prefill {
                offset: if final_chunk { 2 } else { 0 },
                count: 2,
                total_prompt_tokens: 4,
            },
            output: CostRowOutput::Prefill {
                final_logits: final_chunk,
            },
            host_policy_signature: [3; 32],
            mask_upload_required: false,
            host_features: Some(HostCostFeaturesV1 {
                policy: HostCostPolicyV2 {
                    empirical_content_domain: Some(HostContentDomainV1::PlainTextInstalledV2(
                        PlainTextPolicyCapabilityV2 {
                            sampling: PlainTextSamplingRouteV2::FullLogits,
                            model_eos: true,
                            user_stop: true,
                        },
                    )),
                    categorical_signature: [4; 32],
                    decoder_text_bytes_per_token: 4,
                    decoder_scratch_bytes_per_token: 8,
                    raw_token_bytes_bound: 4,
                },
                state: HostCostStateV1 {
                    generated_tokens_before: 0,
                    maximum_output_tokens: 8,
                    sampling_history_tokens: 0,
                    sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                    pending_decoded_utf8: false,
                    completion_state_signature: satisfied_completion_cost_signature(),
                },
            }),
        })
        .unwrap();
    let wave = builder
        .finish_with_captured_structure(
            ActualWaveKind::Prefill,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            32,
        )
        .unwrap();
    let selected = wave.statistical.unwrap();
    StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        &selected,
        selected.structured_capture().unwrap().unwrap(),
        &domain(),
    )
    .unwrap()
}

#[test]
fn nonnegative_body_without_token_production_can_qualify_but_cannot_cover_an_unseen_final_token() {
    let body = prefill_body_input(false);
    assert!(body
        .physical_host_rows()
        .iter()
        .all(|row| row.terminal_expectation == HostTerminalExpectationV1::NoTokenProduced));
    let make_phase = |phase: StructuredPhaseV2| {
        let mut rows = samples(phase, 16, phase.index() * 16);
        for row in &mut rows {
            row.input = body.clone().with_settled_terminal_causes(&[]).unwrap();
            row.wall_ns = 1000;
        }
        rows
    };
    let fit = make_phase(StructuredPhaseV2::Fit);
    let residual = make_phase(StructuredPhaseV2::Residual);
    let heldout = make_phase(StructuredPhaseV2::Qualification);
    let scope = StructuredScopeV2 {
        owner: body.owner().clone(),
        numerical_family: None,
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: Vec::new(),
            authorized_pending_constraints: Vec::new(),
            pending_counts: vec![0],
            length_counts: vec![0],
            pending_positions: Vec::new(),
            length_positions: Vec::new(),
            joint_counts: vec![(0, 0)],
        },
    };
    let fitted = FittedStructuredModelV2::fit_service_window(
        fp(),
        settings(),
        scope,
        physical_contract(),
        close(StructuredPhaseV2::Fit, &fit),
        &fit,
        170,
    )
    .unwrap();
    let final_token = prefill_body_input(true);
    assert_eq!(
        body.owner(),
        final_token.owner(),
        "unseen work must remain rejected even within one structural owner"
    );
    assert_eq!(
        fitted.service_input_membership(&final_token).unwrap(),
        StructuredServiceInputMembershipV1::OutsideFitSupport
    );
    let model = fitted
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850,
        )
        .unwrap()
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &heldout),
            &heldout,
            1530,
        )
        .unwrap();
    assert!(model
        .predict_query(&fp(), &StructuredQueryV2::exact(body), 1600)
        .is_ok());
    assert!(model
        .predict_query(&fp(), &StructuredQueryV2::exact(final_token), 1600)
        .is_err());
}

fn domain() -> CostWorkloadDomainV1 {
    CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: 1,
            model_weights: fp().model_weights,
            numerical_policy: fp().numerical_policy,
            device_runtime: fp().device_runtime,
            execution_config: fp().execution_config,
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(1024).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(4096).unwrap(),
            repetition_slot_capacity: 512,
            fixed_state_bytes_per_row: 32,
        },
    )
    .unwrap()
}
fn physical_contract() -> StructuredServiceWindowContractV2 {
    let mut contract = window_contract();
    contract.domain_policy = StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1;
    contract.nonnegative_envelope = Some(NonNegativeEnvelopeContractV1 {
        algorithm_universe: None,
        planning_estimator: Default::default(),
        population_policy: Default::default(),
        template_policy: Default::default(),
        workload_domain: domain(),
        settings: EnvelopeSettings::default(),
        challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
    });
    contract
}
fn input(terminal: usize, pending: [bool; 2], context: u32, generated: u64) -> StructuredInputV2 {
    let wave = wave_with_history_domain(
        terminal,
        true,
        8,
        "fixture.first",
        pending,
        [3; 32],
        [4; 32],
        context,
        generated,
        HostContentDomainV1::PlainTextInstalledV2(PlainTextPolicyCapabilityV2 {
            sampling: PlainTextSamplingRouteV2::FullLogits,
            model_eos: true,
            user_stop: true,
        }),
    );
    let mut input = project(&wave).unwrap();
    input.physical_domain = Some(*domain().sha256());
    input
}
fn population(phase: StructuredPhaseV2, early: bool) -> Vec<StructuredNumericObservationV2> {
    let mut rows = samples(phase, 16, phase.index() * 16);
    for (i, row) in rows.iter_mut().enumerate() {
        let terminal = [9, 0, 1, 2][i / 4];
        let input = input(
            terminal,
            [i & 1 != 0, i & 2 != 0],
            64 + phase.index() as u32 * 64,
            2 + phase.index() as u64 * 16 + (i % 2) as u64,
        );
        let mut causes = match terminal {
            0 => vec![(0, FinishReason::Length)],
            1 => vec![(1, FinishReason::Length)],
            2 => vec![(0, FinishReason::Length), (1, FinishReason::Length)],
            _ => Vec::new(),
        };
        if early && i == 0 {
            causes.push((0, FinishReason::EOS));
        }
        if early && i == 1 {
            causes.push((1, FinishReason::Stop));
        }
        row.input = input.with_settled_terminal_causes(&causes).unwrap();
        row.wall_ns = 1000;
    }
    rows
}
fn fit_with(rows: &[StructuredNumericObservationV2]) -> FittedStructuredModelV2 {
    FittedStructuredModelV2::fit_service_window(
        fp(),
        settings(),
        scope(rows),
        physical_contract(),
        close(StructuredPhaseV2::Fit, rows),
        rows,
        170,
    )
    .unwrap()
}
fn qualify_with(residual_early: bool) -> QualifiedStructuredModelV2 {
    let fit = population(StructuredPhaseV2::Fit, true);
    let residual = population(StructuredPhaseV2::Residual, residual_early);
    let heldout = population(StructuredPhaseV2::Qualification, true);
    fit_with(&fit)
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850,
        )
        .unwrap()
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &heldout),
            &heldout,
            1530,
        )
        .unwrap()
}
fn future() -> StructuredQueryV2 {
    let mut query = StructuredQueryV2::exact(input(9, [false, false], 320, 70));
    query.pending = Some(PendingQuery {
        eligible: vec![0, 1],
        constraint: HostPendingConstraintV2::AnySubset,
    });
    query.repetition_upper_sum = Some(256);
    query
}

#[test]
fn nonnegative_lifecycle_accepts_real_growing_work_without_empirical_domain_clipping() {
    let fit = population(StructuredPhaseV2::Fit, true);
    let residual = population(StructuredPhaseV2::Residual, true);
    let fitted = fit_with(&fit);
    assert!(residual[0]
        .input
        .support
        .iter()
        .enumerate()
        .any(|(axis, value)| *value > fit.iter().map(|s| s.input.support[axis]).max().unwrap()));
    assert_eq!(
        fitted.service_input_membership(&residual[0].input).unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    let model = qualify_with(true);
    let prediction = model.predict_query(&fp(), &future(), 1600).unwrap();
    assert_eq!(prediction.fitted_lower_ns, 0);
    assert!(prediction.fitted_upper_ns >= 1000);
    assert_eq!(prediction.valid_until_ns, 10_010);
    assert_eq!(
        (
            prediction.fit_samples,
            prediction.residual_samples,
            model.qualification_samples
        ),
        (16, 16, 16)
    );
    assert!(model.retained_payload_bytes().unwrap() > std::mem::size_of_val(&model));
    assert!(matches!(
        model.predict_query(&fp(), &future(), 10_011),
        Err(StructuredUnknown::Stale)
    ));
}

#[test]
fn empirical_fitted_strategy_calibrates_residual_and_rejects_failing_heldout() {
    let fit = population(StructuredPhaseV2::Fit, true);
    let mut residual = population(StructuredPhaseV2::Residual, true);
    for row in &mut residual {
        row.wall_ns = 1200;
    }
    let mut heldout = population(StructuredPhaseV2::Qualification, true);
    for row in &mut heldout {
        row.wall_ns = 1200;
    }
    let fitted = || {
        let mut contract = physical_contract();
        contract
            .nonnegative_envelope
            .as_mut()
            .unwrap()
            .planning_estimator = NonNegativePlanningEstimatorV1::FittedResidualV1;
        FittedStructuredModelV2::fit_service_window(
            fp(),
            settings(),
            scope(&fit),
            contract,
            close(StructuredPhaseV2::Fit, &fit),
            &fit,
            170,
        )
        .unwrap()
    };
    assert_ne!(
        fitted().parameters_signature(),
        fit_with(&fit).parameters_signature()
    );
    let calibrated = || {
        fitted()
            .calibrate_service_window(
                close(StructuredPhaseV2::Residual, &residual),
                &residual,
                850,
            )
            .unwrap()
    };
    let model = calibrated()
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &heldout),
            &heldout,
            1530,
        )
        .unwrap();
    let p = model.predict_query(&fp(), &future(), 1600).unwrap();
    assert!((1000..=1001).contains(&p.fitted_upper_ns));
    assert!((199..=200).contains(&p.residual_ns));
    assert!((1220..=1221).contains(&p.planning_ns));
    assert_eq!(p.fitted_lower_ns, 0);
    assert_eq!(
        (
            p.fit_samples,
            p.residual_samples,
            model.qualification_samples
        ),
        (16, 16, 16)
    );
    assert!(matches!(
        model.predict_query(&fp(), &future(), 10_011),
        Err(StructuredUnknown::Stale)
    ));
    let mut wrong_domain = future();
    wrong_domain.input.physical_domain = Some([99; 32]);
    assert!(matches!(
        model.predict_query(&fp(), &wrong_domain, 1600),
        Err(StructuredUnknown::WrongDomain)
    ));
    heldout[0].wall_ns = 10_000;
    assert!(matches!(
        calibrated().qualify_service_window(
            close(StructuredPhaseV2::Qualification, &heldout),
            &heldout,
            1530,
        ),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

#[test]
fn nonnegative_certificate_replays_same_model_and_rejects_modified_fit_or_domain() {
    let fit = population(StructuredPhaseV2::Fit, true);
    let original = fit_with(&fit);
    let cert = original.nonnegative_fit_certificate().unwrap().clone();
    let replay = |rows: &[StructuredNumericObservationV2], contract, certificate| {
        FittedStructuredModelV2::fit_service_window_from_certificate(
            fp(),
            settings(),
            scope(rows),
            contract,
            close(StructuredPhaseV2::Fit, rows),
            rows,
            170,
            certificate,
        )
    };
    let rebuilt = replay(&fit, physical_contract(), cert.clone()).unwrap();
    assert_eq!(
        original.parameters_signature(),
        rebuilt.parameters_signature()
    );
    let residual = population(StructuredPhaseV2::Residual, true);
    let heldout = population(StructuredPhaseV2::Qualification, true);
    let finish = |model: FittedStructuredModelV2| {
        model
            .calibrate_service_window(
                close(StructuredPhaseV2::Residual, &residual),
                &residual,
                850,
            )
            .unwrap()
            .qualify_service_window(
                close(StructuredPhaseV2::Qualification, &heldout),
                &heldout,
                1530,
            )
            .unwrap()
    };
    let original = finish(original);
    let rebuilt = finish(rebuilt);
    assert_eq!(
        original.parameters_signature(),
        rebuilt.parameters_signature()
    );
    assert_eq!(
        original
            .predict_query(&fp(), &future(), 1600)
            .unwrap()
            .planning_ns,
        rebuilt
            .predict_query(&fp(), &future(), 1600)
            .unwrap()
            .planning_ns
    );
    let mut altered = fit.clone();
    altered[0].wall_ns += 1;
    assert!(matches!(
        replay(&altered, physical_contract(), cert.clone()),
        Err(StructuredUnknown::WrongSource)
    ));
    let mut unstamped = fit.clone();
    unstamped[0].input.physical_domain = None;
    assert!(matches!(
        replay(&unstamped, physical_contract(), cert.clone()),
        Err(StructuredUnknown::WrongDomain)
    ));
    assert!(matches!(
        replay(&fit, window_contract(), cert),
        Err(StructuredUnknown::WrongProtocol)
    ));
}

#[test]
fn nonnegative_joint_pending_completion_repetition_upper_is_one_monotone_certificate() {
    let model = qualify_with(true);
    let all = future();
    let combined = model.predict_query(&fp(), &all, 1600).unwrap();
    let mut pending_only = all.clone();
    pending_only.repetition_upper_sum = None;
    let mut repetition_only = all.clone();
    repetition_only.pending = None;
    for query in [pending_only, repetition_only] {
        assert!(
            combined.fitted_upper_ns
                >= model
                    .predict_query(&fp(), &query, 1600)
                    .unwrap()
                    .fitted_upper_ns
        );
    }
    let mut over_capacity = all.clone();
    over_capacity.repetition_upper_sum = Some(1025);
    assert!(matches!(
        model.predict_query(&fp(), &over_capacity, 1600),
        Err(StructuredUnknown::InvalidInput)
    ));
    let mut wrong_domain = all;
    wrong_domain.input.physical_domain = Some([99; 32]);
    assert!(matches!(
        model.predict_query(&fp(), &wrong_domain, 1600),
        Err(StructuredUnknown::WrongDomain)
    ));
}

#[test]
fn nonnegative_length_does_not_replace_early_and_slow_eligible_heldout_fails() {
    let no_early_fit = population(StructuredPhaseV2::Fit, false);
    assert!(matches!(
        FittedStructuredModelV2::fit_service_window(
            fp(),
            settings(),
            scope(&no_early_fit),
            physical_contract(),
            close(StructuredPhaseV2::Fit, &no_early_fit),
            &no_early_fit,
            170
        ),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    let fit = population(StructuredPhaseV2::Fit, true);
    let no_early_residual = population(StructuredPhaseV2::Residual, false);
    assert!(matches!(
        fit_with(&fit).calibrate_service_window(
            close(StructuredPhaseV2::Residual, &no_early_residual),
            &no_early_residual,
            850
        ),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    let residual = population(StructuredPhaseV2::Residual, true);
    let no_early_heldout = population(StructuredPhaseV2::Qualification, false);
    assert!(matches!(
        fit_with(&fit)
            .calibrate_service_window(
                close(StructuredPhaseV2::Residual, &residual),
                &residual,
                850
            )
            .unwrap()
            .qualify_service_window(
                close(StructuredPhaseV2::Qualification, &no_early_heldout),
                &no_early_heldout,
                1530
            ),
        Err(StructuredUnknown::QualificationCoverage)
    ));
    let fit = population(StructuredPhaseV2::Fit, true);
    let residual = population(StructuredPhaseV2::Residual, true);
    let mut heldout = population(StructuredPhaseV2::Qualification, true);
    let calibrated = fit_with(&fit)
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850,
        )
        .unwrap();
    assert_eq!(
        calibrated
            .service_input_membership(&heldout[0].input)
            .unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    heldout[0].wall_ns = settings().max_wave_ns;
    assert_eq!(
        calibrated
            .service_input_membership(&heldout[0].input)
            .unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    assert!(matches!(
        calibrated.qualify_service_window(
            close(StructuredPhaseV2::Qualification, &heldout),
            &heldout,
            1530
        ),
        Err(StructuredUnknown::QualificationUnderestimate)
    ));
}

#[test]
fn nonnegative_membership_does_not_use_prediction_cap_or_settled_early_outcome() {
    let fit = population(StructuredPhaseV2::Fit, true);
    let mut capped = settings();
    capped.max_wave_ns = 1100;
    let fitted = FittedStructuredModelV2::fit_service_window(
        fp(),
        capped,
        scope(&fit),
        physical_contract(),
        close(StructuredPhaseV2::Fit, &fit),
        &fit,
        170,
    )
    .unwrap();
    let residual = population(StructuredPhaseV2::Residual, true);
    assert_eq!(
        fitted.service_input_membership(&residual[0].input).unwrap(),
        StructuredServiceInputMembershipV1::Eligible
    );
    // Same prepared row may continue or terminate; neither result is a member selector.
    let continued = input(9, [false, false], 128, 18)
        .with_settled_terminal_causes(&[])
        .unwrap();
    let stopped = input(9, [false, false], 128, 18)
        .with_settled_terminal_causes(&[(0, FinishReason::EOS)])
        .unwrap();
    assert_eq!(
        fitted.service_input_membership(&continued),
        fitted.service_input_membership(&stopped)
    );
    assert!(matches!(
        fitted.calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850
        ),
        Err(StructuredUnknown::Capacity)
    ));
}

#[test]
fn nonnegative_unseen_work_is_unknown_and_each_independent_phase_must_challenge_fit_axes() {
    let reset_pending = |rows: &mut [StructuredNumericObservationV2], phase: StructuredPhaseV2| {
        for (i, row) in rows.iter_mut().enumerate() {
            let causes = row.input.settled_terminal_causes().unwrap().to_vec();
            row.input = input(
                [9, 0, 1, 2][i / 4],
                [false, false],
                64 + phase.index() as u32 * 64,
                2 + phase.index() as u64 * 16 + (i % 2) as u64,
            )
            .with_settled_terminal_causes(&causes)
            .unwrap();
        }
    };
    let mut fit = population(StructuredPhaseV2::Fit, true);
    reset_pending(&mut fit, StructuredPhaseV2::Fit);
    let fitted = fit_with(&fit);
    let residual = population(StructuredPhaseV2::Residual, true);
    assert_eq!(
        fitted.service_input_membership(&residual[1].input).unwrap(),
        StructuredServiceInputMembershipV1::OutsideFitSupport
    );

    let fit = population(StructuredPhaseV2::Fit, true);
    let mut residual = residual;
    reset_pending(&mut residual, StructuredPhaseV2::Residual);
    let fitted = fit_with(&fit);
    assert!(residual
        .iter()
        .all(|row| fitted.service_input_membership(&row.input).unwrap()
            == StructuredServiceInputMembershipV1::Eligible));
    assert!(matches!(
        fitted.calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850
        ),
        Err(StructuredUnknown::QualificationCoverage)
    ));
}

#[test]
fn nonnegative_length_only_installed_policy_does_not_require_a_fabricated_early_branch() {
    let make_input = |terminal, pending, context, generated| {
        let wave = wave_with_history_domain(
            terminal,
            true,
            8,
            "fixture.first",
            pending,
            [3; 32],
            [4; 32],
            context,
            generated,
            HostContentDomainV1::PlainTextInstalledV2(PlainTextPolicyCapabilityV2 {
                sampling: PlainTextSamplingRouteV2::FullLogits,
                model_eos: false,
                user_stop: false,
            }),
        );
        let mut input = project(&wave).unwrap();
        input.physical_domain = Some(*domain().sha256());
        input
    };
    let phase_rows = |phase: StructuredPhaseV2| {
        let mut rows = population(phase, false);
        for (i, row) in rows.iter_mut().enumerate() {
            let causes = row.input.settled_terminal_causes().unwrap().to_vec();
            row.input = make_input(
                [9, 0, 1, 2][i / 4],
                [i & 1 != 0, i & 2 != 0],
                64 + phase.index() as u32 * 64,
                2 + phase.index() as u64 * 16 + (i % 2) as u64,
            )
            .with_settled_terminal_causes(&causes)
            .unwrap();
        }
        rows
    };
    let fit = phase_rows(StructuredPhaseV2::Fit);
    let residual = phase_rows(StructuredPhaseV2::Residual);
    let heldout = phase_rows(StructuredPhaseV2::Qualification);
    let model = fit_with(&fit)
        .calibrate_service_window(
            close(StructuredPhaseV2::Residual, &residual),
            &residual,
            850,
        )
        .unwrap()
        .qualify_service_window(
            close(StructuredPhaseV2::Qualification, &heldout),
            &heldout,
            1530,
        )
        .unwrap();
    let query = StructuredQueryV2::exact(make_input(9, [false, false], 320, 70));
    assert!(model.predict_query(&fp(), &query, 1600).is_ok());
}
