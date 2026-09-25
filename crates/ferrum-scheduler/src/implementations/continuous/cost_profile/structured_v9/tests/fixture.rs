//! Typed numerical replay fixture, not a claimed live settlement receipt.
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
pub(super) fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn wave(terminal: bool) -> (CanonicalStructuredWave, StructuredInputV1) {
    let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    selected
        .kernel(
            SelectedAlgorithmClassV1::new("fixture.replay", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 64,
                padded_units: 64,
                inner_units_per_logical_unit: 1,
                grid: [1, 1, 1],
                scratch_bytes: 32,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let selected = selected.finish().unwrap();
    let mut b =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "fixture.replay",
        command_index: 0,
        node_index: None,
        command_phase: DeviceCommandPhase::Compute,
        provider: None,
        path: CostCommandPath::Eager,
        participant_start: 0,
        participant_count: 1,
        token_count: 1,
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: None,
        statistical_evidence: Some(&selected),
    })
    .unwrap();
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    b.row(CanonicalCostRow {
        work: ActualRowWork::Decode { kv_tokens: 64 },
        output: CostRowOutput::Decode {
            requires_full_logits: true,
            repetition_tokens: 0,
            repetition_penalty_bits: 1f32.to_bits(),
        },
        host_policy_signature: [3; 32],
        mask_upload_required: false,
        host_features: Some(HostCostFeaturesV1 {
            policy: HostCostPolicyV2 {
                empirical_content_domain: Some(HostContentDomainV1::PlainTextGreedyV1),
                categorical_signature: [4; 32],
                decoder_text_bytes_per_token: 4,
                decoder_scratch_bytes_per_token: 8,
                raw_token_bytes_bound: 4,
            },
            state: HostCostStateV1 {
                generated_tokens_before: 2,
                maximum_output_tokens: if terminal { 3 } else { 20 },
                sampling_history_tokens: 2,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        }),
    })
    .unwrap();
    let w = b
        .finish_with_captured_structure(
            ActualWaveKind::Decode,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap();
    let structured = w
        .statistical
        .as_ref()
        .unwrap()
        .structured_capture()
        .unwrap()
        .unwrap()
        .as_ref()
        .clone();
    let input =
        StructuredInputV1::from_future(&w.exact, w.statistical.as_ref().unwrap(), &structured)
            .unwrap();
    (
        CanonicalStructuredWave {
            exact: w.exact,
            statistical: w.statistical,
            structured: Ok(structured),
        },
        input,
    )
}
fn header(domain: [u8; 32]) -> Header {
    let mut h = Header {
        artifact_type: "ferrum.structured-live-source".into(),
        schema_version: 2,
        model_revision: MODEL_REVISION.into(),
        population_revision: POPULATION_REVISION.into(),
        capture_identity: [5; 32],
        protocol: [0; 32],
        declared_protocol: [6; 32],
        rule_signature: [0; 32],
        fingerprint: ProfileFingerprint::from(&fingerprint()),
        producer: serde_json::json!({"executable_path":"/fixture/producer","executable_sha256":"11".repeat(32),"executable_bytes":1,"package_version":"test","source_revision":null}),
        opening: PairedClock {
            wall_unix_ns: 1_000_000_000_000,
            monotonic_ns: 100,
        },
        opened_at_ns: 99,
        initial_fifo_cutoff: 10,
        scope: Scope { rows: 1, domain },
        phase_members: [8, 8, 8],
        maximum_offered_waves: 32,
        maximum_file_bytes: 4 * 1024 * 1024,
        settings: Settings {
            min_samples: 8,
            redundancy: 4,
            max_phase_samples: 16,
            max_axes: 64,
            max_rank: 8,
            max_wave_ns: 1_000_000,
            max_age_ns: 1_000_000_000,
            margin_ns: 1000,
        },
    };
    let mut rule = Sha256::new();
    rule.update(POPULATION_REVISION.as_bytes());
    rule.update(b"prepared-wave; all-decode-with-generated-history; before-execute; outcome-independent; one-scope\0");
    rule.update(1u64.to_le_bytes());
    rule.update(domain);
    h.rule_signature = rule.finalize().into();
    let mut p = Sha256::new();
    p.update(b"ferrum.structured-live-source.v1\0");
    p.update(MODEL_REVISION.as_bytes());
    p.update(h.declared_protocol);
    p.update(h.rule_signature);
    for n in [
        8,
        8,
        8,
        8,
        4,
        16,
        64,
        8,
        1_000_000,
        1_000_000_000,
        1000,
        32,
        4 * 1024 * 1024,
    ] {
        p.update((n as u64).to_le_bytes());
    }
    h.protocol = p.finalize().into();
    h
}
fn stages(w: &CanonicalStructuredWave, index: u64, terminal: bool) -> Stages {
    let prepare = index * 1000;
    let recipe: Recipe =
        serde_json::from_value(serde_json::to_value(w.structured.as_ref().unwrap()).unwrap())
            .unwrap();
    let mut s = Stages {
        schema_version: 1,
        call_id: index,
        presubmit_prediction: None,
        fingerprint: Some(ProfileFingerprint::from(&fingerprint())),
        actual_shape: Some(Shape {
            exact: ProfileWaveShape {
                kind: ProfileWaveKind::Decode,
                path: ProfileExecutionPath::PlanRuntime,
                provider_signature: w.exact.provider_signature,
                output_policy_signature: w.exact.output_policy_signature,
                graph_state: ProfileGraphState::Disabled,
                order: ProfileBatchOrder::Ordered,
                decode_kv_tokens: vec![64],
                prefill_chunks: vec![],
                recurrent_state_bytes: 0,
                restore_bytes: 0,
                maintenance_bytes: 0,
                maintenance_units: 0,
            },
            numeric_features: w.exact.numeric_features.clone(),
            host_content_features: w.exact.host_content_features,
            row_multiset_features: w.exact.row_multiset_features.clone(),
        }),
        statistical_evidence: Some(w.statistical.as_ref().unwrap().to_wire_v1()),
        structured_evidence: None,
        prepare_started_at_ns: Some(prepare),
        executor_returned_at_ns: Some(prepare + 50),
        rows: vec![StageRow {
            request_id: format!("request-{index}"),
            owner_incarnation: 1,
            work_generation: 1,
            input_index: 0,
            actual_work: RowWork::Decode { kv_tokens: 64 },
            host_processing_ordinal: Some(0),
            host_started_at_ns: Some(prepare + 50),
            token_committed_at_ns: Some(prepare + 70),
            output_published_at_ns: Some(prepare + 80),
            completion_started_at_ns: terminal.then_some(prepare + 85),
            settled_at_ns: Some(prepare + 100),
            terminal: terminal.then_some(Terminal {
                finish_reason: ferrum_types::FinishReason::Length,
                generated_tokens: 3,
                through_output_ordinal: 3,
                output_failed: false,
                physical_failed: false,
                scheduler_failed: false,
                terminal_handoff_succeeded: true,
                pending_restore_removed: false,
                admission_cancellation_work: "no_additional_work".into(),
                cache_completion_work: "no_additional_work".into(),
                other_physical_resources: false,
                request_slot_closed: true,
                owner_matched: true,
            }),
            completeness: "complete_single_wave".into(),
        }],
        finalized_at_ns: Some(prepare + 110),
        full_wall_ns: Some(100),
        completeness: "complete_single_wave".into(),
    };
    let stage_binding = observation::stage_binding(&s, None).unwrap();
    s.structured_evidence = Some(Ok(Settlement {
        protocol: "ferrum.structured-host-settled-capture.v1".into(),
        call_id: index,
        recipe,
        stage_binding,
        executor_envelope_ns: 50,
        host_settled_after_executor_ns: 50,
        full_wall_ns: 100,
    }));
    s
}
pub(super) struct Source {
    pub bytes: Vec<u8>,
    pub query: StructuredInputV1,
    pub closing: PairedClock,
}
pub(super) fn source() -> Source {
    let (_, query) = wave(false);
    let h = header(*query.domain_signature());
    let mut bytes = Vec::new();
    let mut records = 0u64;
    fn append(bytes: &mut Vec<u8>, records: &mut u64, record: impl Serialize) {
        *records += 1;
        serde_json::to_writer(
            &mut *bytes,
            &serde_json::json!({"source_record_ordinal":*records,"record":record}),
        )
        .unwrap();
        bytes.push(b'\n');
    }
    append(&mut bytes, &mut records, &h);
    // An actual preparation retry remains offered, but never consumes a member.
    append(
        &mut bytes,
        &mut records,
        Record::Offered {
            offered: 1,
            member_candidate: true,
            phase: StructuredProfilePhaseV9::Fit,
            rows: vec![OfferedRow {
                request_id: "warm-resource".into(),
                owner: 1,
                generation: 1,
                generated: 2,
                decode: true,
            }],
        },
    );
    append(
        &mut bytes,
        &mut records,
        Record::PreparationUnavailable {
            offered: 1,
            member_candidate: true,
            phase: StructuredProfilePhaseV9::Fit,
            reason: "resource turnaround".into(),
        },
    );
    let partition = StructuredPartitionV1 {
        source: h.capture_identity,
        protocol: h.protocol,
        population: StructuredPopulationV1::ReservedMembers {
            rule_signature: h.rule_signature,
        },
        fit_through: 8,
        residual_through: 16,
        qualification_through: 24,
    };
    let mut fitted = None;
    let mut calibrated = None;
    for phase in [
        StructuredProfilePhaseV9::Fit,
        StructuredProfilePhaseV9::Residual,
        StructuredProfilePhaseV9::Qualification,
    ] {
        let mut samples = Vec::new();
        for k in 0..8 {
            let index = phase.index() as u64 * 8 + k + 1;
            let terminal = k % 2 == 1;
            let (w, input) = wave(terminal);
            let s = stages(&w, index, terminal);
            let rows = vec![OfferedRow {
                request_id: format!("request-{index}"),
                owner: 1,
                generation: 1,
                generated: 2,
                decode: true,
            }];
            append(
                &mut bytes,
                &mut records,
                Record::Offered {
                    offered: index + 1,
                    member_candidate: true,
                    phase,
                    rows,
                },
            );
            append(
                &mut bytes,
                &mut records,
                Record::Reserved {
                    offered: index + 1,
                    member: Some(index),
                    phase,
                    boundary: "prepared_before_execute".into(),
                },
            );
            let n = Numeric {
                fifo: 10 + index,
                call_id: index,
                observed_at_ns: index * 1000 + 110,
                wall_ns: 100,
                domain: *input.domain_signature(),
                basis: input.regression_axes().to_vec(),
                support: input.joint_support_coordinates().to_vec(),
            };
            let recipe = s
                .structured_evidence
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap()
                .recipe
                .clone();
            append(
                &mut bytes,
                &mut records,
                Record::Completed {
                    offered: index + 1,
                    member: Some(index),
                    phase,
                    queue: Some(Queue {
                        accepted_ordinal: Some(10 + index),
                        disposition: "published".into(),
                    }),
                    reconciled: true,
                    host_stages: Some(s),
                    selected_structured_capture: Some(Ok(recipe)),
                    selected_independent_attention_v2: None,
                    numeric: Some(n),
                    conversion_error: None,
                },
            );
            samples.push(StructuredNumericObservationV1 {
                source: h.capture_identity,
                protocol: h.protocol,
                ordinal: 10 + index,
                membership: Some(
                    StructuredMemberBindingV1::new(h.rule_signature, index + 1, index).unwrap(),
                ),
                call_id: index,
                fingerprint: fingerprint(),
                input,
                boundary: CostBoundary::PreparationToHostSettledV1,
                outcome: WaveObservationOutcome::Completed,
                observed_at_ns: index * 1000 + 110,
                wall_ns: 100,
            });
        }
        let cutoff = (phase.index() as u64 + 1) * 8;
        let now = cutoff * 1000 + 120;
        let signature = match phase {
            StructuredProfilePhaseV9::Fit => {
                let m = FittedStructuredModelV1::fit(
                    fingerprint(),
                    h.settings.native(),
                    partition,
                    &samples,
                    now,
                )
                .unwrap();
                let signature = m.parameters_signature();
                fitted = Some(m);
                signature
            }
            StructuredProfilePhaseV9::Residual => {
                let m = fitted.take().unwrap().calibrate(&samples, now).unwrap();
                let signature = m.parameters_signature();
                calibrated = Some(m);
                signature
            }
            StructuredProfilePhaseV9::Qualification => calibrated
                .take()
                .unwrap()
                .qualify(&samples, now)
                .unwrap()
                .parameters_signature(),
        };
        let r = Freeze {
            capture_identity: h.capture_identity,
            protocol: h.protocol,
            rule_signature: h.rule_signature,
            phase,
            accepted_fifo_cutoff: 10 + cutoff,
            member_cutoff: cutoff,
            source_prefix_bytes: bytes.len() as u64,
            source_prefix_sha256: Sha256::digest(&bytes).into(),
            frozen_at_ns: now,
            parameters_sha256: signature,
        };
        append(&mut bytes, &mut records, Record::PhaseFreeze { receipt: r });
    }
    let closing = PairedClock {
        monotonic_ns: 24_130,
        wall_unix_ns: h.opening.wall_unix_ns + 24_030,
    };
    append(
        &mut bytes,
        &mut records,
        Record::Footer {
            phase: "qualified".into(),
            failure: None,
            offered: 25,
            members: 24,
            failed_members: 0,
            accepted_fifo_cutoff: 34,
            last_captured_fifo: 34,
            fifo_audit_complete: true,
            closing: Some(closing),
        },
    );
    Source {
        bytes,
        query,
        closing,
    }
}
