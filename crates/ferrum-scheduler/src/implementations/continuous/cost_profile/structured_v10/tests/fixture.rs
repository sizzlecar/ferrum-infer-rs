//! Typed source-wire fixture with real canonical producers and synthetic clocks.
//! It never creates the engine's private live qualified receipt.
use super::super::super::super::cost_model::structured_v2::windows::{
    ClosedRangeV2, CohortRequestV2, CohortV2, FrontierWindowV2, RowWindowV2, WorkWindowV2,
};
use super::*;
use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};

pub fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
pub fn prepared(
    id: &str,
    generation: u64,
    generated: u64,
) -> (Prepared, Vec<OfferedRow>, StructuredInputV2) {
    prepared_graph(id, generation, generated, None)
}
pub fn prepared_graph(
    id: &str,
    generation: u64,
    generated: u64,
    resident: Option<&str>,
) -> (Prepared, Vec<OfferedRow>, StructuredInputV2) {
    prepared_route(
        id,
        generation,
        generated,
        resident,
        if resident.is_some() {
            ActualWaveGraphState::Warm
        } else {
            ActualWaveGraphState::Disabled
        },
    )
}
pub fn prepared_route(
    id: &str,
    generation: u64,
    generated: u64,
    resident: Option<&str>,
    graph: ActualWaveGraphState,
) -> (Prepared, Vec<OfferedRow>, StructuredInputV2) {
    let mut command = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    command
        .kernel_with_replay_geometry(
            SelectedAlgorithmClassV1::new("fixture.profile10", 1, [1; 32], [2; 32]).unwrap(),
            KernelNumericWorkV1 {
                logical_units: 16,
                padded_units: 16,
                inner_units_per_logical_unit: 2,
                grid: [1, 1, 1],
                scratch_bytes: 64,
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                fixed_parameters: &[16],
            },
        )
        .unwrap();
    let command = command.finish().unwrap();
    let mut b =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    b.physical_command(CostPhysicalCommand {
        native_op_id: "fixture.profile10",
        command_index: 0,
        node_index: Some(0),
        command_phase: DeviceCommandPhase::Compute,
        provider: Some(CostProviderIdentity {
            provider_id: "fixture.provider",
            implementation_fingerprint: "v1",
            operation_fingerprint: "v1",
        }),
        path: if resident.is_some() {
            CostCommandPath::Replayed
        } else {
            CostCommandPath::Eager
        },
        participant_start: 0,
        participant_count: 1,
        token_count: 1,
        batching_form: "packed",
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        reusable_graph_node_count: resident.map(|_| 1),
        statistical_evidence: resident.is_none().then_some(&command),
    })
    .unwrap();
    if let Some(resident) = resident {
        b.replay_segment(0, resident, 1).unwrap();
        b.logical_command(CostLogicalCommand {
            native_op_id: "fixture.profile10",
            logical_command_ordinal: 0,
            node_index: 0,
            provider: CostProviderIdentity {
                provider_id: "fixture.provider",
                implementation_fingerprint: "v1",
                operation_fingerprint: "v1",
            },
            participant_count: 1,
            token_count: 1,
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: 1,
            statistical_evidence: Some(&command),
        })
        .unwrap();
    }
    b.core_readback_route(CoreReadbackRoute::HostSynchronized)
        .unwrap();
    let first = generated == 0;
    let work = if first {
        ActualRowWork::Prefill {
            offset: 0,
            count: 64,
            total_prompt_tokens: 64,
        }
    } else {
        ActualRowWork::Decode { kv_tokens: 64 }
    };
    b.row(CanonicalCostRow {
        work,
        output: if first {
            CostRowOutput::Prefill { final_logits: true }
        } else {
            CostRowOutput::Decode {
                requires_full_logits: true,
                repetition_tokens: generated,
                repetition_penalty_bits: 1f32.to_bits(),
            }
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
                generated_tokens_before: generated,
                maximum_output_tokens: 3,
                sampling_history_tokens: generated,
                sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                pending_decoded_utf8: false,
                completion_state_signature: satisfied_completion_cost_signature(),
            },
        }),
    })
    .unwrap();
    let wave = b
        .finish_with_captured_structure(
            if first {
                ActualWaveKind::Prefill
            } else {
                ActualWaveKind::Decode
            },
            ActualWavePath::PlanRuntime,
            graph,
            ActualWaveRowOrder::Ordered,
            0,
        )
        .unwrap();
    let stat = wave.statistical.as_ref().unwrap();
    let recipe = stat.structured_capture().unwrap().unwrap();
    if graph != ActualWaveGraphState::Disabled {
        assert!(matches!(
            crate::implementations::continuous::cost_model::statistical::StatisticalModelInputV1::from_future(&wave.exact, stat),
            Err(StatisticalEvidenceUnknown::UnsupportedReplay)
        ));
    }
    let native = StructuredInputV2::from_actual(&wave.exact, stat, recipe).unwrap();
    let facts = StructuredOwnerFactsV2::from_prepared(&wave.exact, stat, recipe).unwrap();
    let one = std::num::NonZeroU32::new(64).unwrap();
    let exact = Shape {
        exact: ProfileWaveShape {
            kind: if first {
                ProfileWaveKind::Prefill
            } else {
                ProfileWaveKind::Decode
            },
            path: ProfileExecutionPath::PlanRuntime,
            provider_signature: wave.exact.provider_signature,
            output_policy_signature: wave.exact.output_policy_signature,
            graph_state: match graph {
                ActualWaveGraphState::Disabled => ProfileGraphState::Disabled,
                ActualWaveGraphState::ConfiguredEager => ProfileGraphState::ConfiguredEager,
                ActualWaveGraphState::Warm => ProfileGraphState::Warm,
                ActualWaveGraphState::Cold => ProfileGraphState::Cold,
            },
            order: ProfileBatchOrder::Ordered,
            decode_kv_tokens: if first { vec![] } else { vec![64] },
            prefill_chunks: if first {
                vec![ProfilePrefillShape {
                    offset: 0,
                    count: one,
                    total_prompt_tokens: one,
                }]
            } else {
                vec![]
            },
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        },
        numeric_features: wave.exact.numeric_features.clone(),
        host_content_features: wave.exact.host_content_features,
        row_multiset_features: wave.exact.row_multiset_features.clone(),
    };
    let frontier_work = if first {
        PreparedWorkV2::Prefill {
            offset: 0,
            count: 64,
            total_prompt_tokens: 64,
        }
    } else {
        PreparedWorkV2::Decode { kv_tokens: 64 }
    };
    let p = Prepared {
        exact,
        selected: serde_json::from_value(serde_json::to_value(stat).unwrap()).unwrap(),
        selected_independent_attention_v2: None,
        recipe: serde_json::from_value(serde_json::to_value(recipe.as_ref()).unwrap()).unwrap(),
        owner_facts: serde_json::to_value(facts).unwrap(),
        rows: vec![PreparedRow {
            request_id: id.into(),
            owner_incarnation: 1,
            work_generation: generation,
            frontier: PreparedRowFactsV2 {
                physical_position: 0,
                work: frontier_work,
                generated_before: generated,
                maximum_output: 3,
                context_before: if first { 0 } else { 64 },
            },
        }],
    };
    let offered = vec![OfferedRow {
        request_id: id.into(),
        owner: 1,
        generation,
        generated,
        work: if first {
            OfferedWork::Prefill {
                offset: 0,
                count: 64,
                total_prompt_tokens: 64,
            }
        } else {
            OfferedWork::Decode { kv_tokens: 64 }
        },
    }];
    (p, offered, native)
}
pub fn header() -> Header {
    header_graph(None)
}
pub fn header_graph(resident: Option<&str>) -> Header {
    let (_, _, input) = prepared_graph("seed", 1, 1, resident);
    let owner = input.owner().clone();
    let scope = StructuredScopeV2 {
        owner: owner.clone(),
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![],
            authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
            pending_counts: vec![0],
            length_counts: vec![0],
            pending_positions: vec![],
            length_positions: vec![],
            joint_counts: vec![(0, 0)],
        },
    };
    let rule = MembershipRuleV2 {
        owner,
        windows: vec![FrontierWindowV2 {
            rows: vec![RowWindowV2 {
                generated_before: ClosedRangeV2 {
                    minimum: 1,
                    maximum: 1,
                },
                remaining_output: ClosedRangeV2::ALL,
                context_before: ClosedRangeV2::ALL,
                work: WorkWindowV2::Decode {
                    kv_tokens: ClosedRangeV2::ALL,
                },
            }],
        }],
    };
    let cohorts = CohortPlanV2 {
        phases: std::array::from_fn(|_| {
            (0..8)
                .map(|repetition| CohortV2 {
                    manifest_case: 0,
                    repetition,
                    requests: vec![CohortRequestV2 {
                        manifest_prompt: 0,
                        maximum_output: 3,
                    }],
                })
                .collect()
        }),
    };
    let payload = serde_json::json!({"fixture":"source3 wire", "output":3});
    let mut h = Header {
        artifact_type: "ferrum.structured-live-source".into(),
        schema_version: 3,
        model_revision: MODEL_REVISION_V2.into(),
        capture_identity: [10; 32],
        protocol: [0; 32],
        declared_protocol: [11; 32],
        rule_signature: rule.signature().unwrap(),
        fingerprint: ProfileFingerprint::from(&fingerprint()),
        producer: serde_json::json!({"executable_path":"fixture","executable_sha256":"a".repeat(64),"executable_bytes":1,"package_version":"fixture","source_revision":null}),
        opening: PairedClock {
            wall_unix_ns: 1_000_000,
            monotonic_ns: 1,
        },
        opened_at_ns: 1,
        initial_fifo_cutoff: 0,
        scope,
        membership_rule: rule,
        cohort_manifest_sha256: cohorts.signature(&payload).unwrap(),
        cohort_plan: cohorts,
        cohort_manifest_payload: payload,
        phase_members: [8; 3],
        maximum_offered_waves: 128,
        maximum_file_bytes: 16 * 1024 * 1024,
        settings: Settings {
            min_samples: 8,
            redundancy: 4,
            max_phase_samples: 32,
            max_axes: 512,
            max_rank: 16,
            max_wave_ns: 1_000_000,
            max_age_ns: 1_000_000_000,
            margin_ns: 10,
        },
    };
    let mut sha = Sha256::new();
    sha.update(b"ferrum.structured-live-source.v2\0");
    sha.update(MODEL_REVISION_V2.as_bytes());
    sha.update(h.declared_protocol);
    sha.update(h.rule_signature);
    sha.update(h.cohort_manifest_sha256);
    sha.update(serde_json::to_vec(&h.scope).unwrap());
    for n in h.phase_members.into_iter().map(|v| v as u64).chain([
        h.settings.min_samples as u64,
        h.settings.redundancy as u64,
        h.settings.max_phase_samples as u64,
        h.settings.max_axes as u64,
        h.settings.max_rank as u64,
        h.settings.max_wave_ns,
        h.settings.max_age_ns,
        h.settings.margin_ns,
        h.maximum_offered_waves as u64,
        h.maximum_file_bytes,
    ]) {
        sha.update(n.to_le_bytes());
    }
    h.protocol = sha.finalize().into();
    h
}
fn stages(h: &Header, p: &Prepared, call: u64, wall_ns: u64) -> Stages {
    let start = call * 2000;
    let last = p.rows[0].frontier.generated_before == 2;
    let mut s = Stages {
        schema_version: 1,
        call_id: call,
        presubmit_prediction: None,
        fingerprint: Some(h.fingerprint.clone()),
        actual_shape: Some(p.exact.clone()),
        statistical_evidence: Some(p.selected.clone()),
        structured_evidence: None,
        prepare_started_at_ns: Some(start),
        executor_returned_at_ns: Some(start + 500),
        rows: vec![StageRow {
            request_id: p.rows[0].request_id.clone(),
            owner_incarnation: 1,
            work_generation: p.rows[0].work_generation,
            input_index: 0,
            actual_work: match p.rows[0].frontier.work {
                PreparedWorkV2::Decode { kv_tokens } => RowWork::Decode { kv_tokens },
                PreparedWorkV2::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => RowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                },
            },
            host_processing_ordinal: Some(0),
            host_started_at_ns: Some(start + 600),
            token_committed_at_ns: Some(start + 700),
            output_published_at_ns: Some(start + 800),
            completion_started_at_ns: if last { Some(start + 900) } else { None },
            settled_at_ns: Some(start + wall_ns),
            terminal: if last {
                Some(Terminal {
                    finish_reason: ferrum_types::FinishReason::Length,
                    generated_tokens: 3,
                    through_output_ordinal: 3,
                    output_failed: false,
                    physical_failed: false,
                    scheduler_failed: false,
                    terminal_handoff_succeeded: true,
                    pending_restore_removed: false,
                    admission_cancellation_work: serde_json::json!("no_additional_work"),
                    cache_completion_work: serde_json::json!("no_additional_work"),
                    other_physical_resources: false,
                    request_slot_closed: true,
                    owner_matched: true,
                })
            } else {
                None
            },
            completeness: "complete_single_wave".into(),
        }],
        finalized_at_ns: Some(start + wall_ns + 100),
        full_wall_ns: Some(wall_ns),
        completeness: "complete_single_wave".into(),
    };
    let binding = observation::stage_binding(&s, None).unwrap();
    s.structured_evidence = Some(Ok(Settlement {
        protocol: "ferrum.structured-host-settled-capture.v1".into(),
        call_id: call,
        recipe: p.recipe.clone(),
        stage_binding: binding,
        executor_envelope_ns: 500,
        host_settled_after_executor_ns: wall_ns - 500,
        full_wall_ns: wall_ns,
    }));
    s
}
fn push(bytes: &mut Vec<u8>, ordinal: &mut u64, record: impl Serialize) {
    *ordinal += 1;
    serde_json::to_writer(
        &mut *bytes,
        &serde_json::json!({"source_record_ordinal":*ordinal,"record":record}),
    )
    .unwrap();
    bytes.push(b'\n');
}
pub fn source() -> (Vec<u8>, StructuredInputV2) {
    source_graph(None)
}
pub fn source_graph(resident: Option<&str>) -> (Vec<u8>, StructuredInputV2) {
    source_graph_fit_tail(resident, false)
}
pub fn source_with_fit_tail() -> (Vec<u8>, StructuredInputV2) {
    source_graph_fit_tail(None, true)
}
fn source_graph_fit_tail(resident: Option<&str>, fit_tail: bool) -> (Vec<u8>, StructuredInputV2) {
    let h = header_graph(resident);
    let mut bytes = Vec::new();
    let mut ordinal = 0;
    push(&mut bytes, &mut ordinal, &h);
    let source = StructuredSourceContractV2 {
        capture_identity: h.capture_identity,
        protocol: h.protocol,
        membership_rule: h.rule_signature,
        cohort_manifest: h.cohort_manifest_sha256,
        phase_members: h.phase_members,
    };
    let mut fit = None;
    let mut calibrated = None;
    let mut offers = 0u64;
    let mut members = 0u64;
    for phase in [
        StructuredProfilePhaseV10::Fit,
        StructuredProfilePhaseV10::Residual,
        StructuredProfilePhaseV10::Qualification,
    ] {
        let mut samples = Vec::new();
        for cohort in 0..8 {
            let id = format!("request-{}-{cohort}", phase.index());
            push(
                &mut bytes,
                &mut ordinal,
                Record::CohortBegin {
                    phase,
                    cohort,
                    manifest_case: 0,
                    repetition: cohort as u32,
                },
            );
            push(
                &mut bytes,
                &mut ordinal,
                Record::RequestAdmitted {
                    phase,
                    cohort,
                    slot: 0,
                    request_id: id.clone(),
                    maximum_output: 3,
                },
            );
            for generated in 0..3 {
                offers += 1;
                let (p, rows, input) = prepared_graph(&id, generated + 1, generated, resident);
                let wall_ns = if fit_tail
                    && phase == StructuredProfilePhaseV10::Fit
                    && cohort == 3
                    && generated == 1
                {
                    1600
                } else {
                    1000
                };
                let s = stages(&h, &p, offers, wall_ns);
                let member = if generated == 1 {
                    members += 1;
                    Some(members)
                } else {
                    None
                };
                push(
                    &mut bytes,
                    &mut ordinal,
                    Record::Offered {
                        offered: offers,
                        phase,
                        cohort,
                        rows,
                    },
                );
                push(
                    &mut bytes,
                    &mut ordinal,
                    Record::Reserved {
                        offered: offers,
                        member,
                        window: member.map(|_| 0),
                        phase,
                        cohort,
                        boundary: "prepared_before_execute".into(),
                        prepared: p.clone(),
                    },
                );
                let n = Numeric {
                    fifo: offers,
                    call_id: offers,
                    observed_at_ns: offers * 2000 + wall_ns + 100,
                    wall_ns,
                    domain: *input.domain_signature(),
                    basis: input.regression_axes().into(),
                    support: input.joint_support_coordinates().into(),
                };
                if let Some(member) = member {
                    samples.push(
                        observation::convert(
                            &h,
                            input,
                            &n,
                            phase,
                            member,
                            offers,
                            offers,
                            wall_ns,
                            n.observed_at_ns,
                        )
                        .unwrap(),
                    );
                }
                let outside = OutsideSettlement {
                    call_id: s.call_id,
                    fingerprint: s.fingerprint.clone(),
                    prepare_started_at_ns: s.prepare_started_at_ns,
                    executor_returned_at_ns: s.executor_returned_at_ns,
                    finalized_at_ns: s.finalized_at_ns,
                    full_wall_ns: s.full_wall_ns,
                    completeness: s.completeness.clone(),
                    rows: s.rows.clone(),
                    stage_binding: observation::stage_binding(&s, None).unwrap(),
                };
                push(
                    &mut bytes,
                    &mut ordinal,
                    Record::Completed {
                        offered: offers,
                        member,
                        phase,
                        cohort,
                        queue: Some(Queue {
                            accepted_ordinal: Some(offers),
                            disposition: "published".into(),
                        }),
                        reconciled: true,
                        host_stages: member.map(|_| s.clone()),
                        outside_settlement: if member.is_none() {
                            Some(outside)
                        } else {
                            None
                        },
                        selected_structured_capture: member.map(|_| Ok(p.recipe.clone())),
                        selected_independent_attention_v2: None,
                        numeric: member.map(|_| n),
                        conversion_error: None,
                    },
                );
                if let Some(t) = &s.rows[0].terminal {
                    push(
                        &mut bytes,
                        &mut ordinal,
                        Record::RequestCompleted {
                            request: CompletedRequest {
                                phase,
                                cohort,
                                slot: 0,
                                request_id: id.clone(),
                                owner_incarnation: 1,
                                call_id: offers,
                                fifo: offers,
                                generated_tokens: 3,
                                terminal: t.clone(),
                            },
                        },
                    );
                }
            }
            push(
                &mut bytes,
                &mut ordinal,
                Record::CohortEnd {
                    phase,
                    cohort,
                    admitted_count: 1,
                    completed_count: 1,
                },
            );
        }
        push(
            &mut bytes,
            &mut ordinal,
            Record::Coverage {
                phase,
                report: serde_json::to_value(h.scope.coverage_report(&samples).unwrap()).unwrap(),
            },
        );
        let frozen = offers * 2000 + 1200;
        let signature = match phase {
            StructuredProfilePhaseV10::Fit => {
                let m = FittedStructuredModelV2::fit(
                    fingerprint(),
                    h.settings.native(),
                    h.scope.clone(),
                    source.clone(),
                    &samples,
                    frozen,
                )
                .unwrap();
                let sig = m.parameters_signature();
                fit = Some(m);
                sig
            }
            StructuredProfilePhaseV10::Residual => {
                let m = fit.take().unwrap().calibrate(&samples, frozen).unwrap();
                let sig = m.parameters_signature();
                calibrated = Some(m);
                sig
            }
            StructuredProfilePhaseV10::Qualification => calibrated
                .take()
                .unwrap()
                .qualify(&samples, frozen)
                .unwrap()
                .parameters_signature(),
        };
        let receipt = Freeze {
            capture_identity: h.capture_identity,
            protocol: h.protocol,
            rule_signature: h.rule_signature,
            phase,
            accepted_fifo_cutoff: offers,
            member_cutoff: members,
            source_prefix_bytes: bytes.len() as u64,
            source_prefix_sha256: Sha256::digest(&bytes).into(),
            frozen_at_ns: frozen,
            parameters_sha256: signature,
        };
        push(&mut bytes, &mut ordinal, Record::PhaseFreeze { receipt });
    }
    let monotonic = offers * 2000 + 1300;
    push(
        &mut bytes,
        &mut ordinal,
        Record::Footer {
            phase: "qualified".into(),
            failure: None,
            offered: offers,
            members,
            failed_members: 0,
            accepted_fifo_cutoff: offers,
            last_captured_fifo: offers,
            fifo_audit_complete: true,
            closing: Some(PairedClock {
                wall_unix_ns: h.opening.wall_unix_ns + monotonic - 1,
                monotonic_ns: monotonic,
            }),
        },
    );
    (bytes, prepared_graph("query", 1, 1, resident).2)
}
