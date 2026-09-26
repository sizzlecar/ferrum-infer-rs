use super::super::super::tests::fixture;
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::windows::{
    ClosedRangeV2, WorkWindowV2,
};

fn limits() -> CostProfileLoadLimits {
    CostProfileLoadLimits::default()
}

fn resign_child(h: &mut Header) {
    h.rule_signature = h.membership_rule.signature().unwrap();
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
}

// Reuse the source3 fixture's canonical physical producer and full lifecycle.
// This is fixture construction, not the production replay strategy. A second
// owner selects original prefill waves that the first owner's source kept as
// outside settlements. The third (terminal) wave remains outside both owners.
fn source4() -> Vec<u8> {
    let (original, _) = fixture::source();
    let mut lines = original.split_inclusive(|b| *b == b'\n');
    let old: Header =
        serde_json::from_value(source_line(lines.next().unwrap()).unwrap().record).unwrap();
    let mut prefill = old.clone();
    prefill.capture_identity = [12; 32];
    prefill.declared_protocol = [13; 32];
    prefill.scope.owner = fixture::prepared("prefill", 1, 0).2.owner().clone();
    prefill.membership_rule.owner = prefill.scope.owner.clone();
    prefill.membership_rule.windows[0].rows[0].generated_before = ClosedRangeV2 {
        minimum: 0,
        maximum: 0,
    };
    prefill.membership_rule.windows[0].rows[0].work = WorkWindowV2::Prefill {
        offset: ClosedRangeV2::ALL,
        count: ClosedRangeV2::ALL,
        total_prompt_tokens: ClosedRangeV2::ALL,
        emits_token: Some(true),
    };
    prefill.settings.margin_ns = 30;
    resign_child(&mut prefill);
    let original_headers = [old, prefill];
    let header: HeaderV4 = serde_json::from_value(
        structured_shared_source_header_v4(
            original_headers
                .iter()
                .map(|h| serde_json::to_value(h).unwrap())
                .collect(),
            original_headers[0].maximum_file_bytes,
            8,
            128 * 1024 * 1024,
            1_048_576,
        )
        .unwrap(),
    )
    .unwrap();
    let mut bytes = Vec::new();
    let mut ordinal = 0;
    fixture::push(&mut bytes, &mut ordinal, &header);
    let mut members = [0u64; 2];
    let mut pending = None;
    let mut samples: [Vec<StructuredNumericObservationV2>; 2] = std::array::from_fn(|_| Vec::new());
    let mut fitted: [Option<FittedStructuredModelV2>; 2] = std::array::from_fn(|_| None);
    let mut calibrated: [Option<CalibratedStructuredModelV2>; 2] = std::array::from_fn(|_| None);
    for line in lines {
        let record: Record = serde_json::from_value(source_line(line).unwrap().record).unwrap();
        let record = match record {
            Record::Reserved {
                offered,
                phase,
                cohort,
                boundary,
                prepared,
                ..
            } => {
                let generated = prepared.rows[0].frontier.generated_before;
                let input =
                    fixture::prepared(&prepared.rows[0].request_id, generated + 1, generated).2;
                let current = std::array::from_fn::<_, 2, _>(|i| {
                    let selected = (i == 0 && generated == 1) || (i == 1 && generated == 0);
                    if selected {
                        members[i] += 1;
                        Some(members[i])
                    } else {
                        None
                    }
                });
                pending = Some((prepared.clone(), input, current));
                RecordV4::Reserved {
                    offered,
                    phase,
                    cohort,
                    boundary,
                    prepared,
                    memberships: current
                        .into_iter()
                        .map(|member| MembershipV4 {
                            window: member.map(|_| 0),
                            member,
                        })
                        .collect(),
                }
            }
            Record::Completed {
                offered,
                phase,
                cohort,
                queue,
                reconciled,
                conversion_error,
                ..
            } => {
                let (prepared, input, current) = pending.take().unwrap();
                let stages = fixture::stages(&original_headers[0], &prepared, offered, 1000);
                let numeric = Numeric {
                    fifo: offered,
                    call_id: offered,
                    observed_at_ns: offered * 2000 + 1100,
                    wall_ns: 1000,
                    domain: *input.domain_signature(),
                    basis: input.regression_axes().into(),
                    support: input.joint_support_coordinates().into(),
                };
                for i in 0..2 {
                    if let Some(member) = current[i] {
                        samples[i].push(
                            observation::convert(
                                &original_headers[i],
                                input.clone(),
                                &numeric,
                                phase,
                                member,
                                offered,
                                offered,
                                1000,
                                numeric.observed_at_ns,
                            )
                            .unwrap(),
                        );
                    }
                }
                let selected = current.iter().any(Option::is_some);
                let outside = (!selected).then(|| OutsideSettlement {
                    call_id: stages.call_id,
                    fingerprint: stages.fingerprint.clone(),
                    prepare_started_at_ns: stages.prepare_started_at_ns,
                    executor_returned_at_ns: stages.executor_returned_at_ns,
                    finalized_at_ns: stages.finalized_at_ns,
                    full_wall_ns: stages.full_wall_ns,
                    completeness: stages.completeness.clone(),
                    rows: stages.rows.clone(),
                    stage_binding: observation::stage_binding(&stages, None).unwrap(),
                });
                RecordV4::Completed {
                    offered,
                    phase,
                    cohort,
                    members: current.into(),
                    queue,
                    reconciled,
                    host_stages: selected.then_some(stages),
                    outside_settlement: outside,
                    selected_structured_capture: selected.then_some(Ok(prepared.recipe)),
                    selected_independent_attention_v2: None,
                    numeric: selected.then_some(numeric),
                    conversion_error,
                }
            }
            Record::Coverage { phase, .. } => RecordV4::Coverage {
                phase,
                reports: header
                    .children
                    .iter()
                    .zip(&samples)
                    .map(|(h, s)| {
                        serde_json::to_value(h.scope.coverage_report(s).unwrap()).unwrap()
                    })
                    .collect(),
            },
            Record::PhaseFreeze { receipt } => {
                let prefix = Sha256::digest(&bytes).into();
                let mut receipts = Vec::new();
                for i in 0..2 {
                    let h = &original_headers[i];
                    let signature = match receipt.phase {
                        StructuredProfilePhaseV10::Fit => {
                            let m = FittedStructuredModelV2::fit(
                                fixture::fingerprint(),
                                h.settings.native(),
                                h.scope.clone(),
                                StructuredSourceContractV2 {
                                    capture_identity: h.capture_identity,
                                    protocol: h.protocol,
                                    membership_rule: h.rule_signature,
                                    cohort_manifest: h.cohort_manifest_sha256,
                                    phase_members: h.phase_members,
                                },
                                &samples[i],
                                receipt.frozen_at_ns,
                            )
                            .unwrap();
                            let sig = m.parameters_signature();
                            fitted[i] = Some(m);
                            sig
                        }
                        StructuredProfilePhaseV10::Residual => {
                            let m = fitted[i]
                                .take()
                                .unwrap()
                                .calibrate(&samples[i], receipt.frozen_at_ns)
                                .unwrap();
                            let sig = m.parameters_signature();
                            calibrated[i] = Some(m);
                            sig
                        }
                        StructuredProfilePhaseV10::Qualification => calibrated[i]
                            .take()
                            .unwrap()
                            .qualify(&samples[i], receipt.frozen_at_ns)
                            .unwrap()
                            .parameters_signature(),
                    };
                    receipts.push(Freeze {
                        capture_identity: h.capture_identity,
                        protocol: h.protocol,
                        rule_signature: h.rule_signature,
                        phase: receipt.phase,
                        accepted_fifo_cutoff: receipt.accepted_fifo_cutoff,
                        member_cutoff: members[i],
                        source_prefix_bytes: bytes.len() as u64,
                        source_prefix_sha256: prefix,
                        frozen_at_ns: receipt.frozen_at_ns,
                        parameters_sha256: signature,
                    });
                    samples[i].clear();
                }
                RecordV4::PhaseFreeze { receipts }
            }
            Record::Footer {
                phase,
                failure,
                offered,
                accepted_fifo_cutoff,
                last_captured_fifo,
                fifo_audit_complete,
                closing,
                ..
            } => RecordV4::Footer {
                phase,
                failure,
                offered,
                members: members.into(),
                failed_members: vec![0; 2],
                accepted_fifo_cutoff,
                last_captured_fifo,
                fifo_audit_complete,
                closing,
            },
            record => RecordV4::Common { record },
        };
        fixture::push(&mut bytes, &mut ordinal, record);
    }
    bytes
}

fn mutate(bytes: &[u8], mut change: impl FnMut(&mut serde_json::Value)) -> Vec<u8> {
    let mut result = Vec::new();
    for line in bytes.split_inclusive(|b| *b == b'\n') {
        let mut line: serde_json::Value = serde_json::from_slice(line).unwrap();
        change(&mut line["record"]);
        serde_json::to_writer(&mut result, &line).unwrap();
        result.push(b'\n');
    }
    result
}

#[test]
fn shared_source4_two_owners_replay_each_original_physical_wave_once() {
    let bytes = source4();
    let mut physical = 0;
    let replayed = replay_with_observer(&bytes, &limits(), || physical += 1).unwrap();
    assert_eq!(physical, 72);
    assert_eq!(replayed.children.len(), 2);
    assert_ne!(
        replayed.children[0].header.scope.owner,
        replayed.children[1].header.scope.owner
    );
    assert!(Arc::ptr_eq(
        &replayed.children[0].header.common,
        &replayed.children[1].header.common
    ));
    for child in &replayed.children {
        assert_eq!(child.offered_attempts, physical);
        assert_eq!(child.total_shape_rows, physical);
        assert_eq!(child.reserved_members, 24);
        assert_eq!(child.phases.each_ref().map(|p| p.members), [8; 3]);
        assert_eq!(child.phases[2].accepted_fifo_cutoff, 72);
        super::super::super::clock::validate_source(
            super::super::profile::clock_evidence(child),
            0,
        )
        .unwrap();
    }
    for phase in 0..3 {
        assert_eq!(
            replayed.children[0].phases[phase].source_prefix_sha256,
            replayed.children[1].phases[phase].source_prefix_sha256
        );
        assert_ne!(
            replayed.children[0].phases[phase].parameters_sha256,
            replayed.children[1].phases[phase].parameters_sha256
        );
    }
    let (old, _) = fixture::source();
    let source3 = super::super::super::replay::replay_source(&old, &limits()).unwrap();
    assert_eq!(
        source3.model.parameters_signature(),
        replayed.children[0].model.parameters_signature()
    );
    assert!(replay_source(&old, &limits()).is_err());
    assert!(super::super::super::replay::replay_source(&bytes, &limits()).is_err());
}

#[test]
fn shared_source4_rejects_membership_sidecar_freeze_fifo_and_cancellation_tampering() {
    let bytes = source4();
    for case in 0..10 {
        let changed = mutate(&bytes, |r| match case {
            0 if r["kind"] == "reserved" => {
                r["memberships"].as_array_mut().unwrap().pop();
            }
            1 if r["kind"] == "reserved" => {
                r["memberships"][0]["window"] = 0.into();
            }
            2 if r["kind"] == "reserved" => {
                r["prepared"]
                    .as_object_mut()
                    .unwrap()
                    .remove("selected_independent_attention_v2");
            }
            3 if r["kind"] == "completed" => {
                r["queue"]["accepted_ordinal"] = 99.into();
            }
            4 if r["kind"] == "completed" => {
                r.as_object_mut()
                    .unwrap()
                    .remove("selected_independent_attention_v2");
            }
            5 if r["kind"] == "phase_freeze" => {
                r["receipts"].as_array_mut().unwrap().pop();
            }
            6 if r["kind"] == "phase_freeze" => {
                r["receipts"][1]["parameters_sha256"][0] = 99.into();
            }
            7 if r["kind"] == "completed" && !r["outside_settlement"].is_null() => {
                r["outside_settlement"]["rows"][0]["terminal"]["finish_reason"] =
                    "cancelled".into();
            }
            8 if r["kind"] == "footer" => {
                r["failed_members"][1] = 1.into();
            }
            9 if r["kind"] == "common" && r["record"]["kind"] == "request_completed" => {
                r["record"]["request"]["generated_tokens"] = 2.into();
            }
            _ => {}
        });
        assert!(replay_source(&changed, &limits()).is_err(), "case {case}");
    }
}

#[test]
fn shared_source4_limits_are_common_bytes_and_rows_but_aggregate_child_samples() {
    let bytes = source4();
    let mut exact = limits();
    exact.max_samples = NonZeroUsize::new(48).unwrap();
    exact.max_total_shape_rows = NonZeroUsize::new(72).unwrap();
    assert!(replay_source(&bytes, &exact).is_ok());
    let mut short = exact.clone();
    short.max_samples = NonZeroUsize::new(47).unwrap();
    assert!(matches!(
        replay_source(&bytes, &short),
        Err(CostProfileError::Limit(_))
    ));
    let mut short = exact.clone();
    short.max_total_shape_rows = NonZeroUsize::new(71).unwrap();
    assert!(replay_source(&bytes, &short).is_err());
    let mut short = exact;
    short.max_file_bytes = NonZeroUsize::new(bytes.len() - 1).unwrap();
    assert!(replay_source(&bytes, &short).is_err());
}

#[test]
fn shared_source4_declarations_bind_common_clock_and_aggregate_memory() {
    let bytes = source4();
    let first = bytes.split_inclusive(|b| *b == b'\n').next().unwrap();
    let original: HeaderV4 = serde_json::from_value(source_line(first).unwrap().record).unwrap();
    for case in 0..7 {
        let mut h = original.clone();
        match case {
            0 => h.common.opening.monotonic_ns = 0,
            1 => h.common.opening.wall_unix_ns = 0,
            2 => h.children[1].opened_at_ns = 2,
            3 => h.common.cohort_manifest_sha256 = [0; 32],
            4 => h.maximum_retained_numeric_bytes = 1,
            5 => h.maximum_retained_coordinates = 1,
            6 => h.children[1] = h.children[0].clone(),
            _ => unreachable!(),
        }
        h.capture_protocol = h.signature().unwrap();
        assert!(
            validate_header(&h, bytes.len(), &limits()).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn shared_source4_one_child_underestimate_rejects_the_complete_import() {
    let original = source4();
    let mut changed = Vec::new();
    let mut modified = false;
    for line in original.split_inclusive(|b| *b == b'\n') {
        let mut v: serde_json::Value = serde_json::from_slice(line).unwrap();
        let r = &mut v["record"];
        if !modified
            && r["kind"] == "completed"
            && r["phase"] == "qualification"
            && !r["members"][1].is_null()
        {
            let mut stages: Stages = serde_json::from_value(r["host_stages"].clone()).unwrap();
            *stages.rows[0].settled_at_ns.as_mut().unwrap() += 200;
            *stages.finalized_at_ns.as_mut().unwrap() += 200;
            *stages.full_wall_ns.as_mut().unwrap() += 200;
            let binding = observation::stage_binding(&stages, None).unwrap();
            let settled = stages
                .structured_evidence
                .as_mut()
                .unwrap()
                .as_mut()
                .unwrap();
            settled.stage_binding = binding;
            settled.host_settled_after_executor_ns += 200;
            settled.full_wall_ns += 200;
            r["host_stages"] = serde_json::to_value(stages).unwrap();
            r["numeric"]["wall_ns"] = 1200.into();
            let observed = r["numeric"]["observed_at_ns"].as_u64().unwrap() + 200;
            r["numeric"]["observed_at_ns"] = observed.into();
            modified = true;
        }
        if r["kind"] == "phase_freeze" {
            for receipt in r["receipts"].as_array_mut().unwrap() {
                receipt["source_prefix_bytes"] = (changed.len() as u64).into();
                receipt["source_prefix_sha256"] =
                    serde_json::to_value(<[u8; 32]>::from(Sha256::digest(&changed))).unwrap();
            }
        }
        serde_json::to_writer(&mut changed, &v).unwrap();
        changed.push(b'\n');
    }
    assert!(modified);
    assert!(matches!(
        replay_source(&changed, &limits()),
        Err(CostProfileError::Metadata(
            "invalid structured numerical replay"
        ))
    ));
}

struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("ferrum-shared-source4-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn shared_profile11_roundtrip_keeps_original_age_and_all_child_identities() {
    let files = Files::new();
    let source = files.0.join("source4.jsonl");
    let profile = files.0.join("profile11.json");
    let bytes = source4();
    std::fs::write(&source, &bytes).unwrap();
    let receipt = export_structured_profile_v11(
        &source,
        Sha256::digest(&bytes).into(),
        &profile,
        &[0, 0],
        &limits(),
    )
    .unwrap();
    let clock = ProfileLoadClock {
        wall_unix_ns: Some(1_000_000 + 72 * 2000 + 1299 + 100),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 7,
    };
    let loaded =
        load_structured_profile_v11(&profile, &fixture::fingerprint(), &limits(), clock).unwrap();
    assert_eq!(loaded.children.len(), 2);
    assert_eq!(loaded.source_bytes, bytes.len() as u64);
    assert_eq!(loaded.total_shape_rows, 72);
    assert_eq!(loaded.offered_attempts, 72);
    assert_eq!(loaded.source_sha256, receipt.source_sha256);
    assert_eq!(loaded.capture_protocol, receipt.capture_protocol);
    for (model, child) in loaded.children.iter().zip(&receipt.children) {
        assert_eq!(model.provenance().schema_version, 11);
        assert_eq!(model.parameters_signature(), child.parameters_sha256);
        assert_eq!(model.model_now_ns(7).unwrap(), 72 * 2000 + 1400);
    }
    assert_eq!(
        loaded.children[1].provenance().oldest_imported_age_ns,
        loaded.children[0].provenance().oldest_imported_age_ns + 2000
    );
    let stale = ProfileLoadClock {
        wall_unix_ns: clock.wall_unix_ns.map(|t| t + 1_000_000_001),
        ..clock
    };
    assert!(matches!(
        load_structured_profile_v11(&profile, &fixture::fingerprint(), &limits(), stale),
        Err(CostProfileError::Clock(_))
    ));
    assert!(export_structured_profile_v11(
        &source,
        receipt.source_sha256,
        &profile,
        &[0, 0],
        &limits()
    )
    .is_err());
    let profile_bytes = std::fs::read(&profile).unwrap();
    let mut altered: serde_json::Value = serde_json::from_slice(&profile_bytes).unwrap();
    altered["children"][1]["parameters_sha256"][0] = 99.into();
    std::fs::write(&profile, serde_json::to_vec(&altered).unwrap()).unwrap();
    assert!(
        load_structured_profile_v11(&profile, &fixture::fingerprint(), &limits(), clock).is_err()
    );
    std::fs::write(&profile, profile_bytes).unwrap();
    let mut changed = bytes;
    changed.push(b'\n');
    std::fs::write(&source, changed).unwrap();
    assert!(
        load_structured_profile_v11(&profile, &fixture::fingerprint(), &limits(), clock).is_err()
    );
}
