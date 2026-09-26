//! Canonical numerical wire fixtures, never live engine execution authority.
use super::super::super::tests::fixture;
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::prefixes::{
    StructuredPrefixCohortV5, StructuredPrefixSlotV5,
};
use ferrum_types::TokenId;
const POLICY: [u8; 32] = [44; 32];

fn source5() -> Vec<u8> {
    let (original, _) = fixture::source_policy(POLICY);
    let mut lines = original.split_inclusive(|b| *b == b'\n');
    let first: serde_json::Value = serde_json::from_slice(lines.next().unwrap()).unwrap();
    let h: Header = serde_json::from_value(first["record"].clone()).unwrap();
    let plan = StructuredPrefixPlanV5 {
        phases: std::array::from_fn(|phase| {
            h.cohort_plan.phases[phase]
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    (i != 0).then(|| StructuredPrefixCohortV5 {
                        release_generated: 1,
                        slots: vec![StructuredPrefixSlotV5 {
                            tokenizer_policy_sha256: [7; 32],
                            token_ids: vec![TokenId::new(7)],
                            token_bytes: vec![b"a".to_vec()],
                        }],
                    })
                })
                .collect()
        }),
    };
    let header = structured_prefix_source_header_v5(
        vec![serde_json::to_value(&h).unwrap()],
        h.maximum_file_bytes,
        8,
        128 * 1024 * 1024,
        1_048_576,
        plan,
    )
    .unwrap();
    let mut bytes = Vec::new();
    let mut ordinal = 0;
    fixture::push(&mut bytes, &mut ordinal, header);
    let mut preparation = false;
    let mut before = None;
    let mut prepared = None;
    for line in lines {
        let wrapper: serde_json::Value = serde_json::from_slice(line).unwrap();
        let record: Record = serde_json::from_value(wrapper["record"].clone()).unwrap();
        let out = match record {
            Record::Offered {
                offered,
                phase,
                cohort,
                rows,
            } => {
                preparation = cohort != 0 && rows[0].generated == 0;
                if preparation {
                    let row = &rows[0];
                    let f = Frontier {
                        request_id: row.request_id.clone(),
                        owner_incarnation: row.owner,
                        work_generation: row.generation,
                        generated_tokens: 0,
                        kv_tokens: 0,
                        model_cache_id: None,
                        pending_utf8: Vec::new(),
                        output_accepted_ordinal: 0,
                    };
                    before = Some(f.clone());
                    serde_json::json!({"kind":"preparation_offered","offered":offered,"phase":phase,"cohort":cohort,
                        "rows":[Offered{before:f,work:row.work.native()}]})
                } else {
                    serde_json::to_value(RecordV4::Common {
                        record: Record::Offered {
                            offered,
                            phase,
                            cohort,
                            rows,
                        },
                    })
                    .unwrap()
                }
            }
            Record::Reserved {
                offered,
                member,
                window,
                phase,
                cohort,
                boundary,
                prepared: p,
            } => {
                if preparation {
                    prepared = Some(p);
                    continue;
                }
                serde_json::to_value(RecordV4::Reserved {
                    offered,
                    phase,
                    cohort,
                    boundary,
                    prepared: p,
                    memberships: vec![MembershipV4 { member, window }],
                })
                .unwrap()
            }
            Record::Completed {
                offered,
                member,
                phase,
                cohort,
                queue,
                reconciled,
                host_stages,
                outside_settlement,
                selected_structured_capture,
                selected_independent_attention_v2,
                numeric,
                conversion_error,
            } => {
                if preparation {
                    let p = prepared.take().unwrap();
                    let before = before.take().unwrap();
                    let after = Frontier {
                        request_id: before.request_id.clone(),
                        owner_incarnation: before.owner_incarnation,
                        work_generation: before.work_generation + 1,
                        generated_tokens: 1,
                        kv_tokens: 64,
                        model_cache_id: Some("fixture-cache".into()),
                        pending_utf8: Vec::new(),
                        output_accepted_ordinal: 1,
                    };
                    let commit = Commit {
                        request_id: before.request_id.clone(),
                        owner_incarnation: before.owner_incarnation,
                        work_generation: before.work_generation,
                        generated_before: 0,
                        generated_after: 1,
                        original_candidate: TokenId::new(9),
                        committed_token: TokenId::new(7),
                        route: Route::FullLogitsSampler,
                        pending_before: Vec::new(),
                        pending_after: Vec::new(),
                    };
                    let mut stages = fixture::stages(&h, &p, offered, 1000);
                    stages.statistical_evidence = None;
                    stages.structured_evidence = None;
                    fixture::push(
                        &mut bytes,
                        &mut ordinal,
                        serde_json::json!({"kind":"preparation_completed","offered":offered,
                        "phase":phase,"cohort":cohort,"queue":queue,"reconciled":true,"host_stages":stages,
                        "rows":[Completed{before,after:Some(after.clone()),preparation_commit:Some(commit)}],"failure":null}),
                    );
                    let mut hash = Sha256::new();
                    hash.update(b"ferrum.calibration.generated-prefix.v1\0");
                    hash.update(7u32.to_le_bytes());
                    let released = Released {
                        frontier: after,
                        original_policy_signature: POLICY,
                        original_numeric_policy: p.recipe.physical_host_rows[0].installed_policy,
                        generated_prefix_sha256: hash.finalize().into(),
                        through_call_id: offered,
                        through_fifo_ordinal: offered,
                        actor_applied_output_ordinal: 1,
                    };
                    serde_json::json!({"kind":"preparation_released","phase":phase,"cohort":cohort,"slot":0,"receipt":released})
                } else {
                    serde_json::to_value(RecordV4::Completed {
                        offered,
                        phase,
                        cohort,
                        members: vec![member],
                        queue,
                        reconciled,
                        host_stages,
                        outside_settlement,
                        selected_structured_capture,
                        selected_independent_attention_v2,
                        numeric,
                        conversion_error,
                    })
                    .unwrap()
                }
            }
            Record::Coverage { phase, report } => serde_json::to_value(RecordV4::Coverage {
                phase,
                reports: vec![report],
            })
            .unwrap(),
            Record::PhaseFreeze { mut receipt } => {
                receipt.source_prefix_bytes = bytes.len() as u64;
                receipt.source_prefix_sha256 = Sha256::digest(&bytes).into();
                serde_json::to_value(RecordV4::PhaseFreeze {
                    receipts: vec![receipt],
                })
                .unwrap()
            }
            Record::Footer {
                phase,
                failure,
                offered,
                members,
                failed_members,
                accepted_fifo_cutoff,
                last_captured_fifo,
                fifo_audit_complete,
                closing,
            } => serde_json::to_value(RecordV4::Footer {
                phase,
                failure,
                offered,
                members: vec![members],
                failed_members: vec![failed_members],
                accepted_fifo_cutoff,
                last_captured_fifo,
                fifo_audit_complete,
                closing,
            })
            .unwrap(),
            record => serde_json::to_value(RecordV4::Common { record }).unwrap(),
        };
        fixture::push(&mut bytes, &mut ordinal, out);
    }
    bytes
}

fn changed(bytes: &[u8], mut edit: impl FnMut(&mut serde_json::Value) -> bool) -> Vec<u8> {
    let mut output = Vec::new();
    let mut ordinal = 0;
    let mut done = false;
    for line in bytes.split_inclusive(|b| *b == b'\n') {
        let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
        if !done {
            done = edit(&mut value["record"]);
        }
        if value["record"]["kind"] == "phase_freeze" {
            let digest: [u8; 32] = Sha256::digest(&output).into();
            for receipt in value["record"]["receipts"].as_array_mut().unwrap() {
                receipt["source_prefix_bytes"] = (output.len() as u64).into();
                receipt["source_prefix_sha256"] = serde_json::to_value(digest).unwrap();
            }
        }
        fixture::push(&mut output, &mut ordinal, &value["record"]);
    }
    assert!(done);
    output
}

#[test]
fn source5_single_pass_keeps_prefix_rows_fifo_length_and_strict_identity() {
    let bytes = source5();
    let limits = CostProfileLoadLimits::default();
    let mut physical = 0;
    let result = replay::replay_driver(&bytes, &limits, || physical += 1, true).unwrap();
    assert_eq!(physical, 72);
    assert_eq!(result.children.len(), 1);
    let child = &result.children[0];
    assert_eq!(child.total_shape_rows, 72);
    assert_eq!(child.offered_attempts, 72);
    assert_eq!(child.reserved_members, 24);
    assert_eq!(
        child.phases.iter().map(|p| p.members).collect::<Vec<_>>(),
        [8, 8, 8]
    );
    assert!(replay::replay_source(&bytes, &limits).is_err());
    assert!(super::super::super::replay::replay_source(&bytes, &limits).is_err());
    let mut short = limits.clone();
    short.max_total_shape_rows = std::num::NonZeroUsize::new(71).unwrap();
    assert!(replay::replay_prefix_source(&bytes, &short).is_err());
}

#[test]
fn source5_tampered_preparation_or_release_cannot_be_repaired_by_suffix() {
    let bytes = source5();
    let limits = CostProfileLoadLimits::default();
    for field in [
        "reconciled",
        "commit",
        "generation",
        "kv",
        "fifo",
        "clock",
        "pending",
        "cancel",
        "physical_work",
    ] {
        let modified = changed(&bytes, |r| {
            if r["kind"] != "preparation_completed" {
                return false;
            }
            match field {
                "reconciled" => r["reconciled"] = false.into(),
                "commit" => r["rows"][0]["preparation_commit"]["committed_token"] = 8.into(),
                "generation" => r["rows"][0]["after"]["work_generation"] = 99.into(),
                "kv" => r["rows"][0]["after"]["kv_tokens"] = 63.into(),
                "fifo" => r["queue"]["accepted_ordinal"] = 999.into(),
                "clock" => r["host_stages"]["executor_returned_at_ns"] = 0.into(),
                "pending" => r["rows"][0]["after"]["pending_utf8"] = serde_json::json!([195]),
                "cancel" => r["failure"] = "cancelled".into(),
                "physical_work" => r["host_stages"]["rows"][0]["actual_work"]["count"] = 63.into(),
                _ => unreachable!(),
            }
            true
        });
        assert!(
            replay::replay_prefix_source(&modified, &limits).is_err(),
            "{field}"
        );
    }
    for field in [
        "policy",
        "numeric",
        "actor",
        "digest",
        "call",
        "slot",
        "missing_release",
    ] {
        let modified = changed(&bytes, |r| {
            if r["kind"] != "preparation_released" {
                return false;
            }
            match field {
                "policy" => r["receipt"]["original_policy_signature"][0] = 9.into(),
                "numeric" => {
                    r["receipt"]["original_numeric_policy"]["raw_token_bytes_bound"] = 99.into()
                }
                "actor" => r["receipt"]["actor_applied_output_ordinal"] = 0.into(),
                "digest" => r["receipt"]["generated_prefix_sha256"][0] = 9.into(),
                "call" => r["receipt"]["through_call_id"] = 99.into(),
                "slot" => r["slot"] = 1.into(),
                "missing_release" => {
                    *r = serde_json::json!({"kind":"common","record":{"kind":"preparation_unavailable",
                    "offered":4,"phase":"fit","cohort":1,"reason":"missing release"}})
                }
                _ => unreachable!(),
            }
            true
        });
        assert!(
            replay::replay_prefix_source(&modified, &limits).is_err(),
            "{field}"
        );
    }
}

struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let p =
            std::env::temp_dir().join(format!("ferrum-prefix-source5-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn profile12_roundtrip_keeps_original_clock_and_does_not_authorize_other_schemas() {
    let files = Files::new();
    let source = files.0.join("source5.jsonl");
    let profile = files.0.join("profile12.json");
    let bytes = source5();
    std::fs::write(&source, &bytes).unwrap();
    let limits = CostProfileLoadLimits::default();
    let receipt = export_structured_profile_v12(
        &source,
        Sha256::digest(&bytes).into(),
        &profile,
        &[0],
        &limits,
    )
    .unwrap();
    let clock = ProfileLoadClock {
        wall_unix_ns: Some(1_000_000 + 72 * 2000 + 1299 + 100),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 7,
    };
    let loaded =
        load_structured_profile_v12(&profile, &fixture::fingerprint(), &limits, clock).unwrap();
    assert_eq!(loaded.source_sha256, receipt.source_sha256);
    assert_eq!(loaded.total_shape_rows, 72);
    let model = &loaded.children[0];
    assert_eq!(model.provenance().schema_version, 12);
    assert_eq!(model.model_now_ns(7).unwrap(), 72 * 2000 + 1400);
    let input = fixture::prepared("query", 2, 1).2;
    assert!(model
        .predict_query_local(&fixture::fingerprint(), &StructuredQueryV2::exact(input), 7)
        .is_ok());
    let stale = ProfileLoadClock {
        wall_unix_ns: clock.wall_unix_ns.map(|v| v + 1_000_000_001),
        ..clock
    };
    assert!(matches!(
        load_structured_profile_v12(&profile, &fixture::fingerprint(), &limits, stale),
        Err(CostProfileError::Clock(_))
    ));
    assert!(
        load_structured_profile_v11(&profile, &fixture::fingerprint(), &limits, clock).is_err()
    );
    assert!(export_structured_profile_v11(
        &source,
        receipt.source_sha256,
        &files.0.join("wrong11.json"),
        &[0],
        &limits
    )
    .is_err());
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&profile).unwrap()).unwrap();
    metadata["children"][0]["parameters_sha256"][0] = 99.into();
    std::fs::write(&profile, serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert!(
        load_structured_profile_v12(&profile, &fixture::fingerprint(), &limits, clock).is_err()
    );
}

#[test]
fn source5_joint_release_tracks_partial_prefill_utf8_and_per_slot_fifo() {
    let bytes = source5();
    let value: serde_json::Value =
        serde_json::from_slice(bytes.split(|b| *b == b'\n').next().unwrap()).unwrap();
    let (mut header, mut plan) = header(value["record"].clone()).unwrap();
    let second = header.common.cohort_plan.phases[0][0].requests[0].clone();
    header.common.cohort_plan.phases[0][0].requests.push(second);
    plan.phases[0][0] = Some(StructuredPrefixCohortV5 {
        release_generated: 1,
        slots: vec![
            StructuredPrefixSlotV5 {
                tokenizer_policy_sha256: [7; 32],
                token_ids: vec![TokenId::new(7)],
                token_bytes: vec![vec![0xc3]],
            };
            2
        ],
    });
    plan.validate(&header.common.cohort_plan).unwrap();
    let common = header.common;
    let mut lifecycle = super::super::super::lifecycle::Lifecycle::new(common.cohort_plan.clone());
    let mut preparation = Preparation::new(plan);
    lifecycle.begin(0, 0, 0, 0).unwrap();
    preparation.begin(0, 0, &common.cohort_plan).unwrap();
    let mut frontiers = (0..2)
        .map(|slot| {
            let id = format!("joint-{slot}");
            lifecycle.admit(0, 0, slot, id.clone(), 3, 256).unwrap();
            preparation.admit(slot, &id).unwrap();
            Frontier {
                request_id: id,
                owner_incarnation: slot as u64 + 1,
                work_generation: 1,
                generated_tokens: 0,
                kv_tokens: 0,
                model_cache_id: None,
                pending_utf8: vec![],
                output_accepted_ordinal: 0,
            }
        })
        .collect::<Vec<_>>();
    let limits = CostProfileLoadLimits::default();
    let (mut offered, mut fifo, mut finalized, mut total) = (0, 0, 1, 0);
    let mut calls = HashSet::new();
    let mut progress = Progress {
        phase: 0,
        offered: &mut offered,
        maximum_offered: 128,
        last_fifo: &mut fifo,
        last_finalized: &mut finalized,
        earliest: 1,
        calls: &mut calls,
        total_rows: &mut total,
        common: &common,
        limits: &limits,
        lifecycle: &mut lifecycle,
    };
    assert!(preparation.ready().is_err());
    let policy = fixture::prepared("policy", 1, 0)
        .0
        .recipe
        .physical_host_rows[0]
        .installed_policy;
    let mut hash = Sha256::new();
    hash.update(b"ferrum.calibration.generated-prefix.v1\0");
    hash.update(7u32.to_le_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    // The first physical wave advances both KV frontiers without a token. The
    // next two waves finish the two slots independently, in a different order.
    for (call, slots) in [(1u64, vec![1usize, 0]), (2, vec![0]), (3, vec![1])] {
        let rows = slots
            .iter()
            .map(|&slot| Offered {
                before: frontiers[slot].clone(),
                work: PreparedWorkV2::Prefill {
                    offset: frontiers[slot].kv_tokens as u32,
                    count: 32,
                    total_prompt_tokens: 64,
                },
            })
            .collect::<Vec<_>>();
        preparation.handle(serde_json::json!({"kind":"preparation_offered","offered":call,"phase":"fit","cohort":0,"rows":rows}), &mut progress).unwrap();
        assert!(preparation.ready().is_err());
        let (p, _, _) = fixture::prepared("stage", 1, 0);
        let mut stages = fixture::stages(&fixture::header(), &p, call, 1000);
        stages.statistical_evidence = None;
        stages.structured_evidence = None;
        let template = stages.rows[0].clone();
        stages.rows = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let mut actual = template.clone();
                actual.request_id = row.before.request_id.clone();
                actual.owner_incarnation = row.before.owner_incarnation;
                actual.work_generation = row.before.work_generation;
                actual.input_index = index as u32;
                actual.host_processing_ordinal = Some(index as u32);
                actual.actual_work = RowWork::Prefill {
                    offset: row.before.kv_tokens as u32,
                    count: 32,
                    total_prompt_tokens: 64,
                };
                let start = call * 2000 + 200 + index as u64 * 300;
                actual.host_started_at_ns = Some(start);
                actual.token_committed_at_ns = Some(start + 50);
                actual.output_published_at_ns = (call != 1).then_some(start + 60);
                actual.settled_at_ns = Some(start + 100);
                actual.completion_started_at_ns = None;
                actual.terminal = None;
                actual
            })
            .collect();
        stages.executor_returned_at_ns = Some(call * 2000 + 100);
        stages.full_wall_ns =
            Some(stages.rows.last().unwrap().settled_at_ns.unwrap() - call * 2000);
        let shape = &mut stages.actual_shape.as_mut().unwrap().exact;
        shape.prefill_chunks = rows
            .iter()
            .map(|row| ProfilePrefillShape {
                offset: row.before.kv_tokens as u32,
                count: std::num::NonZeroU32::new(32).unwrap(),
                total_prompt_tokens: std::num::NonZeroU32::new(64).unwrap(),
            })
            .collect();
        let completed = slots
            .iter()
            .zip(&rows)
            .map(|(&slot, row)| {
                let emits = call != 1;
                let mut after = row.before.clone();
                after.work_generation += 1;
                after.kv_tokens += 32;
                after.model_cache_id = Some(format!("joint-cache-{slot}"));
                let commit = emits.then(|| {
                    after.generated_tokens += 1;
                    after.output_accepted_ordinal += 1;
                    after.pending_utf8 = vec![0xc3];
                    Commit {
                        request_id: row.before.request_id.clone(),
                        owner_incarnation: row.before.owner_incarnation,
                        work_generation: row.before.work_generation,
                        generated_before: 0,
                        generated_after: 1,
                        original_candidate: TokenId::new(9),
                        committed_token: TokenId::new(7),
                        route: Route::FullLogitsSampler,
                        pending_before: vec![],
                        pending_after: vec![0xc3],
                    }
                });
                frontiers[slot] = after.clone();
                Completed {
                    before: row.before.clone(),
                    after: Some(after),
                    preparation_commit: commit,
                }
            })
            .collect::<Vec<_>>();
        // Wire vector order is not an identity. Actual input/host ordinals are.
        stages.rows.reverse();
        assert!(preparation.handle(serde_json::json!({"kind":"preparation_completed","offered":call,"phase":"fit","cohort":0,
            "reconciled":true,"queue":{"accepted_ordinal":call,"disposition":"published"},"host_stages":stages,"rows":completed,"failure":null}), &mut progress).unwrap());
        if call == 2 {
            let receipt = Released {
                frontier: frontiers[0].clone(),
                original_policy_signature: POLICY,
                original_numeric_policy: policy,
                generated_prefix_sha256: digest,
                through_call_id: 2,
                through_fifo_ordinal: 2,
                actor_applied_output_ordinal: 1,
            };
            // Slot 0 is individually ready, but slot 1 has only a partial
            // prefill. The actual common release frontier has not occurred.
            assert!(preparation
                .handle(
                    serde_json::json!({"kind":"preparation_released","phase":"fit","cohort":0,
                "slot":0,"receipt":receipt}),
                    &mut progress
                )
                .is_err());
        }
    }
    for slot in 0..2 {
        let receipt = Released {
            frontier: frontiers[slot].clone(),
            original_policy_signature: POLICY,
            original_numeric_policy: policy,
            generated_prefix_sha256: digest,
            through_call_id: slot as u64 + 2,
            through_fifo_ordinal: slot as u64 + 2,
            actor_applied_output_ordinal: 1,
        };
        let record = serde_json::json!({"kind":"preparation_released","phase":"fit","cohort":0,"slot":slot,"receipt":receipt});
        if slot == 0 {
            let mut wrong = record.clone();
            wrong["receipt"]["through_fifo_ordinal"] = 3.into();
            assert!(preparation.handle(wrong, &mut progress).is_err());
        }
        preparation.handle(record, &mut progress).unwrap();
        assert_eq!(preparation.ready().is_ok(), slot == 1);
    }
    assert_eq!(*progress.last_fifo, 3);
    assert_eq!(*progress.total_rows, 4);
    // Release never fabricates terminal Length or permits the cohort to end.
    assert!(progress.lifecycle.end(0, 0, 2, 2).is_err());
}

#[test]
fn source5_two_owners_reuse_one_physical_pass_for_declared_ordinary_cohorts() {
    let source = replay::tests::source4();
    // All-ordinary is an explicit prefix plan. Its two independently qualified
    // owners still share one physical pass, as they do with prepared cohorts.
    let bytes = changed(&source, |value| {
        if value["schema_version"] != 4 {
            return false;
        }
        let shared: HeaderV4 = serde_json::from_value(value.clone()).unwrap();
        let plan = StructuredPrefixPlanV5 {
            phases: std::array::from_fn(|phase| {
                vec![None; shared.common.cohort_plan.phases[phase].len()]
            }),
        };
        let signature = plan.signature(&shared.common.cohort_plan).unwrap();
        let mut prefix = HeaderV5::new(shared, plan, signature);
        prefix.capture_protocol = prefix.signature().unwrap();
        *value = serde_json::to_value(prefix).unwrap();
        true
    });
    let mut physical = 0;
    let replayed = replay::replay_driver(
        &bytes,
        &CostProfileLoadLimits::default(),
        || physical += 1,
        true,
    )
    .unwrap();
    assert_eq!(physical, 72);
    assert_eq!(replayed.children.len(), 2);
    assert_ne!(
        replayed.children[0].model.owner(),
        replayed.children[1].model.owner()
    );
    for child in replayed.children {
        assert_eq!(child.total_shape_rows, 72);
        assert_eq!(child.reserved_members, 24);
    }
}
