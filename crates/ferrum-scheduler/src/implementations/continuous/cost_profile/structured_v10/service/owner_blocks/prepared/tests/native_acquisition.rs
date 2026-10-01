//! Source DTO replay checks only. Real engine tests separately obtain native
//! Capture -> Restore -> model/engine acknowledgement before issuing this DTO.
use super::*;

const INPUT: [u8; 32] = [72; 32];

fn native_plan() -> StructuredNativePrefixAcquisitionPlanV1 {
    StructuredNativePrefixAcquisitionPlanV1 {
        phases: std::array::from_fn(|_| {
            vec![Some(StructuredNativePrefixAcquisitionCohortV1 {
                prompt_tokens: 4,
                boundary_tokens: 3,
                input_tokens_sha256: INPUT,
                native_scope: StructuredNativePrefixScopeV1 {
                    plan_hash: "plan".into(),
                    layout_fingerprint: "layout".into(),
                    runtime_implementation_fingerprint: "runtime".into(),
                    device_id: "device".into(),
                },
            })]
        }),
    }
}

fn setup(native: bool) -> StructuredPreparedOwnerBlockCollectorV8 {
    setup_width(native, 1)
}

fn setup_width(native: bool, width: usize) -> StructuredPreparedOwnerBlockCollectorV8 {
    let mut h = header();
    for phase in 0..3 {
        let request = h.declaration.cohort_plan.phases[phase][0].requests[0].clone();
        h.declaration.cohort_plan.phases[phase][0]
            .requests
            .resize(width, request);
        let prefix = h.declaration.prefix_plan.phases[phase][0].as_mut().unwrap();
        prefix.slots.resize(width, prefix.slots[0].clone());
    }
    h.declaration.native_prefix_acquisition = native.then(native_plan);
    let h = StructuredPreparedOwnerBlockHeaderV8::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap();
    let mut c =
        StructuredPreparedOwnerBlockCollectorV8::new(h, CostProfileLoadLimits::default()).unwrap();
    c.open_block(2, 0).unwrap();
    let mut events = vec![serde_json::json!({"kind":"cohort_begin","phase":"fit",
        "cohort":0,"manifest_case":0,"repetition":0})];
    for slot in 0..width {
        let request = if slot == 0 {
            "target".to_owned()
        } else {
            format!("target-{slot}")
        };
        events.push(
            serde_json::json!({"kind":"request_admitted","phase":"fit","cohort":0,
            "slot":slot,"request_id":request,"maximum_output":3}),
        );
    }
    for value in events {
        c.push(&StructuredPreparedOwnerBlockRecordV8::Cohort(
            StructuredCohortEventV8::from_diagnostic(value).unwrap(),
        ))
        .unwrap();
    }
    c
}

fn frontier(restored: bool) -> serde_json::Value {
    serde_json::json!({"request_id":"target","owner_incarnation":1,
        "work_generation":if restored { 2 } else { 1 },"generated_tokens":0,
        "kv_tokens":if restored { 3 } else { 0 },
        "model_cache_id":if restored { Some("restored-cache") } else { None },
        "pending_utf8":[],"output_accepted_ordinal":0})
}

fn transfer(capture: bool) -> serde_json::Value {
    serde_json::json!({"slot":if capture { 9 } else { 10 },
        "checkpoint_coordinator":1,"checkpoint_serial":2,
        "sequence_sparse":if capture { 1 } else { 2 },"sequence_generation":1,
        "request_sparse":if capture { 1 } else { 2 },"request_generation":1,
        "boundary_tokens":3,"kind":if capture { "capture" } else { "restore" },
        "plan_hash":"plan","layout_fingerprint":"layout",
        "runtime_implementation_fingerprint":"runtime","device_id":"device"})
}

fn restored() -> serde_json::Value {
    serde_json::json!({"kind":"native_prefix_restored","phase":"fit",
        "cohort":0,"slot":0,"before":frontier(false),"after":frontier(true),
        "input_tokens_sha256":INPUT,"capture":transfer(true),"restore":transfer(false),
        "captured_at_ns":3,"acknowledged_at_ns":4,"expires_at_ns":10,
        "acknowledged":true})
}

fn preparation(value: serde_json::Value) -> StructuredPreparedOwnerBlockRecordV8 {
    StructuredPreparedOwnerBlockRecordV8::Preparation(
        StructuredPreparationEventV8::from_diagnostic(value).unwrap(),
    )
}

fn offer(restored: bool) -> StructuredPreparedOwnerBlockRecordV8 {
    preparation(serde_json::json!({"kind":"preparation_offered","offered":1,
        "phase":"fit","cohort":0,"rows":[{"before":frontier(restored),
        "work":{"kind":"prefill","offset":if restored { 3 } else { 0 },
        "count":if restored { 1 } else { 4 },"total_prompt_tokens":4}}]}))
}

#[test]
fn source8_native_prefix_ack_initializes_only_declared_preparation_frontier() {
    let mut c = setup(true);
    let before = c.source_receipt();
    let record = preparation(restored());
    // The generic source8 wire must preserve the variant, not reinterpret it
    // as a numerical population or an ordinary offered inference wave.
    let bytes = record_bytes_v7(&record).unwrap();
    let replayed: StructuredPreparedOwnerBlockRecordV8 = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(
        replayed,
        StructuredPreparedOwnerBlockRecordV8::Preparation(_)
    ));
    c.push(&replayed).unwrap();
    assert_ne!(
        c.source_receipt(),
        before,
        "original ACK is included in source hash"
    );
    assert_eq!(
        (
            c.offered(),
            c.last_fifo(),
            c.preparation_attempts(),
            c.qualified_children()
        ),
        (0, 0, 0, 0),
        "a native transfer is not an inference observation or qualified member"
    );
    c.push(&offer(true)).unwrap();
    assert_eq!(c.qualified_children(), 0);
}

#[test]
fn source8_native_prefix_none_missing_or_duplicate_ack_is_rejected() {
    let mut cold = setup(false);
    assert!(cold.push(&preparation(restored())).is_err());
    assert!(cold.audit().poisoned);
    // The old None contract still accepts its original genuinely cold start.
    let mut cold = setup(false);
    cold.push(&offer(false)).unwrap();
    for skipped_frontier in [false, true] {
        let mut c = setup(true);
        assert!(c.push(&offer(skipped_frontier)).is_err());
        assert!(c.audit().poisoned);
    }
    let mut c = setup(true);
    c.push(&preparation(restored())).unwrap();
    assert!(c.push(&preparation(restored())).is_err());
    assert!(c.audit().poisoned);
}

#[test]
fn source8_native_prefix_replay_rejects_foreign_or_unacknowledged_receipts() {
    let changes: &[(&str, fn(&mut serde_json::Value))] = &[
        ("missing full ACK", |v| v["acknowledged"] = false.into()),
        ("foreign driver phase", |v| v["phase"] = "residual".into()),
        ("foreign cohort", |v| v["cohort"] = 1.into()),
        ("foreign slot", |v| v["slot"] = 1.into()),
        ("foreign owner", |v| {
            v["before"]["request_id"] = "other".into()
        }),
        ("changed incarnation", |v| {
            v["after"]["owner_incarnation"] = 2.into()
        }),
        ("wrong input tokens", |v| {
            v["input_tokens_sha256"][0] = 71.into()
        }),
        ("wrong capture kind", |v| {
            v["capture"]["kind"] = "restore".into()
        }),
        ("different checkpoint", |v| {
            v["restore"]["checkpoint_serial"] = 3.into()
        }),
        ("different boundary", |v| {
            v["restore"]["boundary_tokens"] = 2.into()
        }),
        ("same transfer reused", |v| v["restore"]["slot"] = 9.into()),
        ("nonfresh target", |v| v["before"]["kv_tokens"] = 1.into()),
        ("fabricated output", |v| {
            v["after"]["generated_tokens"] = 1.into()
        }),
        ("capture after ACK", |v| v["captured_at_ns"] = 5.into()),
        ("expired ACK", |v| v["acknowledged_at_ns"] = 10.into()),
    ];
    for (name, change) in changes {
        let mut c = setup(true);
        let mut value = restored();
        change(&mut value);
        assert!(c.push(&preparation(value)).is_err(), "{name}");
        assert!(c.audit().poisoned, "{name}");
        assert_eq!(c.qualified_children(), 0, "{name}");
    }
    for field in [
        "plan_hash",
        "layout_fingerprint",
        "runtime_implementation_fingerprint",
        "device_id",
    ] {
        for both in [false, true] {
            let mut c = setup(true);
            let mut value = restored();
            value["restore"][field] = "foreign".into();
            if both {
                value["capture"][field] = "foreign".into();
            }
            assert!(
                c.push(&preparation(value)).is_err(),
                "foreign {field}, paired={both}"
            );
            assert!(c.audit().poisoned);
        }
    }
}

#[test]
fn source8_native_prefix_input_and_scope_are_part_of_original_declaration_identity() {
    let mut h = header();
    h.declaration.native_prefix_acquisition = Some(native_plan());
    let original = h.declaration.signature().unwrap();
    let mut changed = h.declaration.clone();
    changed.native_prefix_acquisition.as_mut().unwrap().phases[0][0]
        .as_mut()
        .unwrap()
        .input_tokens_sha256[0] ^= 1;
    assert_ne!(original, changed.signature().unwrap());
    let mut changed = h.declaration.clone();
    changed.native_prefix_acquisition.as_mut().unwrap().phases[0][0]
        .as_mut()
        .unwrap()
        .native_scope
        .layout_fingerprint
        .push_str("-other");
    assert_ne!(original, changed.signature().unwrap());
    let mut malformed = native_plan();
    malformed.phases[0].clear();
    assert!(malformed
        .validate(&h.declaration.cohort_plan, &h.declaration.prefix_plan)
        .is_err());
}

#[test]
fn source8_native_prefix_ack_clock_cannot_regress_between_real_slots() {
    for (second_ack, accepted) in [(5, false), (7, true)] {
        let mut c = setup_width(true, 2);
        let mut first = restored();
        first["acknowledged_at_ns"] = 6.into();
        c.push(&preparation(first)).unwrap();
        let mut second = restored();
        second["slot"] = 1.into();
        second["before"]["request_id"] = "target-1".into();
        second["after"]["request_id"] = "target-1".into();
        second["restore"]["slot"] = 11.into();
        second["restore"]["sequence_sparse"] = 3.into();
        second["restore"]["request_sparse"] = 3.into();
        second["acknowledged_at_ns"] = second_ack.into();
        assert_eq!(c.push(&preparation(second)).is_ok(), accepted);
        assert_eq!(c.audit().poisoned, !accepted);
        assert_eq!(
            (
                c.offered(),
                c.last_fifo(),
                c.preparation_attempts(),
                c.qualified_children()
            ),
            (0, 0, 0, 0)
        );
    }
}
