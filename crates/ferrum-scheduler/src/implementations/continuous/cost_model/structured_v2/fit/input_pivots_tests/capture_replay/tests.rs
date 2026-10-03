//! Synthetic wire/ledger tests only. They do not stand in for hardware capture,
//! checked model recipes or the existing independent geometry-kernel tests.
use super::*;
use serde_json::{json, Value};

fn fixture(limit: u64) -> Vec<Value> {
    fixture_using(limit, StructuredSettingsV2::default())
}

fn fixture_using(limit: u64, settings: StructuredSettingsV2) -> Vec<Value> {
    let mut config = ferrum_types::EngineConfig::default();
    config.scheduler.slo = serde_json::from_value(json!({"mode":"enforce"})).unwrap();
    let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 {
        settings: automatic,
    } = &mut config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    else {
        unreachable!()
    };
    use ferrum_types::SloAutomaticCalibrationInputReadinessV1 as R;
    match &mut automatic.input_readiness {
        R::WorkAxesAndBranchesV1 {
            maximum_geometry_visits,
            ..
        }
        | R::WorkAxesAndBranchesV2 {
            maximum_geometry_visits,
            ..
        }
        | R::WorkAxesAndBranchesV3 {
            maximum_geometry_visits,
            ..
        } => {
            *maximum_geometry_visits = NonZeroU64::new(limit).unwrap();
        }
        R::CountOnlyV1 {} => unreachable!(),
    }
    let cases: Vec<_> = [1, 8, 8]
        .into_iter()
        .map(|width| {
            json!({
                "product":"Full", "template":0, "width":width,
                "maximum_output":21, "release_generated":3, "suffix_tokens":18,
                "preset":ferrum_types::SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
                "prefix":"clean", "route":"full_logits", "reset":false
            })
        })
        .collect();
    let mut records = vec![
        json!({"kind":"ferrum.test.cold_geometry.v1", "input_sha256":vec![7u8;32], "maximum_bytes":CAPTURE_LIMIT, "buffer_bytes":65536}),
        json!({"kind":"actual_startup_inputs", "config":config, "templates":[{"output":"fixture","request_bytes":[]}]}),
        json!({"kind":"original_cases", "cases":cases, "prompts":[61], "chunk":8,"prefill_row_ceiling":2}),
    ];
    let scratch = input_geometry_pivot_scratch_bytes_v1(3, 4, settings.max_rank).unwrap();
    let mut work = StructuredInputGeometryWorkV1::new(NonZeroU64::new(limit).unwrap());
    for ordinal in 0..3 {
        // Column 2 is zero in every row; column 3 is nonzero only in a
        // non-anchor row. Third call changes only one signed-zero bit.
        let mut values = vec![
            vec![1., 0., 0., 0.],
            vec![1., 1., 0., 0.],
            vec![1., 2., 0., 1.],
        ];
        if ordinal == 2 {
            values[0][2] = -0.;
        }
        let bits: Vec<Vec<u64>> = values
            .iter()
            .map(|r| r.iter().map(|v: &f64| v.to_bits()).collect())
            .collect();
        let rows: Vec<_> = values.iter().map(Vec::as_slice).collect();
        records.push(json!({"kind":"population", "population_index":ordinal, "key":{"synthetic_wire_population":ordinal}}));
        records.push(json!({"kind":"original_matrix", "ordinal":ordinal, "cases":[0,1,2], "axis_bits":bits, "mandatory_anchors":[0], "settings":settings,
            "visits_before":work.visits(), "maximum_visits":limit, "exhausted_before":work.exhausted(), "maximum_scratch_bytes":scratch}));
        let before = work.visits();
        let outcome = input_geometry_pivots_original_v1(&rows, &[0], &settings, &mut work, scratch);
        let mut selected = vec![0];
        let (rank, gap) = match outcome {
            Ok(pivots) => {
                selected.extend_from_slice(&pivots.pivot_indices[pivots.anchor_rank..]);
                (Some(pivots.rank), None)
            }
            Err(reason) => (
                None,
                Some(Gap::InputGeometryUnavailable {
                    reason: format!("{reason:?}"),
                    work_exhausted: work.exhausted(),
                }),
            ),
        };
        selected.sort_unstable();
        records.push(json!({"kind":"original_result", "ordinal":ordinal, "audit":GeometryAudit {
            candidate_rank:rank, selected_rank:rank, added_original_cases:selected.len()-1,
            complete:gap.is_none(), visits:work.visits()-before }, "gap":gap,
            "final_selected_cases":selected, "visits_after":work.visits(), "exhausted_after":work.exhausted()}));
    }
    records.push(json!({"kind":"inventory_retired", "complete":true, "deadline_expired":false, "remaining_ns":100,
        "actual_requests":10,"actual_actions":20,"selected_requests":5,"selected_actions":8}));
    records.push(json!({"kind":"completed_after_shutdown","matrices":3}));
    records
}

fn artifact(records: &[Value]) -> (Vec<u8>, CaptureReport) {
    let mut bytes = Vec::new();
    for record in records {
        serde_json::to_writer(&mut bytes, record).unwrap();
        bytes.push(b'\n');
    }
    let report = CaptureReport {
        path: "original/remote/capture.jsonl".into(),
        bytes: bytes.len() as u64,
        sha256: digest(&bytes),
        complete: true,
        failure: None,
        expected_matrices: 3,
        observed_matrices: 3,
        elapsed_ns: 1,
        diagnostic_buffer_bytes: 65536,
        diagnostic_retained_bytes: 66000,
    };
    (bytes, report)
}

#[test]
fn capture_replay_keeps_u64_bits_original_domains_and_strict_duplicate_identity() {
    let (bytes, report) = artifact(&fixture(32_000_000));
    let verified = verify_capture(&bytes, &report, [7; 32]).unwrap();
    assert_eq!(verified.calls[0].matrix.axis_bits[0][0], 1f64.to_bits());
    assert!(verified.calls[0].matrix.axis_bits[0][0] > (1 << 53));
    assert_eq!(verified.calls[2].matrix.axis_bits[0][2], (-0f64).to_bits());
    let stats = statistics(&verified).unwrap();
    assert_eq!(stats.strict_duplicate_groups.len(), 1);
    assert_eq!(stats.strict_duplicate_groups[0].ordinals, [0, 1]);
    for matrix in &stats.matrices {
        assert_eq!(matrix.all_original_rows_zero_axes, [2]);
        assert_eq!(matrix.scan_coordinates, 12);
    }
    assert_eq!(stats.case_domains[1].original.width, 8);
    assert_eq!(stats.case_domains[1].decode_sequence_tokens, Some(64));
}

#[test]
fn capture_replay_retains_optional_final_plan_without_granting_it_authority() {
    let mut records = fixture(32_000_000);
    let selection = json!({"requests": 17, "batches": [], "unqualified": true});
    let position = records.len() - 2;
    records.insert(
        position,
        json!({"kind":"final_selection", "selection":selection}),
    );
    let (bytes, report) = artifact(&records);
    let verified = verify_capture(&bytes, &report, [7; 32]).unwrap();
    let stats = statistics(&verified).unwrap();
    assert_eq!(stats.original_selection, Some(&selection));
    assert!(stats.baseline_matched);
    // The legacy capture above remains valid, but multiple or misplaced final
    // plans cannot silently change which source declaration a reader sees.
    records.insert(position, records[position].clone());
    let (bytes, report) = artifact(&records);
    assert!(verify_capture(&bytes, &report, [7; 32]).is_err());
    records.remove(position);
    let plan = records.remove(position);
    records.insert(3, plan);
    let (bytes, report) = artifact(&records);
    assert!(verify_capture(&bytes, &report, [7; 32]).is_err());
}

#[test]
fn capture_replay_preserves_shared_exhaustion_without_refund_or_reset() {
    let complete = fixture(32_000_000);
    let first_charge = complete[5]["visits_after"].as_u64().unwrap();
    assert!(first_charge > 0);
    let (bytes, report) = artifact(&fixture(first_charge + 1));
    let verified = verify_capture(&bytes, &report, [7; 32]).unwrap();
    assert!(verified.exhausted);
    assert_eq!(verified.visits, first_charge);
    assert!(verified.calls[0].result.audit.complete);
    assert!(!verified.calls[1].matrix.exhausted_before);
    assert!(verified.calls[1].result.exhausted_after);
    assert!(verified.calls[2].matrix.exhausted_before);
    assert!(verified.calls[1..]
        .iter()
        .all(|call| { call.result.gap.is_some() && call.result.visits_after == first_charge }));
}

#[test]
fn capture_replay_rejects_missing_mapping_footer_and_changed_ledger_or_result() {
    let original = fixture(32_000_000);
    for fault in 0..6 {
        let mut records = original.clone();
        match fault {
            0 => {
                records.remove(2);
            }
            1 => {
                records.pop();
            }
            2 => {
                records[5]["visits_after"] =
                    json!(records[5]["visits_after"].as_u64().unwrap() + 1);
            }
            3 => {
                records[5]["final_selected_cases"] = json!([0]);
            }
            4 => {
                records[7]["visits_before"] = json!(0);
            }
            5 => {
                records[4]["cases"] = json!([0, 1, 99]);
            }
            _ => unreachable!(),
        }
        // Deliberately recompute file binding so these failures test original
        // structure/ledger checks, not merely the preceding checksum check.
        let (bytes, report) = artifact(&records);
        assert!(
            verify_capture(&bytes, &report, [7; 32]).is_err(),
            "fault={fault}"
        );
    }
    let (bytes, mut report) = artifact(&original);
    assert!(verify_capture(&bytes, &report, [8; 32]).is_err());
    report.sha256 = "00".repeat(32);
    assert!(verify_capture(&bytes, &report, [7; 32]).is_err());
    report.sha256 = digest(&bytes);
    report.complete = false;
    assert!(verify_capture(&bytes, &report, [7; 32]).is_err());
}

#[test]
fn capture_replay_tagged_retirement_preserves_u64_nanoseconds_and_rejects_overflow() {
    fn wire(remaining_ns: &str) -> String {
        format!(
            r#"{{"kind":"inventory_retired","complete":true,"deadline_expired":false,"remaining_ns":{remaining_ns},"actual_requests":10,"actual_actions":20,"selected_requests":5,"selected_actions":8}}"#
        )
    }
    for value in [0, 1, 120_000_000_000, u64::MAX] {
        let Record::Retirement(retired) =
            serde_json::from_str::<Record>(&wire(&value.to_string())).unwrap()
        else {
            panic!("original tagged retirement changed record kind");
        };
        assert_eq!(retired.remaining_ns, u128::from(value));
    }
    let overflow = (u128::from(u64::MAX) + 1).to_string();
    for invalid in [overflow.as_str(), "-1", "1.5", "\"120\"", "null"] {
        assert!(serde_json::from_str::<Record>(&wire(invalid)).is_err());
    }
}

#[test]
fn capture_reference_measures_complete_demand_without_changing_shared_baseline() {
    let complete = fixture(32_000_000);
    let first_charge = complete[5]["visits_after"].as_u64().unwrap();
    let (bytes, report) = artifact(&fixture(first_charge + 1));
    let verified = verify_capture(&bytes, &report, [7; 32]).unwrap();
    let before = (verified.limit, verified.visits, verified.exhausted);
    let measurement = reference::measure(&verified).unwrap();
    let wire = serde_json::to_value(&measurement).unwrap();
    assert_eq!(wire["successful_calls"], 3);
    assert_eq!(wire["failed_calls"], 0);
    assert_eq!(
        wire["full_required_visits"].as_u64(),
        first_charge.checked_mul(3)
    );
    assert_eq!(
        wire["full_requirement_minus_original_charge"].as_u64(),
        first_charge.checked_mul(2)
    );
    assert_eq!(
        wire["originally_incomplete_calls_full_reference_visits"].as_u64(),
        first_charge.checked_mul(2)
    );
    assert_eq!(wire["extra_original_core_passes"], 3);
    assert!(wire["extra_original_core_visits"].as_u64().unwrap() > 0);
    assert_eq!(
        before,
        (verified.limit, verified.visits, verified.exhausted)
    );
    for call in wire["calls"].as_array().unwrap() {
        assert_eq!(call["rank"], 3);
        assert_eq!(call["anchor_rank"], 1);
        assert_eq!(call["pivot_indices"].as_array().unwrap().len(), 3);
        assert_eq!(call["final_selected_cases"], json!([0, 1, 2]));
    }
}

#[test]
fn capture_reference_groups_only_original_core_bit_equal_normalized_calls() {
    let mut records = fixture(32_000_000);
    // Power-of-two rescaling changes the full raw input, not its original
    // normalized rows. Baseline replay must independently verify all outcomes.
    for row in records[7]["axis_bits"].as_array_mut().unwrap() {
        for bits in row.as_array_mut().unwrap() {
            let value = f64::from_bits(bits.as_u64().unwrap());
            *bits = json!((value * 2.).to_bits());
        }
    }
    let (bytes, report) = artifact(&records);
    let verified = verify_capture(&bytes, &report, [7; 32]).unwrap();
    assert!(statistics(&verified)
        .unwrap()
        .strict_duplicate_groups
        .is_empty());
    let measurement = reference::measure(&verified).unwrap();
    let wire = serde_json::to_value(&measurement).unwrap();
    let groups = wire["normalized_bit_groups"].as_array().unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0]["ordinals"], json!([0, 1]));
    assert_eq!(groups[0]["raw_different_from_first"], json!([1]));
    // The separate signed-zero call must not join a bit-exact group.
    assert_eq!(
        wire["calls"][0]["normalized_call_sha256"],
        wire["calls"][1]["normalized_call_sha256"]
    );
    assert_ne!(
        wire["calls"][0]["normalized_call_sha256"],
        wire["calls"][2]["normalized_call_sha256"]
    );
    assert_ne!(
        wire["calls"][0]["normalization_scale_bits"],
        wire["calls"][1]["normalization_scale_bits"]
    );
    for (ordinal, call) in wire["calls"].as_array().unwrap().iter().enumerate() {
        assert_eq!(call["population_index"], ordinal);
        assert_eq!(call["original_case_indices"], json!([0, 1, 2]));
    }
}

#[test]
fn capture_reference_does_not_call_partial_rank_rejection_complete_demand() {
    let settings = StructuredSettingsV2 {
        max_rank: 1,
        ..Default::default()
    };
    let (bytes, report) = artifact(&fixture_using(32_000_000, settings));
    let verified = verify_capture(&bytes, &report, [7; 32]).unwrap();
    let measurement = reference::measure(&verified).unwrap();
    let wire = serde_json::to_value(&measurement).unwrap();
    assert_eq!(wire["successful_calls"], 0);
    assert_eq!(wire["failed_calls"], 3);
    assert!(wire["full_required_visits"].is_null());
    assert!(wire["full_requirement_minus_original_charge"].is_null());
    assert_eq!(wire["extra_original_core_passes"], 0);
    assert!(wire["normalized_bit_groups"].as_array().unwrap().is_empty());
    for call in wire["calls"].as_array().unwrap() {
        assert_eq!(call["error"], "Capacity");
        assert_eq!(call["reference_exhausted"], false);
        assert!(call["rank"].is_null());
    }
}
