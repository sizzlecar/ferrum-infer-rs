//! Actual Q6 provider execution; the primitive F64/marker oracle lives in the
//! native probe. These gates compare one numerical profile across IP and DS.
use super::*;
use ferrum_types::{InvocationPreparationStrategy, ProgramBindingUploadStrategy};
use std::cell::RefCell;
use std::sync::Barrier;

#[derive(Default)]
struct Preparation(RefCell<Vec<InvocationPreparationStats>>);
impl InvocationPreparationSink for Preparation {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.borrow_mut().push(stats);
    }
}
impl Preparation {
    fn last(&self) -> InvocationPreparationStats {
        *self.0.borrow().last().expect("actual dispatch preparation")
    }
}

fn assert_selected_routes(fixture: &Fixture, kind: AttentionKind, participants: u32) {
    let family = Family::q6_f16(kind);
    let mut eligible = 0;
    let mut strict = 0;
    let mut middle_routes = 0;
    let mut joined_banks = BTreeSet::new();
    for (name, profile) in [
        ("node.attention", family.attention_profile()),
        ("node.swiglu", family.swiglu_profile()),
    ] {
        let node = fixture
            .compilation
            .executable()
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .find(|n| n.id().as_str() == name)
            .unwrap();
        assert_eq!(node.operation_id().as_str(), profile.operation_id());
        let prepared = node.provider_resources().projection_numerics().unwrap();
        assert_eq!(prepared.contract(), &profile.arithmetic());
        for projection in prepared.projections() {
            for leaf in projection.leaves() {
                let WeightEncoding::BlockQuantized(block) = leaf.encoding() else {
                    continue;
                };
                assert_eq!(block.format_id.as_str(), "quantization.gguf.q6-k");
                assert_eq!(block.bytes_per_block, 210);
                let k = projection.input_features();
                let n = leaf.output_features();
                let declaration = prepared
                    .contract()
                    .declared_projection_arithmetic(
                        projection.role(),
                        Some(block),
                        k,
                        n,
                        leaf.has_weight_transform(),
                    )
                    .unwrap();
                let selected = match declaration {
                    DeclaredProjectionArithmetic::Staged(staged) => {
                        assert!(leaf.is_staged());
                        staged.upstream_policy().unwrap().select_arithmetic(
                            UpstreamProjectionLayout::Columns,
                            participants,
                            k,
                            n,
                        )
                    }
                    DeclaredProjectionArithmetic::StrictBase(_) => {
                        assert!(!leaf.is_staged());
                        None
                    }
                };
                let in_domain = k >= 5120
                    && k % 256 == 0
                    && n >= 1024
                    && (4..=32).contains(&participants)
                    && (participants >= 8 || n >= 5120);
                assert_eq!(
                    selected,
                    in_domain.then_some(UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1)
                );
                if in_domain {
                    eligible += 1;
                } else {
                    strict += 1;
                }
                if k >= 5120 && (1024..5120).contains(&n) {
                    middle_routes += 1;
                    assert_eq!(selected.is_some(), participants >= 8);
                }
                if projection.role() == ProjectionRole::SwiGluGateUp {
                    assert_eq!(
                        projection.output_features(),
                        2 * family::q6_f16::INTERMEDIATE
                    );
                    if n == 5120 {
                        joined_banks.insert(leaf.output_offset() / family::q6_f16::INTERMEDIATE);
                        assert!(selected.is_some(), "large leaf selected at both M7 and M8");
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"kind":"q6_f16_prepared_leaf_selection",
                    "attention":format!("{kind:?}"),"node":name,"role":projection.role(),
                    "component":leaf.component_id(),"m":participants,"k":k,"leaf_n":n,
                    "output_offset":leaf.output_offset(),"output_stride":projection.output_features(),
                    "selected":selected,"strict_fallback":selected.is_none(),
                    "scope":"actual prepared physical leaf plus shared selector; CUDA encode validates the native plan, not a launch trace"})
                );
            }
        }
    }
    assert!(eligible > 0 && strict > 0 && middle_routes > 0);
    assert_eq!(joined_banks, BTreeSet::from([0, 1]));
}

fn compare(kind: AttentionKind) {
    geometry::compare_family(
        kind,
        Family::q6_f16,
        InvocationPreparationStrategy::IdentityProjection,
        assert_selected_routes,
        "q6_f16",
    );
}

#[test]
#[ignore = "requires actual Q6 F16 providers and full recurrent state"]
fn q6_f16_gdn_joined_ffn_ip_matches_segment_across_widths_and_requests() {
    compare(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires actual Q6 F16 providers and physical FP16 KV growth"]
fn q6_f16_causal_joined_ffn_ip_matches_segment_across_widths_and_kv_growth() {
    compare(AttentionKind::Causal);
}

#[test]
#[ignore = "requires shared Plan Q6 validation dependencies on two actual CUDA lanes"]
fn q6_f16_shared_plan_two_lanes_preserve_ip_state_and_segment_replay() {
    let kind = AttentionKind::GatedDelta;
    let make = |oracle| {
        Fixture::for_family_with_segment_oracle(
            kind,
            true,
            16,
            Family::q6_f16(kind),
            ProgramBindingUploadStrategy::Sparse,
            oracle,
        )
    };
    let reference = make(SegmentBindingOracleMode::Disabled);
    let shared = make(SegmentBindingOracleMode::CompareReference);
    assert_selected_routes(&shared, kind, 8);
    let second_lane = shared.resources.create_execution_lane().unwrap();
    second_lane
        .configure_reusable_executables(DeviceReusableExecutionPlan::on_demand(8).unwrap())
        .unwrap();
    assert_ne!(shared.lane.id(), second_lane.id());
    let lanes = [Arc::clone(&shared.lane), second_lane];
    let reapers = [Arc::clone(&shared.reaper), CompletionReaper::new()];
    let tokens: Vec<Arc<[u32]>> = (0..16)
        .map(|p| (0..4).map(|i| ((i * 7 + p * 3 + 1) % 32) as u32).collect())
        .collect();
    let admit = |fixture: &Fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(p, t)| fixture.admit(&format!("q6-lane-{p}"), Arc::clone(t)))
            .collect::<Vec<_>>()
    };
    let expected_sessions = admit(&reference);
    let actual_sessions = admit(&shared);
    let mut hits = [0; 2];
    for position in 0..4 {
        let path = if position < 2 {
            Path::Warm
        } else {
            Path::Replay
        };
        let expected = (0..2)
            .map(|lane| {
                let range = lane * 8..lane * 8 + 8;
                reference.execute_participants_with_preparation(
                    &reference.lane,
                    &reference.reaper,
                    &expected_sessions[range.clone()],
                    &tokens[range],
                    position..position + 1,
                    path,
                    InvocationPreparationStrategy::IdentityProjection,
                    &Preparation::default(),
                )
            })
            .collect::<Vec<_>>();
        if position == 0 {
            assert_ne!(expected[0].values, expected[1].values);
        }
        let start = Barrier::new(3);
        let actual = std::thread::scope(|scope| {
            let handles = (0..2)
                .map(|lane| {
                    let fixture = &shared;
                    let lane_ref = &lanes[lane];
                    let reaper = &reapers[lane];
                    let range = lane * 8..lane * 8 + 8;
                    let sessions = &actual_sessions[range.clone()];
                    let input = &tokens[range];
                    let start = &start;
                    scope.spawn(move || {
                        let sink = Preparation::default();
                        start.wait();
                        let observation = fixture.execute_participants_with_preparation(
                            lane_ref,
                            reaper,
                            sessions,
                            input,
                            position..position + 1,
                            path,
                            InvocationPreparationStrategy::DecodeSegment,
                            &sink,
                        );
                        (observation, sink.last())
                    })
                })
                .collect::<Vec<_>>();
            start.wait();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        for (lane, (observation, stats)) in actual.iter().enumerate() {
            expected[lane].assert_same(observation);
            assert_eq!(stats.parts_materialized, 0);
            hits[lane] += stats.segment_hits;
            if position == 3 {
                assert_eq!(stats.segment_hits, 1);
            }
            println!(
                "{}",
                serde_json::json!({"kind":"q6_f16_shared_plan_lane",
                "lane":format!("{:?}",lanes[lane].id()),"wave":position,"actual_m":8,
                "hits":stats.segment_hits,"encoded_nodes":stats.segment_encoded_nodes,
                "same_profile_ip_output_state_equal":true,"shared_plan_weight_flags":true,
                "host_start_barrier":true,"device_overlap_measured":false})
            );
        }
    }
    assert!(hits.iter().all(|&v| v > 0));
    assert!(
        shared
            ._composition
            ._runtime
            .segment_binding_oracle_audited_nodes()
            > 0
    );
    for lane in &lanes {
        assert_eq!(lane.in_flight_count(), 0);
    }
    for session in expected_sessions.into_iter().chain(actual_sessions) {
        session.try_complete().unwrap();
    }
}
