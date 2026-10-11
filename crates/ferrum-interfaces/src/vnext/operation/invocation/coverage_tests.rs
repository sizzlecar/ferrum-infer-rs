//! Real Plan/admission comparison; only the private coverage algorithm differs.
use super::super::view_coverage::with_reference_coverage;
use super::*;

#[test]
fn single_pass_live_wave_matches_reference_and_keeps_per_consumer_checks() {
    for reference in [true, false] {
        with_reference_coverage(reference, || {
            // Retains the actual live descriptor mutation BETWEEN participants,
            // reversed active identity, fresh-wave ownership and unequal windows.
            same_wave_materialization_matches_reference_and_keeps_runtime_checks();
            shared_materialization_preserves_unequal_participant_token_windows();
        });
    }
    for spans in [vec![1], vec![1; 8], vec![1; 32], vec![1, 3, 2]] {
        with_live_wave_spans(spans, |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let make = || {
                BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    &node,
                    OperationInvocationResources::Wave {
                        wave,
                        node_index: 0,
                    },
                    active.iter(),
                    false,
                    true,
                )
                .unwrap()
            };
            let before = with_reference_coverage(true, || make());
            let after = with_reference_coverage(false, || make());
            assert_eq!(snapshot(&before), snapshot(&after));
            assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
            assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
        });
    }
}

#[test]
#[ignore = "optimized paired CPU diagnostic; elapsed results have no speed threshold"]
fn paired_live_wave_single_pass_coverage() {
    for spans in [vec![1], vec![1; 8], vec![1; 32], vec![1, 3, 2]] {
        let participants = spans.len();
        let rows: usize = spans.iter().sum();
        with_live_wave_spans(spans, |fixture, wave, identity, active| {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let node = identity.materialize_node(0).unwrap();
            let make = || {
                BatchedOperationInvocation::from_resources(
                    fixture.runtime.as_ref(),
                    &fixture.resolved,
                    provider.dispatch(),
                    identity,
                    &node,
                    OperationInvocationResources::Wave {
                        wave,
                        node_index: 0,
                    },
                    active.iter(),
                    false,
                    true,
                )
                .unwrap()
            };
            let before = with_reference_coverage(true, || snapshot(&make()));
            let after = with_reference_coverage(false, || snapshot(&make()));
            assert_eq!(before, after);
            let view_count: usize = make().participants().iter().map(|p| p.views().len()).sum();
            let region_count: usize = make()
                .participants()
                .iter()
                .flat_map(|p| p.views())
                .map(|v| {
                    v.translate(0, v.descriptor().size_bytes)
                        .unwrap()
                        .iter()
                        .count()
                })
                .sum();
            for round in 0..14 {
                for reference in if round % 2 == 0 {
                    [true, false]
                } else {
                    [false, true]
                } {
                    // The scoped TLS selector is set once outside the timed loop;
                    // both test arms read it per view. Production has no selector.
                    let (construct_ns, total_ns) = with_reference_coverage(reference, || {
                        let mut construct_ns = 0_u128;
                        let mut total_ns = 0_u128;
                        for _ in 0..128 {
                            let start = Instant::now();
                            let value = black_box(make());
                            construct_ns += start.elapsed().as_nanos();
                            black_box(value.participants().len());
                            drop(value);
                            total_ns += start.elapsed().as_nanos();
                        }
                        (construct_ns, total_ns)
                    });
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({
                                "kind":"invocation_coverage_pair", "participants":participants,
                                "physical_rows":rows, "views":view_count,"regions":region_count,
                                "pair":round-2, "reference":reference, "iterations":128,
                                "construct_ns":construct_ns,"construct_and_drop_ns":total_ns,
                                "scope":"same actual Plan/admitted live wave; U sharing on both arms; no encode/submit; elapsed CPU harness time, not inference throughput"
                            })
                        );
                    }
                }
            }
        });
    }
}
