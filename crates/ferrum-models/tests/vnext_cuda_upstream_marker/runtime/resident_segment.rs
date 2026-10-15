//! Real Indexed/Resident executions, including current request admission,
//! dynamic controls, completion readbacks and retirement. Host stage timers
//! observe the existing dispatch boundaries; nested stages are never added.
use super::*;
use ferrum_types::{InvocationPreparationStrategy, ProgramBindingUploadStrategy};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Default, serde::Serialize)]
struct HostStages {
    // contract, provider parent, binding parent, runtime submit. The binding
    // parent is nested in provider; these entries are not independent costs.
    samples: [u64; 4],
    total_ns: [u64; 4],
}

#[derive(Default)]
struct HostTiming {
    samples: [AtomicU64; 4],
    total_ns: [AtomicU64; 4],
}
impl HostTiming {
    fn snapshot(&self) -> HostStages {
        HostStages {
            samples: std::array::from_fn(|i| self.samples[i].load(Ordering::Relaxed)),
            total_ns: std::array::from_fn(|i| self.total_ns[i].load(Ordering::Relaxed)),
        }
    }
}
impl DeviceSubmissionTimingSink for HostTiming {
    const ENABLED: bool = true;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {}
}
impl SubmissionWaveDispatchTimingSink for HostTiming {
    fn record(&self, stage: SubmissionWaveDispatchStage, elapsed: Duration) {
        let index = match stage {
            SubmissionWaveDispatchStage::ContractValidateAndReserve => 0,
            SubmissionWaveDispatchStage::ProviderNodeEncode => 1,
            SubmissionWaveDispatchStage::SegmentBindingPrepareAndEncode => 2,
            SubmissionWaveDispatchStage::DeviceRuntimeSubmit => 3,
            _ => return,
        };
        let elapsed_ns = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
        let _ = self.samples[index].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            Some(value.saturating_add(1))
        });
        let _ = self.total_ns[index].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            Some(value.saturating_add(elapsed_ns))
        });
    }
}

#[derive(Default)]
struct Preparation(Cell<Option<InvocationPreparationStats>>);
impl InvocationPreparationSink for Preparation {
    fn record_preparation(&self, value: InvocationPreparationStats) {
        self.0.set(Some(value));
    }
}

#[derive(Clone, Copy, Default, serde::Serialize)]
struct Work {
    resident_hits: u64,
    template_installs: u64,
    helper_wall_ns: u64,
    stages: HostStages,
}
impl Work {
    fn add(&mut self, other: Self) {
        self.resident_hits += other.resident_hits;
        self.template_installs += other.template_installs;
        self.helper_wall_ns += other.helper_wall_ns;
        for i in 0..4 {
            self.stages.samples[i] += other.stages.samples[i];
            self.stages.total_ns[i] += other.stages.total_ns[i];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pair(
    fixtures: &[Fixture; 2],
    sessions: &[Vec<Arc<SequenceSession<Runtime>>>; 2],
    tokens: &[Arc<[u32]>],
    ranges: &[Range<usize>],
    kind: AttentionKind,
    phase: &str,
    pass: &str,
    wave: usize,
    path: Path,
    reverse: bool,
    hot_pair: Option<usize>,
) -> [Work; 2] {
    let mut observed: [Option<BatchObservation>; 2] = [None, None];
    let mut work = [Work::default(); 2];
    for arm in if reverse { [1, 0] } else { [0, 1] } {
        let fixture = &fixtures[arm];
        let group = &sessions[arm][..ranges.len()];
        for ((session, tokens), range) in group.iter().zip(tokens).zip(ranges) {
            fixture.extend(session, Arc::from(&tokens[..range.end]));
        }
        let pages = (kind == AttentionKind::Causal).then(|| {
            group
                .iter()
                .zip(tokens)
                .zip(ranges)
                .map(|((session, tokens), range)| {
                    decode_segment::physical_pages(
                        fixture,
                        session,
                        Arc::from(&tokens[..range.end]),
                        range.end,
                    )
                })
                .collect::<Vec<_>>()
        });
        if let Some(pages) = &pages {
            assert_eq!(pages[0], ranges[0].end.div_ceil(64) as u64);
        }
        let runtime = &fixture._composition._runtime;
        let before = (
            runtime.resident_segment_binding_hits(),
            runtime.resident_segment_template_installs(),
        );
        let timing = HostTiming::default();
        let preparation = Preparation::default();
        let started = Instant::now();
        let result = fixture.execute_participant_ranges_observed_with_timing(
            &fixture.lane,
            &fixture.reaper,
            group,
            &tokens[..ranges.len()],
            ranges,
            path,
            InvocationPreparationStrategy::DecodeSegmentProgram,
            &preparation,
            Some(if kind == AttentionKind::Causal {
                88
            } else {
                16
            }),
            batch::BatchObservationOptions::default(),
            &timing,
        );
        let elapsed = started.elapsed();
        let stats = preparation.0.get().expect("actual preparation callback");
        work[arm] = Work {
            resident_hits: runtime.resident_segment_binding_hits() - before.0,
            template_installs: runtime.resident_segment_template_installs() - before.1,
            helper_wall_ns: u64::try_from(elapsed.as_nanos()).unwrap(),
            stages: timing.snapshot(),
        };
        if arm == 0 {
            assert_eq!(
                (work[arm].resident_hits, work[arm].template_installs),
                (0, 0)
            );
        }
        assert_eq!(stats.parts_materialized, 0);
        if hot_pair.is_some() {
            assert!(
                stats.segment_hits > 0,
                "paired sample must execute a warmed program"
            );
            if arm == 1 {
                assert!(
                    work[arm].resident_hits > 0,
                    "a DSR hit or a fallback is not evidence of the Resident path"
                );
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "kind":"resident_segment_wave", "attention":format!("{kind:?}"),
                "pass":pass,"phase":phase,"wave":wave,"arm":arm,
                "owner_mode":runtime.segment_binding_owner_view_mode(),
                "order":if reverse {"BA"} else {"AB"},"hot_pair":hot_pair,
                "participants":ranges.len(),"ranges":ranges,"physical_kv_pages":pages,
                "work":work[arm],"preparation":stats,"published":result.segment_published,
                "stage_order":["contract_validate_reserve","provider_node_encode_parent","segment_binding_prepare_encode_parent","device_runtime_submit"],
                "wall_scope":"complete existing execute helper including admission, dispatch, GPU wait, full output/state/control readback and retirement; extension/diagnostic page query and comparisons excluded",
                "nested_stages_not_additive":true,
            })
        );
        observed[arm] = Some(result);
    }
    let observations = observed.map(Option::unwrap);
    observations[0].assert_same(&observations[1]);
    decode_segment::check_binding_controls(kind, &observations, ranges);
    work
}

fn run(kind: AttentionKind, oracle: SegmentBindingOracleMode, pass: &str) {
    let started = Instant::now();
    let owners = [
        SegmentBindingOwnerViewMode::Indexed,
        SegmentBindingOwnerViewMode::Resident,
    ];
    let fixtures = owners.map(|owner| {
        Fixture::for_family_with_causal_decode_preparation(
            kind,
            true,
            2,
            Family::q6_f16(kind),
            ProgramBindingUploadStrategy::CompactScatter,
            oracle,
            owner,
            ferrum_types::AttentionExecutionPolicy::NativeAdaptive,
            ferrum_types::CausalDecodePreparationMode::Packed,
        )
    });
    for (fixture, owner) in fixtures.iter().zip(owners) {
        assert_eq!(
            fixture
                ._composition
                ._runtime
                .segment_binding_owner_view_mode(),
            owner
        );
    }
    let setup_ns = started.elapsed().as_nanos();
    let initial = if kind == AttentionKind::Causal {
        74
    } else {
        16
    };
    let tokens: Vec<Arc<[u32]>> = (0..2)
        .map(|p| (0..96).map(|i| ((i * 7 + p * 3 + 1) % 32) as u32).collect())
        .collect();
    let mut sessions = fixtures.each_ref().map(|fixture| {
        tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                fixture.admit_with_ceiling(
                    &format!("resident-{pass}-original-{p}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    let mut total = [Work::default(); 2];
    let mut paired = [Work::default(); 2];
    let mut before_growth_hits = 0;
    let mut after_growth_hits = 0;
    let mut growth_fallback = false;
    for position in 0..initial {
        let hot_pair = (position >= initial - 8).then_some(position.saturating_sub(initial - 8));
        let work = pair(
            &fixtures,
            &sessions,
            &tokens,
            &[position..position + 1],
            kind,
            "original_single",
            pass,
            position,
            if position < 2 {
                Path::Warm
            } else {
                Path::Replay
            },
            position % 4 >= 2,
            hot_pair,
        );
        for arm in 0..2 {
            total[arm].add(work[arm]);
            if hot_pair.is_some() {
                paired[arm].add(work[arm]);
            }
        }
        if position < 64 {
            before_growth_hits += work[1].resident_hits;
        }
        if position > 64 {
            after_growth_hits += work[1].resident_hits;
        }
        if kind == AttentionKind::Causal && position == 64 {
            growth_fallback = work[1].resident_hits == 0 && work[1].template_installs > 0;
        }
    }
    assert!(
        before_growth_hits > 0,
        "actual continuous singleton Resident execution is required"
    );
    if kind == AttentionKind::Causal {
        assert!(
            growth_fallback,
            "changed physical KV extent must reinstall through the cold path"
        );
        assert!(
            after_growth_hits > 0,
            "fresh grown backing must later permit Resident execution"
        );
    }
    // A changed cohort is allowed to use the ordinary path. Both implementations
    // still execute and compare every output/state/control, then return to M1.
    for position in 0..4 {
        let ranges = [
            initial + position..initial + position + 1,
            position..position + 1,
        ];
        let work = pair(
            &fixtures,
            &sessions,
            &tokens,
            &ranges,
            kind,
            "two_participants",
            pass,
            position,
            if position < 2 {
                Path::Warm
            } else {
                Path::Replay
            },
            position % 4 >= 2,
            None,
        );
        for arm in 0..2 {
            total[arm].add(work[arm]);
        }
    }
    let mut return_hits = 0;
    for position in 0..4 {
        let start = initial + 4 + position;
        let work = pair(
            &fixtures,
            &sessions,
            &tokens,
            &[start..start + 1],
            kind,
            "return_single",
            pass,
            position,
            Path::Replay,
            position % 4 >= 2,
            None,
        );
        return_hits += work[1].resident_hits;
        for arm in 0..2 {
            total[arm].add(work[arm]);
        }
    }
    assert!(return_hits > 0);
    let old_authorities = sessions.each_ref().map(|group| {
        group
            .iter()
            .map(|s| s.sequence_authority())
            .collect::<Vec<_>>()
    });
    for group in &mut sessions {
        for session in group.drain(..) {
            session.try_complete().unwrap();
        }
    }
    let fresh_tokens: Vec<Arc<[u32]>> = (0..2)
        .map(|p| (0..12).map(|i| ((i * 5 + p + 11) % 32) as u32).collect())
        .collect();
    let fresh = fixtures.each_ref().map(|fixture| {
        fresh_tokens
            .iter()
            .enumerate()
            .map(|(p, t)| {
                fixture.admit_with_ceiling(
                    &format!("resident-{pass}-fresh-{p}"),
                    Arc::from(&t[..1]),
                    t.len(),
                )
            })
            .collect::<Vec<_>>()
    });
    for arm in 0..2 {
        for (p, session) in fresh[arm].iter().enumerate() {
            assert_ne!(session.sequence_authority(), old_authorities[arm][p]);
        }
    }
    let mut fresh_hits = 0;
    for position in 0..8 {
        let work = pair(
            &fixtures,
            &fresh,
            &fresh_tokens,
            &[position..position + 1],
            kind,
            "fresh_single",
            pass,
            position,
            if position < 2 {
                Path::Warm
            } else {
                Path::Replay
            },
            position % 4 >= 2,
            None,
        );
        fresh_hits += work[1].resident_hits;
        for arm in 0..2 {
            total[arm].add(work[arm]);
        }
    }
    assert!(
        fresh_hits > 0,
        "fresh request must reauthorize and actually execute Resident"
    );
    let reordered = fresh
        .each_ref()
        .map(|group| vec![Arc::clone(&group[1]), Arc::clone(&group[0])]);
    let reordered_tokens = vec![Arc::clone(&fresh_tokens[1]), Arc::clone(&fresh_tokens[0])];
    for position in 0..4 {
        let ranges = [position..position + 1, 8 + position..9 + position];
        let work = pair(
            &fixtures,
            &reordered,
            &reordered_tokens,
            &ranges,
            kind,
            "fresh_reordered_participants",
            pass,
            position,
            if position < 2 {
                Path::Warm
            } else {
                Path::Replay
            },
            position % 4 >= 2,
            None,
        );
        for arm in 0..2 {
            total[arm].add(work[arm]);
        }
    }
    drop(reordered);
    for group in fresh {
        for session in group {
            session.try_complete().unwrap();
        }
    }
    println!(
        "{}",
        serde_json::json!({"kind":"resident_segment_complete","attention":format!("{kind:?}"),
            "pass":pass,"oracle":format!("{oracle:?}"),"paired_hot_waves":8,"paired_totals":paired,"all_wave_totals":total,
        "setup_ns":setup_ns,"complete_lifecycle_wall_ns":started.elapsed().as_nanos(),
        "growth_fallback":growth_fallback,"fresh_hits":fresh_hits,"return_single_hits":return_hits,
        "full_output_and_state_equal":true,"binding_controls_valid":true,
        "performance_scope":"synthetic real Core attention/norm/joined-FFN fixture; not full model throughput; only oracle-disabled pass is performance evidence; no threshold asserted"})
    );
}

fn compare(kind: AttentionKind) {
    run(
        kind,
        SegmentBindingOracleMode::CompareReference,
        "correctness_oracle",
    );
    run(
        kind,
        SegmentBindingOracleMode::Disabled,
        "performance_full_path",
    );
}

#[test]
#[ignore = "requires actual CUDA Q6 provider family and resident segment backend"]
fn resident_gdn_joined_ffn_matches_indexed_across_fresh_frames_and_requests() {
    compare(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires actual CUDA Q6 provider family, resident backend and physical KV growth"]
fn resident_causal_joined_ffn_matches_indexed_across_kv_growth_and_widths() {
    compare(AttentionKind::Causal);
}
