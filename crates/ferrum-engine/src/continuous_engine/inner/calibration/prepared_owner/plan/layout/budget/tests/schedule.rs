use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPhaseV2;

fn timeline(waves: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut total = 0;
    for &count in waves {
        starts.push(total);
        total += count;
        ends.push(total);
    }
    (starts, ends)
}

/// Count complete fresh cohorts directly on the repeated original timeline.
/// A cohort present at the cut is excluded even if it has later observations.
fn fresh_completed(
    starts: &[usize],
    ends: &[usize],
    marked: &[usize],
    cut: usize,
    offered: usize,
) -> usize {
    let cycle = *ends.last().unwrap();
    let deadline = cut + offered;
    (0..=deadline / cycle)
        .flat_map(|round| marked.iter().map(move |&index| (round, index)))
        .filter(|&(round, index)| {
            round * cycle + starts[index] > cut && round * cycle + ends[index] <= deadline
        })
        .count()
}

#[test]
fn startup_schedule_preserves_every_anchor_and_fresh_member_at_every_original_cut() {
    let settings = StructuredSettingsV2::default();
    // Vary cohort duration, member sparsity, and a cut at every individual
    // offer, including cuts within the cohort whose later rows are excluded.
    for first in 1..=3 {
        for second in 1..=3 {
            for third in 1..=3 {
                let (starts, ends) = timeline(&[first, second, third]);
                let cycle = *ends.last().unwrap();
                let (schedule, anchor) = startup_schedule(&starts, &ends, &settings).unwrap();
                let mut declared = settings.clone();
                declared.max_phase_samples = *schedule.maximum_phase_members.iter().max().unwrap();
                schedule.validate(&declared).unwrap();
                assert!(schedule.input_readiness.is_none());
                assert!(serde_json::to_value(&schedule)
                    .unwrap()
                    .get("input_readiness")
                    .is_none());
                for rank in 1..=settings.max_rank {
                    assert!(schedule.min_members[0] >= rank + settings.min_fit_redundancy);
                }
                for cut in 0..cycle {
                    for index in 0..ends.len() {
                        assert!(fresh_completed(&starts, &ends, &[index], cut, anchor) >= 1);
                    }
                }
                for mask in 1usize..1 << ends.len() {
                    let marked: Vec<_> = (0..ends.len()).filter(|i| mask & (1 << i) != 0).collect();
                    for (phase_index, phase) in [
                        StructuredPhaseV2::Fit,
                        StructuredPhaseV2::Residual,
                        StructuredPhaseV2::Qualification,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let members = schedule.min_members[phase_index];
                        let span = fresh_span(&starts, &ends, &marked, members).unwrap();
                        let offers =
                            round_block(span.max(schedule.phase_min_offered[phase_index]), cycle)
                                .unwrap();
                        for cut in 0..cycle {
                            assert!(
                                fresh_completed(&starts, &ends, &marked, cut, offers) >= members
                            );
                        }
                        assert!(schedule.is_ready(phase, offers as u64, members).unwrap());
                        assert!(!schedule
                            .is_ready(phase, offers as u64, members - 1)
                            .unwrap());
                    }
                }
            }
        }
    }
}

#[test]
fn startup_complete_horizon_is_bounded_by_declared_rank_and_member_requirements() {
    let settings = StructuredSettingsV2::default();
    let (mut cases, _) = cases_and_population();
    for (case, output) in cases.iter_mut().zip([1, 3, 2, 4]) {
        case.width = 1;
        case.maximum_output = NonZeroUsize::new(output).unwrap();
    }
    let waves: Vec<_> = cases
        .iter()
        .map(|case| case.waves(1, 1).unwrap().0)
        .collect();
    let (starts, ends) = timeline(&waves);
    let cycle = *ends.last().unwrap();
    let (schedule, _) = startup_schedule(&starts, &ends, &settings).unwrap();
    // Each eligible population may occur only once per complete cycle. The
    // worst bound includes discovery, one cut cohort per phase, all required
    // fresh members, and the original final audit block. Native defaults give
    // 1 + (36 + 1) + (8 + 1) + (8 + 1) + 1 = 57 complete cycles.
    let worst_cycles = 2 + schedule
        .min_members
        .iter()
        .map(|members| members + 1)
        .sum::<usize>();
    for index in 0..cases.len() {
        let marked = [index];
        let bound = plan_groups(
            &cases,
            &[1],
            1,
            None,
            &schedule,
            cycle,
            std::iter::once(marked.as_slice()),
        )
        .unwrap();
        let complete_offers = bound.required_original_offers + schedule.block_offered;
        assert!(complete_offers <= worst_cycles * cycle);
        let cycles = complete_offers.div_ceil(cycle);
        assert!(cycles * cycle >= complete_offers);
        assert!((cycles - 1) * cycle < complete_offers);

        // EOS may shorten actual work. Keep the original guaranteed offer
        // floor when declaring the entire source; never refund observed savings
        // into a different source choice or truncate its original cohorts.
        let minimum_cycle = cases.len();
        let early = plan_groups(
            &cases,
            &[1],
            1,
            None,
            &schedule,
            minimum_cycle,
            std::iter::once(marked.as_slice()),
        )
        .unwrap();
        assert_eq!(
            early.required_original_offers,
            bound.required_original_offers
        );
        let early_cycles =
            (early.required_original_offers + schedule.block_offered).div_ceil(minimum_cycle);
        assert!(early_cycles >= cycles);
        assert!(early_cycles * minimum_cycle >= complete_offers);
    }
}
