use super::*;

pub(super) fn issues(
    contract: &FrozenComparisonContract,
    cells: &[&CellComparison],
    configuration: &FrozenPairedBootstrap,
) -> Vec<String> {
    let mut issues = Vec::new();
    let design = &configuration.declared_design;
    if contract.pairs.len() < 2 {
        issues
            .push("a single paired cluster cannot estimate between-repetition uncertainty".into());
    }
    if contract
        .pairs
        .iter()
        .any(|pair| pair.selection.samples != contract.pairs[0].selection.samples)
    {
        issues.push("bootstrap design requires the same fixed ordered ShareGPT samples in every paired block".into());
    }
    let mut windows = Vec::new();
    let mut blocks = vec![(u64::MAX, 0); contract.pairs.len()];
    for cell in cells {
        if cell.status == ComparisonStatus::Unknown || cell.pairs.len() != contract.pairs.len() {
            issues.push(format!(
                "C{} does not contain an unambiguous validated full frozen paired family",
                cell.concurrency
            ));
            continue;
        }
        for (index, (pair, frozen)) in cell.pairs.iter().zip(&contract.pairs).enumerate() {
            if pair.pair_id != frozen.pair_id
                || pair.evidence_status != ComparisonStatus::ObservedPass
                || pair.ratios.len() != ComparisonMetric::ALL.len()
            {
                issues.push(format!(
                    "C{} / {} is not complete validated paired evidence",
                    cell.concurrency, frozen.pair_id
                ));
                continue;
            }
            let (Some(baseline), Some(candidate)) = (&pair.baseline_source, &pair.candidate_source)
            else {
                issues.push(format!(
                    "C{} / {} lacks acquisition/source evidence",
                    cell.concurrency, frozen.pair_id
                ));
                continue;
            };
            for (arm, source) in [("baseline", baseline), ("candidate", candidate)] {
                if source.offered_requests < design.minimum_measured_requests_per_arm as usize
                    || source.visible_gap_requests
                        < design.minimum_gap_bearing_requests_per_arm as usize
                    || (source.observed_visible_gaps as u64) < design.minimum_visible_gaps_per_arm
                {
                    issues.push(format!("C{} / {} / {arm} falls below declared P99 request/gap support (this declaration is not verified pilot adequacy)", cell.concurrency, frozen.pair_id));
                }
                let execution = &source.execution;
                windows.push((
                    execution.measurement_started_unix_ns,
                    execution.measurement_ended_unix_ns,
                ));
                blocks[index].0 = blocks[index].0.min(execution.measurement_started_unix_ns);
                blocks[index].1 = blocks[index].1.max(execution.measurement_ended_unix_ns);
            }
            let baseline_first =
                (design.first_pair_order == PairedArmOrder::BaselineFirst) ^ (index % 2 == 1);
            let (first, second) = if baseline_first {
                (baseline, candidate)
            } else {
                (candidate, baseline)
            };
            if first.execution.measurement_ended_unix_ns
                > second.execution.measurement_started_unix_ns
            {
                issues.push(format!(
                    "C{} / {} violates frozen alternating paired arm order",
                    cell.concurrency, frozen.pair_id
                ));
            }
        }
    }
    windows.sort_unstable();
    if windows.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        issues.push("primary same-hardware measurement windows overlap across cells/pairs".into());
    }
    if blocks.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        issues.push("frozen paired blocks are interleaved/reordered; complete cross-cell clusters were not acquired in frozen order".into());
    }
    issues
}
