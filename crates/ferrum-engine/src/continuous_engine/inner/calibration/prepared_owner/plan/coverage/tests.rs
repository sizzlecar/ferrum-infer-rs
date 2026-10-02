use super::*;
use ferrum_interfaces::vnext::{
    BoundDecodeContextCoverage, DecodeContextBoundaryKind, DecodeContextCoverage, NodeId,
    ProviderId,
};
use std::num::NonZeroU64;

fn cohort(
    width: usize,
    maximum_output: usize,
    prefix: PrefixKind,
    suffix_tokens: usize,
) -> PreparedProbeCohort {
    PreparedProbeCohort {
        pass: 0,
        ordinal: 0,
        template: 0,
        original_template_index: 0,
        width,
        maximum_output: NonZeroUsize::new(maximum_output).unwrap(),
        suffix_tokens,
        preset: SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
        prefix,
        route: CalibrationDecodeRoute::Actual,
        prefill_chunk: None,
        native_acquisition: None,
        acquisition_key: None,
        reset_token_policy: false,
        seed: 0,
    }
}

fn boundary(first: u64) -> DecodeContextBoundary {
    DecodeContextBoundary {
        first_sequence_tokens: NonZeroU64::new(first).unwrap(),
        kind: DecodeContextBoundaryKind::KernelFamily,
    }
}

fn declaration(maximum: u64, boundaries: Vec<DecodeContextBoundary>) -> DecodeContextCoverage {
    DecodeContextCoverage::Declared {
        maximum_sequence_tokens: NonZeroU64::new(maximum).unwrap(),
        boundaries,
    }
}

fn nodes(coverage: Vec<DecodeContextCoverage>) -> ExecutorDecodeContextCoverage {
    ExecutorDecodeContextCoverage {
        nodes: coverage
            .into_iter()
            .enumerate()
            .map(|(i, coverage)| BoundDecodeContextCoverage {
                node_id: NodeId::try_from(format!("node.{i}")).unwrap(),
                provider_id: ProviderId::try_from("provider.test".to_owned()).unwrap(),
                coverage,
            })
            .collect(),
    }
}

#[test]
fn omitted_widths_include_intermediate_widths_before_any_execution() {
    let cohorts = [1, 2, 4].map(|width| cohort(width, 2, PrefixKind::Ordinary, 2));
    let audit = inspect(&cohorts, &[3], 5, 32, &nodes(vec![declaration(31, vec![])])).unwrap();
    assert_eq!(audit.planned_rows, [1, 2, 4]);
    assert_eq!(audit.unprobed_rows, [3, 5]);
    assert!(audit.has_known_gaps());
    assert!(!audit.has_undeclared_routes());
}

#[test]
fn boundary_uses_sequence_frontier_including_current_input_and_checks_both_sides() {
    let declarations = nodes(vec![declaration(31, vec![boundary(9)])]);
    // Prefill at length seven emits token one. The next decode frontier is
    // eight, not seven; output two stops before the boundary at nine.
    let short = inspect(
        &[cohort(1, 2, PrefixKind::Ordinary, 2)],
        &[7],
        1,
        32,
        &declarations,
    )
    .unwrap();
    assert_eq!(short.minimum_planned_decode_sequence_tokens, Some(8));
    assert_eq!(short.maximum_planned_decode_sequence_tokens, Some(8));
    assert!(short.decode_boundaries[0].lower_sequence_planned);
    assert!(!short.decode_boundaries[0].upper_sequence_planned);
    assert!(short.has_known_gaps());

    let crosses = inspect(
        &[cohort(1, 3, PrefixKind::Ordinary, 3)],
        &[7],
        1,
        32,
        &declarations,
    )
    .unwrap();
    assert!(crosses.decode_boundaries[0].lower_sequence_planned);
    assert!(crosses.decode_boundaries[0].upper_sequence_planned);
    assert!(!crosses.has_known_gaps());

    let above = inspect(
        &[cohort(1, 2, PrefixKind::Ordinary, 2)],
        &[8],
        1,
        32,
        &declarations,
    )
    .unwrap();
    assert!(!above.decode_boundaries[0].lower_sequence_planned);
    assert!(above.decode_boundaries[0].upper_sequence_planned);
    assert!(above.has_known_gaps());
}

#[test]
fn prepared_prefix_does_not_claim_its_preparation_as_ordinary_decode_coverage() {
    let declarations = nodes(vec![declaration(31, vec![boundary(9)])]);
    for prefix in [
        PrefixKind::Clean,
        PrefixKind::Pending,
        PrefixKind::Mixed { pending_rows: 1 },
    ] {
        let audit = inspect(&[cohort(2, 5, prefix, 2)], &[7], 2, 32, &declarations).unwrap();
        assert_eq!(audit.minimum_planned_decode_sequence_tokens, Some(10));
        assert_eq!(audit.maximum_planned_decode_sequence_tokens, Some(11));
        assert!(!audit.decode_boundaries[0].lower_sequence_planned);
        assert!(!audit.decode_boundaries[0].upper_sequence_planned);
    }
}

#[test]
fn unknown_nodes_preserve_known_boundaries_and_differ_from_declared_no_switch() {
    let cohorts = [cohort(1, 3, PrefixKind::Ordinary, 3)];
    let complete = inspect(&cohorts, &[7], 1, 32, &nodes(vec![declaration(31, vec![])])).unwrap();
    assert!(!complete.has_undeclared_routes());
    assert!(!complete.has_known_gaps());

    let absent = inspect(
        &cohorts,
        &[7],
        1,
        32,
        &ExecutorDecodeContextCoverage::default(),
    )
    .unwrap();
    assert!(absent.has_undeclared_routes());

    let mixed = inspect(
        &cohorts,
        &[7],
        1,
        32,
        &nodes(vec![
            declaration(31, vec![boundary(9)]),
            DecodeContextCoverage::Unknown {
                maximum_sequence_tokens: NonZeroU64::new(31),
                known_boundaries: vec![boundary(13)],
            },
            DecodeContextCoverage::default(),
        ]),
    )
    .unwrap();
    assert_eq!(mixed.provider_nodes, 3);
    assert_eq!(mixed.undeclared_provider_nodes, 2);
    assert_eq!(mixed.decode_boundaries.len(), 2);
    assert!(mixed.decode_boundaries[0].upper_sequence_planned);
    assert!(!mixed.decode_boundaries[1].upper_sequence_planned);
    assert!(mixed.has_undeclared_routes());
    assert!(mixed.has_known_gaps());
}

#[test]
fn context_clips_unreachable_boundaries_but_reports_a_short_provider_domain() {
    let declarations = nodes(vec![declaration(
        7,
        vec![boundary(1), boundary(8), boundary(9)],
    )]);
    let audit = inspect(
        &[cohort(1, 2, PrefixKind::Ordinary, 2)],
        &[6],
        1,
        8,
        &declarations,
    )
    .unwrap();
    assert!(audit.decode_boundaries.is_empty());
    assert!(!audit.has_known_gaps());
    let short = inspect(
        &[cohort(1, 2, PrefixKind::Ordinary, 2)],
        &[6],
        1,
        9,
        &declarations,
    )
    .unwrap();
    assert_eq!(short.provider_domains_shorter_than_configured, 1);
    assert_eq!(short.decode_boundaries.len(), 1);
    assert!(short.has_known_gaps());
}

#[test]
fn one_output_token_has_no_decode_opportunity_and_invalid_frontiers_are_rejected() {
    let declarations = nodes(vec![declaration(31, vec![boundary(9)])]);
    let audit = inspect(
        &[cohort(1, 1, PrefixKind::Ordinary, 1)],
        &[8],
        1,
        32,
        &declarations,
    )
    .unwrap();
    assert_eq!(audit.minimum_planned_decode_sequence_tokens, None);
    assert_eq!(audit.maximum_planned_decode_sequence_tokens, None);
    assert!(!audit.decode_boundaries[0].upper_sequence_planned);
    assert!(inspect(
        &[cohort(1, 3, PrefixKind::Ordinary, 3)],
        &[8],
        1,
        10,
        &declarations
    )
    .is_err());
    assert!(inspect(
        &[cohort(1, 3, PrefixKind::Ordinary, 3)],
        &[],
        1,
        32,
        &declarations
    )
    .is_err());
}
