use super::*;
use ferrum_interfaces::vnext::{
    BoundDecodeContextCoverage, DecodeContextBoundary, DecodeContextBoundaryKind,
    DecodeContextCoverage, NodeId, ProviderId,
};
use std::num::NonZeroU64;

fn nodes(complete: bool, maximum: u64, boundaries: &[u64]) -> ExecutorDecodeContextCoverage {
    let boundaries = boundaries
        .iter()
        .map(|n| DecodeContextBoundary {
            first_sequence_tokens: NonZeroU64::new(*n).unwrap(),
            kind: DecodeContextBoundaryKind::KernelFamily,
        })
        .collect();
    ExecutorDecodeContextCoverage {
        nodes: vec![BoundDecodeContextCoverage {
            node_id: NodeId::try_from("geometry.attention".to_owned()).unwrap(),
            provider_id: ProviderId::try_from("geometry.provider".to_owned()).unwrap(),
            coverage: if complete {
                DecodeContextCoverage::Declared {
                    maximum_sequence_tokens: NonZeroU64::new(maximum).unwrap(),
                    boundaries,
                }
            } else {
                DecodeContextCoverage::Unknown {
                    maximum_sequence_tokens: NonZeroU64::new(maximum),
                    known_boundaries: boundaries,
                }
            },
        }],
    }
}

#[test]
fn requires_integer_widths_both_switch_sides_and_context_tail_without_execution_claim() {
    let declarations = nodes(true, 2047, &[513, 1025, 1537]);
    let plan = ProbeGeometryRequirements::new(32, 8, 2048, 9, &declarations, 8192).unwrap();
    assert_eq!(
        plan.sequence_tokens,
        [9, 512, 513, 1024, 1025, 1536, 1537, 2047]
    );
    let points: Vec<_> = plan.points().collect();
    assert_eq!(points.len(), plan.point_count().unwrap());
    for width in 1..=8 {
        for tokens in &plan.sequence_tokens {
            assert!(points.contains(&(width, *tokens)));
        }
    }
    assert_eq!(plan.configured_maximum_rows, 32);
    assert_eq!(
        plan.probe_maximum_rows, 8,
        "private capacity is not full serving coverage"
    );
    assert_eq!(plan.undeclared_provider_nodes, 0);
}

#[test]
fn deduplicates_provider_boundaries_but_preserves_unknown_and_short_domains() {
    let mut declarations = nodes(true, 31, &[9, 17]);
    declarations.nodes.extend(nodes(false, 15, &[9, 13]).nodes);
    let plan = ProbeGeometryRequirements::new(3, 3, 32, 9, &declarations, 8192).unwrap();
    assert_eq!(plan.sequence_tokens, [8, 9, 12, 13, 16, 17, 31]);
    assert_eq!(plan.undeclared_provider_nodes, 1);
    assert_eq!(plan.provider_domains_shorter_than_configured, 1);
    assert_eq!(
        plan.intervals
            .iter()
            .filter(|i| **i
                == DecodeInterval {
                    first_sequence_tokens: 8,
                    last_sequence_tokens: 9,
                })
            .count(),
        1
    );
    let absent = ProbeGeometryRequirements::new(1, 1, 32, 9, &Default::default(), 8192).unwrap();
    assert_eq!(absent.provider_nodes, 0);
}

#[test]
fn excludes_impossible_decode_and_no_room_for_final_output() {
    let declarations = nodes(true, 32, &[1, 2, 3, 31, 32, u64::MAX]);
    let plan = ProbeGeometryRequirements::new(1, 1, 32, 9, &declarations, 8192).unwrap();
    assert_eq!(plan.sequence_tokens, [2, 3, 9, 30, 31]);
    assert!(ProbeGeometryRequirements::new(1, 1, 32, 32, &declarations, 8192).is_err());
    assert!(ProbeGeometryRequirements::new(1, 2, 32, 9, &declarations, 8192).is_err());
    assert!(ProbeGeometryRequirements::new(1, 1, 32, 9, &declarations, 1).is_err());
}

#[test]
fn product_prompt_window_covers_both_boundary_sides_with_real_prefix_and_final_output() {
    let interval = DecodeInterval {
        first_sequence_tokens: 512,
        last_sequence_tokens: 513,
    };
    let (minimum, maximum) = interval
        .prompt_window(
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(32).unwrap(),
        )
        .unwrap();
    assert_eq!((minimum.get(), maximum.get()), (482, 509));
    for prompt in minimum.get()..=maximum.get() {
        assert!(prompt + 3 <= 512);
        assert!(prompt + 31 >= 513);
    }
    assert!(interval
        .prompt_window(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(4).unwrap())
        .is_none());
    assert!(interval
        .prompt_window(NonZeroUsize::new(3).unwrap(), NonZeroUsize::new(3).unwrap())
        .is_none());
    let early = DecodeInterval {
        first_sequence_tokens: 2,
        last_sequence_tokens: 3,
    };
    assert!(early
        .prompt_window(
            NonZeroUsize::new(3).unwrap(),
            NonZeroUsize::new(32).unwrap()
        )
        .is_none());
}
