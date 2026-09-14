use super::super::active_coverage::ActivePrefixInput;
use super::*;
use checkpoint_fixture::{Fixture, Spec};
use std::num::NonZeroU64;

fn layout_fixture(dependency: CheckpointInputDependency, suffix_alignment: u64) -> Fixture {
    Fixture::build(Spec {
        dependency,
        boundaries: CheckpointBoundaryConstraint::new(
            CheckpointTokenSpanConstraint::any_positive(),
            CheckpointTokenSpanConstraint::new(
                NonZeroU64::MIN,
                NonZeroU64::new(suffix_alignment).unwrap(),
            )
            .unwrap(),
        )
        .unwrap(),
        checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
        ..Spec::default()
    })
    .unwrap()
}

fn add(
    index: &mut PrefixIndex<Arc<str>>,
    prefix: &[u32],
    input: &[u32],
    original: usize,
    label: &str,
) {
    drop(index.insert(
        Arc::from(prefix),
        Arc::from(input),
        Arc::from(label),
        Some(original),
    ));
}

fn active(tokens: &[u32]) -> ActivePrefixInput {
    ActivePrefixInput {
        request_id: RequestId::new(),
        tokens: tokens.to_vec(),
    }
}

fn evicted(outcome: CaptureEviction<Arc<str>>) -> Arc<str> {
    match outcome {
        CaptureEviction::Evicted(owner) => owner,
        _ => panic!("expected an unprotected victim"),
    }
}

#[test]
fn active_prefix_coverage_preserves_deepest_hit_not_only_a_common_root() {
    let fixture = layout_fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = PrefixIndex::default();
    // Two snapshots along one growth chain are not two observed branches.
    add(&mut index, &[1], &[1, 2, 3], 3, "common-root");
    add(
        &mut index,
        &[1, 2, 3, 4],
        &[1, 2, 3, 4, 5],
        5,
        "deep-active",
    );
    add(&mut index, &[9, 8], &[9, 8, 7], 3, "other-input");
    let inputs = [active(&[1, 2, 3, 4, 5])];
    let purpose = PrefixEvictionPurpose::PromptCapture;
    assert_eq!(
        evicted(index.evict_with_active_coverage(purpose, &inputs, layout)).as_ref(),
        "common-root"
    );
    assert_eq!(
        evicted(index.evict_with_active_coverage(purpose, &inputs, layout)).as_ref(),
        "other-input"
    );
    let CaptureEviction::Protected(coverage) =
        index.evict_with_active_coverage(purpose, &inputs, layout)
    else {
        panic!("a shorter removed root must not permit evicting the deepest hit");
    };
    assert_eq!(coverage.len(), 1);
    assert_eq!(coverage[0].request_id, inputs[0].request_id);
    assert_eq!(coverage[0].prefix_tokens, 4);
    assert_eq!(
        index.longest(&inputs[0].tokens, false, |_| true).as_deref(),
        Some("deep-active")
    );
    assert_eq!(
        evicted(index.evict_with_active_coverage(
            PrefixEvictionPurpose::Foreground,
            &inputs,
            layout
        ))
        .as_ref(),
        "deep-active"
    );
}

#[test]
fn active_prefix_coverage_precedes_generated_victim_preference() {
    let fixture = layout_fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = PrefixIndex::default();
    add(&mut index, &[1, 2, 3], &[1, 2, 3], 2, "active-generated");
    add(&mut index, &[9], &[9, 8], 2, "other-input");
    add(&mut index, &[7, 6], &[7, 6], 1, "other-generated");
    let inputs = [active(&[1, 2, 3, 4])];
    assert_eq!(
        evicted(index.evict_with_active_coverage(
            PrefixEvictionPurpose::PromptCapture,
            &inputs,
            layout
        ))
        .as_ref(),
        "other-generated"
    );
    assert!(matches!(
        index.evict_with_active_coverage(PrefixEvictionPurpose::GeneratedCapture, &inputs, layout),
        CaptureEviction::Protected(_)
    ));
    assert_eq!(
        evicted(index.evict_with_active_coverage(
            PrefixEvictionPurpose::PromptCapture,
            &inputs,
            layout
        ))
        .as_ref(),
        "other-input"
    );
    // Once no other active request needs this prefix, the original class/LRU
    // policy applies again. There is no permanent branch pin or reservation.
    assert_eq!(
        evicted(index.evict_with_active_coverage(
            PrefixEvictionPurpose::GeneratedCapture,
            &[],
            layout
        ))
        .as_ref(),
        "active-generated"
    );
}

#[test]
fn active_prefix_coverage_uses_suffix_and_entire_input_contracts() {
    let tokens = (1..=10).collect::<Vec<_>>();
    let inputs = [active(&tokens[..8])];
    let fixture = layout_fixture(CheckpointInputDependency::ExactTokenPrefix, 3);
    let layout = usable_layout(&fixture.plan).unwrap();
    assert!(layout.permits_capture_from(0, 7, 10));
    assert!(layout.permits_capture_from(0, 5, 8));
    let mut index = PrefixIndex::default();
    add(
        &mut index,
        &tokens[..7],
        &tokens,
        tokens.len(),
        "longer-illegal-suffix",
    );
    add(&mut index, &tokens[..5], &tokens[..8], 8, "legal-suffix");
    assert_eq!(
        evicted(index.evict_with_active_coverage(
            PrefixEvictionPurpose::PromptCapture,
            &inputs,
            layout
        ))
        .as_ref(),
        "longer-illegal-suffix"
    );
    assert!(matches!(
        index.evict_with_active_coverage(PrefixEvictionPurpose::PromptCapture, &inputs, layout),
        CaptureEviction::Protected(_)
    ));
    assert_eq!(
        index
            .longest(&inputs[0].tokens, false, |b| layout
                .permits_suffix(b as u64, 8))
            .as_deref(),
        Some("legal-suffix")
    );

    let fixture = layout_fixture(CheckpointInputDependency::EntireTokenInput, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = PrefixIndex::default();
    add(
        &mut index,
        &tokens[..6],
        &tokens,
        tokens.len(),
        "different-entire-input",
    );
    add(&mut index, &tokens[..8], &tokens[..8], 8, "no-suffix");
    add(
        &mut index,
        &tokens[..4],
        &tokens[..8],
        8,
        "matching-entire-input",
    );
    for expected in ["different-entire-input", "no-suffix"] {
        assert_eq!(
            evicted(index.evict_with_active_coverage(
                PrefixEvictionPurpose::PromptCapture,
                &inputs,
                layout
            ))
            .as_ref(),
            expected
        );
    }
    assert!(matches!(
        index.evict_with_active_coverage(PrefixEvictionPurpose::PromptCapture, &inputs, layout),
        CaptureEviction::Protected(_)
    ));
    assert_eq!(
        index.longest(&inputs[0].tokens, true, |_| true).as_deref(),
        Some("matching-entire-input")
    );
}

#[tokio::test]
async fn active_prefix_coverage_stops_optional_retry_without_releasing_restore_pins() {
    let fixture = layout_fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut entries = PrefixIndex::default();
    add(&mut entries, &[1, 2], &[1, 2, 3], 3, "active-a");
    add(&mut entries, &[9, 8], &[9, 8, 7], 3, "active-b");
    let pin = entries.longest(&[1, 2, 3], false, |_| true).unwrap();
    let inputs = [active(&[1, 2, 3]), active(&[9, 8, 7])];
    let index = Mutex::new(entries);
    let attempts = std::cell::Cell::new(0);
    let maintained = std::cell::Cell::new(0);
    let protected = std::cell::Cell::new(0);
    let result = capture_with_capacity(
        2,
        || {
            attempts.set(attempts.get() + 1);
            std::future::ready(Ok(CaptureAttempt::<(), ()>::NeedsMaintenance(())))
        },
        |()| {
            maintained.set(maintained.get() + 1);
            std::future::ready(Ok(CaptureMaintenanceDecision::CapacityLimited))
        },
        || match index.lock().evict_with_active_coverage(
            PrefixEvictionPurpose::PromptCapture,
            &inputs,
            layout,
        ) {
            CaptureEviction::Protected(coverage) => {
                assert_eq!(coverage.len(), 2);
                protected.set(protected.get() + 1);
                false
            }
            _ => panic!("all eligible owners protect active input coverage"),
        },
    )
    .await
    .unwrap();
    assert!(result.is_none());
    assert_eq!(
        (attempts.get(), maintained.get(), protected.get()),
        (1, 1, 1)
    );
    assert_eq!(Arc::strong_count(&pin), 2);
    drop(index);
    assert_eq!(Arc::strong_count(&pin), 1);
}
