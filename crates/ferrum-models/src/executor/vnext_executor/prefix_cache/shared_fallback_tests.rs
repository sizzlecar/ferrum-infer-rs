use super::super::active_coverage::ActivePrefixInput;
use super::*;
use checkpoint_fixture::{Fixture, Spec};
use std::num::NonZeroU64;

fn fixture(dependency: CheckpointInputDependency, alignment: u64) -> Fixture {
    Fixture::build(Spec {
        dependency,
        boundaries: CheckpointBoundaryConstraint::new(
            CheckpointTokenSpanConstraint::any_positive(),
            CheckpointTokenSpanConstraint::new(
                NonZeroU64::MIN,
                NonZeroU64::new(alignment).unwrap(),
            )
            .unwrap(),
        )
        .unwrap(),
        checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
        ..Spec::default()
    })
    .unwrap()
}

fn add(index: &mut PrefixIndex<Arc<str>>, prefix: &[u32], input: &[u32], label: &str) {
    drop(index.insert(
        Arc::from(prefix),
        Arc::from(input),
        Arc::from(label),
        Some(input.len()),
    ));
}

fn branches() -> PrefixIndex<Arc<str>> {
    let mut index = PrefixIndex::default();
    add(&mut index, &[1, 2], &[1, 2, 3], "root");
    add(&mut index, &[1, 2, 3, 4], &[1, 2, 3, 4, 5], "a");
    add(&mut index, &[1, 2, 9, 8], &[1, 2, 9, 8, 7], "b");
    index
}

#[test]
fn shared_fallback_optional_capture_keeps_root_and_active_deep_hits() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = branches();
    let inputs = [
        ActivePrefixInput {
            request_id: RequestId::new(),
            tokens: vec![1, 2, 3, 4, 5],
        },
        ActivePrefixInput {
            request_id: RequestId::new(),
            tokens: vec![1, 2, 9, 8, 7],
        },
    ];
    assert_eq!(index.shared_fallbacks(&inputs, layout), BTreeSet::from([0]));
    assert!(matches!(
        index.evict_with_active_coverage(PrefixEvictionPurpose::PromptCapture, &inputs, layout),
        CaptureEviction::SharedFallbackProtected(prefixes) if prefixes == vec![2]
    ));
    for (input, expected) in [
        (&inputs[0].tokens[..], "a"),
        (&inputs[1].tokens[..], "b"),
        (&[1, 2, 7][..], "root"),
    ] {
        assert_eq!(
            index
                .longest(input, false, |n| layout
                    .permits_suffix(n as u64, input.len() as u64))
                .as_deref(),
            Some(expected)
        );
    }
    assert!(index.longest(&[1, 8, 7], false, |_| true).is_none());
}

#[test]
fn shared_fallback_foreground_evicts_descendants_then_root_and_releases_owners() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = branches();
    let root = Arc::clone(&index.entries[0].checkpoint);
    for expected in ["a", "b", "root"] {
        let victim = index
            .evict_for_layout(PrefixEvictionPurpose::Foreground, layout)
            .unwrap();
        assert_eq!(victim.as_ref(), expected);
        drop(victim);
    }
    assert!(index.entries.is_empty());
    assert!(index
        .evict_for_layout(PrefixEvictionPurpose::Foreground, layout)
        .is_none());
    assert_eq!(Arc::strong_count(&root), 1);
}

#[test]
fn shared_fallback_finds_divergence_after_a_short_common_prompt() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = PrefixIndex::default();
    add(&mut index, &[1], &[1, 2], "root");
    add(&mut index, &[1, 2, 3], &[1, 2, 3, 4], "a");
    add(&mut index, &[1, 2, 9], &[1, 2, 9, 8], "b");
    assert_eq!(index.shared_fallbacks(&[], layout), BTreeSet::from([0]));
}

#[test]
fn shared_fallback_does_not_count_growth_or_generated_sampling_as_new_prompts() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = PrefixIndex::default();
    add(&mut index, &[1, 2], &[1, 2, 3], "root");
    add(&mut index, &[1, 2, 3, 4], &[1, 2, 3, 4, 5], "growth");
    for (tokens, label) in [
        (&[1, 2, 3, 8][..], "sample-a"),
        (&[1, 2, 3, 9][..], "sample-b"),
    ] {
        drop(index.insert(
            Arc::from(tokens),
            Arc::from(tokens),
            Arc::from(label),
            Some(3),
        ));
    }
    assert!(index.shared_fallbacks(&[], layout).is_empty());
    assert_eq!(
        index
            .evict_for_layout(PrefixEvictionPurpose::PromptCapture, layout)
            .as_deref(),
        Some("sample-a")
    );
    assert_eq!(
        index
            .evict_for_layout(PrefixEvictionPurpose::GeneratedCapture, layout)
            .as_deref(),
        Some("sample-b")
    );
    assert!(index
        .evict_for_layout(PrefixEvictionPurpose::GeneratedCapture, layout)
        .is_none());
    assert_eq!(
        index
            .evict_for_layout(PrefixEvictionPurpose::Foreground, layout)
            .as_deref(),
        Some("root")
    );
}

#[test]
fn shared_fallback_requires_verified_original_prompts_and_legal_suffixes() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 2);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = PrefixIndex::default();
    add(&mut index, &[1, 2], &[1, 2, 3, 4], "root");
    // This state is valid at N=3 for its prompt, but the root's N=2 cannot
    // restore this five-token input under the selected suffix alignment.
    add(&mut index, &[1, 2, 9], &[1, 2, 9, 8, 7], "odd-suffix");
    assert!(index.shared_fallbacks(&[], layout).is_empty());
    let tokens: Arc<[u32]> = Arc::from([1, 2, 8, 7]);
    for origin in [None, Some(0), Some(8)] {
        drop(index.insert(
            Arc::from([1, 2]),
            Arc::clone(&tokens),
            Arc::from("unverified"),
            origin,
        ));
        assert!(index.shared_fallbacks(&[], layout).is_empty());
    }
    add(
        &mut index,
        &[1, 2, 9, 8],
        &[1, 2, 9, 8, 7, 6],
        "even-suffix",
    );
    assert_eq!(index.shared_fallbacks(&[], layout), BTreeSet::from([0]));
}

#[test]
fn shared_fallback_does_not_cross_entire_input_dependency() {
    let fixture = fixture(CheckpointInputDependency::EntireTokenInput, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut index = branches();
    assert!(index.shared_fallbacks(&[], layout).is_empty());
    assert!(index.longest(&[1, 2, 7], true, |_| true).is_none());
    assert_eq!(
        index
            .evict_for_layout(PrefixEvictionPurpose::Foreground, layout)
            .as_deref(),
        Some("root")
    );
}

#[tokio::test]
async fn shared_fallback_stops_optional_maintenance_without_pinning_capacity() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let mut entries = PrefixIndex::default();
    add(&mut entries, &[1, 2], &[1, 2, 3], "root");
    let root = Arc::clone(&entries.entries[0].checkpoint);
    let inputs = [ActivePrefixInput {
        request_id: RequestId::new(),
        tokens: vec![1, 2, 9],
    }];
    let index = Mutex::new(entries);
    let result = capture_with_capacity(
        1,
        || std::future::ready(Ok(CaptureAttempt::<(), ()>::NeedsMaintenance(()))),
        |()| std::future::ready(Ok(CaptureMaintenanceDecision::CapacityLimited)),
        || match index.lock().evict_with_active_coverage(
            PrefixEvictionPurpose::PromptCapture,
            &inputs,
            layout,
        ) {
            CaptureEviction::SharedFallbackProtected(_) => false,
            _ => panic!("optional capture must preserve the only shared fallback"),
        },
    )
    .await
    .unwrap();
    assert!(result.is_none());
    assert_eq!(Arc::strong_count(&root), 2);
    let victim = index
        .lock()
        .evict_for_layout(PrefixEvictionPurpose::Foreground, layout)
        .unwrap();
    drop(victim);
    assert_eq!(Arc::strong_count(&root), 1);
}
