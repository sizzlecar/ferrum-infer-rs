use super::*;

#[test]
fn a_longer_compatible_index_checkpoint_wins_over_a_short_handoff() {
    let prompt = [11, 22, 33, 44, 55, 66];
    let mut index = PrefixIndex::default();
    let indexed = insert(&mut index, &prompt[..4], &prompt, "indexed");
    let handoff: Arc<str> = Arc::from("handoff");
    let (selected, source) =
        index.longest_or_rendezvous(&prompt, false, |_| true, Some((2, &handoff)));
    assert!(matches!(source, PrefixRestoreSource::Index));
    assert!(Arc::ptr_eq(selected.as_ref().unwrap(), &indexed));
    assert_eq!(
        Arc::strong_count(&handoff),
        1,
        "lookup does not retain the losing handoff"
    );
}

#[test]
fn a_longer_handoff_or_equal_boundary_preserves_handoff_and_index_lru() {
    let prompt = [11, 22, 33, 44, 55, 66];
    for indexed_boundary in [2, 4] {
        let mut index = PrefixIndex::default();
        insert(&mut index, &prompt[..indexed_boundary], &prompt, "indexed");
        insert(&mut index, &[77], &[77, 88], "unrelated");
        let handoff: Arc<str> = Arc::from("handoff");
        let (selected, source) =
            index.longest_or_rendezvous(&prompt, false, |_| true, Some((4, &handoff)));
        assert!(matches!(source, PrefixRestoreSource::Rendezvous));
        assert!(Arc::ptr_eq(selected.as_ref().unwrap(), &handoff));
        assert_eq!(index.evict().as_deref(), Some("indexed"));
        assert_eq!(index.evict().as_deref(), Some("unrelated"));
    }
}

#[test]
fn incompatible_index_entries_do_not_displace_a_valid_handoff() {
    use checkpoint_fixture::{Fixture, Spec};
    use std::num::NonZeroU64;

    let fixture = Fixture::build(Spec {
        dependency: CheckpointInputDependency::ExactTokenPrefix,
        boundaries: CheckpointBoundaryConstraint::new(
            CheckpointTokenSpanConstraint::any_positive(),
            CheckpointTokenSpanConstraint::new(NonZeroU64::MIN, NonZeroU64::new(2).unwrap())
                .unwrap(),
        )
        .unwrap(),
        checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
        ..Spec::default()
    })
    .unwrap();
    let layout = usable_layout(&fixture.plan).unwrap();
    let prompt = [11, 22, 33, 44, 55, 66];
    let mut index = PrefixIndex::default();
    insert(
        &mut index,
        &[11, 22, 33, 99],
        &[11, 22, 33, 99, 55, 66],
        "diverged",
    );
    insert(&mut index, &prompt[..5], &prompt, "invalid-suffix");
    insert(&mut index, &prompt, &prompt, "no-suffix");
    let (selected, source) = index.longest_or_rendezvous(
        &prompt,
        false,
        |boundary| layout.permits_suffix(boundary as u64, prompt.len() as u64),
        Some((2, &Arc::from("handoff"))),
    );
    assert!(matches!(source, PrefixRestoreSource::Rendezvous));
    assert_eq!(selected.as_deref(), Some("handoff"));

    // Entire-input-dependent state still requires the complete admitted input
    // to match, even when its captured prefix itself is longer and compatible.
    let mut index = PrefixIndex::default();
    insert(
        &mut index,
        &prompt[..4],
        &[11, 22, 33, 44, 99, 66],
        "different-input",
    );
    let (selected, source) =
        index.longest_or_rendezvous(&prompt, true, |_| true, Some((2, &Arc::from("handoff"))));
    assert!(matches!(source, PrefixRestoreSource::Rendezvous));
    assert_eq!(selected.as_deref(), Some("handoff"));
}

#[test]
fn absence_of_a_usable_handoff_keeps_normal_index_lookup_and_miss_behavior() {
    let prompt = [11, 22, 33, 44];
    let mut index = PrefixIndex::default();
    insert(&mut index, &prompt[..2], &prompt, "indexed");
    let (selected, source) = index.longest_or_rendezvous(&prompt, false, |_| true, None);
    assert!(matches!(source, PrefixRestoreSource::Index));
    assert_eq!(selected.as_deref(), Some("indexed"));
    let (selected, source) = index.longest_or_rendezvous(&[77, 88], false, |_| true, None);
    assert!(matches!(source, PrefixRestoreSource::Index));
    assert!(selected.is_none());
}
