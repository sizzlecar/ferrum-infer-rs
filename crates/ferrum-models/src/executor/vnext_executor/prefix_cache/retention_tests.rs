use super::checkpoint_fixture::{Fixture, Spec};
use super::*;
use ferrum_interfaces::vnext::{
    CheckpointBoundaryConstraint, CheckpointTokenSpanConstraint, ProviderId,
};
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
        completed_input_providers: BTreeSet::from([ProviderId::new("provider.selected").unwrap()]),
        checkpoint_capacity: Some(CheckpointCapacityPolicy::new(1 << 20).unwrap()),
        ..Spec::default()
    })
    .unwrap()
}

fn publish(
    index: &mut PrefixIndex<Arc<str>>,
    layout: &SequenceCheckpointLayout,
    prefix: &[u32],
    input: &[u32],
    original_prompt_tokens: Option<usize>,
    name: &str,
) -> (Arc<str>, Vec<Arc<str>>) {
    let owner: Arc<str> = Arc::from(name);
    drop(index.insert(
        Arc::from(prefix),
        Arc::from(input),
        Arc::clone(&owner),
        original_prompt_tokens,
    ));
    let removed = index.coalesce(layout);
    (owner, removed)
}

#[test]
fn growing_input_coalesces_intermediate_states_and_keeps_repeat_and_endpoint_hits() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let tokens = (1..=20).collect::<Vec<u32>>();
    let mut index = PrefixIndex::default();
    let mut removed_names = Vec::new();
    for prompt in [4, 8, 12, 16] {
        for (boundary, input, name) in [
            (prompt - 1, prompt, format!("partial-{prompt}")),
            (prompt + 2, prompt + 2, format!("end-{prompt}")),
        ] {
            let (_, removed) = publish(
                &mut index,
                layout,
                &tokens[..boundary],
                &tokens[..input],
                Some(prompt),
                &name,
            );
            removed_names.extend(removed.iter().map(|owner| owner.to_string()));
        }
    }
    assert_eq!(
        index.longest(&tokens[..16], false, |_| true).as_deref(),
        Some("partial-16")
    );
    assert_eq!(
        index.longest(&tokens, false, |_| true).as_deref(),
        Some("end-16")
    );
    // Intermediate requests remain eligible for a shorter prefix, deliberately
    // losing their former near-end hit. Never rewind the newer boundary state.
    assert_eq!(
        index.longest(&tokens[..12], false, |_| true).as_deref(),
        Some("partial-4")
    );
    assert!(removed_names.iter().any(|name| name == "partial-12"));
    assert!(removed_names.iter().any(|name| name == "end-12"));
    assert!(index.entries.iter().all(|entry| matches!(
        entry.checkpoint.as_ref(),
        "partial-4" | "partial-16" | "end-16"
    )));
}

#[test]
fn branching_full_input_protects_a_partial_before_the_fork_and_both_frontiers() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let a = (1..=18).collect::<Vec<u32>>();
    let b = [1, 2, 3, 4, 5, 6, 90, 91, 92, 93, 94, 95];
    let mut index = PrefixIndex::default();
    drop(publish(
        &mut index,
        layout,
        &a[..3],
        &a[..4],
        Some(4),
        "early",
    ));
    drop(publish(
        &mut index,
        layout,
        &a[..6],
        &a[..6],
        Some(4),
        "fork",
    ));
    // This prefix has not reached B's divergence. Its full input is already
    // a distinct observed branch and must not be hidden by A's longer prefix.
    drop(publish(
        &mut index,
        layout,
        &b[..5],
        &b[..8],
        Some(8),
        "partial-b",
    ));
    drop(publish(
        &mut index,
        layout,
        &a[..9],
        &a[..10],
        Some(10),
        "partial-a-old",
    ));
    drop(publish(
        &mut index,
        layout,
        &a[..12],
        &a[..12],
        Some(10),
        "end-a-old",
    ));
    assert!(index
        .entries
        .iter()
        .any(|entry| entry.checkpoint.as_ref() == "partial-b"));
    assert_eq!(
        index.longest(&b[..8], false, |_| true).as_deref(),
        Some("fork")
    );
    drop(publish(
        &mut index,
        layout,
        &b[..10],
        &b[..10],
        Some(8),
        "end-b",
    ));
    drop(publish(
        &mut index,
        layout,
        &a[..13],
        &a[..14],
        Some(14),
        "partial-a",
    ));
    drop(publish(
        &mut index,
        layout,
        &a[..16],
        &a[..16],
        Some(14),
        "end-a",
    ));
    assert_eq!(
        index.longest(&a[..14], false, |_| true).as_deref(),
        Some("partial-a")
    );
    assert_eq!(index.longest(&a, false, |_| true).as_deref(), Some("end-a"));
    assert_eq!(index.longest(&b, false, |_| true).as_deref(), Some("end-b"));
    assert_eq!(
        index
            .longest(&[1, 2, 3, 4, 5, 6, 77], false, |_| true)
            .as_deref(),
        Some("fork")
    );
    assert!(index
        .entries
        .iter()
        .any(|entry| entry.checkpoint.as_ref() == "partial-b"));
    assert!(!index
        .entries
        .iter()
        .any(|entry| entry.checkpoint.as_ref() == "partial-a-old"));
}

#[test]
fn suffix_alignment_keeps_distinct_legal_fallbacks() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 4);
    let layout = usable_layout(&fixture.plan).unwrap();
    let tokens = (1..=18).collect::<Vec<u32>>();
    let mut index = PrefixIndex::default();
    for (boundary, prompt, name) in [
        (2, 6, "aligned-two"),
        (3, 7, "aligned-three"),
        (6, 10, "middle"),
        (10, 14, "current-partial"),
    ] {
        assert!(layout.permits_suffix(boundary, prompt));
        drop(publish(
            &mut index,
            layout,
            &tokens[..boundary as usize],
            &tokens[..prompt as usize],
            Some(prompt as usize),
            name,
        ));
    }
    drop(publish(
        &mut index,
        layout,
        &tokens[..14],
        &tokens[..14],
        Some(14),
        "current-end",
    ));
    let lookup = |index: &mut PrefixIndex<Arc<str>>, prompt: usize| {
        index.longest(&tokens[..prompt], false, |n| {
            layout.permits_suffix(n as u64, prompt as u64)
        })
    };
    assert_eq!(lookup(&mut index, 7).as_deref(), Some("aligned-three"));
    assert_eq!(lookup(&mut index, 10).as_deref(), Some("aligned-two"));
    assert_eq!(lookup(&mut index, 14).as_deref(), Some("current-partial"));
    assert!(!index
        .entries
        .iter()
        .any(|entry| entry.checkpoint.as_ref() == "middle"));
}

#[test]
fn entire_input_dependency_does_not_coalesce_different_inputs() {
    let fixture = fixture(CheckpointInputDependency::EntireTokenInput, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    assert_eq!(
        layout.input_dependency(),
        CheckpointInputDependency::EntireTokenInput
    );
    let tokens = (1..=8).collect::<Vec<u32>>();
    let mut index = PrefixIndex::default();
    for (boundary, prompt, name) in [(2, 3, "first"), (4, 5, "second"), (6, 7, "third")] {
        let (_, removed) = publish(
            &mut index,
            layout,
            &tokens[..boundary],
            &tokens[..prompt],
            Some(prompt),
            name,
        );
        assert!(removed.is_empty());
    }
    assert_eq!(
        index.longest(&tokens[..5], true, |_| true).as_deref(),
        Some("second")
    );
    assert!(index.longest(&tokens[..6], true, |_| true).is_none());
}

#[test]
fn missing_origin_or_exact_prompt_partial_does_not_infer_a_replacement() {
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let tokens = (1..=14).collect::<Vec<u32>>();
    for origin in [None, Some(10)] {
        let mut index = PrefixIndex::default();
        drop(index.insert(
            Arc::from(&tokens[..2]),
            Arc::from(&tokens[..3]),
            Arc::<str>::from("early"),
            Some(3),
        ));
        drop(index.insert(
            Arc::from(&tokens[..4]),
            Arc::from(&tokens[..5]),
            Arc::<str>::from("middle"),
            Some(5),
        ));
        // A prefix ancestor alone does not establish the endpoint's original
        // prompt. This partial belongs to an eleven-token, not ten-token input.
        drop(index.insert(
            Arc::from(&tokens[..9]),
            Arc::from(&tokens[..11]),
            Arc::<str>::from("other-prompt"),
            Some(11),
        ));
        let (_, removed) = publish(
            &mut index,
            layout,
            &tokens[..12],
            &tokens[..12],
            origin,
            "endpoint",
        );
        assert!(removed.is_empty());
        assert_eq!(
            index.longest(&tokens[..5], false, |_| true).as_deref(),
            Some("middle")
        );
    }
}

#[test]
fn coalescing_returns_owners_without_releasing_restore_pins_and_has_separate_metrics() {
    struct Owner {
        extent_bytes: u64,
    }
    let fixture = fixture(CheckpointInputDependency::ExactTokenPrefix, 1);
    let layout = usable_layout(&fixture.plan).unwrap();
    let tokens = (1..=10).collect::<Vec<u32>>();
    let index = Mutex::new(PrefixIndex::default());
    let metrics = PrefixCacheMetrics::default();
    let middle = Arc::new(Owner {
        extent_bytes: 131072,
    });
    for (boundary, prompt, owner) in [
        (
            2,
            3,
            Arc::new(Owner {
                extent_bytes: 65536,
            }),
        ),
        (4, 5, Arc::clone(&middle)),
        (
            6,
            7,
            Arc::new(Owner {
                extent_bytes: 196608,
            }),
        ),
    ] {
        drop(index.lock().insert(
            Arc::from(&tokens[..boundary]),
            Arc::from(&tokens[..prompt]),
            owner,
            Some(prompt),
        ));
    }
    let pin = index.lock().longest(&tokens[..5], false, |_| true).unwrap();
    let removed = {
        let mut locked = index.lock();
        drop(locked.insert(
            Arc::from(&tokens[..8]),
            Arc::from(&tokens[..8]),
            Arc::new(Owner {
                extent_bytes: 262144,
            }),
            Some(7),
        ));
        locked.coalesce(layout)
    };
    assert_eq!(
        Arc::strong_count(&middle),
        3,
        "caller, transfer pin and removed owner"
    );
    metrics.record_coalescing(
        removed.len(),
        removed.iter().map(|owner| owner.extent_bytes).sum(),
    );
    // The caller, including production publication, drops these owners only
    // after releasing the index lock. The transfer still retains its own Arc.
    drop(removed);
    assert_eq!(Arc::strong_count(&middle), 2);
    assert_eq!(pin.extent_bytes, 131072);
    let snapshot = index
        .lock()
        .snapshot(&fixture.plan, true, &metrics, |owner| owner.extent_bytes);
    assert_eq!(snapshot["coalesced_entries"], 1);
    assert_eq!(snapshot["coalesced_bytes"], 131072);
    assert_eq!(snapshot["evictions"], 0);
    assert_eq!(snapshot["bytes"], 65536 + 196608 + 262144);
    drop(pin);
    assert_eq!(Arc::strong_count(&middle), 1);
    metrics.reset();
    let reset = index
        .lock()
        .snapshot(&fixture.plan, true, &metrics, |owner| owner.extent_bytes);
    assert_eq!(reset["coalesced_entries"], 0);
    assert_eq!(reset["coalesced_bytes"], 0);
    assert_eq!(reset["entries"], 3);
}
