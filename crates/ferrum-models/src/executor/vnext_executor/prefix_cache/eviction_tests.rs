use super::*;

fn add(
    index: &mut PrefixIndex<Arc<str>>,
    prefix: &[u32],
    input: &[u32],
    original_prompt_tokens: Option<usize>,
    name: &str,
) {
    drop(index.insert(
        Arc::from(prefix),
        Arc::from(input),
        Arc::from(name),
        original_prompt_tokens,
    ));
}

#[test]
fn prefix_eviction_preserves_input_when_generated_output_is_rendered_differently() {
    for purpose in [
        PrefixEvictionPurpose::Foreground,
        PrefixEvictionPurpose::PromptCapture,
    ] {
        let mut index = PrefixIndex::default();
        add(&mut index, &[1, 2, 3], &[1, 2, 3, 4], Some(4), "input");
        add(&mut index, &[9, 8], &[9, 8, 7], Some(3), "other-input");
        add(
            &mut index,
            &[1, 2, 3, 4, 5, 6],
            &[1, 2, 3, 4, 5, 6],
            Some(4),
            "generated",
        );
        // The most recently used generated checkpoint is still the preferred
        // victim. Both exact repeats and a re-rendered continuation need input.
        assert_eq!(
            index
                .longest(&[1, 2, 3, 4, 5, 6, 7], false, |_| true)
                .as_deref(),
            Some("generated")
        );
        assert_eq!(index.evict_for(purpose).as_deref(), Some("generated"));
        assert_eq!(
            index
                .longest(&[1, 2, 3, 4, 99, 7], false, |_| true)
                .as_deref(),
            Some("input")
        );
        assert_eq!(
            index.longest(&[1, 2, 3, 4], false, |_| true).as_deref(),
            Some("input")
        );
        // Hard capacity pressure can still release every owner, in LRU order.
        assert_eq!(index.evict_for(purpose).as_deref(), Some("other-input"));
        assert_eq!(index.evict_for(purpose).as_deref(), Some("input"));
        assert!(index.evict_for(purpose).is_none());
    }
}

#[test]
fn prefix_eviction_requires_positive_generation_provenance() {
    let mut index = PrefixIndex::default();
    add(&mut index, &[1, 2], &[1, 2, 3], Some(3), "partial");
    add(&mut index, &[4, 5], &[4, 5], Some(2), "prompt-end");
    add(&mut index, &[6, 7], &[6, 7], None, "unknown-origin");
    add(&mut index, &[8, 9], &[8, 9], Some(3), "invalid-origin");
    add(
        &mut index,
        &[10, 11],
        &[10, 11, 12],
        Some(1),
        "invalid-partial",
    );
    add(&mut index, &[13, 14], &[13, 14], Some(1), "generated-old");
    add(&mut index, &[15, 16], &[15, 16], Some(1), "generated-new");

    let purpose = PrefixEvictionPurpose::GeneratedCapture;
    assert_eq!(index.evict_for(purpose).as_deref(), Some("generated-old"));
    assert_eq!(index.evict_for(purpose).as_deref(), Some("generated-new"));
    assert!(index.evict_for(purpose).is_none());
    for expected in [
        "partial",
        "prompt-end",
        "unknown-origin",
        "invalid-origin",
        "invalid-partial",
    ] {
        assert_eq!(index.evict().as_deref(), Some(expected));
    }
}

#[tokio::test]
async fn prefix_generated_capture_skips_under_pressure_without_discarding_input() {
    let index = Mutex::new(PrefixIndex::default());
    add(
        &mut index.lock(),
        &[1, 2, 3],
        &[1, 2, 3, 4],
        Some(4),
        "input",
    );
    add(
        &mut index.lock(),
        &[1, 2, 3, 4, 5],
        &[1, 2, 3, 4, 5],
        Some(4),
        "generated",
    );
    let pin = index
        .lock()
        .longest(&[1, 2, 3, 4, 5, 6], false, |_| true)
        .unwrap();
    let entries = index.lock().entries.len();
    let result = capture_with_capacity(
        entries,
        || std::future::ready(Ok(CaptureAttempt::<(), ()>::NeedsMaintenance(()))),
        |()| std::future::ready(Ok(CaptureMaintenanceDecision::CapacityLimited)),
        || {
            index
                .lock()
                .evict_for(PrefixEvictionPurpose::GeneratedCapture)
                .is_some()
        },
    )
    .await
    .unwrap();
    // Eviction cannot free the restore pin, nor justify sacrificing input when
    // the next real maintenance attempt still finds insufficient capacity.
    assert!(result.is_none());
    assert_eq!(Arc::strong_count(&pin), 1);
    assert_eq!(
        index
            .lock()
            .longest(&[1, 2, 3, 4, 99], false, |_| true)
            .as_deref(),
        Some("input")
    );
    assert_eq!(index.lock().evict().as_deref(), Some("input"));
}
