use super::*;
use crate::vnext::{selection_mask_bytes_match, ProductTokenMaskContent};

fn selection(source: &Arc<[i8]>) -> ProductTokenMaskContent {
    // Deliberately identical fingerprints exercise the full-content contract.
    ProductTokenMaskContent::selection(5, 19, source)
}

#[test]
fn planning_selection_masks_preserve_raw_contents_order_and_mode_transitions() {
    let slot = real_mask_step_identity();
    let a: Arc<[i8]> = Arc::from([0, 1, -1, 0, 1]);
    let equal_a: Arc<[i8]> = Arc::from(a.as_ref());
    let b: Arc<[i8]> = Arc::from([1, 0, 0, 1, 1]);
    let normalized_equal_a: Arc<[i8]> = Arc::from([0, 2, -1, 0, 1]);
    let initial = ProductTokenMaskResidencySnapshot::new(true, 4, vec![], &mut || true).unwrap();
    let mut state = initial.clone();
    let mut project = |rows: &[ProductTokenMaskContent]| {
        state
            .project_contents(Some(&slot), 5, rows, &mut || true)
            .unwrap()
    };
    assert_eq!(project(&[selection(&a), selection(&b)]), [true, true]);
    assert_eq!(
        project(&[selection(&equal_a), selection(&b)]),
        [false, false]
    );
    assert_eq!(project(&[selection(&b), selection(&a)]), [true, true]);
    // Normalized upload bytes can be equal while the actual source comparison
    // intentionally misses. A u64 fingerprint cannot authorize skipping it.
    assert_eq!(
        project(&[selection(&b), selection(&normalized_equal_a)]),
        [false, true]
    );
    let full = ProductTokenMaskContent::AllValid { vocabulary_size: 5 };
    assert_eq!(project(&[full.clone(), full.clone()]), [true, true]);
    assert_eq!(project(&[full.clone(), full]), [false, false]);
    assert_eq!(project(&[selection(&b), selection(&a)]), [true, true]);
    let mut sibling = initial;
    assert_eq!(
        sibling
            .project_contents(Some(&slot), 5, &[selection(&a)], &mut || true)
            .unwrap(),
        [true]
    );
}

#[test]
fn planning_selection_masks_keep_full_source_length_and_slot_identity() {
    let slot = real_mask_step_identity();
    let other_slot = real_mask_step_identity();
    let a: Arc<[i8]> = Arc::from([1, 0, 1, 0, 1, 8]);
    let b: Arc<[i8]> = Arc::from([1, 0, 1, 0, 1, 9]);
    let short: Arc<[i8]> = Arc::from([1, 0]);
    let padded: Arc<[i8]> = Arc::from([1, 0, 0, 0, 0]);
    let mut state = ProductTokenMaskResidencySnapshot::new(true, 4, vec![], &mut || true).unwrap();
    for rows in [
        [selection(&a)],
        [selection(&b)],
        [selection(&short)],
        [selection(&padded)],
    ] {
        assert_eq!(
            state
                .project_contents(Some(&slot), 5, &rows, &mut || true)
                .unwrap(),
            [true]
        );
    }
    assert_eq!(
        state
            .project_contents(Some(&slot), 5, &[selection(&padded)], &mut || true)
            .unwrap(),
        [false]
    );
    assert_eq!(
        state
            .project_contents(Some(&other_slot), 5, &[selection(&padded)], &mut || true)
            .unwrap(),
        [true]
    );
}

#[test]
fn planning_selection_masks_do_not_pin_sources_and_expiration_is_honest() {
    let slot = real_mask_step_identity();
    let a: Arc<[i8]> = Arc::from([0, 1, 0, 1, 0]);
    let weak = Arc::downgrade(&a);
    let content = ProductTokenMaskContent::capture_selection(5, 19, 5, &weak).unwrap();
    let mut state = ProductTokenMaskResidencySnapshot::new(
        true,
        4,
        vec![ProductTokenMaskResidencyEntry::with_content(
            slot.clone(),
            0,
            content,
        )],
        &mut || true,
    )
    .unwrap();
    drop(a);
    assert!(
        weak.upgrade().is_none(),
        "a retained cost view must not preserve actual residency"
    );
    let b: Arc<[i8]> = Arc::from([0, 1, 0, 1, 0]);
    assert_eq!(
        state.project_contents(Some(&slot), 5, &[selection(&b)], &mut || true),
        Err(ExecutionCostRouteUnknown::StaleView)
    );
    let expired = ProductTokenMaskContent::capture_selection(5, 19, 5, &weak).unwrap();
    let mut fresh = ProductTokenMaskResidencySnapshot::new(
        true,
        4,
        vec![ProductTokenMaskResidencyEntry::with_content(
            slot.clone(),
            0,
            expired,
        )],
        &mut || true,
    )
    .unwrap();
    assert_eq!(
        fresh
            .project_contents(Some(&slot), 5, &[selection(&b)], &mut || true)
            .unwrap(),
        [true]
    );
}

#[test]
fn planning_selection_masks_poll_full_content_comparison_and_reject_missing_proof() {
    let left: Arc<[i8]> = vec![1; 16 * 1024].into();
    let right: Arc<[i8]> = vec![1; 16 * 1024].into();
    let mut polls = 0;
    assert_eq!(
        selection_mask_bytes_match(
            &left,
            &right,
            Some(&mut || {
                polls += 1;
                polls < 3
            })
        ),
        Err(ExecutionCostRouteUnknown::BudgetExhausted)
    );
    assert!(
        polls > 1,
        "full equality must not be a single uninterruptible read"
    );
    assert!(selection_mask_bytes_match(&left, &right, None).unwrap());
    let slot = real_mask_step_identity();
    let mut unknown = ProductTokenMaskResidencySnapshot::new(
        true,
        4,
        vec![ProductTokenMaskResidencyEntry::new(slot.clone(), 0, None)],
        &mut || true,
    )
    .unwrap();
    assert_eq!(
        unknown.project_contents(Some(&slot), 5, &[selection(&left)], &mut || true),
        Err(ExecutionCostRouteUnknown::OutputBranch)
    );
    assert!(ProductTokenMaskContent::capture_selection(
        5,
        19,
        left.len() + 1,
        &Arc::downgrade(&left)
    )
    .is_err());
}
