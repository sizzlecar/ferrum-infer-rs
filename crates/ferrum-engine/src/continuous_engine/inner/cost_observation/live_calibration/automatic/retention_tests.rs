use super::*;

fn select(
    origins: VecDeque<[u8; 32]>,
    fresh: &BTreeSet<[u8; 32]>,
    charges: &BTreeMap<[u8; 32], Option<usize>>,
    retained: usize,
    bytes: usize,
) -> Result<(VecDeque<[u8; 32]>, usize), FerrumError> {
    let owners = origins.iter().copied().map(|origin| (origin, 1)).collect();
    retained_origins_within_budget(origins, fresh, charges, retained, bytes, &owners, 128, &[])
}

#[test]
fn owner_capacity_evicts_whole_old_origin_without_selecting_fresh_children() {
    let origins = VecDeque::from([[1; 32], [2; 32]]);
    let charges = BTreeMap::from([([1; 32], Some(20)), ([2; 32], Some(20))]);
    let owners = BTreeMap::from([([1; 32], 80), ([2; 32], 80)]);
    let (kept, bytes) = retained_origins_within_budget(
        origins,
        &BTreeSet::from([[2; 32]]),
        &charges,
        4,
        100,
        &owners,
        128,
        &[],
    )
    .unwrap();
    assert_eq!(kept, VecDeque::from([[2; 32]]));
    assert_eq!(bytes, 20);
}

#[test]
fn oversized_old_origin_is_evicted_in_order_while_fresh_population_is_whole() {
    let origins = VecDeque::from([[1; 32], [2; 32], [3; 32]]);
    let charges = BTreeMap::from([
        ([1; 32], Some(20)),
        ([2; 32], Some(101)),
        ([3; 32], Some(40)),
    ]);
    let (kept, bytes) = select(origins, &BTreeSet::from([[3; 32]]), &charges, 3, 100).unwrap();
    assert_eq!(kept, VecDeque::from([[3; 32]]));
    assert_eq!(bytes, 40);
}

#[test]
fn fresh_payload_overflow_or_excess_is_rejected_without_dropping_fresh_members() {
    for bytes in [None, Some(101)] {
        let origins = VecDeque::from([[1; 32], [2; 32]]);
        let charges = BTreeMap::from([([1; 32], Some(20)), ([2; 32], bytes)]);
        assert!(select(origins, &BTreeSet::from([[2; 32]]), &charges, 2, 100).is_err());
    }
}

#[test]
fn retained_count_and_payload_boundary_keep_the_newest_complete_origins() {
    let origins = VecDeque::from([[1; 32], [2; 32], [3; 32]]);
    let charges = BTreeMap::from([
        ([1; 32], Some(100)),
        ([2; 32], Some(100)),
        ([3; 32], Some(100)),
    ]);
    let (kept, bytes) = select(origins, &BTreeSet::from([[3; 32]]), &charges, 2, 100).unwrap();
    assert_eq!(kept, VecDeque::from([[2; 32], [3; 32]]));
    assert_eq!(bytes, 200);
}

#[test]
fn unknown_old_payload_cannot_be_kept_as_zero_bytes() {
    let origins = VecDeque::from([[1; 32], [2; 32]]);
    let charges = BTreeMap::from([([1; 32], None), ([2; 32], Some(100))]);
    let (kept, bytes) = select(origins, &BTreeSet::from([[2; 32]]), &charges, 2, 100).unwrap();
    assert_eq!(kept, VecDeque::from([[2; 32]]));
    assert_eq!(bytes, 100);
}

#[test]
fn startup_coverage_retention_protects_only_this_pass_without_charging_old_as_fresh() {
    let origins = VecDeque::from([[1; 32], [2; 32], [3; 32]]);
    let charges = BTreeMap::from([
        ([1; 32], Some(100)),
        ([2; 32], Some(100)),
        ([3; 32], Some(100)),
    ]);
    let owners = BTreeMap::from([([1; 32], 1), ([2; 32], 1), ([3; 32], 1)]);
    let fresh = BTreeSet::from([[3; 32]]);
    let (kept, bytes) = retained_origins_within_budget(
        origins.clone(),
        &fresh,
        &charges,
        2,
        100,
        &owners,
        128,
        &[[1; 32]],
    )
    .unwrap();
    assert_eq!(kept, VecDeque::from([[1; 32], [3; 32]]));
    assert_eq!(
        bytes, 200,
        "two existing per-origin allowances; only origin 3 is fresh"
    );
    let (unprotected, _) = select(origins, &fresh, &charges, 2, 100).unwrap();
    assert_eq!(
        unprotected,
        VecDeque::from([[2; 32], [3; 32]]),
        "protection is not remembered across calls"
    );
}

#[test]
fn startup_coverage_retention_cannot_relax_count_owner_or_per_origin_bounds() {
    let origins = VecDeque::from([[1; 32], [2; 32]]);
    let fresh = BTreeSet::from([[2; 32]]);
    for (retained, old_bytes, fresh_bytes, owners_each, owner_limit) in [
        (1, Some(100), Some(100), 1, 128),
        (2, Some(100), Some(100), 65, 128),
        (2, Some(101), Some(100), 1, 128),
        (2, None, Some(100), 1, 128),
        (2, Some(100), Some(101), 1, 128),
    ] {
        let charges = BTreeMap::from([([1; 32], old_bytes), ([2; 32], fresh_bytes)]);
        let owners = BTreeMap::from([([1; 32], owners_each), ([2; 32], owners_each)]);
        assert!(retained_origins_within_budget(
            origins.clone(),
            &fresh,
            &charges,
            retained,
            100,
            &owners,
            owner_limit,
            &[[1; 32]],
        )
        .is_err());
    }
}
