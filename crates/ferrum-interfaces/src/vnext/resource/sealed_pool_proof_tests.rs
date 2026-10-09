use super::*;

#[test]
fn sealed_pool_proof_invalidates_every_mutable_summary_before_revalidation() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;

    macro_rules! check_changed {
        ($view:ident, $change:expr) => {{
            let mut $view = pools.view_with_pool_version(&authority).unwrap();
            assert!(pools
                .revalidate_view_with_pool_version(&authority, &$view)
                .unwrap());
            assert!(pools
                .revalidate_view_with_pool_version_late_reference(&authority, &$view)
                .unwrap());
            pools.revalidate_view(&authority, &$view).unwrap();
            $change;
            assert!(!$view.has_pool_version_proof());
            let expected = pools.revalidate_view(&authority, &$view).unwrap_err();
            let actual = pools
                .revalidate_view_with_pool_version(&authority, &$view)
                .unwrap_err();
            let late = pools
                .revalidate_view_with_pool_version_late_reference(&authority, &$view)
                .unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
            assert_eq!(late.to_string(), expected.to_string());
        }};
    }

    check_changed!(view, view.payload.logical_size_bytes -= 1);
    check_changed!(view, view.payload.capacity_size_bytes += 1);
    check_changed!(view, view.payload.alignment_bytes *= 2);
    check_changed!(view, view.payload.usage = BufferUsage::Transfer);
    check_changed!(view, view.payload.element_type = ElementType::Bool);
    check_changed!(view, view.payload.storage_profile = linear_profile());
    check_changed!(view, view.payload.bindings.pop());

    let pool = &pools.pools[&harness.pool_ids[0]];
    let mut view = pools.view_with_pool_version(&authority).unwrap();
    view.payload.logical_size_bytes -= 1;
    pool.state.lock().unwrap().poisoned = true;
    let expected = pools.revalidate_view(&authority, &view).unwrap_err();
    let actual = pools
        .revalidate_view_with_pool_version(&authority, &view)
        .unwrap_err();
    let late = pools
        .revalidate_view_with_pool_version_late_reference(&authority, &view)
        .unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(late.to_string(), expected.to_string());
    assert!(actual.to_string().contains("fail-closed"));
    pool.state.lock().unwrap().poisoned = false;
}

#[test]
fn sealed_pool_proof_moves_only_with_its_exact_authority_and_pool_payload() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let equal_authority = authority.retained();
    assert_eq!(authority.evidence(), equal_authority.evidence());
    let mut original = pools.view_with_pool_version(&authority).unwrap();
    let mut equal = pools.view_with_pool_version(&equal_authority).unwrap();

    std::mem::swap(&mut original.payload, &mut equal.payload);
    // Moving all sealed data preserves its own proof, never the receiving
    // variable's old authority. Equal evidence is not the same borrowed slice.
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &original)
        .is_err());
    assert!(pools
        .revalidate_view_with_pool_version(&equal_authority, &equal)
        .is_err());
    assert!(pools
        .revalidate_view_with_pool_version(&equal_authority, &original)
        .unwrap());
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &equal)
        .unwrap());

    let (foreign, _foreign_authority) = paged_window_fixture();
    assert_eq!(harness.pool_ids, foreign.pool_ids);
    assert!(foreign
        .root
        .dynamic_pools
        .revalidate_view_with_pool_version(&equal_authority, &original)
        .is_err());

    std::mem::swap(&mut original.payload, &mut equal.payload);
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &original)
        .unwrap());
}

#[test]
fn sealed_pool_proof_mutable_escape_and_restoration_never_reauthorize_a_stamp() {
    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let mut view = pools.view_with_pool_version(&authority).unwrap();
    let size = view.size_bytes();
    // The mutable borrow alone must invalidate, even if the caller restores
    // exactly the old value or performs no writes through the borrow.
    let metadata = &mut view.payload.logical_size_bytes;
    *metadata -= 1;
    *metadata = size;
    assert!(!view.has_pool_version_proof());
    pools.revalidate_view(&authority, &view).unwrap();
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &view)
        .unwrap());
    assert!(!view.has_pool_version_proof());

    let mut checkpoint_view = pools.view_with_pool_version(&authority).unwrap();
    let before = checkpoint_view.segment_bindings().len();
    assert_eq!(checkpoint_view.payload.bindings.iter_mut().count(), before);
    assert!(!checkpoint_view.has_pool_version_proof());
    assert!(!pools
        .revalidate_view_with_pool_version(&authority, &checkpoint_view)
        .unwrap());
    let fresh = pools.view_with_pool_version(&authority).unwrap();
    assert!(pools
        .revalidate_view_with_pool_version(&authority, &fresh)
        .unwrap());
}

#[test]
#[ignore = "local paired diagnostic of complete admitted backing revalidation; no performance assertion"]
fn sealed_pool_proof_complete_revalidation_paired_diagnostic() {
    use std::hint::black_box;
    use std::time::Instant;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        Full,
        Late,
        Early,
    }
    use Arm::{Early, Full, Late};

    let (harness, authority) = paged_window_fixture();
    let pools = &harness.root.dynamic_pools;
    let view = pools.view_with_pool_version(&authority).unwrap();
    for _ in 0..1024 {
        pools.revalidate_view(&authority, &view).unwrap();
        assert!(pools
            .revalidate_view_with_pool_version(&authority, &view)
            .unwrap());
        assert!(pools
            .revalidate_view_with_pool_version_late_reference(&authority, &view)
            .unwrap());
    }

    let iterations = 100_000_u64;
    let orders = [
        [Full, Late, Early],
        [Early, Late, Full],
        [Late, Early, Full],
        [Full, Early, Late],
        [Early, Full, Late],
        [Late, Full, Early],
    ];
    for (repetition, order) in orders.into_iter().enumerate() {
        for (position, arm) in order.into_iter().enumerate() {
            let mut hits = 0_u64;
            let start = Instant::now();
            for _ in 0..iterations {
                match arm {
                    Full => pools
                        .revalidate_view(black_box(&authority), black_box(&view))
                        .unwrap(),
                    Late => {
                        hits += u64::from(
                            pools
                                .revalidate_view_with_pool_version_late_reference(
                                    black_box(&authority),
                                    black_box(&view),
                                )
                                .unwrap(),
                        );
                    }
                    Early => {
                        hits += u64::from(
                            pools
                                .revalidate_view_with_pool_version(
                                    black_box(&authority),
                                    black_box(&view),
                                )
                                .unwrap(),
                        );
                    }
                }
            }
            let elapsed = start.elapsed();
            println!(
                "{}",
                serde_json::json!({
                    "diagnostic": "complete_admitted_backing_revalidation",
                    "mode": match arm {
                        Full => "full_locked",
                        Late => "r1_late_hit_reference",
                        Early => "r2_sealed_early_hit",
                    },
                    "repetition": repetition,
                    "position": position,
                    "iterations": iterations,
                    "physical_bindings": view.segment_bindings().len(),
                    "logical_bytes": view.size_bytes(),
                    "successful_hits": hits,
                    "elapsed_ns": elapsed.as_nanos(),
                    "ns_per_call": elapsed.as_secs_f64() * 1e9 / iterations as f64,
                    "scope": "one real admitted CPU paged view; same complete function with const-specialized early/late hit placement; excludes construction, runtime buffer coverage, providers and submission"
                })
            );
            assert_eq!(hits, if arm == Full { 0 } else { iterations });
        }
    }
}
