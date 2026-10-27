use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Plan(Arc<AtomicUsize>);
impl Drop for Plan {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn live_compiled_and_command_leases_cannot_be_evicted() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut cache = RetainedPlanCache::new(1);
    let compiled = cache
        .get_or_try_insert_with(1, || Ok::<_, ()>(Arc::new(Plan(drops.clone()))))
        .unwrap();
    let command = Arc::clone(&compiled);
    drop(compiled);
    assert!(matches!(
        cache.get_or_try_insert_with(2, || -> Result<Arc<Plan>, ()> {
            panic!("must reject before preparing an over-capacity native plan")
        }),
        Err(PlanCacheError::AllPlansRetained)
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(Arc::ptr_eq(&cache.get(&1).unwrap(), &command));
    drop(command);
    let replacement = cache
        .get_or_try_insert_with(2, || {
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            Ok::<_, ()>(Arc::new(Plan(drops.clone())))
        })
        .unwrap();
    drop(cache);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(replacement);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn failed_candidates_and_shape_switching_remain_bounded() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut cache = RetainedPlanCache::new(2);
    let valid = cache
        .get_or_try_insert_with(0, || Ok::<_, ()>(Arc::new(Plan(drops.clone()))))
        .unwrap();
    for shape in 1..64 {
        let candidate = cache
            .get_or_try_insert_with(shape, || Ok::<_, ()>(Arc::new(Plan(drops.clone()))))
            .unwrap();
        // Simulate preparation/capacity failure after this native stage.
        drop(candidate);
        assert!(cache.plans.len() <= 2);
        assert!(Arc::ptr_eq(&cache.get(&0).unwrap(), &valid));
    }
    assert_eq!(drops.load(Ordering::SeqCst), 62);
    assert!(matches!(
        cache.get_or_try_insert_with(64, || Err("native compilation failed")),
        Err(PlanCacheError::Prepare("native compilation failed"))
    ));
    assert_eq!(cache.plans.len(), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 63);
    assert!(Arc::ptr_eq(&cache.get(&0).unwrap(), &valid));
}

#[test]
fn cache_hit_reuses_the_exact_plan_and_reclaims_least_recent_unleased() {
    let mut cache = RetainedPlanCache::new(2);
    for key in [1, 2] {
        cache
            .get_or_try_insert_with(key, || Ok::<_, ()>(Arc::new(key)))
            .unwrap();
    }
    let same = cache
        .get_or_try_insert_with(1, || -> Result<Arc<i32>, ()> { panic!("cache hit") })
        .unwrap();
    drop(same);
    cache
        .get_or_try_insert_with(3, || Ok::<_, ()>(Arc::new(3)))
        .unwrap();
    assert!(cache.get(&1).is_some());
    assert!(cache.get(&2).is_none());
    assert!(cache.get(&3).is_some());
}
