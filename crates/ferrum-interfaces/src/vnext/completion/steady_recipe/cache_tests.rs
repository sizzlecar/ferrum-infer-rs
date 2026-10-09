//! Cache boundary tests use a recipe sealed from real admitted wave resources.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Weak;

struct DropProbe {
    lane: Weak<ExecutionLane<TestRuntime>>,
    dropped: Arc<AtomicUsize>,
    locks_free: Arc<AtomicBool>,
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        let free = self
            .lane
            .upgrade()
            .is_none_or(|lane| lane.steady_recipe_test_locks_available());
        self.locks_free.fetch_and(free, Ordering::AcqRel);
        self.dropped.fetch_add(1, Ordering::AcqRel);
    }
}

#[test]
fn cache_publication_requires_the_current_entry_and_live_provider() {
    with_recipe_wave(|fixture, wave, identity, active| {
        let lane = wave.step_resources().execution_lane();
        let owner = Arc::new(());
        let entry = DeviceReusableExecutionEntryIdentity::new();
        let recipe = Arc::new(
            seal_recipe(
                fixture,
                wave,
                identity,
                active,
                0,
                Arc::new(7u64),
                &owner,
                entry.clone(),
            )
            .unwrap()
            .unwrap(),
        );
        let program = recipe.program.clone();

        // No backend entry: a successful caller cannot publish stale learning.
        lane.record_successful_steady_recipe(Arc::clone(&recipe))
            .unwrap();
        assert!(lane
            .steady_recipe(0, &owner, &program, &entry)
            .unwrap()
            .is_none());

        fixture.runtime_trace.lock().unwrap().steady_recipe_entry = Some(entry.clone());
        lane.record_successful_steady_recipe(Arc::clone(&recipe))
            .unwrap();
        assert!(Arc::ptr_eq(
            &lane
                .steady_recipe(0, &owner, &program, &entry)
                .unwrap()
                .unwrap(),
            &recipe
        ));

        let replacement = DeviceReusableExecutionEntryIdentity::new();
        fixture.runtime_trace.lock().unwrap().steady_recipe_entry = Some(replacement.clone());
        lane.record_successful_steady_recipe(Arc::clone(&recipe))
            .unwrap();
        assert!(lane
            .steady_recipe(0, &owner, &program, &replacement)
            .unwrap()
            .is_none());

        drop(owner);
        fixture.runtime_trace.lock().unwrap().steady_recipe_entry = Some(entry);
        let candidate = Arc::downgrade(&recipe);
        lane.record_successful_steady_recipe(recipe).unwrap();
        assert!(candidate.upgrade().is_none());
        let new_owner = Arc::new(());
        assert!(lane
            .steady_recipe(0, &new_owner, &program, &replacement)
            .unwrap()
            .is_none());
    });
}

#[test]
fn cache_rejects_foreign_lane_and_drops_mismatched_provider_state_outside_locks() {
    with_recipe_wave(|fixture, wave, identity, active| {
        let lane = wave.step_resources().execution_lane();
        let owner = Arc::new(());
        let entry = DeviceReusableExecutionEntryIdentity::new();
        let dropped = Arc::new(AtomicUsize::new(0));
        let locks_free = Arc::new(AtomicBool::new(true));
        let recipe = Arc::new(
            seal_recipe(
                fixture,
                wave,
                identity,
                active,
                0,
                Arc::new(DropProbe {
                    lane: Arc::downgrade(lane),
                    dropped: Arc::clone(&dropped),
                    locks_free: Arc::clone(&locks_free),
                }),
                &owner,
                entry.clone(),
            )
            .unwrap()
            .unwrap(),
        );
        let program = recipe.program.clone();
        let foreign = ExecutionLane::create(Arc::clone(&fixture.runtime)).unwrap();
        assert!(foreign
            .record_successful_steady_recipe(Arc::clone(&recipe))
            .is_err());
        fixture.runtime_trace.lock().unwrap().steady_recipe_entry = Some(entry.clone());
        lane.record_successful_steady_recipe(recipe).unwrap();

        assert!(lane
            .steady_recipe(0, &Arc::new(()), &program, &entry)
            .unwrap()
            .is_none());
        assert_eq!(dropped.load(Ordering::Acquire), 1);
        assert!(locks_free.load(Ordering::Acquire));
    });
}

#[test]
fn replacement_and_trim_release_cold_state_after_releasing_lane_and_cache() {
    with_recipe_wave(|fixture, wave, identity, active| {
        let lane = wave.step_resources().execution_lane();
        let owner = Arc::new(());
        let entry = DeviceReusableExecutionEntryIdentity::new();
        fixture.runtime_trace.lock().unwrap().steady_recipe_entry = Some(entry.clone());
        let dropped = Arc::new(AtomicUsize::new(0));
        let locks_free = Arc::new(AtomicBool::new(true));
        let mut program = None;
        for _ in 0..2 {
            let recipe = Arc::new(
                seal_recipe(
                    fixture,
                    wave,
                    identity,
                    active,
                    0,
                    Arc::new(DropProbe {
                        lane: Arc::downgrade(lane),
                        dropped: Arc::clone(&dropped),
                        locks_free: Arc::clone(&locks_free),
                    }),
                    &owner,
                    entry.clone(),
                )
                .unwrap()
                .unwrap(),
            );
            program = Some(recipe.program.clone());
            lane.record_successful_steady_recipe(recipe).unwrap();
        }
        assert_eq!(dropped.load(Ordering::Acquire), 1);
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .steady_recipe_trim_released = 1;
        assert!(lane.trim_reusable_executables_if_quiescent().unwrap());
        assert_eq!(dropped.load(Ordering::Acquire), 2);
        assert!(locks_free.load(Ordering::Acquire));
        assert!(lane
            .steady_recipe(0, &owner, &program.unwrap(), &entry)
            .unwrap()
            .is_none());
    });
}

#[test]
fn trim_between_completion_and_publication_prevents_stale_candidate_repopulation() {
    with_recipe_wave(|fixture, wave, identity, active| {
        let lane = wave.step_resources().execution_lane();
        let owner = Arc::new(());
        let entry = DeviceReusableExecutionEntryIdentity::new();
        let recipe = Arc::new(
            seal_recipe(
                fixture,
                wave,
                identity,
                active,
                0,
                Arc::new(7u64),
                &owner,
                entry.clone(),
            )
            .unwrap()
            .unwrap(),
        );
        let program = recipe.program.clone();
        {
            let mut trace = fixture.runtime_trace.lock().unwrap();
            trace.steady_recipe_entry = Some(entry.clone());
            trace.steady_recipe_trim_released = 1;
        }
        assert!(lane.trim_reusable_executables_if_quiescent().unwrap());
        // Even a backend marker returned again cannot restore the old epoch.
        fixture.runtime_trace.lock().unwrap().steady_recipe_entry = Some(entry.clone());
        lane.record_successful_steady_recipe(recipe).unwrap();
        assert!(lane
            .steady_recipe(0, &owner, &program, &entry)
            .unwrap()
            .is_none());
    });
}
