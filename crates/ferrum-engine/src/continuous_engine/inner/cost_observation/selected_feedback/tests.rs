use super::*;
use ferrum_types::SloSelectedFeedbackStorageV1 as Storage;
use std::{
    fs,
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
};
mod structured_scope;

fn policy() -> SloSelectedFeedbackSettingsV1 {
    SloSelectedFeedbackSettingsV1 {
        window_samples: NonZeroUsize::new(2).unwrap(),
        minimum_underestimates: NonZeroUsize::new(2).unwrap(),
        minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
        trigger_excess_ns: NonZeroU64::new(2).unwrap(),
        correction_padding_ns: 5,
        maximum_family_margin_ns: NonZeroU64::new(100).unwrap(),
        maximum_consumption_lag_ns: NonZeroU64::new(50).unwrap(),
        maximum_uncomparable_observations: 3,
        maximum_failed_or_partial: 1,
        maximum_queue_drops: 1,
        maximum_state_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
    }
}
fn binding() -> Binding {
    Binding {
        profile_sha256: [1; 32],
        source_sha256: [2; 32],
        fit_sha256: [3; 32],
        protocol_sha256: [4; 32],
        policy_sha256: Sha256::digest(serde_json::to_vec(&policy()).unwrap()).into(),
    }
}
fn comparison(actual: u64) -> Comparison {
    Comparison {
        family: [7; 32],
        base_planning_ns: 100,
        actual_ns: actual,
        observed_at_ns: 1,
        consumed_at_ns: 2,
    }
}
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("ferrum-feedback-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> PathBuf {
        self.0.join("receipt.json")
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn selected_feedback_requires_declared_consecutive_errors_and_never_compounds_margin() {
    let p = policy();
    let mut state = State::new(binding());
    state.compare(&p, 1, comparison(120));
    assert_eq!(
        state.margin(&[7; 32]),
        0,
        "one noise sample is below this declared trigger"
    );
    state.compare(&p, 1, comparison(100));
    state.compare(&p, 1, comparison(120));
    assert_eq!(state.margin(&[7; 32]), 0);
    state.compare(&p, 1, comparison(120));
    assert_eq!(state.margin(&[7; 32]), 25);
    let before = state.clone();
    for _ in 0..4 {
        state.compare(&p, 1, comparison(120));
    }
    assert!(state.same_prediction(&before));
    state.compare(&p, 1, comparison(140));
    state.compare(&p, 1, comparison(140));
    assert_eq!(
        state.margin(&[7; 32]),
        45,
        "base 100 -> 140+padding; not prior25+45"
    );
    assert_eq!(state.corrections, 2);
}

#[test]
fn selected_feedback_excess_and_observation_failures_revoke_without_learning_unknowns() {
    let p = policy();
    let mut high = State::new(binding());
    high.compare(&p, 1, comparison(200));
    high.compare(&p, 1, comparison(200));
    assert_eq!(high.revoked, Some(Revocation::CorrectionLimit));
    assert_eq!(high.margin(&[7; 32]), 0);
    high.compare(&p, 1, comparison(100));
    assert_eq!(high.revoked, Some(Revocation::CorrectionLimit));
    let mut late = State::new(binding());
    let mut value = comparison(120);
    value.consumed_at_ns = 52;
    late.compare(&p, 1, value);
    assert_eq!(late.revoked, Some(Revocation::ObservationLag));
    assert_eq!(late.compared, 0);
    let mut missing = State::new(binding());
    for _ in 0..4 {
        missing.uncomparable(&p);
    }
    assert_eq!(missing.revoked, Some(Revocation::UncomparableObservation));
    assert!(missing.families.is_empty());
    let mut loss = State::new(binding());
    loss.dropped(&p, 2);
    assert_eq!(loss.revoked, Some(Revocation::QueueLoss));
}

#[test]
fn selected_feedback_receipt_resumes_margin_and_sticky_revocation_but_rejects_new_policy() {
    let d = Directory::new();
    let p = policy();
    let (mut store, mut state) =
        store::Store::open(&Storage::CreateNew { path: d.path() }, &p, binding(), 1).unwrap();
    state.compare(&p, 1, comparison(120));
    state.compare(&p, 1, comparison(120));
    state.epoch += 1;
    state.revoke(Revocation::QueueLoss);
    store.finish(&state).unwrap();
    drop(store);
    let (mut resumed, loaded) =
        store::Store::open(&Storage::Resume { path: d.path() }, &p, binding(), 1).unwrap();
    assert_eq!(loaded.margin(&[7; 32]), 25);
    assert_eq!(loaded.epoch, 2);
    assert_eq!(loaded.session, 2);
    assert_eq!(loaded.revoked, Some(Revocation::QueueLoss));
    resumed.finish(&loaded).unwrap();
    drop(resumed);
    let mut changed = binding();
    changed.policy_sha256 = [99; 32];
    assert!(store::Store::open(&Storage::Resume { path: d.path() }, &p, changed, 1).is_err());
}

#[test]
fn selected_feedback_unclean_session_corruption_and_interrupted_write_fail_closed() {
    let p = policy();
    let dirty = Directory::new();
    let (owner, _) =
        store::Store::open(&Storage::CreateNew { path: dirty.path() }, &p, binding(), 1).unwrap();
    drop(owner); // Deliberately omit the acknowledged clean-shutdown operation.
    assert!(store::Store::open(&Storage::Resume { path: dirty.path() }, &p, binding(), 1).is_err());
    let corrupt = Directory::new();
    fs::write(corrupt.path(), b"{not-a-receipt}").unwrap();
    assert!(store::Store::open(
        &Storage::Resume {
            path: corrupt.path()
        },
        &p,
        binding(),
        1
    )
    .is_err());
    let altered = Directory::new();
    let (mut owner, state) = store::Store::open(
        &Storage::CreateNew {
            path: altered.path(),
        },
        &p,
        binding(),
        1,
    )
    .unwrap();
    owner.finish(&state).unwrap();
    let mut receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(altered.path()).unwrap()).unwrap();
    receipt["state"]["epoch"] = 17.into();
    fs::write(altered.path(), serde_json::to_vec(&receipt).unwrap()).unwrap();
    assert!(store::Store::open(
        &Storage::Resume {
            path: altered.path()
        },
        &p,
        binding(),
        1
    )
    .is_err());
    let interrupted = Directory::new();
    let (mut owner, state) = store::Store::open(
        &Storage::CreateNew {
            path: interrupted.path(),
        },
        &p,
        binding(),
        1,
    )
    .unwrap();
    fs::create_dir(interrupted.0.join("receipt.json.pending")).unwrap();
    assert!(
        owner.persist(&state).is_err(),
        "actual filesystem refuses an occupied pending publication"
    );
    assert!(owner.finish(&state).is_err());
    drop(owner);
    assert!(store::Store::open(
        &Storage::Resume {
            path: interrupted.path()
        },
        &p,
        binding(),
        1
    )
    .is_err());
}

#[test]
fn selected_feedback_session_is_exclusive_across_real_processes() {
    let d = Directory::new();
    let (mut owner, state) = store::Store::open(
        &Storage::CreateNew { path: d.path() },
        &policy(),
        binding(),
        1,
    )
    .unwrap();
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!(
                "{}::selected_feedback_child_cannot_open_owned_session",
                module_path!().split_once("::").unwrap().1
            ),
            "--ignored",
        ])
        .env("FERRUM_TEST_SELECTED_FEEDBACK_RECEIPT", d.path())
        .status()
        .unwrap();
    assert!(child.success());
    // Do not accept a harness that selected zero tests as evidence of exclusion.
    assert_eq!(
        fs::read(d.0.join("child-result")).unwrap(),
        b"live-owner-excluded"
    );
    owner.finish(&state).unwrap();
}

#[test]
#[ignore = "subprocess helper invoked with a parent-owned temporary receipt"]
fn selected_feedback_child_cannot_open_owned_session() {
    let path = PathBuf::from(
        std::env::var_os("FERRUM_TEST_SELECTED_FEEDBACK_RECEIPT").expect("test parent path"),
    );
    let result = store::Store::open(
        &Storage::Resume { path: path.clone() },
        &policy(),
        binding(),
        1,
    );
    assert!(
        result.is_err(),
        "another process may not acquire the live feedback receipt"
    );
    fs::write(
        path.parent().unwrap().join("child-result"),
        b"live-owner-excluded",
    )
    .unwrap();
}
