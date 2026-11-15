use super::*;

#[tokio::test]
async fn delete_and_shutdown_keep_pending_waiter_and_slot_until_real_terminal() {
    let registry = Arc::new(Registry::new(1));
    let slot = registry.reserve().unwrap();
    let id = Uuid::new_v4();
    let host = super::super::tests::host();
    let owner = host
        .allocate_zeroed(ferrum_runtime::HostMemoryClass::Output, 128)
        .unwrap();
    // Exercise the HTTP record/terminal boundary with an actual blocked waiter
    // and admitted payload. Engine/codec cancellation itself is qualified by
    // their tests; this fixture never constructs or returns a synthetic video.
    registry.state().entries.insert(
        id,
        Entry {
            snapshot: Snapshot {
                id,
                status: Status::InProgress,
                created_at: 0,
                seed: 42,
                cancel_requested: false,
                delete_requested: false,
                error: None,
                encoded_bytes: None,
            },
            cancel: None,
            outcome: None,
            _slot: slot.clone(),
        },
    );
    let (release, wait) = tokio::sync::oneshot::channel::<()>();
    let worker = registry.clone();
    registry.state().tasks.push(tokio::spawn(async move {
        let _slot = slot;
        let _owner = owner;
        wait.await.unwrap();
        worker.complete(id, Outcome::Finished(Err(VideoJobError::Cancelled)));
    }));
    assert_eq!(registry.delete(id).unwrap().status, Status::InProgress);
    assert!(registry.reserve().is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 128);
    let tasks = registry.close();
    assert!(registry.reserve().is_err());
    assert_eq!(tasks.len(), 1);
    assert!(!tasks[0].is_finished());
    release.send(()).unwrap();
    for task in tasks {
        task.await.unwrap();
    }
    assert!(registry.get(id).is_none());
    assert_eq!(registry.slots.available_permits(), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn ingress_reservations_and_readers_share_atomic_shutdown_boundary() {
    let registry = Registry::new(1);
    let first = registry.reserve().unwrap();
    assert!(registry.reserve().is_err());
    drop(first);
    assert!(registry.reserve().is_ok());
    let readers = Arc::new(Semaphore::new(1));
    let reader = registry.admit_reader(&readers).unwrap();
    assert!(registry.admit_reader(&readers).is_err());
    assert!(registry.close().is_empty());
    drop(reader);
    assert_eq!(readers.available_permits(), 1);
    assert!(registry.admit_reader(&readers).is_err());
}
