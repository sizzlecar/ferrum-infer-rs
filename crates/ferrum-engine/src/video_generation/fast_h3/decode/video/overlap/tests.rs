use super::*;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                weight_backing_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: bytes,
            },
        )
        .unwrap(),
    )
}

#[test]
fn rendezvous_bounds_live_readbacks_and_releases_original_host_credit() {
    let host = host(256);
    let (entered, receiving) = std::sync::mpsc::channel();
    let (release, proceed) = std::sync::mpsc::channel();
    let mut received = Vec::new();
    let received_ref = &mut received;
    run::<_, anyhow::Error>(
        |sender| {
            let first = host.allocate_zeroed(HostMemoryClass::Output, 128).unwrap();
            sender.send(first.freeze()).unwrap();
            receiving
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            let mut second = host.allocate_zeroed(HostMemoryClass::Output, 128).unwrap();
            second.as_mut()[0] = 1;
            let second = match sender.try_send(second.freeze()) {
                Err(std::sync::mpsc::TrySendError::Full(value)) => value,
                _ => panic!("CPU processing must not permit an extra queued readback"),
            };
            assert_eq!(host.usage().unwrap().allocated_bytes(), 256);
            assert!(host.allocate_zeroed(HostMemoryClass::Output, 1).is_err());
            release.send(()).unwrap();
            sender.send(second).unwrap();
            Ok(())
        },
        move |value| {
            received_ref.push(value.bytes()[0]);
            if received_ref.len() == 1 {
                entered.send(()).unwrap();
                proceed
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
            Ok(())
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(received, [0, 1]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[derive(Debug)]
enum OwningFailure {
    Native(Arc<()>),
    Host(anyhow::Error),
}
impl From<anyhow::Error> for OwningFailure {
    fn from(error: anyhow::Error) -> Self {
        Self::Host(error)
    }
}

#[test]
fn native_failure_keeps_its_owner_and_joins_the_failed_consumer() {
    let owner = Arc::new(());
    let finished = AtomicBool::new(false);
    let result = run(
        |sender| {
            sender.send(()).unwrap();
            Err(ProductionError::Original(OwningFailure::Native(
                owner.clone(),
            )))
        },
        |_| {
            finished.store(true, Ordering::SeqCst);
            Err(anyhow::anyhow!("CPU validation failed"))
        },
    )
    .unwrap();
    assert!(finished.load(Ordering::SeqCst));
    match result.unwrap_err() {
        OwningFailure::Native(retained) => {
            assert!(Arc::ptr_eq(&retained, &owner));
            assert_eq!(Arc::strong_count(&owner), 2);
        }
        OwningFailure::Host(error) => panic!("native owner lost: {error}"),
    }
    assert_eq!(Arc::strong_count(&owner), 1);
}

#[test]
fn consumer_failure_returns_original_diagnostic_and_drains_host_owners() {
    let host = host(256);
    let result = run::<_, anyhow::Error>(
        |sender| {
            for _ in 0..2 {
                let bytes = host
                    .allocate_zeroed(HostMemoryClass::Output, 128)
                    .unwrap()
                    .freeze();
                sender
                    .send(bytes)
                    .map_err(|_| ProductionError::ConsumerStopped)?;
            }
            Ok(())
        },
        |_| Err(anyhow::anyhow!("invalid decoded pixel")),
    )
    .unwrap();
    assert_eq!(result.unwrap_err().to_string(), "invalid decoded pixel");
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
