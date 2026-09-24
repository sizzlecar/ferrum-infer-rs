use super::*;

#[tokio::test]
async fn credited_evidence_shutdown_reservation_and_seal_are_ordered() {
    let observers = Arc::new(EvidenceObservers::default());
    let reservation = observers.register().unwrap();
    observers.seal();
    assert!(observers.register().is_err());
    // No completion obligation was accepted: admission/startup failed.
    drop(reservation);
    observers.drain().await.unwrap();
    observers.drain().await.unwrap();
}

#[tokio::test]
async fn credited_evidence_shutdown_cancelled_task_is_not_a_successful_drain() {
    let observers = Arc::new(EvidenceObservers::default());
    let mut ticket = observers.register().unwrap();
    ticket.arm();
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let _ticket = ticket;
        entered_tx.send(()).unwrap();
        std::future::pending::<()>().await;
    });
    entered_rx.await.unwrap();
    observers.seal();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(observers.drain().await.unwrap_err().contains("cancelled"));
    assert!(observers.drain().await.is_err());
}
