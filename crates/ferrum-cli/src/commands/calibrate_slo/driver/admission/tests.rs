use super::*;
use ferrum_interfaces::output_credit::{
    OutputAccountLimits, OutputCreditAmount, OutputCreditAttempt, OutputCreditLane,
    OutputCreditPool, OutputPoolLimits,
};
use ferrum_interfaces::output_flow::{
    CreditedOutputFrame, CreditedOutputSession, OutputCompletion, OutputConsumerControl,
    OutputFrameMetadata,
};
use ferrum_types::{FinishReason, RequestId, TokenUsage};
use futures::FutureExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{mpsc, oneshot};

#[test]
fn rolling_arrivals_replace_completed_owner_before_slow_peer_and_keep_order() {
    let prompts = [9, 7, 9, 4, 2];
    let mut window = AdmissionWindow::new(&prompts, 2).unwrap();
    let mut admitted = Vec::new();
    let mut admit = |window: &mut AdmissionWindow<'_, usize>| {
        let (ordinal, source) = window.next_prompt().unwrap();
        admitted.push(source);
        window.admitted(ordinal, ordinal).unwrap();
        assert!(window.active().len() <= 2);
    };
    admit(&mut window);
    admit(&mut window);
    assert_eq!(window.next_prompt(), None);
    assert_eq!(window.completed(&1).unwrap().source_index, 7);
    admit(&mut window);
    assert_eq!(
        window.active().iter().map(|x| x.id).collect::<Vec<_>>(),
        [0, 2]
    );
    // Owner zero remains slow while three later full requests complete.
    for id in 2..4 {
        window.completed(&id).unwrap();
        admit(&mut window);
    }
    window.completed(&4).unwrap();
    assert!(!window.drained());
    window.completed(&0).unwrap();
    assert!(window.drained());
    assert_eq!(admitted, prompts);
    assert!(window.completed(&0).is_err());
}

#[test]
fn rolling_arrival_identity_and_limit_do_not_become_admission_authority() {
    let mut window = AdmissionWindow::new(&[3, 3, 3], 2).unwrap();
    assert!(window.admitted(1, 100).is_err());
    assert_eq!(window.next_prompt(), Some((0, 3)));
    window.admitted(0, 100).unwrap();
    assert!(window.admitted(1, 100).is_err());
    window.admitted(1, 101).unwrap();
    assert!(window.admitted(2, 102).is_err());
    assert!(window.completed(&999).is_err());
    assert_eq!(window.active().len(), 2);
    assert!(window.next_prompt().is_none());
}

#[derive(Default)]
struct Control(AtomicUsize);
impl OutputConsumerControl for Control {
    fn consumer_dropped(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn lease<T>(
    account: &ferrum_interfaces::output_credit::OutputCreditAccount,
    lane: OutputCreditLane,
    credit: OutputCreditAmount,
    value: T,
) -> ferrum_interfaces::output_credit::LeasedOutput<T> {
    match account.try_reserve(lane, credit).unwrap() {
        OutputCreditAttempt::Reserved(reservation) => reservation.into_output(value),
        _ => panic!("fixture exhausted its declared output account"),
    }
}

fn pool() -> OutputCreditPool {
    OutputCreditPool::new(OutputPoolLimits {
        maximum: OutputCreditAmount {
            events: 4,
            bytes: 128,
            projection_bytes: 64,
        },
        max_total_bytes: 192,
        max_open_accounts: 1,
    })
    .unwrap()
}

fn limits() -> OutputAccountLimits {
    OutputAccountLimits {
        maximum: OutputCreditAmount {
            events: 4,
            bytes: 128,
            projection_bytes: 64,
        },
        terminal: OutputCreditAmount {
            events: 1,
            bytes: 16,
            projection_bytes: 0,
        },
    }
}

#[tokio::test]
async fn rolling_refill_waits_for_real_terminal_and_completion_lease_consumption() {
    let id = RequestId::new();
    let pool = pool();
    let account = pool.open_request(id.clone(), limits()).unwrap();
    let control = Arc::new(Control::default());
    let (frames, receiver) = mpsc::channel(1);
    let (completion, completion_receiver) = oneshot::channel();
    let session =
        CreditedOutputSession::from_receivers(receiver, completion_receiver, control.clone());
    let mut window = AdmissionWindow::new(&[8, 4], 1).unwrap();
    window.admitted(0, id.clone()).unwrap();
    let mut consuming = Box::pin(super::super::consume(id.clone(), session));
    assert!(consuming.as_mut().now_or_never().is_none());
    let terminal = lease(
        &account,
        OutputCreditLane::Terminal,
        OutputCreditAmount {
            events: 1,
            bytes: 4,
            projection_bytes: 0,
        },
        b"done".to_vec(),
    );
    assert!(frames
        .send(CreditedOutputFrame::new(
            terminal,
            OutputFrameMetadata {
                ordinal: 73,
                token: None,
                generated_tokens: 73,
                terminal: true,
            }
        ))
        .await
        .is_ok());
    let retained = lease(
        &account,
        OutputCreditLane::Data,
        OutputCreditAmount {
            events: 0,
            bytes: 0,
            projection_bytes: 16,
        },
        OutputCompletion::Succeeded {
            history: None,
            reason: FinishReason::Length,
            usage: TokenUsage::new(3, 73),
            execution_evidence: None,
        },
    );
    account.close();
    assert!(consuming.as_mut().now_or_never().is_none());
    assert!(
        window.next_prompt().is_none(),
        "terminal wire alone cannot refill"
    );
    assert_eq!(pool.snapshot().retained_accounts, 1);
    assert!(completion.send(retained).is_ok());
    let record = consuming.await.unwrap();
    assert_eq!(record["usage"]["completion_tokens"], 73);
    assert_eq!(pool.snapshot().retained_accounts, 0);
    window.completed(&id).unwrap();
    assert_eq!(window.next_prompt(), Some((1, 4)));
    assert_eq!(control.0.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rolling_failed_or_cancelled_consumer_never_refills_and_releases_real_leases() {
    let id = RequestId::new();
    let mut window = AdmissionWindow::new(&[0, 1], 1).unwrap();
    window.admitted(0, id.clone()).unwrap();
    let control = Arc::new(Control::default());
    let (frames, receiver) = mpsc::channel(1);
    let (_completion, completion_receiver) = oneshot::channel();
    let session =
        CreditedOutputSession::from_receivers(receiver, completion_receiver, control.clone());
    drop(frames);
    assert!(super::super::consume(id.clone(), session).await.is_err());
    assert!(window.next_prompt().is_none());
    assert_eq!(control.0.load(Ordering::SeqCst), 1);

    let pool = pool();
    let account = pool.open_request(id.clone(), limits()).unwrap();
    let (frames, receiver) = mpsc::channel(1);
    let (completion, completion_receiver) = oneshot::channel();
    let session =
        CreditedOutputSession::from_receivers(receiver, completion_receiver, control.clone());
    let retained = lease(
        &account,
        OutputCreditLane::Data,
        OutputCreditAmount {
            events: 0,
            bytes: 0,
            projection_bytes: 16,
        },
        OutputCompletion::Succeeded {
            history: None,
            reason: FinishReason::Length,
            usage: TokenUsage::new(3, 73),
            execution_evidence: None,
        },
    );
    assert!(completion.send(retained).is_ok());
    account.close();
    let consuming = Box::pin(super::super::consume(id, session));
    assert_eq!(pool.snapshot().retained_accounts, 1);
    drop(consuming);
    drop(frames);
    assert_eq!(pool.snapshot().retained_accounts, 0);
    assert_eq!(control.0.load(Ordering::SeqCst), 2);
    assert!(window.next_prompt().is_none());
}

#[tokio::test]
async fn rolling_terminal_failure_does_not_count_as_success_or_retain_credit() {
    let id = RequestId::new();
    let pool = pool();
    let account = pool.open_request(id.clone(), limits()).unwrap();
    let control = Arc::new(Control::default());
    let (frames, receiver) = mpsc::channel(1);
    let (completion, completion_receiver) = oneshot::channel();
    let session =
        CreditedOutputSession::from_receivers(receiver, completion_receiver, control.clone());
    let mut window = AdmissionWindow::new(&[0, 1], 1).unwrap();
    window.admitted(0, id.clone()).unwrap();
    let terminal = lease(
        &account,
        OutputCreditLane::Terminal,
        OutputCreditAmount {
            events: 1,
            bytes: 4,
            projection_bytes: 0,
        },
        b"fail".to_vec(),
    );
    assert!(frames
        .send(CreditedOutputFrame::new(
            terminal,
            OutputFrameMetadata {
                ordinal: 1,
                token: None,
                generated_tokens: 0,
                terminal: true,
            }
        ))
        .await
        .is_ok());
    let failed = lease(
        &account,
        OutputCreditLane::Data,
        OutputCreditAmount {
            events: 0,
            bytes: 0,
            projection_bytes: 16,
        },
        OutputCompletion::Failed(ferrum_interfaces::output_flow::BoundedOutputError::new(
            "fixture failure",
        )),
    );
    assert!(completion.send(failed).is_ok());
    account.close();
    assert!(super::super::consume(id, session).await.is_err());
    assert!(window.next_prompt().is_none());
    assert_eq!(pool.snapshot().retained_accounts, 0);
    assert_eq!(
        control.0.load(Ordering::SeqCst),
        0,
        "terminal was consumed, failure is separate"
    );
}
