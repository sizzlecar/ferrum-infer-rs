//! Real Reference execution through the opt-in serial admission contract.
use super::*;
use ferrum_execution::{TensorAdmissionControl, TensorSerialActor};
use std::{sync::mpsc, time::Duration};

#[test]
fn capacity_wait_resumes_one_execution_or_cancels_without_poisoning_its_actor() {
    for cancel in [false, true] {
        let fixture = Fixture::new(Config::standard(), account(2 * BYTE_LEN, BYTE_LEN));
        let binding = fixture
            .resources
            .bind_tensor_execution(fixture.plan(), fixture.host.clone())
            .unwrap();
        let input = input(&fixture.host, &fixture.config);
        let work = binding.bind_inputs(vec![input.clone()]).unwrap();
        let sequence = admit_initial_sequence(&binding, &work);
        let session = sequence.open_session().unwrap();
        let runtime = fixture.composition.runtime().clone();
        let mut executor = fixture.executor();
        let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
        let control = TensorAdmissionControl::new();
        actor.set_admission_control(control.clone());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let before = runtime.snapshot();
        let thread = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            let result = executor.execute_on_actor(
                &mut actor,
                id("run.weightless.capacity-wait"),
                id("request.weightless.capacity-wait"),
                vec![input.clone()],
            );
            assert!(done_tx.send((actor, executor, input, result)).is_ok());
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(30)).is_err());
        let waiting = runtime.snapshot();
        assert_eq!(waiting.submissions, before.submissions);
        assert_eq!(waiting.uploaded_bytes, before.uploaded_bytes);
        if cancel {
            control.cancel();
            let (mut actor, mut executor, input, result) =
                done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            assert!(matches!(
                result,
                Err(TensorExecutionError::CancelledBeforeSubmission(_))
            ));
            assert!(!actor.is_failed());
            assert_eq!(runtime.snapshot().submissions, before.submissions);
            assert_eq!(
                fixture
                    .host
                    .usage()
                    .unwrap()
                    .class_committed_bytes(HostMemoryClass::Output),
                0
            );
            session.try_abort_if_quiescent().unwrap();
            drop((session, sequence, work, binding));
            actor.set_admission_control(TensorAdmissionControl::new());
            let output = executor
                .execute_on_actor(
                    &mut actor,
                    id("run.weightless.after-cancel"),
                    id("request.weightless.after-cancel"),
                    vec![input],
                )
                .unwrap();
            check_output(output[0].bytes().bytes());
            drop((output, executor, actor));
        } else {
            session.try_abort_if_quiescent().unwrap();
            drop((session, sequence, work, binding));
            let (actor, executor, input, result) =
                done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            assert!(!actor.is_failed());
            let output = result.unwrap();
            check_output(output[0].bytes().bytes());
            drop((output, input, executor, actor));
        }
        thread.join().unwrap();
        assert_eq!(
            runtime.snapshot().silu_launches - before.silu_launches,
            1,
            "one admitted invocation, never a replay of completed work"
        );
        assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
        fixture.close();
        assert_eq!(runtime.snapshot().live_allocations, 0);
    }
}
