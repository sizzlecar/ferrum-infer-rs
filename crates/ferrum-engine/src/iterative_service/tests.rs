//! The common worker accepts a typed temporal input without image ingress.
//! This fixture proves lifecycle routing, not native batching performance.
use super::*;
use std::sync::mpsc;

mod cancellation;
mod capacity_fallback;
mod decoder_cancellation;

struct TimelineRequest {
    id: u64,
    frames: Vec<u64>,
    transitions: usize,
}
impl ServiceRequest for TimelineRequest {
    fn pending_bytes(&self) -> usize {
        self.frames.capacity() * std::mem::size_of::<u64>()
    }
    fn admission_bytes(&self) -> usize {
        self.pending_bytes()
    }
    fn kind(&self) -> &'static str {
        "timeline"
    }
}
#[derive(Debug, PartialEq)]
struct TimelineOutput {
    id: u64,
    frames: Vec<u64>,
}
struct Lane {
    output: TimelineOutput,
    cursor: usize,
    transitions: usize,
    cancelled: Option<Arc<AtomicBool>>,
}
type Memberships = Arc<Mutex<Vec<Vec<(u64, usize)>>>>;
#[derive(Default)]
struct Observed(Mutex<Vec<observations::LifecycleEvent>>);
impl LifecycleObserver for Observed {
    fn record_at(&self, _: Instant, event: observations::LifecycleEvent) -> Option<u64> {
        let mut events = self.0.lock().unwrap();
        events.push(event);
        Some(events.len() as u64)
    }
}
struct TimelineWorker {
    first_boundary: Option<mpsc::Sender<()>>,
    release: mpsc::Receiver<()>,
    memberships: Memberships,
    observed: Arc<Observed>,
    cancellation_seen: Option<mpsc::Sender<bool>>,
}
impl BatchWorker for TimelineWorker {
    type Request = TimelineRequest;
    type Output = TimelineOutput;
    fn lifecycle_observer(&self) -> Option<LifecycleObserverHandle> {
        Some(self.observed.clone())
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn DenoisingGenerator<Request = Self::Request, Output = Self::Output>> {
        Some(self)
    }
    fn execute(
        &mut self,
        _: BatchPreparation,
        _: Vec<Self::Request>,
    ) -> Result<Vec<Self::Output>, BatchError> {
        panic!("declared iterative requests must use the common step worker")
    }
    fn compatible(&self, _: &Self::Request, _: &Self::Request) -> bool {
        true
    }
    fn maximum_batch_size(&self, _: &Self::Request, configured: usize) -> usize {
        configured
    }
    fn shutdown_evidence(&mut self) -> Result<Option<Box<dyn Any + Send>>, GeneratorError> {
        Ok(Some(Box::new(self.memberships.lock().unwrap().clone())))
    }
}

#[tokio::test]
async fn finite_temporal_run_uses_completed_steps_and_preserves_retirement_evidence() {
    let (_, released) = mpsc::channel();
    let memberships = Arc::new(Mutex::new(Vec::new()));
    let worker = TimelineWorker {
        first_boundary: None,
        release: released,
        memberships: memberships.clone(),
        observed: Arc::new(Observed::default()),
        cancellation_seen: None,
    };
    let service = ServiceCore::unstarted(
        &worker,
        ServiceConfig {
            batching: BatchPolicy::Configured,
            maximum_pending: 1,
            maximum_pending_bytes: 4096,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
        None,
    );
    let job = service
        .try_submit(TimelineRequest {
            id: 7,
            frames: vec![10, 20],
            transitions: 3,
        })
        .unwrap();
    let retired = service.run_to_completion(Box::new(worker));
    assert!(retired.failure.is_none());
    assert_eq!(
        job.completed().unwrap(),
        TimelineOutput {
            id: 7,
            frames: vec![16, 26]
        }
    );
    let receipt = retired
        .evidence
        .as_ref()
        .unwrap()
        .downcast_ref::<Vec<Vec<(u64, usize)>>>()
        .unwrap();
    assert_eq!(receipt, &*memberships.lock().unwrap());
    assert_eq!(service.snapshot().executed_requests, 1);
}
impl DenoisingGenerator for TimelineWorker {
    type Request = TimelineRequest;
    type Output = TimelineOutput;
    fn bind_request_cancellation(
        &mut self,
        prepared: &mut BatchPreparation,
        cancelled: Arc<AtomicBool>,
    ) {
        prepared.get_mut::<Lane>().unwrap().cancelled = Some(cancelled);
    }
    fn supports_request(&mut self, _: &Self::Request) -> bool {
        true
    }
    fn prepare_request(&mut self, request: &Self::Request) -> Result<BatchPreparation, BatchError> {
        Ok(BatchPreparation::new(Lane {
            output: TimelineOutput {
                id: request.id,
                frames: request.frames.clone(),
            },
            cursor: 0,
            transitions: request.transitions,
            cancelled: None,
        }))
    }
    fn try_admit_request(
        &mut self,
        _: &mut BatchPreparation,
        _: PreparationResources,
    ) -> Result<bool, BatchError> {
        Ok(true)
    }
    fn start_request(
        &mut self,
        prepared: BatchPreparation,
        _: &Self::Request,
    ) -> Result<DenoisingRequest, BatchError> {
        Ok(DenoisingRequest::new(prepared.into_inner::<Lane>()?))
    }
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut DenoisingRequest],
        _: Option<&mut DenoisingBatch>,
    ) -> Result<Option<DenoisingBatch>, BatchError> {
        let ids = requests
            .iter()
            .map(|request| Ok(request.get::<Lane>()?.output.id))
            .collect::<Result<Vec<_>, BatchError>>()?;
        Ok(Some(DenoisingBatch::new(ids)))
    }
    fn pause_batch(
        &mut self,
        _: &mut DenoisingBatch,
        _: &mut [&mut DenoisingRequest],
    ) -> Result<(), BatchError> {
        Ok(())
    }
    fn advance_batch(
        &mut self,
        batch: &mut DenoisingBatch,
        requests: &mut [&mut DenoisingRequest],
    ) -> Result<Vec<bool>, BatchError> {
        let members = requests
            .iter()
            .map(|request| {
                let lane = request.get::<Lane>()?;
                Ok((lane.output.id, lane.cursor))
            })
            .collect::<Result<Vec<_>, BatchError>>()?;
        assert_eq!(
            *batch.get_mut::<Vec<u64>>()?,
            members.iter().map(|member| member.0).collect::<Vec<_>>()
        );
        self.memberships.lock().unwrap().push(members);
        let completed = requests
            .iter_mut()
            .map(|request| {
                let lane = request.get_mut::<Lane>()?;
                assert!(lane.cursor < lane.transitions);
                lane.cursor += 1;
                for frame in &mut lane.output.frames {
                    *frame += lane.cursor as u64;
                }
                Ok(lane.cursor == lane.transitions)
            })
            .collect::<Result<Vec<_>, BatchError>>()?;
        if let Some(first) = self.first_boundary.take() {
            first.send(()).unwrap();
            self.release.recv_timeout(Duration::from_secs(10)).unwrap();
        }
        if let Some(observed) = self.cancellation_seen.take() {
            observed
                .send(
                    requests[0]
                        .get::<Lane>()?
                        .cancelled
                        .as_ref()
                        .unwrap()
                        .load(Ordering::Acquire),
                )
                .unwrap();
        }
        Ok(completed)
    }
    fn finish_request(
        &mut self,
        request: &mut DenoisingRequest,
        _: &mut DenoisingBatch,
    ) -> Result<Self::Output, BatchError> {
        let lane = request.get_mut::<Lane>()?;
        assert_eq!(lane.cursor, lane.transitions);
        Ok(TimelineOutput {
            id: lane.output.id,
            frames: std::mem::take(&mut lane.output.frames),
        })
    }
}

#[tokio::test]
async fn typed_temporal_requests_join_the_original_worker_with_independent_clocks() {
    let (reached, boundary) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let memberships = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::new(Observed::default());
    let service = ServiceCore::start(
        Box::new(TimelineWorker {
            first_boundary: Some(reached),
            release: released,
            memberships: memberships.clone(),
            observed: observed.clone(),
            cancellation_seen: None,
        }),
        ServiceConfig {
            batching: BatchPolicy::Auto,
            maximum_pending: 4,
            maximum_pending_bytes: 4096,
            max_num_seqs: 2,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    let older = service
        .try_submit(TimelineRequest {
            id: 7,
            frames: vec![10, 20],
            transitions: 3,
        })
        .unwrap();
    let older_id = older.correlation.request_id;
    boundary.recv_timeout(Duration::from_secs(10)).unwrap();
    let newer = service
        .try_submit(TimelineRequest {
            id: 9,
            frames: vec![100, 200, 300],
            transitions: 1,
        })
        .unwrap();
    let newer_id = newer.correlation.request_id;
    release.send(()).unwrap();
    let (older, newer) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(older.wait(), newer.wait())
    })
    .await
    .unwrap();
    assert_eq!(
        older.unwrap(),
        TimelineOutput {
            id: 7,
            frames: vec![16, 26]
        }
    );
    assert_eq!(
        newer.unwrap(),
        TimelineOutput {
            id: 9,
            frames: vec![101, 201, 301]
        }
    );
    assert!(memberships
        .lock()
        .unwrap()
        .iter()
        .any(|members| members == &[(7, 1), (9, 0)]));
    assert!(service.shutdown().await.unwrap().failure.is_none());
    let events = observed.0.lock().unwrap();
    for request_id in [older_id, newer_id] {
        use observations::{LifecycleEvent as Event, RequestOutcome};
        let queued = events
            .iter()
            .position(|event| {
                matches!(event,
            Event::Queued { correlation } if correlation.request_id == request_id)
            })
            .unwrap();
        let dequeued = events
            .iter()
            .position(|event| {
                matches!(event,
            Event::Dequeued { request_id: id } if *id == request_id)
            })
            .unwrap();
        let finished = events.iter().position(|event| matches!(event,
            Event::Finished { request_id: id, outcome: RequestOutcome::Completed } if *id == request_id)).unwrap();
        assert!(queued < dequeued && dequeued < finished);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event,
            Event::Finished { request_id: id, .. } if *id == request_id))
                .count(),
            1
        );
    }
}
