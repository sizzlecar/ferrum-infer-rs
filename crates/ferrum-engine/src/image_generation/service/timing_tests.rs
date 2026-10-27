use super::*;
use std::collections::BTreeMap;
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
    Event, Metadata, Subscriber,
};

type Fields = BTreeMap<String, u64>;

#[derive(Default)]
struct CaptureState {
    spans: BTreeMap<u64, Fields>,
    entered: Vec<u64>,
    events: Vec<Fields>,
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<CaptureState>>);

struct Numbers<'a>(&'a mut Fields);
impl Visit for Numbers<'_> {
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_owned(), value);
    }
    fn record_debug(&mut self, _: &Field, _: &dyn fmt::Debug) {}
}

// Tests drive the real synchronous scheduler on this thread. Capture entered
// span context and typed event fields without depending on formatted log text.
impl Subscriber for Capture {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.target() == "ferrum_engine::image_generation::timing"
    }
    fn new_span(&self, attributes: &Attributes<'_>) -> Id {
        let mut fields = Fields::new();
        attributes.record(&mut Numbers(&mut fields));
        let mut state = self.0.lock().unwrap();
        let id = state.spans.len() as u64 + 1;
        state.spans.insert(id, fields);
        Id::from_u64(id)
    }
    fn record(&self, span: &Id, values: &Record<'_>) {
        let mut state = self.0.lock().unwrap();
        values.record(&mut Numbers(state.spans.get_mut(&span.into_u64()).unwrap()));
    }
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields::new();
        let mut state = self.0.lock().unwrap();
        for id in &state.entered {
            fields.extend(state.spans[id].clone());
        }
        event.record(&mut Numbers(&mut fields));
        state.events.push(fields);
    }
    fn enter(&self, span: &Id) {
        self.0.lock().unwrap().entered.push(span.into_u64());
    }
    fn exit(&self, span: &Id) {
        assert_eq!(self.0.lock().unwrap().entered.pop(), Some(span.into_u64()));
    }
}

struct Worker {
    shared: Arc<Shared<u64>>,
    fail_preparation: bool,
}
impl BatchWorker for Worker {
    type Output = u64;

    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }

    fn prepare(
        &mut self,
        requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        tracing::debug!(
            target: "ferrum_engine::image_generation::timing",
            preparation_requests = requests.len(),
        );
        if self.fail_preparation {
            return Err(ImageBatchError::Fatal(Box::new(std::io::Error::other(
                "fixture preparation failed",
            ))));
        }
        Ok(ImageBatchPreparation::default())
    }

    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        // This nested event models the generator's batch/stage records and
        // must carry the scheduler's dispatch, including with duplicate seeds.
        tracing::debug!(
            target: "ferrum_engine::image_generation::timing",
            execution_requests = requests.len(),
        );
        let finished = self.shared.lock().queue.is_empty();
        if finished {
            self.shared.stop();
        }
        Ok(requests.into_iter().map(|request| request.seed).collect())
    }
}

fn run(fail_preparation: bool) -> (Capture, Arc<Shared<u64>>, Retirement<u64>) {
    let mut queue = VecDeque::new();
    for width in [32, 32, 64] {
        let (response, _) = oneshot::channel();
        queue.push_back(Queued {
            request: QueuedRequest::Generate(ImageGenerationRequest {
                prompt: "same prompt and seed".into(),
                width,
                height: 32,
                seed: 7,
                n: 1,
            }),
            submitted: Instant::now(),
            cancelled: Arc::new(AtomicBool::new(false)),
            response,
            _prompt_reservation: None,
            preprocessing_bytes: 0,
            preparation_duration: Duration::ZERO,
        });
    }
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            queue,
            accepting: true,
            closing: false,
            active: 0,
            preparing: 0,
            preparing_bytes: 0,
            rejected: 0,
            executed_batches: 0,
            executed_requests: 0,
            last_batch_size: 0,
            capacity_reductions: 0,
        }),
        changed: Condvar::new(),
        config: ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            maximum_pending: 3,
            maximum_pending_bytes: 4096,
            maximum_prompt_bytes: 64,
            max_num_seqs: 2,
            request_batch_max_wait_ms: 0,
        },
        host: None,
    });
    let capture = Capture::default();
    let retirement = tracing::subscriber::with_default(capture.clone(), || {
        worker_loop(
            Box::new(Worker {
                shared: shared.clone(),
                fail_preparation,
            }),
            shared.clone(),
        )
    });
    (capture, shared, retirement)
}

#[test]
fn queue_and_nested_execution_events_share_dispatch_and_original_batch_indices() {
    let (capture, shared, retirement) = run(false);
    assert!(retirement.failure.is_none());
    assert_eq!(shared.lock().executed_requests, 3);
    let state = capture.0.lock().unwrap();
    let requests = state
        .events
        .iter()
        .filter(|fields| fields.contains_key("queue_wait_us"))
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["dispatch_id"], requests[1]["dispatch_id"]);
    assert_ne!(requests[0]["dispatch_id"], requests[2]["dispatch_id"]);
    for (fields, (index, size, width)) in requests.iter().zip([(0, 2, 32), (1, 2, 32), (0, 1, 64)])
    {
        assert_eq!(fields["batch_index"], index);
        assert_eq!(fields["batch_size"], size);
        assert_eq!(fields["width"], width);
        assert_eq!(fields["height"], 32);
        assert_eq!(fields["seed"], 7);
        assert!(fields.contains_key("queue_and_prepare_us"));
        assert!(fields.contains_key("preparation_us"));
    }
    let executions = state
        .events
        .iter()
        .filter(|fields| fields.contains_key("execution_requests"))
        .collect::<Vec<_>>();
    assert_eq!(executions.len(), 2);
    for (event, request) in executions.iter().zip([requests[0], requests[2]]) {
        assert_eq!(event["dispatch_id"], request["dispatch_id"]);
        assert_eq!(event["execution_requests"], request["batch_size"]);
    }
}

#[test]
fn fatal_preparation_has_no_dispatch_or_execution_wait_samples() {
    let (capture, shared, retirement) = run(true);
    assert!(retirement.failure.is_some());
    assert_eq!(shared.lock().executed_requests, 0);
    let state = capture.0.lock().unwrap();
    assert!(state
        .events
        .iter()
        .any(|fields| fields.contains_key("preparation_requests")));
    assert!(state.spans.is_empty());
    assert!(state.events.iter().all(|fields| {
        !fields.contains_key("queue_wait_us") && !fields.contains_key("execution_requests")
    }));
}
