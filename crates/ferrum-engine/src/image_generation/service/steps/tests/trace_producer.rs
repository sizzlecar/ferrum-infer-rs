//! Real service queue observations around the existing Stored Reference DiT
//! barrier. Preconditioned initialization remains explicit; this does not
//! qualify full H3, HTTP transport, CUDA timing or an image-serving SLO.
use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
};
use serde_json::Value;

struct Traced(Preconditioned);
impl ImageGenerator for Traced {
    fn continuous_trace_handle(&self) -> Option<ImageContinuousTraceHandle> {
        self.0
            .pipeline
            .execution
            .continuous_trace
            .as_ref()
            .map(|trace| trace.handle())
    }
    fn retained_output_bytes(&self) -> u64 {
        self.continuous_trace_handle()
            .unwrap()
            .status()
            .backing_bytes as u64
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        Some(self)
    }
    fn host_account(&self) -> &HostMemoryAccount {
        self.0.host_account()
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        self.0.capabilities()
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        unreachable!()
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        ImageGenerator::shutdown(&mut self.0)
    }
}
impl ImageDenoisingGenerator for Traced {
    type Output = ImageGenerationOutput;
    fn prepare_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.0.prepare_request(request)
    }
    fn try_admit_request(
        &mut self,
        p: &mut ImageBatchPreparation,
        r: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.0.try_admit_request(p, r)
    }
    fn start_request(
        &mut self,
        p: ImageBatchPreparation,
        r: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        self.0.start_request(p, r)
    }
    fn try_prepare_batch(
        &mut self,
        r: &mut [&mut ImageDenoisingRequest],
        b: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        self.0.try_prepare_batch(r, b)
    }
    fn pause_batch(
        &mut self,
        b: &mut ImageDenoisingBatch,
        r: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        self.0.pause_batch(b, r)
    }
    fn advance_batch(
        &mut self,
        b: &mut ImageDenoisingBatch,
        r: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        self.0.advance_batch(b, r)
    }
    fn finish_request(
        &mut self,
        r: &mut ImageDenoisingRequest,
        b: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        self.0.finish_request(r, b)
    }
}
fn event_for<'a>(records: &'a [Value], kind: &str, id: Uuid) -> Vec<&'a Value> {
    records
        .iter()
        .filter(|record| {
            record["event"]["kind"] == kind
                && (record["event"]["request_id"] == id.to_string()
                    || record["event"]["correlation"]["request_id"] == id.to_string())
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_trace_producer_correlates_real_queue_initialization_cancellation_and_return() {
    let (mut pipeline, source) = fixture_pipeline();
    let host = pipeline.host_account().clone();
    let runtime = pipeline.execution.runtime.clone();
    let (recorder, handle) = ImageContinuousTraceRecorder::new(
        &host,
        ImageContinuousTraceLimits {
            maximum_bytes: 32 << 10,
        },
    )
    .unwrap();
    pipeline.execution.continuous_trace = Some(recorder);
    let boundary = Arc::new(Boundary::default());
    let _release = ReleaseOnDrop(boundary.clone());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let service = ImageService::new(
        Traced(Preconditioned {
            pipeline,
            _source: source,
            boundary: boundary.clone(),
            trace: trace.clone(),
            denied_native_join: Arc::new(AtomicBool::new(false)),
        }),
        ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            maximum_pending: 4,
            maximum_pending_bytes: 1 << 18,
            maximum_prompt_bytes: 1024,
            max_num_seqs: 2,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    assert_eq!(
        service.continuous_trace_handle().unwrap().session_uuid(),
        handle.session_uuid()
    );
    let a = service.try_submit(request(0)).unwrap();
    let ca = a.correlation();
    assert!(ca.client_attempt_id.is_none());
    boundary.wait(1).await;
    let attempt = Uuid::new_v4();
    let b = service
        .try_submit_with_client_attempt(request(1), Some(attempt))
        .unwrap();
    let c = service
        .try_submit_with_client_attempt(request(2), Some(attempt))
        .unwrap();
    let cb = b.correlation();
    let cc = c.correlation();
    assert_eq!(cb.client_attempt_id, Some(attempt));
    assert_eq!(cc.client_attempt_id, Some(attempt));
    assert_ne!(cb.request_id, cc.request_id);
    assert_ne!(cb.request_id, attempt);
    c.cancel();
    let mut invalid = request(3);
    invalid.width = 0;
    let rejected = service
        .try_submit_with_client_attempt(invalid, Some(attempt))
        .err()
        .unwrap();
    let rejected = rejected.correlation().unwrap();
    assert_ne!(rejected.request_id, cb.request_id);
    boundary.release(1);
    boundary.wait(2).await;
    boundary.release(2);
    assert_eq!(
        c.wait().await.err().unwrap().kind(),
        ImageServiceErrorKind::Cancelled
    );
    drop((a.wait().await.unwrap(), b.wait().await.unwrap()));
    let retired = service.shutdown().await.unwrap();
    assert!(!retired.has_failure(), "{:?}", retired.failure());
    drop((retired, service));
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert!(handle.status().sealed && !handle.status().truncated);
    let mut bytes = Vec::new();
    handle.write_jsonl(&mut bytes).unwrap();
    let records = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let events = records
        .iter()
        .filter(|record| record.get("event").is_some())
        .cloned()
        .collect::<Vec<_>>();
    assert!(events
        .windows(2)
        .all(|pair| pair[0]["monotonic_ns"].as_u64().unwrap()
            <= pair[1]["monotonic_ns"].as_u64().unwrap()));
    for correlation in [ca, cb] {
        let queued = event_for(&events, "request_queued", correlation.request_id);
        let dequeued = event_for(&events, "request_dequeued", correlation.request_id);
        let started = event_for(&events, "initialization_started", correlation.request_id);
        let finished = event_for(&events, "request_service_finished", correlation.request_id);
        assert_eq!((queued.len(), started.len(), finished.len()), (1, 1, 1));
        assert!(!dequeued.is_empty());
        assert!(
            queued[0]["event_ns"].as_u64().unwrap() <= started[0]["event_ns"].as_u64().unwrap()
        );
        assert_eq!(finished[0]["event"]["outcome"], "completed");
        for phase in ["metadata_preparation", "initialization"] {
            let start = event_for(&events, "phase_started", correlation.request_id)
                .into_iter()
                .find(|record| record["event"]["phase"] == phase)
                .expect("actual worker phase start");
            let finish = events
                .iter()
                .find(|record| {
                    record["event"]["kind"] == "phase_finished"
                        && record["event"]["start_sequence"] == start["sequence"]
                })
                .expect("actual worker phase return");
            assert_eq!(finish["event"]["outcome"], "completed");
            assert!(start["event_ns"].as_u64().unwrap() <= finish["event_ns"].as_u64().unwrap());
        }
    }
    assert_eq!(
        event_for(&events, "request_service_finished", cc.request_id)[0]["event"]["outcome"],
        "cancelled"
    );
    assert!(event_for(&events, "initialization_started", cc.request_id).is_empty());
    assert!(event_for(&events, "request_queued", rejected.request_id).is_empty());
    assert_eq!(
        event_for(&events, "request_submission_rejected", rejected.request_id).len(),
        1
    );
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        handle.status().backing_bytes
    );
    drop(handle);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
