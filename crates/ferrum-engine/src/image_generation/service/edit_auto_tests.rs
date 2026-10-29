use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits, ManagedHostBytes};
use std::{collections::BTreeMap, sync::mpsc};

const PREPROCESSED_BYTES: usize = 256;
type Seeds = Vec<u64>;

struct RetainedEdit {
    seed: u64,
    _backing: ManagedHostBytes,
}
struct Generator {
    host: HostMemoryAccount,
    preprocess_calls: Arc<Mutex<BTreeMap<u64, usize>>>,
}
impl ImageGenerator for Generator {
    fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        Ok(ImageGeneratorCapabilities {
            maximum_batch_size: 8,
            maximum_width: 1024,
            maximum_height: 1024,
            dimension_multiple: 1,
        })
    }
    fn edit_preparation_bytes(&self, _: &ImageEditRequest) -> Result<usize, ImageBatchError> {
        Ok(PREPROCESSED_BYTES)
    }
    fn preprocess_edit(
        &mut self,
        request: &ImageEditRequest,
    ) -> Result<ImageEditPreparation, ImageBatchError> {
        assert!(self.host.owns(&request.image));
        *self
            .preprocess_calls
            .lock()
            .unwrap()
            .entry(request.generation.seed)
            .or_default() += 1;
        let backing = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, PREPROCESSED_BYTES)
            .unwrap()
            .freeze();
        Ok(ImageEditPreparation::new(
            RetainedEdit {
                seed: request.generation.seed,
                _backing: backing,
            },
            PREPROCESSED_BYTES,
        ))
    }
    fn prepare_edits(
        &mut self,
        requests: &[(&ImageEditRequest, &ImageEditPreparation)],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        let seeds = requests
            .iter()
            .map(|(request, prepared)| {
                let input = prepared.get::<RetainedEdit>()?;
                assert_eq!(request.generation.seed, input.seed);
                Ok(input.seed)
            })
            .collect::<Result<Seeds, ImageBatchError>>()?;
        Ok(ImageBatchPreparation::new(seeds))
    }
    fn prepare_generation(
        &mut self,
        requests: &[ImageGenerationRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        Ok(ImageBatchPreparation::new(
            requests
                .iter()
                .map(|request| request.seed)
                .collect::<Seeds>(),
        ))
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        panic!("the scalar worker executes the prepared batch")
    }
}

/// Use the production GeneratorAdapter for preprocessing and reuse; only the
/// device execution/output is replaced with scalar seeds and deterministic gates.
struct Worker {
    adapter: GeneratorAdapter<Generator>,
    capacity: usize,
    preparing: mpsc::Sender<Seeds>,
    prepare_release: mpsc::Receiver<()>,
    executing: mpsc::Sender<Seeds>,
    execute_release: mpsc::Receiver<()>,
}
impl BatchWorker for Worker {
    type Output = u64;
    fn preparation_bytes(&self, request: &QueuedRequest) -> Result<usize, ImageBatchError> {
        self.adapter.preparation_bytes(request)
    }
    fn prepare(
        &mut self,
        requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        let prepared = self.adapter.prepare(requests)?;
        let seeds = requests
            .iter()
            .map(|request| request.generation().seed)
            .collect();
        self.preparing.send(seeds).unwrap();
        self.prepare_release
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        if requests.len() > self.capacity {
            let proof = super::super::ImageDeviceCapacityExceeded {
                report: super::super::ImageDeviceCapacityReport {
                    complete_batch: true,
                    required_bytes: requests.len() as u64,
                    capacity_bytes: self.capacity as u64,
                    retained_static_bytes: 0,
                    active_stage_peak_bytes: requests.len() as u64,
                    retained_dynamic_allowance_bytes: 0,
                    native_allowance_bytes: 0,
                },
            };
            return Err(ImageBatchError::Capacity {
                diagnostic: proof.to_string(),
                proof,
            });
        }
        Ok(prepared)
    }
    fn execute(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<QueuedRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        let seeds = prepared.into_inner::<Seeds>()?;
        assert_eq!(
            seeds,
            requests
                .iter()
                .map(|request| request.generation().seed)
                .collect::<Seeds>()
        );
        self.executing.send(seeds.clone()).unwrap();
        self.execute_release
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        drop(requests);
        Ok(seeds)
    }
    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }
    fn maximum_edit_batch_size(&self) -> usize {
        8
    }
    fn compatible_edits(&self, a: &ImageEditRequest, b: &ImageEditRequest) -> bool {
        compatible_edit_geometry(a, b)
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        panic!("execution must consume the successful preparation")
    }
}
struct Controls {
    preparing: mpsc::Receiver<Seeds>,
    prepare_release: mpsc::Sender<()>,
    executing: mpsc::Receiver<Seeds>,
    execute_release: mpsc::Sender<()>,
    preprocess_calls: Arc<Mutex<BTreeMap<u64, usize>>>,
}
fn fixture(host: &HostMemoryAccount, capacity: usize) -> (Worker, Controls) {
    let (preparing, p) = mpsc::channel();
    let (pr, prepare_release) = mpsc::channel();
    let (executing, e) = mpsc::channel();
    let (er, execute_release) = mpsc::channel();
    let preprocess_calls = Arc::new(Mutex::new(BTreeMap::new()));
    (
        Worker {
            adapter: GeneratorAdapter(Generator {
                host: host.clone(),
                preprocess_calls: preprocess_calls.clone(),
            }),
            capacity,
            preparing,
            prepare_release,
            executing,
            execute_release,
        },
        Controls {
            preparing: p,
            prepare_release: pr,
            executing: e,
            execute_release: er,
            preprocess_calls,
        },
    )
}
fn recv(channel: &mpsc::Receiver<Seeds>) -> Seeds {
    channel
        .recv_timeout(Duration::from_secs(10))
        .expect("worker progress")
}
fn account() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            128 * 1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 128 * 1024,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn config() -> ImageServiceConfig {
    ImageServiceConfig {
        batching: ImageBatchPolicy::Auto,
        maximum_pending: 8,
        maximum_pending_bytes: 128 * 1024,
        maximum_prompt_bytes: 128,
        max_num_seqs: 8,
        request_batch_max_wait_ms: 0,
    }
}
fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "image".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}
fn edit(host: &HostMemoryAccount, seed: u64, width: u32, height: u32) -> QueuedRequest {
    let mut encoded = Vec::new();
    ferrum_output::png::write_rgba8_png(
        &mut encoded,
        ferrum_output::png::PngDimensions::new(u64::from(width), u64::from(height)).unwrap(),
        ferrum_output::png::PngLimits::default(),
        |writer| {
            for _ in 0..width * height {
                writer.write_all(&[1, 2, 3, 255])?;
            }
            Ok(())
        },
    )
    .unwrap();
    let mut image = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, encoded.len())
        .unwrap();
    image.as_mut().copy_from_slice(&encoded);
    QueuedRequest::Edit(
        ImageEditRequest {
            generation: request(seed),
            image: image.freeze(),
        },
        None,
    )
}

#[tokio::test]
async fn auto_edit_preprocesses_once_and_keeps_reduced_tail_bytes_and_fifo() {
    let host = account();
    let (worker, c) = fixture(&host, 2);
    let edits = (1..=3)
        .map(|seed| edit(&host, seed, 16, 16))
        .collect::<Vec<_>>();
    let raw_bytes = edits
        .iter()
        .map(QueuedRequest::pending_bytes)
        .collect::<Vec<_>>();
    let waiting_bytes = raw_bytes.iter().sum::<usize>() + 3 * PREPROCESSED_BYTES;
    let service = ServiceCore::start(
        Box::new(worker),
        ImageServiceConfig {
            maximum_pending_bytes: waiting_bytes,
            ..config()
        },
    )
    .unwrap();
    let active = service.try_submit(request(0)).unwrap();
    assert_eq!(recv(&c.preparing), [0]);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.executing), [0]);
    let jobs = edits
        .into_iter()
        .map(|edit| service.try_submit_request(edit).unwrap())
        .collect::<Vec<_>>();
    let tail_arrival = service.shared.lock().queue.back().unwrap().submitted;
    c.execute_release.send(()).unwrap();
    assert_eq!(active.wait().await.unwrap(), 0);
    assert_eq!(recv(&c.preparing), [1, 2, 3]);
    assert_eq!(service.snapshot().pending_bytes, waiting_bytes);
    assert_eq!(
        service.try_submit(request(4)).err().unwrap().kind(),
        ImageServiceErrorKind::QueueFull
    );
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.preparing), [1]);
    assert_eq!(service.snapshot().pending_bytes, waiting_bytes);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.preparing), [1, 2]);
    assert_eq!(service.snapshot().pending_bytes, waiting_bytes);
    assert_eq!(
        *c.preprocess_calls.lock().unwrap(),
        BTreeMap::from([(1, 1), (2, 1), (3, 1)])
    );
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.executing), [1, 2]);
    {
        let state = service.shared.lock();
        let tail = state.queue.front().unwrap();
        assert_eq!(tail.request.generation().seed, 3);
        assert_eq!(tail.submitted, tail_arrival);
        assert_eq!(tail.preprocessing_bytes, PREPROCESSED_BYTES);
        assert_eq!(tail.pending_bytes(), raw_bytes[2] + PREPROCESSED_BYTES);
    }
    assert_eq!(
        *c.preprocess_calls.lock().unwrap(),
        BTreeMap::from([(1, 1), (2, 1), (3, 1)])
    );
    assert_eq!(
        service.snapshot().pending_bytes,
        raw_bytes[2] + PREPROCESSED_BYTES
    );
    c.execute_release.send(()).unwrap();
    assert_eq!(recv(&c.preparing), [3]);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.executing), [3]);
    c.execute_release.send(()).unwrap();
    for (index, job) in jobs.into_iter().enumerate() {
        assert_eq!(job.wait().await.unwrap(), index as u64 + 1);
    }
    assert_eq!(
        *c.preprocess_calls.lock().unwrap(),
        BTreeMap::from([(1, 1), (2, 1), (3, 1)])
    );
    assert_eq!(service.snapshot().capacity_reductions, 1);
    assert!(service.shutdown().await.unwrap().failure.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn waiting_host_budget_rejects_before_preprocessing_without_capacity_retry() {
    let host = account();
    let (worker, c) = fixture(&host, 8);
    let request = edit(&host, 1, 16, 16);
    let service = ServiceCore::start(
        Box::new(worker),
        ImageServiceConfig {
            maximum_pending_bytes: request.pending_bytes() + PREPROCESSED_BYTES - 1,
            ..config()
        },
    )
    .unwrap();
    let job = service.try_submit_request(request).unwrap();
    let error = tokio::time::timeout(Duration::from_secs(10), job.wait())
        .await
        .unwrap()
        .unwrap_err();
    // QueueFull is the service error mapped to HTTP 429 by the image endpoint.
    assert_eq!(error.kind(), ImageServiceErrorKind::QueueFull);
    assert!(c.preprocess_calls.lock().unwrap().is_empty());
    assert!(c.preparing.try_recv().is_err());
    assert!(c.executing.try_recv().is_err());
    let snapshot = service.snapshot();
    assert_eq!(
        (
            snapshot.rejected_requests,
            snapshot.capacity_reductions,
            snapshot.executed_batches
        ),
        (1, 0, 0)
    );
    assert_eq!((snapshot.pending_requests, snapshot.pending_bytes), (0, 0));
    assert!(snapshot.accepting);
    assert!(service.shutdown().await.unwrap().failure.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn auto_preserves_generation_edit_and_reference_geometry_fifo_boundaries() {
    let host = account();
    let (worker, c) = fixture(&host, 8);
    let service = ServiceCore::start(Box::new(worker), config()).unwrap();
    let first = service.try_submit(request(0)).unwrap();
    assert_eq!(recv(&c.preparing), [0]);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.executing), [0]);
    let queued = [
        QueuedRequest::Generate(request(1)),
        edit(&host, 2, 16, 16),
        edit(&host, 3, 32, 16),
        edit(&host, 4, 16, 16),
        QueuedRequest::Generate(request(5)),
        QueuedRequest::Generate(request(6)),
    ];
    let jobs = queued
        .into_iter()
        .map(|request| service.try_submit_request(request).unwrap())
        .collect::<Vec<_>>();
    c.execute_release.send(()).unwrap();
    assert_eq!(first.wait().await.unwrap(), 0);
    for expected in [vec![1], vec![2], vec![3], vec![4], vec![5, 6]] {
        assert_eq!(recv(&c.preparing), expected);
        c.prepare_release.send(()).unwrap();
        assert_eq!(recv(&c.executing), expected);
        c.execute_release.send(()).unwrap();
    }
    for (index, job) in jobs.into_iter().enumerate() {
        assert_eq!(job.wait().await.unwrap(), index as u64 + 1);
    }
    assert_eq!(
        *c.preprocess_calls.lock().unwrap(),
        BTreeMap::from([(2, 1), (3, 1), (4, 1)])
    );
    assert_eq!(service.snapshot().capacity_reductions, 0);
    assert!(service.shutdown().await.unwrap().failure.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
