//! A finite image request uses the serving worker's admission, completed-step
//! lifecycle, output publication and native retirement on the caller's thread.
use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[must_use = "inspect the request result and retain any original retirement failure"]
pub struct ImageServiceRun {
    pub result: Result<ImageGenerationOutput, ImageServiceError>,
    pub shutdown: ImageServiceShutdown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
    use std::sync::atomic::AtomicUsize;

    struct EditGenerator {
        host: HostMemoryAccount,
        preprocessed: Arc<AtomicUsize>,
        closed: Arc<AtomicUsize>,
        capability_failure: Option<(Arc<AtomicUsize>, bool)>,
    }
    impl ImageGenerator for EditGenerator {
        fn host_account(&self) -> &HostMemoryAccount {
            &self.host
        }
        fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
            if let Some((dropped, panic)) = &self.capability_failure {
                let source = OriginalCapabilityFailure(dropped.clone());
                if *panic {
                    std::panic::panic_any(source);
                }
                return Err(Box::new(source));
            }
            Ok(ImageGeneratorCapabilities {
                maximum_batch_size: 1,
                maximum_width: 32,
                maximum_height: 32,
                dimension_multiple: 1,
            })
        }
        fn supports_edits(&self) -> bool {
            true
        }
        fn edit_preparation_bytes(&self, _: &ImageEditRequest) -> Result<usize, ImageBatchError> {
            Ok(4096)
        }
        fn preprocess_edit(
            &mut self,
            request: &ImageEditRequest,
        ) -> Result<ImageEditPreparation, ImageBatchError> {
            assert!(self.host.owns(&request.image));
            self.preprocessed.fetch_add(1, Ordering::SeqCst);
            let owner = self
                .host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, 4096)
                .unwrap();
            Ok(ImageEditPreparation::new(owner, 4096))
        }
        fn prepare_edits(
            &mut self,
            requests: &[(&ImageEditRequest, &ImageEditPreparation)],
        ) -> Result<ImageBatchPreparation, ImageBatchError> {
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].0.generation.seed, 17);
            // An owner-free rejection still retires the real generator. The
            // test exercises preparation/Host accounting, not image arithmetic.
            Err(ImageBatchError::Rejected(
                "fixture stops after admitted preprocessing".into(),
            ))
        }
        fn generate_batch(
            &mut self,
            _: Vec<ImageGenerationRequest>,
        ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
            panic!("edit must retain the existing edit preparation branch")
        }
        fn shutdown(&mut self) -> Result<(), GeneratorError> {
            self.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
    #[derive(Debug)]
    struct OriginalCapabilityFailure(Arc<AtomicUsize>);
    impl std::fmt::Display for OriginalCapabilityFailure {
        fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            panic!("opaque capability errors must not be formatted before custody transfer")
        }
    }
    impl std::error::Error for OriginalCapabilityFailure {}
    impl Drop for OriginalCapabilityFailure {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn host() -> HostMemoryAccount {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                16384,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 16384,
                    transfer_staging_bytes: 0,
                    output_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
    }
    fn request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: "cat".into(),
            width: 32,
            height: 32,
            seed: 17,
            n: 1,
        }
    }

    #[tokio::test]
    async fn finite_edit_retains_preparation_capacity_and_releases_the_original_host_claims() {
        let host = host();
        let preprocessed = Arc::new(AtomicUsize::new(0));
        let closed = Arc::new(AtomicUsize::new(0));
        let image = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
            .unwrap()
            .freeze();
        let outcome = ImageService::run_edit(
            EditGenerator {
                host: host.clone(),
                preprocessed: preprocessed.clone(),
                closed: closed.clone(),
                capability_failure: None,
            },
            ImageEditRequest {
                generation: request(),
                image,
                references: vec![],
            },
            1024,
        );
        assert!(outcome.result.is_err());
        assert!(!outcome.shutdown.has_failure());
        assert_eq!(outcome.shutdown.stage_receipts().unwrap().len(), 0);
        assert_eq!(preprocessed.load(Ordering::SeqCst), 1);
        assert_eq!(closed.load(Ordering::SeqCst), 1);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn finite_metadata_failure_keeps_original_error_or_panic_and_skips_close() {
        for panic in [false, true] {
            let dropped = Arc::new(AtomicUsize::new(0));
            let closed = Arc::new(AtomicUsize::new(0));
            let outcome = ImageService::run_generation(
                EditGenerator {
                    host: host(),
                    preprocessed: Arc::new(AtomicUsize::new(0)),
                    closed: closed.clone(),
                    capability_failure: Some((dropped.clone(), panic)),
                },
                request(),
                1024,
            );
            assert!(outcome.result.is_err());
            assert!(outcome.shutdown.has_failure());
            assert!(outcome.shutdown.stage_receipts().is_none());
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
            assert_eq!(closed.load(Ordering::SeqCst), 0);
            let (owner, failure) = outcome.shutdown.into_parts();
            assert!(match failure.as_ref().unwrap() {
                ImageWorkerFailure::Panic(payload) => payload
                    .downcast_ref::<OriginalCapabilityFailure>()
                    .is_some(),
                ImageWorkerFailure::Generation(error) =>
                    error.downcast_ref::<OriginalCapabilityFailure>().is_some(),
                _ => false,
            });
            drop(failure);
            drop(owner);
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn finite_invalid_ingress_retires_without_execution() {
        let closed = Arc::new(AtomicUsize::new(0));
        let outcome = ImageService::run_generation(
            EditGenerator {
                host: host(),
                preprocessed: Arc::new(AtomicUsize::new(0)),
                closed: closed.clone(),
                capability_failure: None,
            },
            request(),
            0,
        );
        assert_eq!(
            outcome.result.err().unwrap().kind(),
            ImageServiceErrorKind::InvalidRequest
        );
        assert!(!outcome.shutdown.has_failure());
        assert!(outcome.shutdown.stage_receipts().is_some());
        assert_eq!(closed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn finite_qwen_run_rejects_missing_provider_and_closes_the_real_adapter() {
        let (pipeline, _sources) = crate::image_generation::tests::adapter::pipeline(1);
        let runtime = pipeline.execution.runtime.clone();
        let before = runtime.snapshot();
        let outcome = ImageService::run_generation(pipeline, request(), 1024);
        let error = outcome
            .result
            .err()
            .expect("Reference has no real Philox provider");
        assert!(error.to_string().contains("philox"), "{error}");
        assert!(!outcome.shutdown.has_failure());
        assert!(outcome.shutdown.stage_receipts().is_some());
        let after = runtime.snapshot();
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.submissions, before.submissions);
    }
}

impl ImageService {
    pub fn run_generation<P: ImageGenerator>(
        generator: P,
        request: ImageGenerationRequest,
        maximum_prompt_bytes: usize,
    ) -> ImageServiceRun {
        Self::run_request(
            generator,
            QueuedRequest::Generate(request),
            maximum_prompt_bytes,
        )
    }

    pub fn run_edit<P: ImageGenerator>(
        generator: P,
        request: ImageEditRequest,
        maximum_prompt_bytes: usize,
    ) -> ImageServiceRun {
        Self::run_request(
            generator,
            QueuedRequest::Edit(request, None),
            maximum_prompt_bytes,
        )
    }

    fn run_request<P: ImageGenerator>(
        generator: P,
        request: QueuedRequest,
        maximum_prompt_bytes: usize,
    ) -> ImageServiceRun {
        let worker = GeneratorAdapter(Box::new(generator));
        enum PreflightError {
            Invalid(ImageServiceError),
            Generator(GeneratorError),
        }
        let prepared = catch_unwind(AssertUnwindSafe(|| {
            let config = ImageServiceConfig {
                batching: ImageBatchPolicy::Configured,
                maximum_pending: 1,
                // Preparation can retain more than the original encoded input.
                // Use the real Host class ceiling; actual reservations remain
                // authoritative for this request and all coexisting owners.
                maximum_pending_bytes: worker
                    .0
                    .host_account()
                    .limits()
                    .class_bytes(HostMemoryClass::InputPreprocess),
                maximum_prompt_bytes,
                max_num_seqs: 1,
                request_batch_max_wait_ms: 0,
            };
            let capabilities = worker.0.capabilities().map_err(PreflightError::Generator)?;
            Self::prepare_with_capabilities(&worker, config, capabilities)
                .map_err(PreflightError::Invalid)
        }));
        let service = match prepared {
            Ok(Ok(service)) => service,
            Ok(Err(PreflightError::Invalid(error))) => return rejected(worker, error, None),
            Ok(Err(PreflightError::Generator(error))) => {
                return rejected(
                    worker,
                    service_error(
                        ImageServiceErrorKind::Generation,
                        "image capability query failed; original owner retained",
                    ),
                    Some(ImageWorkerFailure::Generation(error)),
                )
            }
            Err(panic) => {
                return rejected(
                    worker,
                    service_error(
                        ImageServiceErrorKind::WorkerPanic,
                        "image ingress panicked; original owner retained",
                    ),
                    Some(ImageWorkerFailure::Panic(panic)),
                )
            }
        };
        let job = match catch_unwind(AssertUnwindSafe(|| {
            service.submit_request(request, None, None)
        })) {
            Ok(job) => job,
            Err(panic) => {
                return rejected(
                    worker,
                    service_error(
                        ImageServiceErrorKind::WorkerPanic,
                        "image submission panicked; original owner retained",
                    ),
                    Some(ImageWorkerFailure::Panic(panic)),
                )
            }
        };
        let inner = service.core.run_to_completion(Box::new(worker));
        ImageServiceRun {
            result: job.and_then(|job| job.inner.completed()),
            shutdown: ImageServiceShutdown { inner },
        }
    }
}

fn rejected(
    worker: GeneratorAdapter,
    error: ImageServiceError,
    failure: Option<ImageWorkerFailure>,
) -> ImageServiceRun {
    ImageServiceRun {
        result: Err(error),
        shutdown: ImageServiceShutdown {
            inner: crate::iterative_service::retire_worker(Box::new(worker), failure),
        },
    }
}
