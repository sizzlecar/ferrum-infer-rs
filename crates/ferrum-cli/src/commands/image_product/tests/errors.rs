use super::*;
use ferrum_engine::image_generation::{service::*, ImageGenerationOutput, ImageGenerationRequest};
use ferrum_execution::{ExecutorBuildError, ExecutorShutdownObserver, ExecutorShutdownPhase};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use std::{error::Error, sync::Arc};

struct StartupFailure {
    original: ExecutorBuildError,
    formatter_panics: bool,
}
impl fmt::Display for StartupFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert!(
            !self.formatter_panics,
            "controlled original formatter panic"
        );
        fmt::Display::fmt(&self.original, f)
    }
}
impl fmt::Debug for StartupFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl Error for StartupFailure {}

struct FailingShutdown {
    host: HostMemoryAccount,
    failure: Option<StartupFailure>,
}
impl ImageGenerator for FailingShutdown {
    fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, Box<dyn Error + Send>> {
        Ok(ImageGeneratorCapabilities {
            maximum_batch_size: 1,
            maximum_width: 1,
            maximum_height: 1,
            dimension_multiple: 1,
        })
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        panic!("shutdown formatting must not submit generation")
    }
    fn shutdown(&mut self) -> Result<(), Box<dyn Error + Send>> {
        Err(Box::new(self.failure.take().unwrap()))
    }
}

#[tokio::test]
async fn image_server_error_preserves_original_cause_and_shutdown_custody() {
    for formatter_panics in [false, true] {
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                65536,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 65536,
                    output_bytes: 65536,
                    transfer_staging_bytes: 65536,
                    weight_backing_bytes: 65536,
                },
            )
            .unwrap(),
            1,
        )
        .unwrap();
        let phase = host
            .reserve_scope(&[(HostMemoryClass::WeightBacking, 64)])
            .unwrap();
        let backing = phase
            .account()
            .allocate_zeroed(HostMemoryClass::WeightBacking, 32)
            .unwrap();
        let (retirement, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
        observer.retain_until_retired((backing, phase));
        let original = ExecutorBuildError::with_cleanup(
            ferrum_types::FerrumError::backend("original startup configuration failed"),
            observer.clone(),
        );
        let service = Arc::new(
            ImageService::new(
                FailingShutdown {
                    host: host.clone(),
                    failure: Some(StartupFailure {
                        original,
                        formatter_panics,
                    }),
                },
                ImageServiceConfig {
                    batching: ferrum_types::ImageBatchPolicy::Configured,
                    maximum_pending: 1,
                    maximum_pending_bytes: 8192,
                    maximum_prompt_bytes: 64,
                    max_num_seqs: 1,
                    request_batch_max_wait_ms: 0,
                },
            )
            .unwrap(),
        );
        let shutdown = service.shutdown().await.unwrap();
        let source = anyhow::anyhow!("image worker has a retained failure");
        let message = image_server_error_message(&source, Some(&shutdown));
        assert!(message.contains("image worker has a retained failure"));
        assert!(message.contains(if formatter_panics {
            "error formatter panicked; original owner retained"
        } else {
            "original startup configuration failed"
        }));
        assert_eq!(
            image_server_error_message(&source, None),
            "image server: image worker has a retained failure"
        );
        #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
        let shutdown = {
            let error = ImageProductError::Server {
                source,
                service: ferrum_server::ImageHttpService::new(
                    "test-image",
                    service.clone(),
                    ferrum_server::ImageHttpConfig {
                        output: ferrum_types::OutputConfig::default(),
                        default_width: 1,
                        default_height: 1,
                        default_seed: 0,
                        rgba_scratch_bytes: 8,
                        png_limits: ferrum_output::png::PngLimits::default(),
                    },
                )
                .unwrap(),
                shutdown: Some(shutdown),
            };
            assert_eq!(error.to_string(), message);
            assert!(Error::source(&error).unwrap().is::<ImageWorkerFailure>());
            let ImageProductError::Server {
                shutdown: Some(shutdown),
                ..
            } = error
            else {
                unreachable!()
            };
            shutdown
        };
        assert!(shutdown.failure().is_some());
        assert_ne!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::WeightBacking),
            64
        );
        let (generator, failure) = shutdown.into_parts();
        let Some(ImageWorkerFailure::Generation(failure)) = failure else {
            panic!("the original failure must still be available for typed recovery")
        };
        let failure = failure.downcast::<StartupFailure>().unwrap();
        assert!(failure
            .original
            .to_string()
            .contains("original startup configuration failed"));
        assert_ne!(
            failure.original.cleanup().unwrap().snapshot().phase(),
            ExecutorShutdownPhase::Retired
        );
        drop((failure, generator, service));
        assert_eq!(host.usage().unwrap().committed_bytes(), 64);
        retirement.retired(None);
        assert_eq!(
            observer.wait().await.phase(),
            ExecutorShutdownPhase::Retired
        );
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    }
}
