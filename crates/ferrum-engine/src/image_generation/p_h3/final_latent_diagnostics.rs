//! Borrow the completed target sample; no extra readback or retained owner.
use anyhow::{ensure, Context, Result};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostTensor, RequestIdentity,
    ResolvedTensorLayout, ResolvedTensorSpec,
};
use std::{num::NonZeroUsize, panic::AssertUnwindSafe};

pub struct PH3ImageFinalLatentObservation<'a> {
    pub request: &'a RequestIdentity,
    pub job: u64,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub completed_steps: usize,
    /// Original normalized F32 NCTHW sample, not denormalized VAE tile rows.
    pub tensor: &'a ResolvedTensorSpec,
    pub bytes: &'a [u8],
}
impl PH3ImageFinalLatentObservation<'_> {
    pub fn payload_bytes(&self) -> Result<usize> {
        ensure!(
            self.completed_steps > 0
                && self.tensor.element_type() == ElementType::F32
                && self.tensor.layout() == &ResolvedTensorLayout::Contiguous
                && matches!(self.tensor.dimensions(), [1, 24, 1, h, w]
                    if *h > 0 && *w > 0
                    && h.checked_mul(16) == Some(u64::from(self.height))
                    && w.checked_mul(16) == Some(u64::from(self.width)))
                && self.tensor.minimum_storage_bytes()? == u64::try_from(self.bytes.len())?,
            "final target capture requires the complete normalized B1 F32 image sample"
        );
        ensure!(
            self.bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "final target capture contains nonfinite values"
        );
        Ok(self.bytes.len())
    }
}

/// A synchronous diagnostic sink may only borrow the existing host payload.
pub trait PH3ImageFinalLatentObserver: Send + Sync {
    fn observe(&mut self, view: PH3ImageFinalLatentObservation<'_>) -> Result<()>;
}
pub(super) struct FinalLatentDiagnostics {
    limit: NonZeroUsize,
    observer: Box<dyn PH3ImageFinalLatentObserver>,
}
impl FinalLatentDiagnostics {
    pub(super) fn new(limit: NonZeroUsize, observer: Box<dyn PH3ImageFinalLatentObserver>) -> Self {
        Self { limit, observer }
    }
    fn deliver(&mut self, view: PH3ImageFinalLatentObservation<'_>) -> Result<()> {
        ensure!(
            view.payload_bytes()? <= self.limit.get(),
            "final target capture exceeds raw payload limit"
        );
        std::panic::catch_unwind(AssertUnwindSafe(|| self.observer.observe(view)))
            .map_err(|_| anyhow::anyhow!("final target observer panicked"))?
            .context("final target capture is incomplete")
    }
}

/// The caller has already completed decoding and set its original decoded
/// state. Failure is a diagnostic processing error, never admission/retry.
#[allow(clippy::too_many_arguments)]
pub(super) fn observe_final_latent(
    diagnostics: &mut Option<FinalLatentDiagnostics>,
    host: &HostMemoryAccount,
    request: &RequestIdentity,
    job: u64,
    width: u32,
    height: u32,
    seed: u64,
    completed_steps: usize,
    image: &ManagedHostTensor,
) -> Result<()> {
    let Some(diagnostics) = diagnostics else {
        return Ok(());
    };
    ensure!(
        host.owns(image.bytes()) && image.bytes().class() == HostMemoryClass::InputPreprocess,
        "final target capture belongs to a foreign Host owner"
    );
    diagnostics.deliver(PH3ImageFinalLatentObservation {
        request,
        job,
        width,
        height,
        seed,
        completed_steps,
        tensor: image.tensor(),
        bytes: image.bytes().bytes(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_generation::ImageGenerationFailure;
    use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits, ProgramValueId};

    struct Observer {
        pointer: usize,
        panics: bool,
    }
    impl PH3ImageFinalLatentObserver for Observer {
        fn observe(&mut self, view: PH3ImageFinalLatentObservation<'_>) -> Result<()> {
            assert_eq!(view.bytes.as_ptr() as usize, self.pointer);
            assert_eq!(view.completed_steps, 8);
            if self.panics {
                panic!("capture panic fixture");
            }
            anyhow::bail!("capture I/O fixture")
        }
    }
    #[test]
    fn final_latent_observer_failures_leave_original_host_owner_and_cannot_retry_decode() {
        for panics in [false, true] {
            let host = HostMemoryAccount::with_allocation_alignment(
                HostMemoryLimits::new(
                    256,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: 256,
                        output_bytes: 0,
                        transfer_staging_bytes: 0,
                        weight_backing_bytes: 0,
                    },
                )
                .unwrap(),
                64,
            )
            .unwrap();
            let image = ManagedHostTensor::new(
                ProgramValueId::new("value.final.sample").unwrap(),
                ResolvedTensorSpec::new(
                    vec![1, 24, 1, 1, 1],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                host.allocate_zeroed(HostMemoryClass::InputPreprocess, 96)
                    .unwrap()
                    .freeze(),
            )
            .unwrap();
            let charged = host.usage().unwrap().committed_bytes();
            let request = RequestIdentity::new("request.final.sample").unwrap();
            let mut observer = Some(FinalLatentDiagnostics::new(
                NonZeroUsize::new(96).unwrap(),
                Box::new(Observer {
                    pointer: image.bytes().bytes().as_ptr() as usize,
                    panics,
                }),
            ));
            let error =
                observe_final_latent(&mut observer, &host, &request, 71, 16, 16, 42, 8, &image)
                    .unwrap_err();
            assert_eq!(host.usage().unwrap().committed_bytes(), charged);
            let failure: ImageGenerationFailure<ReferenceDeviceRuntime> =
                ImageGenerationFailure::Processing {
                    source: error,
                    active: None,
                };
            assert!(super::super::clean_decoder_admission(failure).is_err());
            assert_eq!(host.usage().unwrap().committed_bytes(), charged);
            observe_final_latent(&mut None, &host, &request, 71, 0, 0, 0, 0, &image).unwrap();
            drop(image);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
    }
}
