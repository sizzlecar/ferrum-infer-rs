//! Synchronous observation of the real prepared inputs, before actor execution.
use anyhow::{ensure, Context, Result};
use ferrum_models::p_h3_image::{PH3ImageCurveBlockRowRun, PH3ImageSamplingStep};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, ManagedHostTensor, PreparedModelFamily, RequestIdentity,
    ResolvedTensorLayout,
};
use std::{num::NonZeroUsize, panic::AssertUnwindSafe};

#[derive(Clone, Copy)]
pub(super) struct DitInputIdentity {
    pub job: u64,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
}

/// Original Host owners remain exclusively with the production request. This
/// callback is preparation evidence, not a claim that the forward completed.
pub struct PH3ImageDitInputObservation<'a> {
    pub request: &'a RequestIdentity,
    pub job: u64,
    pub cohort_job: u64,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub step: PH3ImageSamplingStep,
    pub family: &'a PreparedModelFamily,
    pub row_runs: &'a [PH3ImageCurveBlockRowRun],
    pub times: &'a [f32],
    pub inputs: &'a [ManagedHostTensor],
    pub image_state: &'a ManagedHostTensor,
    pub audio_state: &'a ManagedHostTensor,
}
impl PH3ImageDitInputObservation<'_> {
    /// Conservative per-callback extent includes every borrowed tensor; a sink
    /// can store fixed ports once. Neither limit is an extra Host reservation.
    pub fn payload_bytes(&self) -> Result<usize> {
        ensure!(
            self.inputs.len() == 6 && !self.row_runs.is_empty(),
            "DiT capture requires six real B1 inputs and routing"
        );
        ensure!(
            self.times.iter().all(|v| v.is_finite()),
            "DiT capture time is nonfinite"
        );
        tensor_bytes(
            self.inputs
                .iter()
                .chain([self.image_state, self.audio_state]),
        )
    }
}
fn tensor_bytes<'a>(mut tensors: impl Iterator<Item = &'a ManagedHostTensor>) -> Result<usize> {
    tensors.try_fold(0usize, |n, t| {
        ensure!(
            t.tensor().layout() == &ResolvedTensorLayout::Contiguous
                && t.tensor().minimum_storage_bytes()? == u64::try_from(t.bytes().bytes().len())?,
            "DiT capture extent differs"
        );
        let finite = match t.tensor().element_type() {
            ElementType::F32 => t
                .bytes()
                .bytes()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            ElementType::Bf16 => t
                .bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80),
            _ => false,
        };
        ensure!(
            finite,
            "DiT capture requires finite original F32/BF16 tensors"
        );
        n.checked_add(t.bytes().bytes().len())
            .context("DiT capture extent overflows")
    })
}
pub trait PH3ImageDitInputObserver: Send + Sync {
    fn observe(&mut self, view: PH3ImageDitInputObservation<'_>) -> Result<()>;
}
pub(super) struct DitInputDiagnostics {
    limit: NonZeroUsize,
    observer: Box<dyn PH3ImageDitInputObserver>,
}
impl DitInputDiagnostics {
    pub(super) fn new(limit: NonZeroUsize, observer: Box<dyn PH3ImageDitInputObserver>) -> Self {
        Self { limit, observer }
    }
    pub(super) fn deliver(
        &mut self,
        host: &HostMemoryAccount,
        view: PH3ImageDitInputObservation<'_>,
    ) -> Result<()> {
        ensure!(
            view.inputs
                .iter()
                .chain([view.image_state, view.audio_state])
                .all(|t| host.owns(t.bytes())),
            "DiT capture belongs to a foreign Host owner"
        );
        ensure!(
            view.payload_bytes()? <= self.limit.get(),
            "DiT capture exceeds raw payload limit"
        );
        invoke(|| self.observer.observe(view))
    }
}
fn invoke(callback: impl FnOnce() -> Result<()>) -> Result<()> {
    std::panic::catch_unwind(AssertUnwindSafe(callback))
        .map_err(|_| anyhow::anyhow!("DiT input observer panicked"))?
        .context("DiT input capture is incomplete")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{
        HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits, ProgramValueId,
        ResolvedTensorSpec,
    };
    #[test]
    fn borrowed_dit_payload_rejects_nonfinite_and_observer_failure_keeps_owner() {
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                64,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 64,
                    output_bytes: 0,
                    weight_backing_bytes: 0,
                    transfer_staging_bytes: 0,
                },
            )
            .unwrap(),
        );
        let make = |value: f32| {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
                .unwrap();
            bytes.as_mut().copy_from_slice(&value.to_le_bytes());
            ManagedHostTensor::new(
                ProgramValueId::new("value.diagnostic.original").unwrap(),
                ResolvedTensorSpec::new(
                    vec![1],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
            .unwrap()
        };
        let tensor = make(-0.0);
        assert_eq!(tensor_bytes([&tensor].into_iter()).unwrap(), 4);
        let pointer = tensor.bytes().bytes().as_ptr();
        let charged = host.usage().unwrap().committed_bytes();
        for panics in [false, true] {
            assert!(invoke(|| {
                assert_eq!(tensor.bytes().bytes().as_ptr(), pointer);
                if panics {
                    panic!("observer failure");
                }
                anyhow::bail!("disk failure")
            })
            .is_err());
            assert_eq!(host.usage().unwrap().committed_bytes(), charged);
            assert_eq!(tensor.bytes().bytes(), (-0.0f32).to_le_bytes());
        }
        let invalid = make(f32::NAN);
        assert!(tensor_bytes([&invalid].into_iter()).is_err());
        drop(invalid);
        drop(tensor);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
