//! Loaded projection ownership and its optional graph-capture contract.

use crate::{backend::Backend, Linear, LinearMetadata};

/// A loaded projection and the capture contract of that exact owned object.
///
/// Safe extensions use [`Self::new`]. They remain available for eager execution;
/// a model explicitly requesting graph capture rejects unqualified projections.
/// The contract is checked once during model construction, never per token.
pub struct LoadedLinear<B: Backend> {
    inner: Box<dyn Linear<B>>,
    capture_stable: bool,
}

impl<B: Backend> LoadedLinear<B> {
    pub fn new(inner: Box<dyn Linear<B>>) -> Self {
        Self {
            inner,
            capture_stable: false,
        }
    }

    /// Qualify this owned projection for retained capture and replay.
    ///
    /// # Safety
    /// Every persistent dependency encoded by `forward` must be owned by this
    /// object or the execution context supplied to that invocation (not a borrowed
    /// loader/upload context). Its address and interpretation
    /// must remain stable until the caller has retired every captured graph and
    /// its actual executions. Temporary input/output storage belongs to the
    /// caller and remains subject to that same retirement requirement. No encoded
    /// operation may refer to temporary host storage that dies after `forward`.
    /// Replaying the encoded work without calling `forward` again must preserve
    /// semantics: required host side effects, mutable external weights, or
    /// independently replaceable dependencies make this contract invalid.
    /// The caller retaining this wrapper must retire graphs before dropping it.
    pub unsafe fn new_capture_stable(inner: Box<dyn Linear<B>>) -> Self {
        Self {
            inner,
            capture_stable: true,
        }
    }

    pub fn is_capture_stable(&self) -> bool {
        self.capture_stable
    }

    /// Consume the wrapper for a consumer that does not retain captured graphs.
    /// The capture credential is destroyed and cannot qualify another object.
    pub fn into_eager_box(self) -> Box<dyn Linear<B>> {
        self.inner
    }

    #[inline]
    pub fn as_ref(&self) -> &dyn Linear<B> {
        self.inner.as_ref()
    }
    #[inline]
    pub fn in_features(&self) -> usize {
        self.inner.in_features()
    }
    #[inline]
    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }
    #[inline]
    pub fn metadata(&self) -> LinearMetadata {
        self.inner.metadata()
    }
    #[inline]
    pub fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        self.inner.forward(ctx, input, out, m);
    }
    #[cfg(feature = "cuda")]
    #[inline]
    pub fn cuda_marlin_touch_ref(
        &self,
    ) -> Option<crate::quant_linear::cuda_marlin::CudaMarlinTouchRef<'_>> {
        self.inner.cuda_marlin_touch_ref()
    }
}

impl<B: Backend> Linear<B> for LoadedLinear<B> {
    #[inline]
    fn in_features(&self) -> usize {
        self.in_features()
    }
    #[inline]
    fn out_features(&self) -> usize {
        self.out_features()
    }
    #[inline]
    fn metadata(&self) -> LinearMetadata {
        self.metadata()
    }
    #[inline]
    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        self.forward(ctx, input, out, m);
    }
    #[cfg(feature = "cuda")]
    #[inline]
    fn cuda_marlin_touch_ref(
        &self,
    ) -> Option<crate::quant_linear::cuda_marlin::CudaMarlinTouchRef<'_>> {
        self.cuda_marlin_touch_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::cpu::CpuBackend, LinearProjectionRole};
    use std::sync::Arc;

    struct OwnedProjection {
        weights: Vec<f32>,
        _owner: Arc<()>,
    }
    impl Linear<CpuBackend> for OwnedProjection {
        fn in_features(&self) -> usize {
            2
        }
        fn out_features(&self) -> usize {
            2
        }
        fn metadata(&self) -> LinearMetadata {
            LinearMetadata::new(Some(3), Some(LinearProjectionRole::Output))
        }
        fn forward(
            &self,
            ctx: &mut <CpuBackend as Backend>::Context,
            input: &Vec<f32>,
            output: &mut Vec<f32>,
            m: usize,
        ) {
            CpuBackend::gemm(ctx, input, &self.weights, output, m, 2, 2);
        }
    }

    #[test]
    fn loaded_linear_credential_stays_with_exact_owned_projection() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        // SAFETY: the projection owns fixed weights and delegates only actual
        // backend GEMM; no required host side effect is skipped by replay.
        let loaded = unsafe {
            LoadedLinear::new_capture_stable(Box::new(OwnedProjection {
                weights: vec![1.0, 2.0, -1.0, 3.0],
                _owner: owner,
            }))
        };
        assert!(loaded.is_capture_stable());
        assert_eq!((loaded.in_features(), loaded.out_features()), (2, 2));
        assert_eq!(
            loaded.metadata(),
            LinearMetadata::new(Some(3), Some(LinearProjectionRole::Output))
        );
        let mut ctx = CpuBackend::new_context(&Default::default()).unwrap();
        let input = vec![2.0, 5.0, -3.0, 4.0];
        let mut output = vec![f32::NAN; 4];
        loaded.forward(&mut ctx, &input, &mut output, 2);
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(output, vec![12.0, 13.0, 5.0, 15.0]);
        let eager = LoadedLinear::new(loaded.into_eager_box());
        assert!(
            !eager.is_capture_stable(),
            "consuming the owner cannot transfer a detached credential"
        );
        assert!(weak.upgrade().is_some());
        eager.forward(&mut ctx, &input, &mut output, 2);
        assert_eq!(output, vec![12.0, 13.0, 5.0, 15.0]);
        CpuBackend::retire_context(&mut ctx).unwrap();
        drop(eager);
        assert!(weak.upgrade().is_none());
    }
}
