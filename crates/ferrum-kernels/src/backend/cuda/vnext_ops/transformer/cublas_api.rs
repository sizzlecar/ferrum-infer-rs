//! The existing GemmEx API contract, shared by execution and future evidence.
//! DEFAULT_TENSOR_OP is a selection policy, not a private kernel algorithm ID.
use super::*;
use ferrum_interfaces::execution_cost::{
    LibraryApiNumericWorkV1, LibraryReplayParametersV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, StatisticalEvidenceUnknown,
};
use sha2::{Digest, Sha256};

/// Observed once at a controlled handle-lifecycle boundary. The owner must
/// retain the handle policy unchanged and check each subsequently created
/// handle before using this identity. Not executable or deserializable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::backend::cuda) struct CublasHandleApiIdentity {
    version: u32,
    math_mode: u32,
    atomics_mode: u32,
    pointer_mode: u32,
    sm_count_target: u32,
}

impl CublasHandleApiIdentity {
    pub(in crate::backend::cuda) fn read(blas: &CudaBlas) -> Result<Self, CudaDeviceRuntimeError> {
        use cudarc::cublas::sys;
        let (mut version, mut math, mut atomics, mut pointer, mut target) =
            (0i32, 0i32, 0i32, 0i32, 0i32);
        // SAFETY: the borrowed CudaBlas owns this live handle throughout all
        // getters. Each output is an initialized C int-sized stack slot.
        // Read mode bits as integers so combined math flags are not assumed
        // to be a single Rust enum discriminant.
        unsafe {
            sys::cublasGetVersion_v2(*blas.handle(), &mut version)
                .result()
                .map_err(|e| CudaDeviceRuntimeError::blas("cuBLAS version", e))?;
            sys::cublasGetMathMode(*blas.handle(), (&mut math as *mut i32).cast())
                .result()
                .map_err(|e| CudaDeviceRuntimeError::blas("cuBLAS math mode", e))?;
            sys::cublasGetAtomicsMode(*blas.handle(), (&mut atomics as *mut i32).cast())
                .result()
                .map_err(|e| CudaDeviceRuntimeError::blas("cuBLAS atomics mode", e))?;
            sys::cublasGetPointerMode_v2(*blas.handle(), (&mut pointer as *mut i32).cast())
                .result()
                .map_err(|e| CudaDeviceRuntimeError::blas("cuBLAS pointer mode", e))?;
            sys::cublasGetSmCountTarget(*blas.handle(), &mut target)
                .result()
                .map_err(|e| CudaDeviceRuntimeError::blas("cuBLAS SM target", e))?;
        }
        Self::checked(version, math, atomics, pointer, target).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("cuBLAS API handle identity unavailable")
        })
    }

    fn checked(version: i32, math: i32, atomics: i32, pointer: i32, target: i32) -> Option<Self> {
        // The original launcher passes host alpha/beta addresses. No mode is
        // changed here to make an unsupported handle fit this cost contract.
        if version <= 0 || math < 0 || !(0..=1).contains(&atomics) || pointer != 0 || target < 0 {
            return None;
        }
        Some(Self {
            version: version as u32,
            math_mode: math as u32,
            atomics_mode: atomics as u32,
            pointer_mode: pointer as u32,
            sm_count_target: target as u32,
        })
    }

    #[cfg(test)]
    pub(super) fn fixture_identity() -> Self {
        Self::checked(120900, 0, 0, 0, 0).unwrap()
    }

    fn signature(self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"ferrum.cuda.cublas-observed-api-policy.v1\0");
        // This is the unchanged CudaBlas::new default workspace lifecycle.
        // It is not a claim that vendor-private scratch has size zero.
        hash.update(b"one-owned-handle-per-stream.default-library-workspace\0");
        for value in [
            self.version,
            self.math_mode,
            self.atomics_mode,
            self.pointer_mode,
            self.sm_count_target,
        ] {
            hash.update(value.to_le_bytes());
        }
        hash.finalize().into()
    }
}

/// Shared startup observation, never a substitute for the actual stream's
/// separate observation. Disabled retains no Arc, allocation or getter call.
#[derive(Clone, Default)]
pub(in crate::backend::cuda) struct CublasCostIdentitySource(
    Option<std::sync::Arc<std::sync::OnceLock<Option<CublasHandleApiIdentity>>>>,
);
impl CublasCostIdentitySource {
    pub(in crate::backend::cuda) fn new(capture: ferrum_types::SloStructuredCostCapture) -> Self {
        Self((!capture.is_disabled()).then(|| std::sync::Arc::new(std::sync::OnceLock::new())))
    }
    pub(in crate::backend::cuda) fn observe_created_handle(
        &self,
        blas: &CudaBlas,
    ) -> Option<CublasHandleApiIdentity> {
        let shared = self.0.as_ref()?;
        let observed = CublasHandleApiIdentity::read(blas).ok();
        // Freeze the first attempt, including failure. Later successful handles
        // cannot silently grant a contract unavailable during initial planning.
        let _ = shared.set(observed);
        observed
    }
    pub(in crate::backend::cuda) fn frozen(&self) -> Option<CublasHandleApiIdentity> {
        self.0.as_ref()?.get().copied().flatten()
    }
}

/// Required(None) is deliberately different from NotRequired: a lost library
/// contract cannot make a cost-witness gate treat the command as legacy work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::backend::cuda) enum CublasCostRequirement {
    #[default]
    NotRequired,
    Required(Option<CublasHandleApiIdentity>),
}
impl CublasCostRequirement {
    pub(in crate::backend::cuda) fn matches_observed(
        self,
        observed: Option<CublasHandleApiIdentity>,
    ) -> bool {
        match self {
            Self::NotRequired => true,
            Self::Required(Some(expected)) => observed == Some(expected),
            Self::Required(None) => false,
        }
    }
    pub(in crate::backend::cuda) fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::NotRequired, rhs) => rhs,
            (lhs, Self::NotRequired) => lhs,
            (Self::Required(Some(a)), Self::Required(Some(b))) if a == b => Self::Required(Some(a)),
            _ => Self::Required(None),
        }
    }
}

/// Sealed with the retained capture handle, not a caller-supplied current
/// stream identity. Cloning this value grants no graph or resource authority.
#[derive(Debug, Clone, Copy)]
pub(in crate::backend::cuda) struct CapturedCublasCostContract {
    requirement: CublasCostRequirement,
    observed: Option<CublasHandleApiIdentity>,
}
impl CapturedCublasCostContract {
    pub(in crate::backend::cuda) fn new(
        requirement: CublasCostRequirement,
        observed: Option<CublasHandleApiIdentity>,
    ) -> Self {
        Self {
            requirement,
            observed,
        }
    }
    pub(in crate::backend::cuda) fn is_valid(self) -> bool {
        self.requirement.matches_observed(self.observed)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GemmF16ApiPlan {
    rows: i32,
    columns: i32,
    reduction: i32,
}
impl GemmF16ApiPlan {
    pub(super) fn new(
        rows: i32,
        columns: i32,
        reduction: i32,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if rows <= 0 || columns <= 0 || reduction <= 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "cuBLAS GemmEx dimensions must be positive",
            ));
        }
        Ok(Self {
            rows,
            columns,
            reduction,
        })
    }

    fn parameters(self) -> [u64; 16] {
        [
            cublasOperation_t::CUBLAS_OP_T as u64,
            cublasOperation_t::CUBLAS_OP_N as u64,
            self.columns as u64,
            self.rows as u64,
            self.reduction as u64,
            cudaDataType_t::CUDA_R_16F as u64,
            self.reduction as u64,
            cudaDataType_t::CUDA_R_16F as u64,
            self.reduction as u64,
            cudaDataType_t::CUDA_R_16F as u64,
            self.columns as u64,
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F as u64,
            // Signed public selection-mode enum retained byte-for-byte.
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP as i32 as u32 as u64,
            u64::from(CUDA_GEMM_ALPHA_F32.to_bits()),
            u64::from(CUDA_GEMM_BETA_F32.to_bits()),
            cudaDataType_t::CUDA_R_32F as u64,
        ]
    }

    fn algorithm(
        self,
        identity: CublasHandleApiIdentity,
    ) -> Result<SelectedAlgorithmClassV1, StatisticalEvidenceUnknown> {
        let mut layout = Sha256::new();
        layout.update(b"ferrum.cuda.cublas.GemmEx.dense-f16-f32-fast.v1\0");
        // N/K/leading dimensions/types/compute policy are fixed within each
        // statistical class. Only M is a numeric axis. This prevents equal
        // MN/MNK products from conflating different library matrix geometries.
        for (index, parameter) in self.parameters().into_iter().enumerate() {
            if index != 3 {
                layout.update(parameter.to_le_bytes());
            }
        }
        SelectedAlgorithmClassV1::library_api(
            "cuda.cublasGemmEx",
            1,
            identity.signature(),
            layout.finalize().into(),
        )
    }

    /// Caller first proves the same owned runtime/handle policy and original
    /// physical operand bindings. No code currently enables RN cost via this
    /// passive helper; a missing identity must remain missing evidence.
    pub(super) fn append_selected(
        self,
        builder: &mut SelectedCommandCostBuilderV1,
        identity: CublasHandleApiIdentity,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        builder.library_call_with_replay_parameters(
            self.algorithm(identity)?,
            LibraryApiNumericWorkV1 {
                output_elements: (self.rows as u64)
                    .checked_mul(self.columns as u64)
                    .ok_or(StatisticalEvidenceUnknown::Overflow)?,
                reduction_units_per_output: self.reduction as u64,
            },
            LibraryReplayParametersV1 {
                fixed_parameters: &self.parameters(),
            },
        )
    }

    pub(super) fn launch(
        self,
        blas: &CudaBlas,
        input: cudarc::driver::sys::CUdeviceptr,
        weight: cudarc::driver::sys::CUdeviceptr,
        output: cudarc::driver::sys::CUdeviceptr,
        operation: &'static str,
    ) -> Result<(), CudaDeviceRuntimeError> {
        // SAFETY: identical original checked projection spans and GemmEx ABI.
        unsafe {
            gemm_ex(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                self.columns,
                self.rows,
                self.reduction,
                &CUDA_GEMM_ALPHA_F32 as *const f32 as *const c_void,
                weight as *const c_void,
                cudaDataType_t::CUDA_R_16F,
                self.reduction,
                input as *const c_void,
                cudaDataType_t::CUDA_R_16F,
                self.reduction,
                &CUDA_GEMM_BETA_F32 as *const f32 as *const c_void,
                output as *mut c_void,
                cudaDataType_t::CUDA_R_16F,
                self.columns,
                cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .map_err(|error| CudaDeviceRuntimeError::blas(operation, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::execution_cost::{
        AlgorithmWorkKindV1, SelectedReplayAlgorithmTemplateV1,
    };

    fn identity() -> CublasHandleApiIdentity {
        CublasHandleApiIdentity::checked(120900, 0, 0, 0, 0).unwrap()
    }

    #[test]
    fn cublas_required_identity_cannot_disappear_or_borrow_a_different_capture_handle() {
        let a = identity();
        let b = CublasHandleApiIdentity::checked(120901, 0, 0, 0, 0).unwrap();
        assert!(CublasCostRequirement::NotRequired.matches_observed(None));
        assert!(!CublasCostRequirement::Required(None).matches_observed(Some(a)));
        assert!(!CublasCostRequirement::Required(Some(a)).matches_observed(None));
        assert!(!CublasCostRequirement::Required(Some(a)).matches_observed(Some(b)));
        let captured =
            CapturedCublasCostContract::new(CublasCostRequirement::Required(Some(a)), Some(a));
        assert!(captured.is_valid());
        assert!(!CapturedCublasCostContract::new(
            CublasCostRequirement::Required(Some(a)),
            Some(b)
        )
        .is_valid());
        assert_eq!(
            CublasCostRequirement::Required(Some(a)).merge(CublasCostRequirement::Required(None)),
            CublasCostRequirement::Required(None)
        );
        assert_eq!(
            CublasCostRequirement::Required(Some(a))
                .merge(CublasCostRequirement::Required(Some(b))),
            CublasCostRequirement::Required(None)
        );
        let disabled =
            CublasCostIdentitySource::new(ferrum_types::SloStructuredCostCapture::Disabled);
        assert!(disabled.0.is_none());
        let enabled =
            CublasCostIdentitySource::new(ferrum_types::SloStructuredCostCapture::HostSettledV1);
        assert!(enabled.frozen().is_none());
        let shared = enabled.0.as_ref().unwrap();
        shared.set(None).unwrap();
        assert!(shared.set(Some(a)).is_err());
        assert!(
            enabled.frozen().is_none(),
            "a failed initial observation cannot be renewed silently"
        );
    }

    #[test]
    fn cublas_api_classes_fix_n_k_handle_policy_and_preserve_numeric_m() {
        let a = GemmF16ApiPlan::new(4, 64, 32).unwrap();
        let b = GemmF16ApiPlan::new(8, 64, 32).unwrap();
        assert_eq!(
            a.algorithm(identity()).unwrap(),
            b.algorithm(identity()).unwrap()
        );
        for other in [
            GemmF16ApiPlan::new(2, 128, 32).unwrap(),
            GemmF16ApiPlan::new(4, 64, 64).unwrap(),
        ] {
            assert_ne!(
                a.algorithm(identity()).unwrap(),
                other.algorithm(identity()).unwrap()
            );
        }
        for changed in [
            CublasHandleApiIdentity::checked(120901, 0, 0, 0, 0).unwrap(),
            CublasHandleApiIdentity::checked(120900, 1, 0, 0, 0).unwrap(),
            CublasHandleApiIdentity::checked(120900, 0, 1, 0, 0).unwrap(),
            CublasHandleApiIdentity::checked(120900, 0, 0, 0, 8).unwrap(),
        ] {
            assert_ne!(
                a.algorithm(identity()).unwrap(),
                a.algorithm(changed).unwrap()
            );
        }
        assert!(CublasHandleApiIdentity::checked(120900, 0, 0, 1, 0).is_none());
        assert!(CublasHandleApiIdentity::checked(0, 0, 0, 0, 0).is_none());
        assert!(GemmF16ApiPlan::new(0, 64, 32).is_err());
        let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(4);
        a.append_selected(&mut builder, identity()).unwrap();
        let value = builder.finish().unwrap();
        assert_eq!(value.work().logical_units, 256);
        assert_eq!(value.work().grid_blocks, 0);
        assert_eq!(
            value.algorithm_work().unwrap().unwrap().entries()[0].kind(),
            AlgorithmWorkKindV1::LibraryCall
        );
        SelectedReplayAlgorithmTemplateV1::from_selected(&value, 4, 1, 0)
            .unwrap()
            .validate_binding(&value)
            .unwrap();
    }

    #[test]
    #[ignore = "requires an actual CUDA device"]
    fn cublas_api_identity_reads_real_unchanged_owned_handle() {
        use cudarc::driver::CudaContext;
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let blas = CudaBlas::new(stream).unwrap();
        let first = CublasHandleApiIdentity::read(&blas).unwrap();
        assert!(first.version > 0);
        assert_eq!(first.pointer_mode, 0);
        assert_eq!(first, CublasHandleApiIdentity::read(&blas).unwrap());
    }
}
