//! Typed selection from one fixed native module. Selection never grants a
//! capability absent from the runtime descriptor and is checked before load.

use super::*;

/// BF16 fixed-tensor operations implemented by the shared CUDA native module.
/// Dense remains a separate provider because it uses the admitted lane BLAS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum CudaTensorOperation {
    Silu,
    SiluMul,
    Tanh,
    ModulationScale,
    GatedResidual,
    LayerNorm,
    HeadRmsNorm,
    AdjacentRope,
    StructuredAttention,
    ZeroCenteredRmsNorm,
    GeluTanh,
    RowGather,
    BatchedRowConcat,
    PlainResidualAdd,
    HalfSplitRope,
    StructuredGqaAttention,
}

impl CudaTensorOperation {
    pub const ALL: [Self; 16] = [
        Self::Silu,
        Self::SiluMul,
        Self::Tanh,
        Self::ModulationScale,
        Self::GatedResidual,
        Self::LayerNorm,
        Self::HeadRmsNorm,
        Self::AdjacentRope,
        Self::StructuredAttention,
        Self::ZeroCenteredRmsNorm,
        Self::GeluTanh,
        Self::RowGather,
        Self::BatchedRowConcat,
        Self::PlainResidualAdd,
        Self::HalfSplitRope,
        Self::StructuredGqaAttention,
    ];

    fn pointwise(self) -> Option<Kind> {
        match self {
            Self::Silu => Some(Kind::Silu),
            Self::SiluMul => Some(Kind::SiluMul),
            Self::Tanh => Some(Kind::Tanh),
            Self::ModulationScale => Some(Kind::Modulation),
            Self::GatedResidual => Some(Kind::GatedResidual),
            Self::GeluTanh => Some(Kind::GeluTanh),
            Self::PlainResidualAdd => Some(Kind::PlainResidualAdd),
            _ => None,
        }
    }

    fn normalization(self) -> Option<norm::Kind> {
        match self {
            Self::LayerNorm => Some(norm::Kind::Layer),
            Self::HeadRmsNorm => Some(norm::Kind::HeadRms),
            Self::ZeroCenteredRmsNorm => Some(norm::Kind::ZeroCenteredRms),
            _ => None,
        }
    }

    fn descriptor(
        self,
        device: &DeviceDescriptor,
    ) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
        if let Some(kind) = self.pointwise() {
            return descriptor(device, kind);
        }
        if let Some(kind) = self.normalization() {
            return norm::descriptor(device, kind);
        }
        match self {
            Self::AdjacentRope => rope::descriptor(device),
            Self::StructuredAttention => attention::descriptor(device),
            Self::HalfSplitRope => half_split_rope::descriptor(device),
            Self::StructuredGqaAttention => gqa::descriptor(device),
            Self::RowGather => transport::descriptor(device, transport::Kind::RowGather),
            Self::BatchedRowConcat => {
                transport::descriptor(device, transport::Kind::BatchedRowConcat)
            }
            _ => Err(CudaDeviceRuntimeError::contract(
                "unknown CUDA tensor operation",
            )),
        }
    }

    fn provider(
        self,
        descriptor: OperationProviderDescriptor,
        kernels: &Arc<RetainedFunctions>,
    ) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
        if let Some(kind) = self.pointwise() {
            return Box::new(CudaTensorPointwiseProvider {
                descriptor,
                kind,
                kernels: Arc::clone(kernels),
            });
        }
        if let Some(kind) = self.normalization() {
            return norm::provider(kind, descriptor, kernels);
        }
        match self {
            Self::AdjacentRope => rope::provider(descriptor, kernels),
            Self::StructuredAttention => attention::provider(descriptor, kernels),
            Self::HalfSplitRope => half_split_rope::provider(descriptor, kernels),
            Self::StructuredGqaAttention => gqa::provider(descriptor, kernels),
            Self::RowGather => transport::provider(transport::Kind::RowGather, descriptor, kernels),
            Self::BatchedRowConcat => {
                transport::provider(transport::Kind::BatchedRowConcat, descriptor, kernels)
            }
            _ => unreachable!("operation kind was exhaustively classified"),
        }
    }
}

pub(super) fn selected_descriptors(
    device: &DeviceDescriptor,
    selected: &[CudaTensorOperation],
) -> Result<Vec<(CudaTensorOperation, OperationProviderDescriptor)>, CudaDeviceRuntimeError> {
    if selected.is_empty() || selected.len() > CudaTensorOperation::ALL.len() {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA tensor operation selection must be nonempty and bounded by the module",
        ));
    }
    for (index, operation) in selected.iter().enumerate() {
        if selected[..index].contains(operation) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor operation selection contains a duplicate",
            ));
        }
    }
    selected
        .iter()
        .copied()
        .map(|operation| operation.descriptor(device).map(|d| (operation, d)))
        .collect()
}

pub(super) fn providers(
    runtime: &CudaDeviceRuntime,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    let descriptors = selected_descriptors(runtime.descriptor(), selected)?;
    let kernels = RetainedFunctions::load(runtime)?;
    Ok(descriptors
        .into_iter()
        .map(|(operation, descriptor)| operation.provider(descriptor, &kernels))
        .collect())
}

#[cfg(test)]
mod tests;
