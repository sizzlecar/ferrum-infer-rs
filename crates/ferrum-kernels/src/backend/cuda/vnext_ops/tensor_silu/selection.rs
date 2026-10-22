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
    ChannelL2Norm,
    SiluNchw,
    PlainResidualAddNchw,
    ChannelL2NormImage,
    SpatialAttention,
    SpatialUpsampleGather,
    /// Explicit-state normal generation. Its launch geometry is queried from
    /// the existing runtime only when this operation is selected.
    PhiloxNormal,
}

impl CudaTensorOperation {
    pub const ALL: [Self; 23] = [
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
        Self::ChannelL2Norm,
        Self::SiluNchw,
        Self::PlainResidualAddNchw,
        Self::ChannelL2NormImage,
        Self::SpatialAttention,
        Self::SpatialUpsampleGather,
        Self::PhiloxNormal,
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
            Self::ChannelL2NormImage => {
                spatial::descriptor(device, spatial::SpatialKind::ChannelL2Image)
            }
            Self::SpatialAttention => {
                spatial_extended::descriptor(device, spatial_extended::Kind::Attention)
            }
            Self::SpatialUpsampleGather => {
                spatial_extended::descriptor(device, spatial_extended::Kind::Gather)
            }
            Self::ChannelL2Norm => spatial::descriptor(device, spatial::SpatialKind::ChannelL2),
            Self::SiluNchw => spatial::descriptor(device, spatial::SpatialKind::Silu),
            Self::PlainResidualAddNchw => {
                spatial::descriptor(device, spatial::SpatialKind::ResidualAdd)
            }
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
            Self::ChannelL2NormImage => {
                spatial::provider(spatial::SpatialKind::ChannelL2Image, descriptor, kernels)
            }
            Self::SpatialAttention => {
                spatial_extended::provider(spatial_extended::Kind::Attention, descriptor, kernels)
            }
            Self::SpatialUpsampleGather => {
                spatial_extended::provider(spatial_extended::Kind::Gather, descriptor, kernels)
            }
            Self::ChannelL2Norm => {
                spatial::provider(spatial::SpatialKind::ChannelL2, descriptor, kernels)
            }
            Self::SiluNchw => spatial::provider(spatial::SpatialKind::Silu, descriptor, kernels),
            Self::PlainResidualAddNchw => {
                spatial::provider(spatial::SpatialKind::ResidualAdd, descriptor, kernels)
            }
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

/// A Philox descriptor cannot lose or substitute its fingerprinted geometry
/// between metadata validation and sharing the admitted module owner.
pub(super) enum SelectedDescriptor {
    Fixed(CudaTensorOperation, OperationProviderDescriptor),
    Philox(philox::PreparedProvider),
}

impl SelectedDescriptor {
    #[cfg(test)]
    fn operation(&self) -> CudaTensorOperation {
        match self {
            Self::Fixed(operation, _) => *operation,
            Self::Philox(_) => CudaTensorOperation::PhiloxNormal,
        }
    }

    #[cfg(test)]
    fn descriptor(&self) -> &OperationProviderDescriptor {
        match self {
            Self::Fixed(_, descriptor) => descriptor,
            Self::Philox(prepared) => prepared.descriptor(),
        }
    }

    fn into_provider(
        self,
        kernels: &Arc<RetainedFunctions>,
    ) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
        match self {
            Self::Fixed(operation, descriptor) => operation.provider(descriptor, kernels),
            Self::Philox(prepared) => prepared.into_provider(kernels),
        }
    }
}

fn validate_selection(selected: &[CudaTensorOperation]) -> Result<(), CudaDeviceRuntimeError> {
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
    Ok(())
}

/// Pure metadata preparation. Philox needs explicitly supplied, checked device
/// geometry; absence never falls back to creating a context or querying a GPU.
pub(super) fn selected_descriptors(
    device: &DeviceDescriptor,
    selected: &[CudaTensorOperation],
    philox_geometry: Option<CudaPhiloxDeviceGeometry>,
) -> Result<Vec<SelectedDescriptor>, CudaDeviceRuntimeError> {
    validate_selection(selected)?;
    selected
        .iter()
        .copied()
        .map(|operation| {
            if operation == CudaTensorOperation::PhiloxNormal {
                let geometry = philox_geometry.ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "Philox selection requires explicit device launch geometry",
                    )
                })?;
                philox::PreparedProvider::new(device, geometry).map(SelectedDescriptor::Philox)
            } else {
                operation
                    .descriptor(device)
                    .map(|d| SelectedDescriptor::Fixed(operation, d))
            }
        })
        .collect()
}

pub(super) fn runtime_descriptors(
    runtime: &CudaDeviceRuntime,
    selected: &[CudaTensorOperation],
) -> Result<Vec<SelectedDescriptor>, CudaDeviceRuntimeError> {
    runtime_descriptors_with_query(runtime.descriptor(), selected, || {
        CudaPhiloxDeviceGeometry::query(runtime)
    })
}

fn runtime_descriptors_with_query(
    device: &DeviceDescriptor,
    selected: &[CudaTensorOperation],
    query: impl FnOnce() -> Result<CudaPhiloxDeviceGeometry, CudaDeviceRuntimeError>,
) -> Result<Vec<SelectedDescriptor>, CudaDeviceRuntimeError> {
    validate_selection(selected)?;
    let geometry = if selected.contains(&CudaTensorOperation::PhiloxNormal) {
        Some(query()?)
    } else {
        None
    };
    selected_descriptors(device, selected, geometry)
}

pub(super) fn providers(
    runtime: &CudaDeviceRuntime,
    selected: &[CudaTensorOperation],
) -> Result<Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>>, CudaDeviceRuntimeError> {
    let descriptors = runtime_descriptors(runtime, selected)?;
    let kernels = RetainedFunctions::load(runtime)?;
    Ok(providers_from_descriptors(descriptors, &kernels))
}

pub(super) fn providers_from_descriptors(
    descriptors: Vec<SelectedDescriptor>,
    kernels: &Arc<RetainedFunctions>,
) -> Vec<Box<dyn OperationProvider<CudaDeviceRuntime>>> {
    descriptors
        .into_iter()
        .map(|descriptor| descriptor.into_provider(kernels))
        .collect()
}

#[cfg(test)]
mod tests;
