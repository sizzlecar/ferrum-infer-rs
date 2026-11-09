//! Managed single-round RowRMS and addcmul, with independent numerical IDs.
use super::*;
use ferrum_runtime::{
    addcmul_bf16_contract, row_rms_norm_single_round_bf16_contract, AddcmulBf16Geometry,
    PhysicalWeightLayout, RowBroadcast, RowRmsNormSingleRoundBf16Geometry, WeightFormatId,
    ADDCMUL_BF16_CAPABILITY_ID, ROW_RMS_NORM_SINGLE_ROUND_BF16_CAPABILITY_ID,
};
pub const ROW_RMS_PROFILE_ID: &str = "cuda.ieee.row_rms_norm_single_round.bf16.warp32_precise.v1";
pub const ADDCMUL_PROFILE_ID: &str = "cuda.ieee.addcmul.bf16.separate_f32_single_round.v1";
pub(super) const FUNCTIONS: [&str; 2] = [
    "ferrum_vnext_row_rms_norm_single_round_bf16",
    "ferrum_vnext_addcmul_bf16",
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    RowRms,
    Addcmul,
}
impl Kind {
    fn index(self) -> usize {
        if self == Self::RowRms {
            0
        } else {
            1
        }
    }
    fn inputs(self) -> u32 {
        if self == Self::RowRms {
            2
        } else {
            3
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::RowRms => row_rms_norm_single_round_bf16_contract(),
            Self::Addcmul => addcmul_bf16_contract(),
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::RowRms => ROW_RMS_NORM_SINGLE_ROUND_BF16_CAPABILITY_ID,
            Self::Addcmul => ADDCMUL_BF16_CAPABILITY_ID,
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::RowRms => ROW_RMS_PROFILE_ID,
            Self::Addcmul => ADDCMUL_PROFILE_ID,
        }
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: Kind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(kind.capability()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "runtime lacks requested single-round operation",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let name = if kind == Kind::RowRms {
        "row_rms_norm_single_round"
    } else {
        "addcmul"
    };
    let estimator = format!("resource-estimator.cuda.{name}.bf16");
    let fingerprint=implementation_fingerprint(&[kind.profile().as_bytes(),numerical_fingerprint().as_bytes(),include_bytes!("row_rms_addcmul.rs"),include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),"/kernels/vnext_row_rms_addcmul.cuh")),b"BF16-final-only;RN32-explicit-no-contraction-no-FTZ;RowRMS-warp32-FMA-squares-fixed-tree-sqrt-reciprocal;NoAlias;zero-global-scratch;not-Torch-binary-parity"]);
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{name}.bf16")).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(kind.inputs()),
        &estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    kind: Kind,
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        kind,
        descriptor,
        kernels: Arc::clone(kernels),
    })
}
struct Provider {
    kind: Kind,
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        r: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if r.operation().id != *self.descriptor.operation_id()
            || r.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan("estimator received another operation"));
        }
        geometry(self.kind, r.values(), r.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            r.input_fingerprint(),
            ALIGNMENT,
            None,
            None,
        ))
    }
}
impl OperationProvider<CudaDeviceRuntime> for Provider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        i: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            i.participants()[0].identity().clone(),
            "cuda.row_rms_addcmul.encode",
            "requires fixed tensor invocation".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        self.encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|s| provider_failure(identity, "cuda.row_rms_addcmul.encode", s))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one tensor participant".into());
        };
        if i.operation().id != *self.descriptor.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires its fixed tensor contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let elements = g.rows() * g.width();
        let mut regions = Vec::with_capacity(self.kind.inputs() as usize + 1);
        for ordinal in 0..self.kind.inputs() {
            let bytes = if ordinal == 1 {
                g.parameter_elements() * 2
            } else {
                elements * 2
            };
            regions.push(retained_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, ordinal)?,
                bytes,
            )?);
        }
        let output = retained_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            elements * 2,
        )?;
        for read in &regions {
            nonoverlapping_ranges(
                read.device_ptr(),
                read.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let kind = self.kind;
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(FUNCTIONS[kind.index()], regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let parameter = regions[1].device_ptr();
            let output = regions[kind.inputs() as usize].device_ptr();
            let width = g.width();
            let mut builder = stream.launch_builder(kernels.row_rms_addcmul_function(kind.index()));
            match g {
                Geometry::RowRms(a) => {
                    let rows = a.rows();
                    let epsilon = a.effective_epsilon();
                    builder
                        .arg(&x)
                        .arg(&parameter)
                        .arg(&output)
                        .arg(&rows)
                        .arg(&width)
                        .arg(&epsilon);
                    unsafe {
                        builder.launch(LaunchConfig {
                            grid_dim: (rows.div_ceil(4).min(MAX_BLOCKS) as u32, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
                Geometry::Addcmul(a) => {
                    let update = regions[2].device_ptr();
                    let broadcast = u32::from(a.mode() == RowBroadcast::Row);
                    builder
                        .arg(&x)
                        .arg(&parameter)
                        .arg(&update)
                        .arg(&output)
                        .arg(&elements)
                        .arg(&width)
                        .arg(&broadcast);
                    unsafe {
                        builder.launch(LaunchConfig {
                            grid_dim: (
                                elements.div_ceil(u64::from(THREADS)).min(MAX_BLOCKS) as u32,
                                1,
                                1,
                            ),
                            block_dim: (THREADS, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("single-round RowRMS/addcmul launch", e))
        })
        .and_then(|c| c.with_tensor_work(elements, g.operations().map_err(contract_error)?, 1, 0))
        .map_err(|e| e.to_string())
    }
}
#[derive(Clone, Copy)]
enum Geometry {
    RowRms(RowRmsNormSingleRoundBf16Geometry),
    Addcmul(AddcmulBf16Geometry),
}
impl Geometry {
    fn rows(self) -> u64 {
        match self {
            Self::RowRms(g) => g.rows(),
            Self::Addcmul(g) => g.rows(),
        }
    }
    fn width(self) -> u64 {
        match self {
            Self::RowRms(g) => g.width(),
            Self::Addcmul(g) => g.width(),
        }
    }
    fn parameter_elements(self) -> u64 {
        match self {
            Self::RowRms(g) => g.width(),
            Self::Addcmul(g) => {
                if g.mode() == RowBroadcast::Row {
                    g.width()
                } else {
                    g.rows() * g.width()
                }
            }
        }
    }
    fn operations(self) -> Result<u64, VNextError> {
        match self {
            Self::RowRms(g) => g.logical_arithmetic_operations(),
            Self::Addcmul(g) => g.logical_arithmetic_operations(),
        }
    }
}
fn geometry(
    kind: Kind,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Geometry, String> {
    if b.len() != kind.inputs() as usize + 1 || attrs.len() != 1 {
        return Err("requires exact operands and one semantic attribute".into());
    }
    for a in b {
        if a.tensor()
            .minimum_storage_bytes()
            .map_err(|e| e.to_string())?
            > isize::MAX as u64
            || a.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("requires addressable dense operands".into());
        }
    }
    let input = |n| binding(b, ResolvedValueRole::Input, n).map(|b| b.tensor());
    let output = binding(b, ResolvedValueRole::Output, 0)?.tensor();
    let get = |name| {
        attrs
            .iter()
            .find(|(k, _)| k.as_str() == name)
            .map(|(_, v)| v)
    };
    match kind {
        Kind::RowRms => {
            let Some(SemanticValue::Rational(eps)) = get("epsilon") else {
                return Err("requires rational epsilon".into());
            };
            Ok(Geometry::RowRms(
                RowRmsNormSingleRoundBf16Geometry::new(input(0)?, input(1)?, output, *eps)
                    .map_err(|e| e.to_string())?,
            ))
        }
        Kind::Addcmul => {
            let Some(SemanticValue::Text(mode)) = get("broadcast") else {
                return Err("requires explicit broadcast".into());
            };
            Ok(Geometry::Addcmul(
                AddcmulBf16Geometry::new(
                    input(0)?,
                    input(1)?,
                    input(2)?,
                    output,
                    RowBroadcast::parse(mode).ok_or("invalid broadcast")?,
                )
                .map_err(|e| e.to_string())?,
            ))
        }
    }
}
