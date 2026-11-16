//! Bounded BF16 bit transport through admitted contiguous storage views.
//! No source data is interpreted as floating point and no payload is staged
//! on the host. Gather metadata is a bounded kernel argument, not an allocation.

use ferrum_runtime::{
    batched_row_concat_bf16_contract, row_gather_bf16_contract, BatchedRowConcatBf16Geometry,
    LeadingBatchBf16Geometry, LeadingBatchBf16Kind, PhysicalWeightLayout, RowGatherBf16Geometry,
    WeightFormatId, BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID, BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
    ROW_GATHER_BF16_CAPABILITY_ID, ROW_GATHER_BF16_OPERATION_ID,
};

use super::*;
use ferrum_runtime::{
    row_index_gather_or_zero_bf16_contract, RowIndexGatherBf16Geometry,
    ROW_INDEX_GATHER_OR_ZERO_BF16_CAPABILITY_ID, ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID,
};
use ferrum_runtime::{
    row_write_bf16_contract, RowWriteBf16Geometry, ROW_WRITE_BF16_CAPABILITY_ID,
    ROW_WRITE_BF16_OPERATION_ID,
};

pub const ROW_GATHER_PROFILE_ID: &str = "cuda.bitcopy.row_gather.bf16.v1";
pub const BATCHED_ROW_CONCAT_PROFILE_ID: &str = "cuda.bitcopy.batched_row_concat.bf16.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "BF16-bit-transport-v1;uint16-load-store;all-bits-preserved-including-NaN-payload-and-signed-zero;",
    "rank2-row-gather-concat;output-no-alias;gather-max32-runs;row-stride=0-or-1;",
    "row-write=initialized-base-exact-output-alias;disjoint-update-source;untouched-bits-preserved;",
    "rank3-indexed-gather-I32-shared-map;invalid-indices=positive-zero;valid-source-bits-preserved;",
    "indexed-gather=aligned-eight-BF16-row-vectors-else-scalar;bounds-before-source-address;",
    "leading-batch-rank2-rank3-copy=preserve-axis-order-and-all-bits;",
    "source-row=start+batch*batch-stride+local-row*row-stride;",
    "column-subrange-explicit;concat=left-then-right-within-each-batch;",
    "kernel-parameters=gather1088-bytes-including-pointers-and-count;",
    "zero-global-scratch;zero-dynamic-shared;logical-arithmetic=0;",
    "target-execution-evidence-required"
);
const MAX_RUNS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    RowGather,
    RowWrite,
    BatchedRowConcat,
    RowIndexGather,
    MergeLeadingBatch,
    SplitLeadingBatch,
    MergeLeadingBatchView,
    SplitLeadingBatchView,
}

impl Kind {
    pub(super) const fn index(self) -> usize {
        match self {
            Self::RowGather => 0,
            Self::RowWrite => 4,
            Self::BatchedRowConcat => 1,
            Self::RowIndexGather => 2,
            Self::MergeLeadingBatch
            | Self::SplitLeadingBatch
            | Self::MergeLeadingBatchView
            | Self::SplitLeadingBatchView => 3,
        }
    }
    pub(super) const fn function(self) -> &'static str {
        match self {
            Self::RowGather => "ferrum_vnext_row_gather_bf16",
            Self::RowWrite => "ferrum_vnext_row_write_bf16",
            Self::BatchedRowConcat => "ferrum_vnext_batched_row_concat_bf16",
            Self::RowIndexGather => "ferrum_vnext_row_index_gather_or_zero_bf16",
            Self::MergeLeadingBatch
            | Self::SplitLeadingBatch
            | Self::MergeLeadingBatchView
            | Self::SplitLeadingBatchView => "ferrum_vnext_transport_copy_bf16",
        }
    }
    fn is_view(self) -> bool {
        matches!(
            self,
            Self::MergeLeadingBatchView | Self::SplitLeadingBatchView
        )
    }
    fn operation(self) -> &'static str {
        match self {
            Self::RowGather => ROW_GATHER_BF16_OPERATION_ID,
            Self::RowWrite => ROW_WRITE_BF16_OPERATION_ID,
            Self::BatchedRowConcat => BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
            Self::RowIndexGather => ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID,
            Self::MergeLeadingBatch => LeadingBatchBf16Kind::Merge.operation_id(),
            Self::MergeLeadingBatchView => LeadingBatchBf16Kind::Merge.view_operation_id(),
            Self::SplitLeadingBatch => LeadingBatchBf16Kind::Split.operation_id(),
            Self::SplitLeadingBatchView => LeadingBatchBf16Kind::Split.view_operation_id(),
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::RowGather => ROW_GATHER_BF16_CAPABILITY_ID,
            Self::RowWrite => ROW_WRITE_BF16_CAPABILITY_ID,
            Self::BatchedRowConcat => BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID,
            Self::RowIndexGather => ROW_INDEX_GATHER_OR_ZERO_BF16_CAPABILITY_ID,
            Self::MergeLeadingBatch => LeadingBatchBf16Kind::Merge.capability_id(),
            Self::MergeLeadingBatchView => LeadingBatchBf16Kind::Merge.view_capability_id(),
            Self::SplitLeadingBatch => LeadingBatchBf16Kind::Split.capability_id(),
            Self::SplitLeadingBatchView => LeadingBatchBf16Kind::Split.view_capability_id(),
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::RowGather => row_gather_bf16_contract(),
            Self::RowWrite => row_write_bf16_contract(),
            Self::BatchedRowConcat => batched_row_concat_bf16_contract(),
            Self::RowIndexGather => row_index_gather_or_zero_bf16_contract(),
            Self::MergeLeadingBatch => LeadingBatchBf16Kind::Merge.contract(),
            Self::MergeLeadingBatchView => LeadingBatchBf16Kind::Merge.view_contract(),
            Self::SplitLeadingBatch => LeadingBatchBf16Kind::Split.contract(),
            Self::SplitLeadingBatchView => LeadingBatchBf16Kind::Split.view_contract(),
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::RowGather => ROW_GATHER_PROFILE_ID,
            Self::RowWrite => "cuda.bitcopy.row_write.bf16.v1",
            Self::BatchedRowConcat => BATCHED_ROW_CONCAT_PROFILE_ID,
            Self::RowIndexGather => "cuda.bitcopy.row_index_gather_or_zero.bf16.i32.total.v1",
            Self::MergeLeadingBatch => "cuda.bitcopy.merge_leading_batch.bf16.v1",
            Self::MergeLeadingBatchView => "cuda.bitcopy.view.merge_leading_batch.bf16.v1",
            Self::SplitLeadingBatch => "cuda.bitcopy.split_leading_batch.bf16.v1",
            Self::SplitLeadingBatchView => "cuda.bitcopy.view.split_leading_batch.bf16.v1",
        }
    }
    fn provider(self) -> &'static str {
        match self {
            Self::RowGather => "provider.cuda.row_gather.bf16.bitcopy",
            Self::RowWrite => "provider.cuda.row_write.bf16.bitcopy",
            Self::BatchedRowConcat => "provider.cuda.batched_row_concat.bf16.bitcopy",
            Self::RowIndexGather => "provider.cuda.row_index_gather_or_zero.bf16.bitcopy",
            Self::MergeLeadingBatch => "provider.cuda.merge_leading_batch.bf16.bitcopy",
            Self::MergeLeadingBatchView => "provider.cuda.view.merge_leading_batch.bf16.bitcopy",
            Self::SplitLeadingBatch => "provider.cuda.split_leading_batch.bf16.bitcopy",
            Self::SplitLeadingBatchView => "provider.cuda.view.split_leading_batch.bf16.bitcopy",
        }
    }
    fn estimator(self) -> &'static str {
        match self {
            Self::RowGather => "resource-estimator.cuda.row_gather.bf16.bitcopy",
            Self::RowWrite => "resource-estimator.cuda.row_write.bf16.bitcopy",
            Self::BatchedRowConcat => "resource-estimator.cuda.batched_row_concat.bf16.bitcopy",
            Self::RowIndexGather => "resource-estimator.cuda.row_index_gather_or_zero.bf16.bitcopy",
            Self::MergeLeadingBatch => "resource-estimator.cuda.merge_leading_batch.bf16.bitcopy",
            Self::MergeLeadingBatchView => {
                "resource-estimator.cuda.view.merge_leading_batch.bf16.bitcopy"
            }
            Self::SplitLeadingBatch => "resource-estimator.cuda.split_leading_batch.bf16.bitcopy",
            Self::SplitLeadingBatchView => {
                "resource-estimator.cuda.view.split_leading_batch.bf16.bitcopy"
            }
        }
    }
    fn inputs(self) -> u32 {
        match self {
            Self::RowGather
            | Self::MergeLeadingBatch
            | Self::SplitLeadingBatch
            | Self::MergeLeadingBatchView
            | Self::SplitLeadingBatchView => 1,
            Self::BatchedRowConcat | Self::RowIndexGather | Self::RowWrite => 2,
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
            "CUDA runtime lacks selected BF16 transport capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        if kind.is_view() {
            b"read-only-contiguous-full-extent-view-v1;unchanged-bits;zero-device-commands"
        } else {
            NUMERICAL_IDENTITY.as_bytes()
        },
        kind.profile().as_bytes(),
        kind.provider().as_bytes(),
    ]);
    let estimator =
        implementation_fingerprint(&[kind.estimator().as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(kind.provider()).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(kind.inputs()),
        kind.estimator(),
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)
    .map(|d| d.with_fixed_tensor_work(true))
}

pub(super) fn provider(
    kind: Kind,
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(TransportProvider {
        kind,
        descriptor,
        kernels: Arc::clone(kernels),
    })
}

struct TransportProvider {
    kind: Kind,
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
}

impl OperationResourceEstimator for TransportProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id != *self.descriptor.operation_id()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(
                "CUDA transport estimator received another operation",
            ));
        }
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            ALIGNMENT,
            None,
            None,
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for TransportProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "cuda.transport.bf16.encode",
            "CUDA BF16 transport supports fixed tensors only".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        if self.kind.is_view() {
            return EncodedDeviceOperation::read_only_view(invocation.participants()[0].bindings())
                .map(Some)
                .map_err(|e| {
                    provider_failure(identity, "cuda.transport.view.encode", e.to_string())
                });
        }
        self.encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|message| provider_failure(identity, "cuda.transport.bf16.encode", message))
    }
}

impl TransportProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA transport requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != self.kind.operation()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA transport requires its fixed tensor contract".into());
        }
        let shape = geometry(self.kind, participant.bindings(), participant.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.inputs() as usize + 1);
        for ordinal in 0..self.kind.inputs() {
            regions.push(retained_typed_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
                shape.input_bytes[ordinal as usize],
                if self.kind == Kind::RowIndexGather && ordinal == 1 {
                    ElementType::I32
                } else {
                    ElementType::Bf16
                },
            )?);
        }
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.output_bytes,
        )?;
        for (index, input) in regions.iter().enumerate() {
            if self.kind == Kind::RowWrite && index == 0 {
                if input.device_ptr() != output.device_ptr()
                    || input.length_bytes() != output.length_bytes()
                {
                    return Err("row write output must exactly alias base".into());
                }
                continue;
            }
            nonoverlapping_ranges(
                input.device_ptr(),
                input.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let kernels = Arc::clone(&self.kernels);
        let kind = self.kind;
        CudaDeviceCommand::operation(kind.function(), regions, move |stream, regions| {
            let first = regions[0].device_ptr();
            let output = regions[kind.inputs() as usize].device_ptr();
            let second = regions[1].device_ptr();
            let mut builder = stream.launch_builder(kernels.transport_function(kind));
            builder.arg(&first);
            match &shape.parameters {
                Parameters::RowWrite { offset } => {
                    builder
                        .arg(&second)
                        .arg(&output)
                        .arg(&shape.elements)
                        .arg(offset);
                }
                Parameters::Copy => {
                    builder.arg(&output).arg(&shape.elements);
                }
                Parameters::IndexGather {
                    source_rows,
                    output_rows,
                    columns,
                } => {
                    builder
                        .arg(&second)
                        .arg(&output)
                        .arg(&shape.elements)
                        .arg(source_rows)
                        .arg(output_rows)
                        .arg(columns);
                }
                Parameters::Gather(runs) => {
                    builder.arg(&output).arg(&shape.elements).arg(runs);
                }
                Parameters::Concat {
                    left_rows,
                    right_rows,
                    columns,
                } => {
                    builder
                        .arg(&second)
                        .arg(&output)
                        .arg(&shape.elements)
                        .arg(left_rows)
                        .arg(right_rows)
                        .arg(columns);
                }
            }
            let blocks = match shape.parameters {
                Parameters::IndexGather { columns, .. } if columns % 8 == 0 => {
                    shape.elements / columns
                }
                _ => shape.elements.div_ceil(u64::from(THREADS)),
            };
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (blocks.min(MAX_BLOCKS) as u32, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("BF16 transport launch", e))
        })
        .and_then(|command| command.with_tensor_work(shape.elements, 0, 1, 0))
        .map_err(|e| e.to_string())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NativeGather {
    run_ends: [u64; MAX_RUNS],
    source_starts: [u64; MAX_RUNS],
    source_batch_strides: [u64; MAX_RUNS],
    source_row_strides: [u64; MAX_RUNS],
    rows_per_batch: u64,
    source_width: u64,
    column_start: u64,
    column_count: u64,
    run_count: u32,
    padding: u32,
}

// SAFETY: repr(C), fixed-width integer fields, and explicit initialized padding match
// FerrumRowGatherRuns. No pointers or host-owned references cross this ABI.
unsafe impl cudarc::driver::DeviceRepr for NativeGather {}
const _: [(); 1064] = [(); std::mem::size_of::<NativeGather>()];

impl NativeGather {
    fn new(g: &RowGatherBf16Geometry) -> Result<Self, String> {
        let count = g.run_lengths().len();
        if count == 0 || count > MAX_RUNS {
            return Err("CUDA row gather requires 1..=32 runs".into());
        }
        let mut result = Self {
            run_ends: [0; MAX_RUNS],
            source_starts: [0; MAX_RUNS],
            source_batch_strides: [0; MAX_RUNS],
            source_row_strides: [0; MAX_RUNS],
            rows_per_batch: g.rows_per_batch(),
            source_width: g.source_width(),
            column_start: g.column_start(),
            column_count: g.column_count(),
            run_count: count as u32,
            padding: 0,
        };
        let mut end = 0_u64;
        for i in 0..count {
            end = end
                .checked_add(g.run_lengths()[i])
                .ok_or("CUDA gather run sum overflows")?;
            result.run_ends[i] = end;
            result.source_starts[i] = g.source_starts()[i];
            result.source_batch_strides[i] = g.source_batch_strides()[i];
            result.source_row_strides[i] = g.source_row_strides()[i];
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug)]
enum Parameters {
    RowWrite {
        offset: u64,
    },
    Copy,
    IndexGather {
        source_rows: u64,
        output_rows: u64,
        columns: u64,
    },
    Gather(NativeGather),
    Concat {
        left_rows: u64,
        right_rows: u64,
        columns: u64,
    },
}
#[derive(Clone, Copy, Debug)]
struct Shape {
    input_bytes: [u64; 2],
    output_bytes: u64,
    elements: u64,
    parameters: Parameters,
}

fn geometry(
    kind: Kind,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if values.len() != kind.inputs() as usize + 1 {
        return Err("CUDA transport has an incorrect port count".into());
    }
    let source = binding(values, ResolvedValueRole::Input, 0)?;
    let output = binding(values, ResolvedValueRole::Output, 0)?;
    let mut input_bytes = [0; 2];
    for ordinal in 0..kind.inputs() {
        let value = binding(values, ResolvedValueRole::Input, ordinal)?;
        if value
            .weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("CUDA transport requires unquantized dense weight storage".into());
        }
        input_bytes[ordinal as usize] = bounded_bytes(value.tensor())?;
    }
    let output_bytes = bounded_bytes(output.tensor())?;
    let (elements, parameters) = match kind {
        Kind::MergeLeadingBatch
        | Kind::SplitLeadingBatch
        | Kind::MergeLeadingBatchView
        | Kind::SplitLeadingBatchView => {
            if !attrs.is_empty() {
                return Err("leading batch copy has no attributes".into());
            }
            let direction = if matches!(kind, Kind::MergeLeadingBatch | Kind::MergeLeadingBatchView)
            {
                LeadingBatchBf16Kind::Merge
            } else {
                LeadingBatchBf16Kind::Split
            };
            let g = LeadingBatchBf16Geometry::new(direction, source.tensor(), output.tensor())
                .map_err(|e| e.to_string())?;
            (g.elements(), Parameters::Copy)
        }
        Kind::RowIndexGather => {
            if !attrs.is_empty() {
                return Err("indexed gather has no attributes".into());
            }
            let g = RowIndexGatherBf16Geometry::new(
                source.tensor(),
                binding(values, ResolvedValueRole::Input, 1)?.tensor(),
                output.tensor(),
            )
            .map_err(|e| e.to_string())?;
            (
                g.element_count(),
                Parameters::IndexGather {
                    source_rows: g.source_rows(),
                    output_rows: g.output_rows(),
                    columns: g.columns(),
                },
            )
        }
        Kind::RowWrite => {
            row_write_bf16_contract()
                .map_err(|e| e.to_string())?
                .descriptor()
                .validate_attributes(attrs)
                .map_err(|e| e.to_string())?;
            let Some(SemanticValue::Unsigned(start)) = attrs
                .iter()
                .find(|(k, _)| k.as_str() == "row_start")
                .map(|(_, v)| v)
            else {
                return Err("row write requires row_start".into());
            };
            let g = RowWriteBf16Geometry::new(
                source.tensor(),
                binding(values, ResolvedValueRole::Input, 1)?.tensor(),
                output.tensor(),
                *start,
            )
            .map_err(|e| e.to_string())?;
            (
                g.update_elements(),
                Parameters::RowWrite {
                    offset: g.offset_elements(),
                },
            )
        }
        Kind::RowGather => {
            let g = RowGatherBf16Geometry::new(source.tensor(), output.tensor(), attrs)
                .map_err(|e| e.to_string())?;
            (
                g.element_count(),
                Parameters::Gather(NativeGather::new(&g)?),
            )
        }
        Kind::BatchedRowConcat => {
            if attrs.len() != 1 {
                return Err("CUDA batched concat requires only batch".into());
            }
            let Some(SemanticValue::Unsigned(batch)) = attrs
                .iter()
                .find(|(key, _)| key.as_str() == "batch")
                .map(|(_, value)| value)
            else {
                return Err("CUDA batched concat requires unsigned batch".into());
            };
            let second = binding(values, ResolvedValueRole::Input, 1)?;
            let g = BatchedRowConcatBf16Geometry::new(
                source.tensor(),
                second.tensor(),
                output.tensor(),
                *batch,
            )
            .map_err(|e| e.to_string())?;
            (
                g.element_count(),
                Parameters::Concat {
                    left_rows: g.left_rows_per_batch(),
                    right_rows: g.right_rows_per_batch(),
                    columns: g.columns(),
                },
            )
        }
    };
    Ok(Shape {
        input_bytes,
        output_bytes,
        elements,
        parameters,
    })
}

fn bounded_bytes(tensor: &ferrum_runtime::ResolvedTensorSpec) -> Result<u64, String> {
    let bytes = tensor.minimum_storage_bytes().map_err(|e| e.to_string())?;
    if bytes > isize::MAX as u64 {
        return Err("CUDA transport extent exceeds host address space".into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;
