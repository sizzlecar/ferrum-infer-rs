//! Independent oracle for the explicit half-operands, F32-logits head.

use std::collections::BTreeSet;

use ferrum_interfaces::vnext::{
    last_token_dense_linear_f32_f16_operands_contract, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceBatchingForm, DeviceRuntime, ElementType, EncodedDeviceOperation,
    OperationContract, OperationFailure, OperationInvocation, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, PhysicalStorageLayout, PhysicalWeightLayout, PhysicalWeightPadding,
    ProfilePhase, ProviderExecutionSemantics, ProviderId, QuantizationFormatId,
    ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, VNextError, WeightEncoding, WeightFormatId,
    LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_CAPABILITY_ID,
    LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID,
};
use half::f16;

use crate::gguf_blocks::GgufBlockFormat;

use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region_range, contiguous_token_region, dimension,
    implementation_fingerprint,
};
use super::runtime::{
    ReferenceBufferRegion, ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError,
};

const PROVIDER_ID: &str = "provider.reference.last_token_dense_linear.f32.f16-operands.q6-k";
const ESTIMATOR_ID: &str =
    "resource-estimator.reference.last_token_dense_linear.f32.f16-operands.q6-k";

pub(super) struct ReferenceLastTokenF16OperandsProvider {
    descriptor: OperationProviderDescriptor,
}

impl ReferenceLastTokenF16OperandsProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract =
            last_token_dense_linear_f32_f16_operands_contract().map_err(contract_error)?;
        let capability = CapabilityId::new(LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_CAPABILITY_ID)
            .map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime lacks the half-operands head capability",
            ));
        }
        let fingerprint = implementation_fingerprint(&[
            include_str!("last_token_f16_operands.rs").as_bytes(),
            include_str!("../../gguf_blocks/block_decode.rs").as_bytes(),
            include_str!("../../gguf_blocks/mod.rs").as_bytes(),
            PROVIDER_ID.as_bytes(),
        ]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            fingerprint.clone(),
            ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new("weight-format.gguf.native-block").map_err(contract_error)?
            ]),
            BTreeSet::from([QuantizationFormatId::new(GgufBlockFormat::Q6K.format_id())
                .map_err(contract_error)?]),
            contiguous_bindings(2),
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            implementation_fingerprint(&[ESTIMATOR_ID.as_bytes(), fingerprint.as_bytes()]),
        )
        .map_err(contract_error)?;
        Ok(Self { descriptor })
    }
}

impl OperationResourceEstimator for ReferenceLastTokenF16OperandsProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "reference half-operands head estimator received another operation".into(),
            });
        }
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            None,
        ))
    }
}

impl OperationProvider<ReferenceDeviceRuntime> for ReferenceLastTokenF16OperandsProvider {
    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| {
                OperationFailure::new(
                    identity,
                    ProfilePhase::Forward,
                    "reference.last_token_f16_operands.encode",
                    message.chars().take(2048).collect::<String>(),
                    false,
                )
                .expect("bounded reference provider failure")
            })
    }
}

fn encode(
    invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    if invocation.operation().id.as_str() != LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID {
        return Err("reference half-operands head received another operation".into());
    }
    let first = invocation
        .participants()
        .first()
        .ok_or("reference head has no participants")?;
    let hidden = dimension(first.attributes(), "hidden_size")?;
    let vocabulary = dimension(first.attributes(), "out_features")?;
    let weight = resolve_q6_weight(first, hidden, vocabulary)?;
    let input_packed = invocation
        .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
        .map_err(|error| error.to_string())?;
    let ranges = invocation.participant_token_ranges();
    if ranges.len() != invocation.participants().len() {
        return Err("reference head participant ranges are incomplete".into());
    }
    let mut launches = Vec::with_capacity(ranges.len());
    for (participant, range) in invocation.participants().iter().zip(ranges) {
        if dimension(participant.attributes(), "hidden_size")? != hidden
            || dimension(participant.attributes(), "out_features")? != vocabulary
            || !weight.same_physical_region(&resolve_q6_weight(participant, hidden, vocabulary)?)
        {
            return Err("reference head participant attributes or shared weight disagree".into());
        }
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if input.tensor().dimensions().len() != 2
            || input.tensor().dimensions()[0] == 0
            || input.tensor().dimensions()[1] != hidden as u64
            || output.tensor().dimensions() != [1, vocabulary as u64]
            || [input, output].iter().any(|value| {
                value.tensor().element_type() != ElementType::F32
                    || !matches!(value.tensor().layout(), ResolvedTensorLayout::Contiguous)
            })
        {
            return Err("reference head activation/logit signature differs".into());
        }
        let selected = if input_packed {
            range.immediate_token_range()
        } else {
            range.source_token_range()
        };
        if selected.is_empty() {
            return Err("reference head cannot select an empty token span".into());
        }
        let input =
            contiguous_token_region(participant, input, ElementType::F32, selected.end - 1, 1)?;
        let [stored] = output.storage().components() else {
            return Err("reference head output needs one contiguous component".into());
        };
        let output = contiguous_region_range(
            participant,
            output,
            ElementType::F32,
            stored.offset_bytes(),
            stored.length_bytes(),
        )?;
        launches.push(ReferenceHalfHeadLaunch::new(
            input,
            weight.clone(),
            output,
            hidden,
            vocabulary,
        )?);
    }
    let participants =
        u32::try_from(launches.len()).map_err(|_| "reference head batch exceeds u32")?;
    ReferenceDeviceCommand::half_operands_head(
        launches,
        if participants == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        participants,
        invocation.work_shape().immediate_tokens(),
    )
    .map_err(|error| error.to_string())
}

fn resolve_q6_weight(
    participant: &OperationInvocation<'_, ReferenceDeviceBuffer>,
    hidden: usize,
    vocabulary: usize,
) -> Result<ReferenceBufferRegion, String> {
    let binding = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
    validate_q6_weight(binding, hidden, vocabulary)?;
    let [stored] = binding.storage().components() else {
        unreachable!()
    };
    contiguous_region_range(
        participant,
        binding,
        ElementType::U8,
        stored.offset_bytes(),
        stored.length_bytes(),
    )
}

fn validate_q6_weight(
    binding: &ResolvedValueBinding,
    hidden: usize,
    vocabulary: usize,
) -> Result<(), String> {
    if hidden == 0
        || vocabulary == 0
        || !hidden.is_multiple_of(256)
        || binding.tensor().dimensions() != [vocabulary as u64, hidden as u64]
        || binding.tensor().element_type() != ElementType::F16
        || !matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
    {
        return Err(
            "reference half-operands head requires a Q6_K matrix with complete K blocks".into(),
        );
    }
    let weight = binding
        .weight()
        .ok_or("reference head lacks typed weight metadata")?;
    let ([component], [stored]) = (weight.components(), binding.storage().components()) else {
        return Err("reference head requires one native weight component".into());
    };
    let PhysicalWeightLayout::BlockQuantized {
        blocks,
        block_axis: 1,
        block_padding: PhysicalWeightPadding::Exact,
    } = weight.physical_layout()
    else {
        return Err("reference head requires an untransformed native Q6_K layout".into());
    };
    let WeightEncoding::BlockQuantized(spec) = component.encoding() else {
        return Err("reference head requires native block encoding".into());
    };
    if GgufBlockFormat::from_spec(spec)? != GgufBlockFormat::Q6K
        || blocks.storage != PhysicalStorageLayout::exact_contiguous()
        || component.physical_dimensions() != [vocabulary as u64, (hidden / 256) as u64]
        || &blocks.component_id != component.component_id()
        || stored.component_id() != Some(component.component_id())
        || stored.element_type() != ElementType::U8
        || component.physical_element_type() != ElementType::U8
        || stored.length_bytes()
            != component
                .physical_bytes()
                .map_err(|error| error.to_string())?
    {
        return Err("reference head Q6_K component metadata disagrees".into());
    }
    Ok(())
}

pub(super) struct ReferenceHalfHeadLaunch {
    pub(super) input: ReferenceBufferRegion,
    pub(super) weight: ReferenceBufferRegion,
    pub(super) output: ReferenceBufferRegion,
    hidden: usize,
    vocabulary: usize,
}

impl ReferenceHalfHeadLaunch {
    pub(super) fn new(
        input: ReferenceBufferRegion,
        weight: ReferenceBufferRegion,
        output: ReferenceBufferRegion,
        hidden: usize,
        vocabulary: usize,
    ) -> Result<Self, String> {
        if hidden == 0
            || vocabulary == 0
            || !hidden.is_multiple_of(256)
            || hidden.checked_mul(4) != Some(input.length_bytes())
            || vocabulary.checked_mul(4) != Some(output.length_bytes())
            || vocabulary
                .checked_mul(hidden / 256)
                .and_then(|n| n.checked_mul(210))
                != Some(weight.length_bytes())
            || input.element_type() != ElementType::F32
            || output.element_type() != ElementType::F32
            || weight.element_type() != ElementType::U8
        {
            return Err("reference half-operands head ranges differ from native Q6_K work".into());
        }
        Ok(Self {
            input,
            weight,
            output,
            hidden,
            vocabulary,
        })
    }

    pub(super) fn execute(&self) {
        let input = self.input.read();
        let weight = self.weight.read();
        let mut output = vec![0_u8; self.output.length_bytes()];
        let mut decoded = [0.0_f32; 256];
        let blocks_per_row = self.hidden / 256;
        for (row, destination) in output.chunks_exact_mut(4).enumerate().take(self.vocabulary) {
            let mut sum = 0.0_f32;
            for block in 0..blocks_per_row {
                let offset = (row * blocks_per_row + block) * 210;
                GgufBlockFormat::Q6K.decode_block(&weight[offset..offset + 210], &mut decoded);
                for (index, &coefficient) in decoded.iter().enumerate() {
                    let offset = (block * 256 + index) * 4;
                    let activation =
                        f32::from_le_bytes(input[offset..offset + 4].try_into().unwrap());
                    sum += f16::from_f32(activation).to_f32() * f16::from_f32(coefficient).to_f32();
                }
            }
            destination.copy_from_slice(&sum.to_le_bytes());
        }
        self.output.write(&output);
    }
}

fn contract_error(error: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{
        AliasPolicy, BlockQuantizationSpec, BufferUsage, PhysicalWeightComponentBinding,
        ResolvedStorageComponent, ResolvedTensorSpec, ResolvedValueStorage, ResolvedWeightBinding,
        TensorAccess, WeightComponentRole, WeightComponentSpec, WeightId, WeightSchema,
        WeightTensorSpec,
    };

    fn weight_binding(
        format: GgufBlockFormat,
        hidden: usize,
        vocabulary: usize,
        block_axis: u32,
        storage: PhysicalStorageLayout,
    ) -> ResolvedValueBinding {
        let component_id = WeightId::new("weight.reference.half-head.component").unwrap();
        let weight_id = WeightId::new("weight.reference.half-head").unwrap();
        let mut physical_dimensions = vec![vocabulary as u64, hidden as u64];
        physical_dimensions[block_axis as usize] /= format.block_values() as u64;
        let physical_bytes =
            physical_dimensions.iter().product::<u64>() * format.block_bytes() as u64;
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.gguf.native-block").unwrap(),
            layout_id: "weight-layout.reference.half-head"
                .to_owned()
                .try_into()
                .unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: component_id.clone(),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["output.weight".into()],
                dimensions: physical_dimensions,
                encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: QuantizationFormatId::new(format.format_id()).unwrap(),
                    logical_values_per_block: format.block_values() as u32,
                    bytes_per_block: format.block_bytes() as u32,
                }),
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: weight_id.clone(),
                dimensions: vec![vocabulary as u64, hidden as u64],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding {
                        component_id: component_id.clone(),
                        storage,
                    },
                    block_axis,
                    block_padding: PhysicalWeightPadding::Exact,
                },
                required: true,
            }],
        };
        ResolvedValueBinding::new(
            "value.reference.half-head.weight"
                .to_owned()
                .try_into()
                .unwrap(),
            ResolvedValueRole::Input,
            1,
            ResolvedTensorSpec::new(
                vec![vocabulary as u64, hidden as u64],
                ElementType::F16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            TensorAccess::Read,
            AliasPolicy::NoAlias,
            BufferUsage::Weights,
            Some(ResolvedWeightBinding::from_schema(&schema, &weight_id).unwrap()),
            ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
                Some(component_id),
                "resource.reference.half-head.weight"
                    .to_owned()
                    .try_into()
                    .unwrap(),
                0,
                physical_bytes,
                ElementType::U8,
            )
            .unwrap()])
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn half_operands_head_reference_accepts_only_its_declared_q6_row_abi() {
        for (hidden, vocabulary) in [(256, 1), (512, 17), (768, 65)] {
            let binding = weight_binding(
                GgufBlockFormat::Q6K,
                hidden,
                vocabulary,
                1,
                PhysicalStorageLayout::exact_contiguous(),
            );
            validate_q6_weight(&binding, hidden, vocabulary).unwrap();
            assert!(validate_q6_weight(&binding, hidden + 1, vocabulary).is_err());
            assert!(validate_q6_weight(&binding, hidden, vocabulary + 1).is_err());
        }
        let q4 = weight_binding(
            GgufBlockFormat::Q4K,
            256,
            7,
            1,
            PhysicalStorageLayout::exact_contiguous(),
        );
        assert!(validate_q6_weight(&q4, 256, 7).is_err());
        let column_blocks = weight_binding(
            GgufBlockFormat::Q6K,
            256,
            512,
            0,
            PhysicalStorageLayout::exact_contiguous(),
        );
        assert!(validate_q6_weight(&column_blocks, 256, 512).is_err());
        let transposed_blocks = weight_binding(
            GgufBlockFormat::Q6K,
            512,
            2,
            1,
            PhysicalStorageLayout::Strided {
                strides_in_elements: vec![1, 2],
                padding: PhysicalWeightPadding::Exact,
            },
        );
        assert!(validate_q6_weight(&transposed_blocks, 512, 2).is_err());
    }
}
