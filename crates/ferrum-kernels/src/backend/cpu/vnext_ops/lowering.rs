use ferrum_interfaces::vnext::{
    BatchedOperationInvocation, DeviceBatchingForm, ElementType, EncodedDeviceOperation,
    NodeWorkContract, ResolvedValueRole, SemanticValue,
};

use super::super::vnext_runtime::{
    CpuBufferRegion, CpuDeviceBuffer, CpuDeviceCommand, CpuKernelLaunch, CpuRuntimeError,
};
use super::bindings::{binding, scratch_region, token_region, value_region};
use super::launch::{CpuAction, CpuOperatorLaunch, ResidualStorage};
use super::provider::{unsigned, CpuOperation};
use super::scalar::CpuFloat;
use super::weights::{matrix_parts, CpuMatrixPart};

impl CpuFloat {
    pub(super) const fn element_type(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32 => ElementType::F32,
        }
    }
}

fn size(value: u64) -> Result<usize, CpuRuntimeError> {
    usize::try_from(value)
        .map_err(|_| CpuRuntimeError::new("CPU operation dimension exceeds addressable memory"))
}

fn push(regions: &mut Vec<CpuBufferRegion>, region: CpuBufferRegion) -> usize {
    let index = regions.len();
    regions.push(region);
    index
}

pub(super) fn encode(
    operation: CpuOperation,
    invocation: &BatchedOperationInvocation<'_, CpuDeviceBuffer>,
) -> Result<EncodedDeviceOperation<CpuDeviceCommand>, CpuRuntimeError> {
    let first = &invocation.participants()[0];
    let mut launches: Vec<Box<dyn CpuKernelLaunch>> = Vec::new();
    let mut bindings: Vec<Box<dyn CpuKernelLaunch>> = Vec::new();
    let mut workspace_offset = 0_u64;
    for (index, participant) in invocation.participants().iter().enumerate() {
        if participant.attributes() != first.attributes() {
            return Err(CpuRuntimeError::new(
                "CPU batched operation attributes disagree",
            ));
        }
        let mut regions = Vec::new();
        let mut actions = Vec::new();
        let rows = if matches!(participant.work(), NodeWorkContract::Fixed) {
            size(
                binding(participant, ResolvedValueRole::Input, 0)?
                    .tensor()
                    .dimensions()
                    .first()
                    .copied()
                    .unwrap_or(0),
            )?
        } else {
            size(
                invocation
                    .participant_token_ranges()
                    .get(index)
                    .ok_or_else(|| {
                        CpuRuntimeError::new("CPU batch lacks participant token ranges")
                    })?
                    .immediate_tokens(),
            )?
        };
        let attribute =
            |name| unsigned(participant.attributes(), name).map_err(CpuRuntimeError::from);
        let token = |role, ordinal, dtype: CpuFloat, last| {
            token_region(invocation, index, role, ordinal, dtype.element_type(), last)
        };
        let value = |role, ordinal, dtype| {
            value_region(
                participant,
                binding(participant, role, ordinal)?,
                dtype,
                None,
            )
        };
        use ResolvedValueRole::{Input, Output};
        match operation {
            CpuOperation::CausalAttention(dtype) => {
                let (launch, preparation) = super::causal_attention_launch::prepare(
                    invocation,
                    index,
                    rows,
                    dtype,
                    workspace_offset,
                    &mut regions,
                )?;
                let bytes = super::causal_attention_launch::workspace_bytes(
                    super::causal_attention::CausalShape::from_attributes(
                        participant.attributes(),
                    )?,
                )?;
                workspace_offset = workspace_offset.checked_add(bytes).ok_or_else(|| {
                    CpuRuntimeError::new("CPU attention batch workspace overflows")
                })?;
                actions.push(CpuAction::CausalAttention(Box::new(launch)));
                bindings.push(Box::new(preparation));
            }
            CpuOperation::GatedDelta(dtype) => {
                let launch = super::gated_delta_launch::prepare(
                    invocation,
                    index,
                    rows,
                    dtype,
                    workspace_offset,
                    &mut regions,
                )?;
                let bytes = super::gated_delta_launch::workspace_bytes(
                    super::gated_delta::GatedDeltaShape::from_attributes(participant.attributes())?,
                )?;
                workspace_offset = workspace_offset
                    .checked_add(bytes)
                    .ok_or_else(|| CpuRuntimeError::new("CPU GDN batch workspace overflows"))?;
                actions.push(CpuAction::GatedDelta(Box::new(launch)));
            }
            CpuOperation::Embedding(dtype) => {
                let vocabulary = attribute("vocab_size")?;
                let hidden = attribute("hidden_size")?;
                let parts = matrix_parts(
                    participant,
                    binding(participant, Input, 1)?,
                    &[vocabulary, hidden],
                    &mut regions,
                )?;
                let [part] = parts.as_slice() else {
                    return Err(CpuRuntimeError::new(
                        "CPU embedding requires a complete physical matrix",
                    ));
                };
                let tokens = push(
                    &mut regions,
                    token_region(invocation, index, Input, 0, ElementType::U32, false)?,
                );
                let output = push(&mut regions, token(Output, 0, dtype, false)?);
                actions.push(CpuAction::Embedding {
                    weight: part.region,
                    tokens,
                    output,
                    format: part.format,
                    output_type: dtype,
                    vocabulary: size(vocabulary)?,
                    hidden: size(hidden)?,
                });
            }
            CpuOperation::RmsNorm {
                input: input_type,
                output: output_type,
            } => {
                let width = size(attribute("hidden_size")?)?;
                let epsilon = match participant
                    .attributes()
                    .iter()
                    .find(|(id, _)| id.as_str() == "epsilon")
                    .map(|(_, value)| value)
                {
                    Some(SemanticValue::Rational(value)) => {
                        (value.numerator() as f64 / value.denominator() as f64) as f32
                    }
                    _ => return Err(CpuRuntimeError::new("CPU RMSNorm lacks rational epsilon")),
                };
                if !epsilon.is_finite() || epsilon <= 0.0 {
                    return Err(CpuRuntimeError::new(
                        "CPU RMSNorm epsilon must be finite and positive",
                    ));
                }
                let input = push(&mut regions, token(Input, 0, input_type, false)?);
                let weight = push(&mut regions, value(Input, 1, ElementType::F16)?);
                let output = push(&mut regions, token(Output, 0, output_type, false)?);
                actions.push(CpuAction::RmsNorm {
                    input,
                    weight,
                    output,
                    input_type,
                    output_type,
                    width,
                    epsilon,
                });
            }
            CpuOperation::Residual(left_type) => {
                let left = push(&mut regions, token(Input, 0, left_type, false)?);
                let right = push(&mut regions, token(Input, 1, CpuFloat::F16, false)?);
                let output = push(&mut regions, token(Output, 0, left_type, false)?);
                let storage = if regions[left].same_physical_region(&regions[output]) {
                    if regions[right].same_physical_region(&regions[output]) {
                        ResidualStorage::BothInPlace
                    } else {
                        ResidualStorage::LeftInPlace
                    }
                } else {
                    ResidualStorage::Separate
                };
                actions.push(CpuAction::Residual {
                    left,
                    right,
                    output,
                    left_type,
                    storage,
                });
            }
            CpuOperation::Linear | CpuOperation::LastTokenLinear(_) => {
                let (dtype, last, width_name) = match operation {
                    CpuOperation::LastTokenLinear(dtype) => (dtype, true, "hidden_size"),
                    _ => (CpuFloat::F16, false, "in_features"),
                };
                let columns = attribute(width_name)?;
                let output_width = attribute("out_features")?;
                let parts = matrix_parts(
                    participant,
                    binding(participant, Input, 1)?,
                    &[output_width, columns],
                    &mut regions,
                )?;
                let input = push(&mut regions, token(Input, 0, dtype, last)?);
                let output = push(
                    &mut regions,
                    if last {
                        value(Output, 0, dtype.element_type())?
                    } else {
                        token(Output, 0, dtype, false)?
                    },
                );
                linear_actions(
                    &mut actions,
                    &parts,
                    input,
                    output,
                    dtype,
                    if last { 1 } else { rows },
                    size(output_width)?,
                );
            }
            CpuOperation::SwiGlu => {
                let hidden = attribute("hidden_size")?;
                let width = attribute("intermediate_size")?;
                let gate_up = matrix_parts(
                    participant,
                    binding(participant, Input, 1)?,
                    &[2, width, hidden],
                    &mut regions,
                )?;
                let down = matrix_parts(
                    participant,
                    binding(participant, Input, 2)?,
                    &[hidden, width],
                    &mut regions,
                )?;
                let input = push(&mut regions, token(Input, 0, CpuFloat::F16, false)?);
                let output = push(&mut regions, token(Output, 0, CpuFloat::F16, false)?);
                let activation_bytes = width
                    .checked_mul(rows as u64)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(|| CpuRuntimeError::new("CPU SwiGLU workspace size overflows"))?;
                let gate_bytes = activation_bytes
                    .checked_mul(2)
                    .ok_or_else(|| CpuRuntimeError::new("CPU gate workspace overflows"))?;
                let activation_offset =
                    workspace_offset.checked_add(gate_bytes).ok_or_else(|| {
                        CpuRuntimeError::new("CPU activation workspace offset overflows")
                    })?;
                let gate = push(
                    &mut regions,
                    scratch_region(first, workspace_offset, gate_bytes)?,
                );
                let activation = push(
                    &mut regions,
                    scratch_region(first, activation_offset, activation_bytes)?,
                );
                workspace_offset = activation_offset
                    .checked_add(activation_bytes)
                    .ok_or_else(|| CpuRuntimeError::new("CPU workspace offset overflows"))?;
                let gate_width = size(
                    width
                        .checked_mul(2)
                        .ok_or_else(|| CpuRuntimeError::new("CPU gate width overflows"))?,
                )?;
                linear_actions(
                    &mut actions,
                    &gate_up,
                    input,
                    gate,
                    CpuFloat::F16,
                    rows,
                    gate_width,
                );
                actions.push(CpuAction::SwiGlu {
                    gate_up: gate,
                    output: activation,
                    width: size(width)?,
                });
                linear_actions(
                    &mut actions,
                    &down,
                    activation,
                    output,
                    CpuFloat::F16,
                    rows,
                    size(hidden)?,
                );
            }
            CpuOperation::Argmax(dtype) => {
                let vocabulary = attribute("vocab_size")?;
                let logits = push(&mut regions, value(Input, 0, dtype.element_type())?);
                let mask = push(&mut regions, value(Input, 1, ElementType::U8)?);
                let repeated_ids = push(&mut regions, value(Input, 2, ElementType::U32)?);
                let repeated_offsets = push(&mut regions, value(Input, 3, ElementType::U32)?);
                let penalty = push(&mut regions, value(Input, 4, ElementType::F32)?);
                let output = push(&mut regions, value(Output, 0, ElementType::U32)?);
                let scratch = push(
                    &mut regions,
                    scratch_region(first, workspace_offset, vocabulary)?,
                );
                workspace_offset = workspace_offset
                    .checked_add(vocabulary)
                    .ok_or_else(|| CpuRuntimeError::new("CPU argmax workspace offset overflows"))?;
                actions.push(CpuAction::Argmax {
                    logits,
                    mask,
                    repeated_ids,
                    repeated_offsets,
                    penalty,
                    scratch,
                    output,
                    dtype,
                });
            }
        }
        launches.push(Box::new(CpuOperatorLaunch { regions, actions }));
    }
    let participants = u32::try_from(invocation.participants().len())
        .map_err(|_| CpuRuntimeError::new("CPU participant count exceeds u32"))?;
    let command = CpuDeviceCommand::compute(
        operation.native_id(),
        launches,
        if participants == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        participants,
        invocation.work_shape().immediate_tokens(),
    )?;
    let operation = EncodedDeviceOperation::compute(command);
    if bindings.is_empty() {
        Ok(operation)
    } else {
        let binding = CpuDeviceCommand::compute(
            "cpu.causal_attention.bind",
            bindings,
            if participants == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants,
            invocation.work_shape().immediate_tokens(),
        )?;
        Ok(invocation.attach_binding_command(operation, binding))
    }
}

fn linear_actions(
    actions: &mut Vec<CpuAction>,
    parts: &[CpuMatrixPart],
    input: usize,
    output: usize,
    dtype: CpuFloat,
    rows: usize,
    output_stride: usize,
) {
    for part in parts {
        actions.push(CpuAction::Linear {
            input,
            weight: part.region,
            output,
            format: part.format,
            input_type: dtype,
            output_type: dtype,
            rows,
            input_width: part.columns,
            output_width: part.rows,
            output_stride,
            output_column_offset: part.output_offset,
        });
    }
}
