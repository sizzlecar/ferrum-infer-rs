use super::super::vnext_runtime::{
    CpuBufferRegion, CpuKernelLaunch, CpuRegionSet, CpuRuntimeError,
};
use super::elementwise;
use super::matrix::{CpuMatrix, CpuMatrixFormat};
use super::scalar::CpuFloat;

pub(super) struct CpuOperatorLaunch {
    pub(super) regions: Vec<CpuBufferRegion>,
    pub(super) actions: Vec<CpuAction>,
}

#[derive(Clone, Copy)]
pub(super) enum ResidualStorage {
    Separate,
    LeftInPlace,
    BothInPlace,
}

pub(super) enum CpuAction {
    GatedDelta(Box<super::gated_delta_launch::CpuGatedDeltaLaunch>),
    CausalAttention(Box<super::causal_attention_launch::CpuCausalLaunch>),
    CausalBinding {
        binding: usize,
        source_start: u64,
        tokens: u64,
    },
    Linear {
        input: usize,
        weight: usize,
        output: usize,
        format: CpuMatrixFormat,
        input_type: CpuFloat,
        output_type: CpuFloat,
        rows: usize,
        input_width: usize,
        output_width: usize,
        output_stride: usize,
        output_column_offset: usize,
    },
    Embedding {
        weight: usize,
        tokens: usize,
        output: usize,
        format: CpuMatrixFormat,
        output_type: CpuFloat,
        vocabulary: usize,
        hidden: usize,
    },
    RmsNorm {
        input: usize,
        weight: usize,
        output: usize,
        input_type: CpuFloat,
        output_type: CpuFloat,
        width: usize,
        epsilon: f32,
    },
    Residual {
        left: usize,
        right: usize,
        output: usize,
        left_type: CpuFloat,
        storage: ResidualStorage,
    },
    SwiGlu {
        gate_up: usize,
        output: usize,
        width: usize,
    },
    Argmax {
        logits: usize,
        mask: usize,
        repeated_ids: usize,
        repeated_offsets: usize,
        penalty: usize,
        scratch: usize,
        output: usize,
        dtype: CpuFloat,
    },
}

impl CpuKernelLaunch for CpuOperatorLaunch {
    fn validate_runtime(&self, instance: u64) -> Result<(), CpuRuntimeError> {
        if self.actions.is_empty() {
            return Err(CpuRuntimeError::new("CPU operation has no work"));
        }
        for region in &self.regions {
            region.validate_runtime(instance)?;
        }
        Ok(())
    }

    fn execute(&self) -> Result<(), CpuRuntimeError> {
        CpuBufferRegion::with_regions(&self.regions, |views| {
            for action in &self.actions {
                action.execute(views)?;
            }
            Ok(())
        })
    }
}

impl CpuAction {
    fn execute(&self, views: &mut CpuRegionSet<'_>) -> Result<(), CpuRuntimeError> {
        match *self {
            Self::GatedDelta(ref launch) => launch.execute(views),
            Self::CausalAttention(ref launch) => launch.execute(views),
            Self::CausalBinding {
                binding,
                source_start,
                tokens,
            } => views.with_io([], [binding], |[], [output]| {
                if output.len() != 16 {
                    return Err(CpuRuntimeError::new(
                        "CPU attention binding has an incompatible scalar ABI",
                    ));
                }
                output[..8].copy_from_slice(&source_start.to_le_bytes());
                output[8..].copy_from_slice(&tokens.to_le_bytes());
                Ok(())
            }),
            Self::Linear {
                input,
                weight,
                output,
                format,
                input_type,
                output_type,
                rows,
                input_width,
                output_width,
                output_stride,
                output_column_offset,
            } => views.with_io([input, weight], [output], |[input, weight], [output]| {
                CpuMatrix::new(weight, format, output_width, input_width)?.linear(
                    input,
                    input_type,
                    output,
                    output_type,
                    rows,
                    output_stride,
                    output_column_offset,
                )
            }),
            Self::Embedding {
                weight,
                tokens,
                output,
                format,
                output_type,
                vocabulary,
                hidden,
            } => views.with_io([weight, tokens], [output], |[weight, tokens], [output]| {
                CpuMatrix::new(weight, format, vocabulary, hidden)?.embedding(
                    tokens,
                    output,
                    output_type,
                )
            }),
            Self::RmsNorm {
                input,
                weight,
                output,
                input_type,
                output_type,
                width,
                epsilon,
            } => views.with_io([input, weight], [output], |[input, weight], [output]| {
                elementwise::rms_norm(
                    input,
                    input_type,
                    weight,
                    output,
                    output_type,
                    width,
                    epsilon,
                )
            }),
            Self::Residual {
                left,
                right,
                output,
                left_type,
                storage,
            } => match storage {
                ResidualStorage::Separate => {
                    views.with_io([left, right], [output], |[left, right], [output]| {
                        elementwise::residual_add(left, left_type, right, output)
                    })
                }
                ResidualStorage::LeftInPlace => {
                    views.with_io([right], [output], |[right], [output]| {
                        elementwise::residual_add_in_place(output, left_type, Some(right))
                    })
                }
                ResidualStorage::BothInPlace => views.with_io([], [output], |[], [output]| {
                    elementwise::residual_add_in_place(output, left_type, None)
                }),
            },
            Self::SwiGlu {
                gate_up,
                output,
                width,
            } => views.with_io([gate_up], [output], |[input], [output]| {
                elementwise::swiglu(input, output, width)
            }),
            Self::Argmax {
                logits,
                mask,
                repeated_ids,
                repeated_offsets,
                penalty,
                scratch,
                output,
                dtype,
            } => views.with_io(
                [logits, mask, repeated_ids, repeated_offsets, penalty],
                [scratch, output],
                |[logits, mask, ids, offsets, penalty], [scratch, output]| {
                    if penalty.len() != 4 || output.len() != 4 {
                        return Err(CpuRuntimeError::new("CPU argmax scalar bindings differ"));
                    }
                    let selected = elementwise::masked_argmax(
                        logits,
                        dtype,
                        mask,
                        ids,
                        offsets,
                        CpuFloat::F32.read(penalty, 0),
                        scratch,
                    )?;
                    output.copy_from_slice(&selected.to_le_bytes());
                    Ok(())
                },
            ),
        }
    }
}
