//! Live time-component outputs retained for later block modulation.
use super::*;
use crate::fast_h3::FastH3ModulationConfig;
use ferrum_execution::TensorExecutionOutput;
use half::bf16;

/// A compact, immutable result of one block's time component. Output owners
/// keep their original Host class and accounting after the component closes.
///
/// The model adapter binds the configuration and schedule it actually used.
/// This checks output shape, arithmetic storage, finite values and downstream
/// routing; it is not an attestation for arbitrary cached files or input data.
#[derive(Debug)]
pub struct FastH3ConditioningTable {
    config: FastH3TimeConditioningConfig,
    times: Vec<f32>,
    outputs: [TensorExecutionOutput; 2],
}

impl FastH3ConditioningTable {
    pub fn from_execution(
        config: FastH3TimeConditioningConfig,
        schedule: &FastH3ConditioningSchedule,
        mut outputs: Vec<TensorExecutionOutput>,
        limits: FastH3TimeConditioningLimits,
    ) -> Result<Self, VNextError> {
        config.validate(limits)?;
        if config.time_rows() != schedule.timesteps().len() as u64 || outputs.len() != 2 {
            return Err(invalid("time outputs do not cover the declared schedule"));
        }
        if outputs[1].value_id().as_str() == FastH3TimeConditioningFamily::EMBEDDING_ID {
            outputs.swap(0, 1);
        }
        let modulation_width = config
            .hidden_size()
            .checked_mul(18)
            .ok_or_else(|| invalid("modulation width overflows"))?;
        for (output, name, dtype, width) in [
            (
                &outputs[0],
                FastH3TimeConditioningFamily::EMBEDDING_ID,
                ElementType::F32,
                config.embedding_dim(),
            ),
            (
                &outputs[1],
                FastH3TimeConditioningFamily::OUTPUT_ID,
                ElementType::Bf16,
                modulation_width,
            ),
        ] {
            if output.value_id().as_str() != name
                || output.tensor().dimensions() != [config.time_rows(), width]
                || output.tensor().element_type() != dtype
                || output.tensor().layout() != &ResolvedTensorLayout::Contiguous
                || output.bytes().class() != HostMemoryClass::Output
                || output.tensor().minimum_storage_bytes()? != output.bytes().len() as u64
            {
                return Err(invalid("time output identity, storage or geometry differs"));
            }
            let finite = if dtype == ElementType::F32 {
                output
                    .bytes()
                    .bytes()
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
            } else {
                output.bytes().bytes().chunks_exact(2).all(|b| {
                    bf16::from_bits(u16::from_le_bytes([b[0], b[1]]))
                        .to_f32()
                        .is_finite()
                })
            };
            if !finite {
                return Err(invalid("time component produced nonfinite output"));
            }
        }
        Ok(Self {
            config,
            times: schedule.timesteps().to_vec(),
            outputs: outputs
                .try_into()
                .map_err(|_| invalid("time output inventory differs"))?,
        })
    }

    pub fn configuration(&self) -> &FastH3TimeConditioningConfig {
        &self.config
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.times
    }

    /// Embedding followed by modulation, retaining their original output IDs.
    pub fn outputs(&self) -> &[TensorExecutionOutput; 2] {
        &self.outputs
    }

    /// Make one explicitly admitted compact input copy for a matching block.
    /// Changing request progress changes row routing, never the table contents.
    /// The copy has an independent InputPreprocess owner; no Output allocation
    /// is relabelled, and no per-token expanded table is constructed.
    pub fn modulation_input(
        &self,
        config: &FastH3ModulationConfig,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor, VNextError> {
        if config.block() != self.config.block()
            || config.width() != self.config.hidden_size()
            || config.time_values().len() != self.times.len()
            || !config
                .time_values()
                .iter()
                .zip(&self.times)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Err(invalid(
                "modulation block, width or exact time rows differ from the live table",
            ));
        }
        config.managed_table(host, self.outputs[1].bytes().bytes())
    }
}
