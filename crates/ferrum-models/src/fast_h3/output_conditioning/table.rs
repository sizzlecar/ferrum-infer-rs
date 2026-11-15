use super::*;
use ferrum_execution::TensorExecutionOutput;
use half::bf16;

/// Live compact final-output table, retaining its original Output-class Host
/// owner. Many request evaluations can borrow this immutable table. It binds
/// the model adapter's actual execution configuration; it does not authenticate
/// arbitrary serialized cache contents or mint execution completion evidence.
#[derive(Debug)]
pub struct FastH3OutputConditioningTable {
    config: FastH3OutputTimeConfig,
    output: TensorExecutionOutput,
}
impl FastH3OutputConditioningTable {
    pub fn from_execution(
        config: FastH3OutputTimeConfig,
        mut outputs: Vec<TensorExecutionOutput>,
        limits: FastH3OutputConditioningLimits,
    ) -> Result<Self, VNextError> {
        config.validate(limits)?;
        if outputs.len() != 1 {
            return Err(invalid("final table requires exactly one output"));
        }
        let output = outputs.pop().expect("validated one output");
        if output.value_id().as_str() != FastH3OutputTimeFamily::OUTPUT_ID
            || output.tensor().dimensions() != [config.times.len() as u64, 2 * config.hidden]
            || output.tensor().element_type() != ElementType::Bf16
            || output.tensor().layout() != &ResolvedTensorLayout::Contiguous
            || output.bytes().class() != HostMemoryClass::Output
            || output.tensor().minimum_storage_bytes()? != output.bytes().len() as u64
            || !output
                .bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite())
        {
            return Err(invalid(
                "final table identity, dtype, shape, ownership class or finite values differ",
            ));
        }
        Ok(Self { config, output })
    }
    pub fn configuration(&self) -> &FastH3OutputTimeConfig {
        &self.config
    }
    pub fn output(&self) -> &TensorExecutionOutput {
        &self.output
    }
    /// One admitted compact copy, without per-token expansion or relabelling.
    pub fn modulation_input(
        &self,
        c: &FastH3OutputNormConfig,
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor, VNextError> {
        if c.hidden != self.config.hidden || !same_times(&c.times, &self.config.times) {
            return Err(invalid(
                "final normalization width or exact times differ from live table",
            ));
        }
        managed_copy(
            host,
            FastH3OutputNormFamily::TABLE_ID,
            vec![self.config.times.len() as u64, 2 * self.config.hidden],
            ElementType::Bf16,
            self.output.bytes().bytes(),
        )
    }
}
