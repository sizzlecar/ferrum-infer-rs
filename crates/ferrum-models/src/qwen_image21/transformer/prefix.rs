//! Request-local, BF16 prefix handoff. These descriptors carry no resource
//! authority: the executor must admit, initialize and retire the decode states.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenImage21TransformerPhase {
    #[default]
    Full,
    PrefixPrefill,
    PrefixDecode,
}
impl QwenImage21TransformerPhase {
    pub const fn is_full(&self) -> bool {
        matches!(self, Self::Full)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenImage21PrefixPart {
    /// BF16 key after learned head RMS normalization and adjacent RoPE.
    Key,
    /// BF16 value projection, without RMS normalization or RoPE.
    Value,
}
impl QwenImage21PrefixPart {
    const fn name(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Value => "value",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenImage21PrefixCachePort {
    layer: u64,
    part: QwenImage21PrefixPart,
    value_id: ProgramValueId,
    tensor: ProgramTensorSpec,
}
impl QwenImage21PrefixCachePort {
    pub const fn layer(&self) -> u64 {
        self.layer
    }
    pub const fn part(&self) -> QwenImage21PrefixPart {
        self.part
    }
    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }
    pub fn tensor(&self) -> &ProgramTensorSpec {
        &self.tensor
    }
    pub fn state_spec(&self) -> Result<StateSpec, VNextError> {
        Ok(StateSpec {
            id: StateId::new(format!(
                "state.qwen_image21.transformer.prefix.layer{}.{}",
                self.layer,
                self.part.name()
            ))?,
            value_id: self.value_id.clone(),
            tensor: self.tensor.clone(),
            lifetime: StateLifetime::Request,
            capacity_demand: StateCapacityDemand::FixedPerScope,
            initialization: StateInitialization::None,
            checkpoint: StateCheckpointCapability::Unsupported,
        })
    }
}

pub(super) fn cache_value(
    layer: u64,
    part: QwenImage21PrefixPart,
) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!(
        "value.qwen_image21.transformer.prefix.layer{layer}.{}",
        part.name()
    ))
}
pub(super) fn joined_value(
    layer: u64,
    part: QwenImage21PrefixPart,
) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!(
        "value.qwen_image21.transformer.layer{layer}.joined_{}",
        part.name()
    ))
}

impl QwenImage21TransformerConfig {
    pub fn with_execution_phase(
        mut self,
        phase: QwenImage21TransformerPhase,
    ) -> Result<Self, VNextError> {
        self.phase = phase;
        self.validate()?;
        Ok(self)
    }
    pub const fn execution_phase(&self) -> QwenImage21TransformerPhase {
        self.phase
    }
    pub(super) const fn is_decode(&self) -> bool {
        matches!(self.phase, QwenImage21TransformerPhase::PrefixDecode)
    }
    pub(super) fn execution_rows(&self) -> u64 {
        if self.is_decode() {
            self.layout().batch() * self.layout().target_tokens()
        } else {
            self.layout().rows()
        }
    }
    /// Stable, layer-ordered ports shared by prefill outputs and decode states.
    /// The caller must bind the exact successful prefill/request identity, not
    /// merely tensors with equal dimensions or a matching plan hash.
    pub fn prefix_cache_ports(&self) -> Result<Vec<QwenImage21PrefixCachePort>, VNextError> {
        if !self.transformer().causal_condition() {
            return Err(invalid(
                "prefix_cache",
                "prefix reuse requires causal_condition=true",
            ));
        }
        let tensor = ProgramTensorSpec {
            dimensions: vec![
                self.layout().batch() * self.layout().target_start(),
                self.transformer().hidden_size(),
            ],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        };
        tensor.validate("prefix_cache")?;
        (0..self.transformer().layers())
            .flat_map(|layer| {
                [QwenImage21PrefixPart::Key, QwenImage21PrefixPart::Value]
                    .map(move |part| (layer, part))
            })
            .map(|(layer, part)| {
                Ok(QwenImage21PrefixCachePort {
                    layer,
                    part,
                    value_id: cache_value(layer, part)?,
                    tensor: tensor.clone(),
                })
            })
            .collect()
    }
    pub fn prefix_cache_bytes(&self) -> Result<u64, VNextError> {
        self.prefix_cache_ports()?.iter().try_fold(0u64, |sum, p| {
            sum.checked_add(p.tensor.byte_len()?)
                .ok_or_else(|| invalid("prefix_cache", "cache byte extent overflows"))
        })
    }
    pub(super) fn prefix_states(&self) -> Result<Vec<StateSpec>, VNextError> {
        if self.is_decode() {
            self.prefix_cache_ports()?
                .iter()
                .map(QwenImage21PrefixCachePort::state_spec)
                .collect()
        } else {
            Ok(vec![])
        }
    }
    pub(super) fn cross_attention_attributes(
        &self,
    ) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let mut attrs = QwenImage21AttentionConfig::new(
            self.transformer().clone(),
            self.layout().attention_layout()?,
        )?
        .attributes()?;
        for key in [
            "sequence_length",
            "segment_lengths",
            "segment_bidirectional",
        ] {
            attrs.remove(&AttributeId::new(key)?);
        }
        attrs.insert(
            AttributeId::new("batch")?,
            SemanticValue::Unsigned(self.layout().batch()),
        );
        Ok(attrs)
    }
}

#[cfg(test)]
mod tests;
