//! Borrowed preparation inputs for a causal-language execution driver.
//!
//! These values bind product metadata to the prepared family and frozen source
//! evidence. They do not grant resource, submission or completion authority.
//! Initialization consumes the original borrowed weight source; no source path
//! is reopened and no second model or sequence registry is created here.

use ferrum_runtime::{
    ElementType, OriginalModelSources, PreparedModelFamily, ResolvedModelPlan, ResolvedModelSource,
    ResolvedModelSources, WeightComponentSource,
};
use ferrum_types::{DataType, FerrumError, ModelInfo, Result};

/// Execution geometry supplied by the typed model definition. Head dimension
/// is explicit: valid architectures need not use hidden_size / head_count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalExecutionFacts {
    hidden_size: usize,
    layer_count: usize,
    head_count: usize,
    kv_head_count: usize,
    attention_head_dimension: usize,
    vocabulary_size: usize,
    maximum_sequence_tokens: usize,
    execution_dtype: DataType,
}

impl CausalExecutionFacts {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        layer_count: usize,
        head_count: usize,
        kv_head_count: usize,
        attention_head_dimension: usize,
        vocabulary_size: usize,
        maximum_sequence_tokens: usize,
        execution_dtype: DataType,
    ) -> Result<Self> {
        for (name, value) in [
            ("hidden_size", hidden_size),
            ("layer_count", layer_count),
            ("head_count", head_count),
            ("kv_head_count", kv_head_count),
            ("attention_head_dimension", attention_head_dimension),
            ("vocabulary_size", vocabulary_size),
            ("maximum_sequence_tokens", maximum_sequence_tokens),
        ] {
            if value == 0 {
                return Err(invalid(format!("{name} must be positive")));
            }
        }
        if kv_head_count > head_count || head_count % kv_head_count != 0 {
            return Err(invalid("attention heads must be divisible by KV heads"));
        }
        if !matches!(
            execution_dtype,
            DataType::FP16 | DataType::BF16 | DataType::FP32
        ) {
            return Err(invalid("execution dtype must be FP16, BF16 or FP32"));
        }
        let projection_width = head_count
            .checked_mul(attention_head_dimension)
            .ok_or_else(|| invalid("attention projection width overflows usize"))?;
        let kv_width = kv_head_count
            .checked_mul(attention_head_dimension)
            .and_then(|width| width.checked_mul(2))
            .ok_or_else(|| invalid("key/value projection width overflows usize"))?;
        for (name, elements) in [
            ("hidden activation", hidden_size),
            ("attention projection", projection_width),
            ("key/value projection", kv_width),
            ("vocabulary logits", vocabulary_size),
        ] {
            elements
                .checked_mul(execution_dtype.size_bytes())
                .ok_or_else(|| invalid(format!("{name} storage size overflows usize")))?;
        }
        Ok(Self {
            hidden_size,
            layer_count,
            head_count,
            kv_head_count,
            attention_head_dimension,
            vocabulary_size,
            maximum_sequence_tokens,
            execution_dtype,
        })
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }
    pub const fn head_count(&self) -> usize {
        self.head_count
    }
    pub const fn kv_head_count(&self) -> usize {
        self.kv_head_count
    }
    pub const fn attention_head_dimension(&self) -> usize {
        self.attention_head_dimension
    }
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }
    pub const fn maximum_sequence_tokens(&self) -> usize {
        self.maximum_sequence_tokens
    }
    pub const fn execution_dtype(&self) -> DataType {
        self.execution_dtype
    }
}

/// One borrowed preparation, kept alive through compilation and initialization.
/// The source records describe the already-frozen inputs, not new I/O locations.
#[derive(Clone, Copy)]
pub struct PreparedExecutionInput<'a> {
    family: &'a PreparedModelFamily,
    weights: &'a dyn WeightComponentSource,
    facts: CausalExecutionFacts,
    original_sources: &'a OriginalModelSources,
    resolved_sources: &'a ResolvedModelSources,
}

impl<'a> PreparedExecutionInput<'a> {
    pub fn new(
        family: &'a PreparedModelFamily,
        weights: &'a dyn WeightComponentSource,
        facts: CausalExecutionFacts,
        original_sources: &'a OriginalModelSources,
        resolved_sources: &'a ResolvedModelSources,
    ) -> Result<Self> {
        family
            .metadata()
            .causal_language()
            .map_err(|error| invalid(error.to_string()))?;
        let activation = family
            .numerical_profile()
            .activation_type()
            .map_err(|error| invalid(format!("invalid numerical activation contract: {error}")))?;
        let expected = match facts.execution_dtype {
            DataType::FP16 => ElementType::F16,
            DataType::BF16 => ElementType::Bf16,
            DataType::FP32 => ElementType::F32,
            _ => return Err(invalid("unsupported execution dtype")),
        };
        if activation != expected {
            return Err(invalid(
                "execution dtype differs from the prepared numerical profile",
            ));
        }
        Ok(Self {
            family,
            weights,
            facts,
            original_sources,
            resolved_sources,
        })
    }

    pub const fn family(&self) -> &'a PreparedModelFamily {
        self.family
    }
    pub fn weights(&self) -> &'a dyn WeightComponentSource {
        self.weights
    }
    pub const fn facts(&self) -> &CausalExecutionFacts {
        &self.facts
    }
    pub const fn original_sources(&self) -> &'a OriginalModelSources {
        self.original_sources
    }
    pub const fn resolved_sources(&self) -> &'a ResolvedModelSources {
        self.resolved_sources
    }

    /// Validate the transitional product metadata before using it to compile
    /// storage or language I/O. ModelInfo has no independent head-dimension
    /// field; the driver must consume the explicit value in `facts()`.
    pub fn validate_info(&self, info: &ModelInfo) -> Result<()> {
        for (name, actual, expected) in [
            ("hidden_size", info.hidden_size, self.facts.hidden_size),
            ("num_layers", info.num_layers, self.facts.layer_count),
            ("num_heads", info.num_heads, self.facts.head_count),
            ("num_kv_heads", info.num_kv_heads, self.facts.kv_head_count),
            ("vocab_size", info.vocab_size, self.facts.vocabulary_size),
            (
                "max_sequence_length",
                info.max_sequence_length,
                self.facts.maximum_sequence_tokens,
            ),
        ] {
            if actual != expected {
                return Err(invalid(format!(
                    "ModelInfo.{name} differs from prepared execution facts"
                )));
            }
        }
        if info.dtype != self.facts.execution_dtype {
            return Err(invalid(
                "ModelInfo.dtype differs from prepared execution facts",
            ));
        }
        Ok(())
    }

    /// A valid resolved plan can still belong to a different preparation. Check
    /// the complete family and source bindings before provider binding or any
    /// resource provisioning. Fingerprints are evidence, never native authority.
    pub fn validate_resolved_plan(&self, plan: &ResolvedModelPlan) -> Result<()> {
        let parts = plan.parts();
        if &parts.prepared_family != self.family {
            return Err(invalid("resolved plan uses a different prepared family"));
        }
        if &parts.external_metadata_id != self.family.external_metadata_id() {
            return Err(invalid(
                "resolved plan uses a different external metadata identity",
            ));
        }
        if &parts.original_sources != self.original_sources {
            return Err(invalid(
                "resolved plan uses different original model sources",
            ));
        }
        if &parts.resolved_sources != self.resolved_sources {
            return Err(invalid("resolved plan uses different frozen model sources"));
        }
        if parts.config.typed_config_sha256 != self.family.config_fingerprint() {
            return Err(invalid(
                "resolved typed config differs from the prepared family",
            ));
        }
        validate_file_binding(
            &self.resolved_sources.semantic,
            &parts.config.source_file,
            &parts.config.sha256,
            "config",
        )?;
        validate_file_binding(
            &self.resolved_sources.tokenizer,
            &parts.tokenizer.source_file,
            &parts.tokenizer.sha256,
            "tokenizer",
        )?;
        if usize::try_from(parts.tokenizer.vocabulary_size).ok() != Some(self.facts.vocabulary_size)
        {
            return Err(invalid(
                "resolved tokenizer vocabulary differs from execution facts",
            ));
        }
        Ok(())
    }
}

fn validate_file_binding(
    source: &ResolvedModelSource,
    relative_path: &str,
    sha256: &str,
    role: &str,
) -> Result<()> {
    let mut matching = source
        .files
        .iter()
        .filter(|file| file.relative_path == relative_path);
    let file = matching.next().ok_or_else(|| {
        invalid(format!(
            "resolved {role} file is absent from the frozen source"
        ))
    })?;
    if matching.next().is_some() || file.sha256 != sha256 {
        return Err(invalid(format!(
            "resolved {role} hash does not uniquely match the frozen source"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> FerrumError {
    FerrumError::model(format!("prepared execution input: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_facts_preserve_explicit_head_dimension_and_supported_dtype() {
        for dtype in [DataType::FP16, DataType::BF16, DataType::FP32] {
            let facts = CausalExecutionFacts::new(1152, 4, 8, 2, 256, 48, 128, dtype).unwrap();
            assert_eq!(facts.hidden_size(), 1152);
            assert_eq!(facts.attention_head_dimension(), 256);
            assert_eq!(facts.execution_dtype(), dtype);
        }
    }

    #[test]
    fn causal_facts_reject_invalid_geometry_and_storage_overflow() {
        let valid = [32, 2, 4, 2, 8, 48, 128];
        for index in 0..valid.len() {
            let mut values = valid;
            values[index] = 0;
            assert!(CausalExecutionFacts::new(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                DataType::FP32
            )
            .is_err());
        }
        for (heads, kv_heads, head_dim) in [(3, 2, 8), (2, 4, 8), (usize::MAX, 1, 2)] {
            assert!(CausalExecutionFacts::new(
                32,
                2,
                heads,
                kv_heads,
                head_dim,
                48,
                128,
                DataType::FP32
            )
            .is_err());
        }
        assert!(
            CausalExecutionFacts::new(usize::MAX, 2, 4, 2, 8, 48, 128, DataType::FP32).is_err()
        );
        for dtype in [
            DataType::FP8,
            DataType::INT8,
            DataType::UINT32,
            DataType::BOOL,
        ] {
            assert!(CausalExecutionFacts::new(32, 2, 4, 2, 8, 48, 128, dtype).is_err());
        }
    }
}
