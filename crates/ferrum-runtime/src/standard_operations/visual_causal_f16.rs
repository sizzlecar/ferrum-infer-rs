//! Explicit visual prompting over the original F16 causal state ABI.
use super::*;
pub const ROW_SELECT_F16_OPERATION_ID: &str = "operation.row_select.f16";
pub const ROW_SELECT_F16_CAPABILITY_ID: &str = "capability.operation.row_select.f16";
pub const CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID: &str =
    "operation.causal_paged_attention.explicit-rope.f16";
pub const CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_CAPABILITY_ID: &str =
    "capability.operation.causal_paged_attention.explicit-rope.f16";

/// Copy exactly one complete F16 row. A zero U8 mask selects the original row
/// without reading its visual source row; every nonzero value selects visual.
/// All U8 values are valid predicates. Selection copies bits, never multiplies
/// floating values by a mask.
pub fn row_select_f16_contract() -> Result<StandardOperationContract, VNextError> {
    let descriptor = OperationDescriptor {
        id: OperationId::new(ROW_SELECT_F16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![symbol("tokens")],
                [ElementType::U8],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::F16],
            TensorAccess::Write,
        )?],
        attributes: AttributeSchema::new(BTreeMap::from([unsigned_attribute("hidden_size")?]))?,
        resources: no_auxiliary_resources(),
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(ROW_SELECT_F16_CAPABILITY_ID, ContractVersion::new(1, 0))?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

/// The original nine F16 causal-attention ports, plus per-token F16 cosine
/// and sine half tables. Position in these tables is independent of the
/// physical paged KV cursor. Only half-split active-prefix rotation is allowed.
pub fn causal_paged_attention_explicit_rope_f16_contract(
) -> Result<StandardOperationContract, VNextError> {
    let mut contract = causal_paged_attention_contract()?;
    let d = &mut contract.descriptor;
    d.id = OperationId::new(CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_OPERATION_ID)?;
    d.version = ContractVersion::new(1, 0);
    d.provider = provider_requirement(
        CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_F16_CAPABILITY_ID,
        d.version,
    )?;
    d.inputs.extend([
        contiguous_tensor(
            vec![symbol("tokens"), symbol("rope_pairs")],
            [ElementType::F16],
            TensorAccess::Read,
        )?,
        contiguous_tensor(
            vec![symbol("tokens"), symbol("rope_pairs")],
            [ElementType::F16],
            TensorAccess::Read,
        )?,
    ]);
    let mut attrs = d.attributes.entries().clone();
    attrs.insert(
        unsigned_attribute("rope_pairs")?.0,
        unsigned_attribute("rope_pairs")?.1,
    );
    attrs.insert(
        AttributeId::new("rope_interleaved")?,
        AttributeSpec {
            value_kind: AttributeValueKind::Bool,
            required: true,
            constraint: AttributeConstraint::BoolEquals(false),
        },
    );
    d.attributes = AttributeSchema::new(attrs)?;
    d.validate()?;
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_rope_preserves_the_original_state_and_projection_abi() {
        let old = causal_paged_attention_contract().unwrap();
        let new = causal_paged_attention_explicit_rope_f16_contract().unwrap();
        assert_eq!(&new.descriptor.inputs[..9], &old.descriptor.inputs);
        assert_eq!(new.descriptor.outputs, old.descriptor.outputs);
        assert_eq!(new.descriptor.resources, old.descriptor.resources);
        // The explicit descriptor must reject pair-interleaving, independent
        // of frequency-axis THW interleaving performed by the producer.
        assert_eq!(
            new.descriptor.attributes.entries()[&AttributeId::new("rope_interleaved").unwrap()]
                .constraint,
            AttributeConstraint::BoolEquals(false)
        );
    }
}
