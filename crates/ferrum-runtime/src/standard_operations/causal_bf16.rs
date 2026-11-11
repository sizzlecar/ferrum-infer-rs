//! Native BF16 causal boundaries with separate operation identities.
//!
//! Dense coefficients, activations, projection results, convolution history and
//! paged KV use BF16. Arithmetic accumulates in F32 and rounds to nearest,
//! ties-to-even at each declared BF16 boundary. The DeltaNet recurrent state,
//! decay parameters and gated-normalization coefficients remain F32. These are
//! distinct numerical contracts: an F16 provider cannot satisfy them by treating
//! BF16 bit patterns as IEEE binary16, nor by rounding whole weights to F16.
//!
//! Shapes, operand order, residual aliasing, rotary attributes, masking and
//! repetition-penalty semantics follow the corresponding causal operations.
//! Embedding and row selection copy BF16 bits. Argmax converts eligible logits
//! to F32 for the declared penalty/ordering rules. Providers must identify their
//! reduction and intermediate-rounding behavior; vendor bitwise parity is not
//! implied by the storage type.
use super::*;

fn bf16_boundaries(
    mut contract: StandardOperationContract,
    operation: &str,
    capability: &str,
) -> Result<StandardOperationContract, VNextError> {
    let descriptor = &mut contract.descriptor;
    descriptor.id = OperationId::new(operation)?;
    descriptor.version = ContractVersion::new(1, 0);
    descriptor.provider = provider_requirement(capability, descriptor.version)?;
    for tensor in descriptor.inputs.iter_mut().chain(&mut descriptor.outputs) {
        if tensor.element_types().contains(&ElementType::F16) {
            let types = tensor
                .element_types()
                .iter()
                .map(|&dtype| {
                    if dtype == ElementType::F16 {
                        ElementType::Bf16
                    } else {
                        dtype
                    }
                })
                .collect();
            *tensor = TensorContract::new(
                tensor.dimensions().to_vec(),
                types,
                tensor.layouts().to_vec(),
                tensor.access(),
                tensor.alias().clone(),
            )?;
        }
    }
    // The old F16 numerical tolerance is not a BF16 oracle. Each native BF16
    // implementation is qualified against the explicitly rounded operation.
    descriptor.oracle = OracleSpec::OperationDefined;
    descriptor.validate()?;
    Ok(contract)
}

macro_rules! bf16_operation {
    ($function:ident, $operation:ident, $capability:ident, $base:ident, $name:literal) => {
        pub const $operation: &str = concat!("operation.", $name, ".bf16");
        pub const $capability: &str = concat!("capability.operation.", $name, ".bf16");
        pub fn $function() -> Result<StandardOperationContract, VNextError> {
            bf16_boundaries($base()?, $operation, $capability)
        }
    };
}

bf16_operation!(
    token_embedding_bf16_contract,
    TOKEN_EMBEDDING_BF16_OPERATION_ID,
    TOKEN_EMBEDDING_BF16_CAPABILITY_ID,
    token_embedding_contract,
    "token_embedding"
);
bf16_operation!(
    last_token_dense_linear_bf16_contract,
    LAST_TOKEN_DENSE_LINEAR_BF16_OPERATION_ID,
    LAST_TOKEN_DENSE_LINEAR_BF16_CAPABILITY_ID,
    last_token_dense_linear_contract,
    "last_token_dense_linear"
);
bf16_operation!(
    last_token_masked_argmax_bf16_contract,
    LAST_TOKEN_MASKED_ARGMAX_BF16_OPERATION_ID,
    LAST_TOKEN_MASKED_ARGMAX_BF16_CAPABILITY_ID,
    last_token_masked_argmax_contract,
    "last_token_masked_argmax"
);
bf16_operation!(
    rms_norm_bf16_contract,
    RMS_NORM_BF16_OPERATION_ID,
    RMS_NORM_BF16_CAPABILITY_ID,
    rms_norm_contract,
    "rms_norm"
);
bf16_operation!(
    dense_swiglu_bf16_contract,
    DENSE_SWIGLU_BF16_OPERATION_ID,
    DENSE_SWIGLU_BF16_CAPABILITY_ID,
    dense_swiglu_contract,
    "dense_swiglu"
);
bf16_operation!(
    residual_add_bf16_contract,
    RESIDUAL_ADD_BF16_OPERATION_ID,
    RESIDUAL_ADD_BF16_CAPABILITY_ID,
    residual_add_contract,
    "residual_add"
);
bf16_operation!(
    gated_delta_recurrent_attention_bf16_contract,
    GATED_DELTA_RECURRENT_ATTENTION_BF16_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_BF16_CAPABILITY_ID,
    gated_delta_recurrent_attention_contract,
    "gated_delta_recurrent_attention"
);
bf16_operation!(
    causal_paged_attention_bf16_contract,
    CAUSAL_PAGED_ATTENTION_BF16_OPERATION_ID,
    CAUSAL_PAGED_ATTENTION_BF16_CAPABILITY_ID,
    causal_paged_attention_contract,
    "causal_paged_attention"
);
bf16_operation!(
    row_select_bf16_contract,
    ROW_SELECT_BF16_OPERATION_ID,
    ROW_SELECT_BF16_CAPABILITY_ID,
    row_select_f16_contract,
    "row_select"
);
bf16_operation!(
    causal_paged_attention_explicit_rope_bf16_contract,
    CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_BF16_OPERATION_ID,
    CAUSAL_PAGED_ATTENTION_EXPLICIT_ROPE_BF16_CAPABILITY_ID,
    causal_paged_attention_explicit_rope_f16_contract,
    "causal_paged_attention.explicit-rope"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrent_bf16_preserves_f32_parameters_state_and_access() {
        let contract = gated_delta_recurrent_attention_bf16_contract().unwrap();
        let descriptor = contract.descriptor();
        for (index, input) in descriptor.inputs.iter().enumerate() {
            let dtype = if [4, 5, 6, 9].contains(&index) {
                ElementType::F32
            } else {
                ElementType::Bf16
            };
            assert_eq!(input.element_types(), &BTreeSet::from([dtype]));
        }
        assert_eq!(descriptor.inputs[8].access(), TensorAccess::ReadWrite);
        assert_eq!(descriptor.inputs[9].access(), TensorAccess::ReadWrite);
        let original = gated_delta_recurrent_attention_contract().unwrap();
        assert!(contract
            .validate_signature(
                &original.descriptor().inputs,
                &original.descriptor().outputs,
            )
            .is_err());
    }

    #[test]
    fn bf16_visual_contract_preserves_mask_rotary_and_residual_rules() {
        let select = row_select_bf16_contract().unwrap();
        assert_eq!(
            select.descriptor().inputs[2].element_types(),
            &BTreeSet::from([ElementType::U8])
        );
        let attention = causal_paged_attention_explicit_rope_bf16_contract().unwrap();
        for port in attention
            .descriptor()
            .inputs
            .iter()
            .chain(&attention.descriptor().outputs)
        {
            assert_eq!(port.element_types(), &BTreeSet::from([ElementType::Bf16]));
        }
        assert_eq!(
            attention.descriptor().inputs[8].access(),
            TensorAccess::ReadWrite
        );
        assert_eq!(
            attention.descriptor().attributes.entries()
                [&AttributeId::new("rope_interleaved").unwrap()]
                .constraint,
            AttributeConstraint::BoolEquals(false)
        );
        let residual = residual_add_bf16_contract().unwrap();
        assert_eq!(
            residual.descriptor().outputs[0].alias(),
            &AliasPolicy::MayAlias { tensor_index: 0 }
        );
    }
}
