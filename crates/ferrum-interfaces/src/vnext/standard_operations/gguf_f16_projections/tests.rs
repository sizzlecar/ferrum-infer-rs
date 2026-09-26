use super::*;

#[test]
fn gguf_f16_projection_contracts_preserve_resource_state_and_tensor_abi() {
    for (original, changed, operation, capability) in [
        (
            dense_swiglu_contract().unwrap(),
            dense_swiglu_gguf_f16_weights_contract().unwrap(),
            DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID,
            DENSE_SWIGLU_GGUF_F16_WEIGHTS_CAPABILITY_ID,
        ),
        (
            gated_delta_recurrent_attention_f32_master_contract().unwrap(),
            gated_delta_recurrent_attention_f32_master_gguf_f16_projections_contract().unwrap(),
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID,
        ),
        (
            causal_paged_attention_f32_master_contract().unwrap(),
            causal_paged_attention_f32_master_gguf_f16_projections_contract().unwrap(),
            CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
            CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID,
        ),
    ] {
        assert_ne!(original.descriptor.id, changed.descriptor.id);
        assert_eq!(changed.descriptor.id.as_str(), operation);
        assert_eq!(changed.descriptor.version, ContractVersion::new(1, 0));
        let mut expected = original.descriptor.clone();
        expected.id = OperationId::new(operation).unwrap();
        expected.version = ContractVersion::new(1, 0);
        expected.provider = provider_requirement(capability, ContractVersion::new(1, 0)).unwrap();
        assert_eq!(expected, changed.descriptor);
        changed
            .validate_signature(&original.descriptor.inputs, &original.descriptor.outputs)
            .unwrap();
        let mut invalid = original.descriptor.inputs.clone();
        invalid[0] = contiguous_tensor(
            token_hidden_dimensions(),
            [ElementType::I8],
            TensorAccess::Read,
        )
        .unwrap();
        assert!(changed
            .validate_signature(&invalid, &original.descriptor.outputs)
            .is_err());
    }
}

#[test]
fn gguf_f16_projection_authority_is_exact_operation_and_weight_ordinal() {
    use GgufF16ProjectionRoleV1::*;
    for (operation, expected) in [
        (
            DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID,
            vec![(1, FfnGateUp), (2, FfnDown)],
        ),
        (
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
            vec![(2, GdnInput), (7, GdnOutput)],
        ),
        (
            CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
            vec![
                (2, CausalQuery),
                (3, CausalKey),
                (4, CausalValue),
                (5, CausalOutput),
            ],
        ),
    ] {
        let operation = OperationId::new(operation).unwrap();
        for ordinal in 0..16 {
            assert_eq!(
                gguf_f16_projection_role_v1(&operation, ordinal),
                expected
                    .iter()
                    .find_map(|&(index, role)| (index == ordinal).then_some(role))
            );
        }
        assert_eq!(gguf_f16_projection_role_v1(&operation, u32::MAX), None);
    }
    for operation in [
        DENSE_SWIGLU_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
        TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
        LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
        DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID,
    ] {
        for ordinal in 0..16 {
            assert_eq!(
                gguf_f16_projection_role_v1(&OperationId::new(operation).unwrap(), ordinal),
                None
            );
        }
    }
}
