use super::*;

pub const TAIL_TWO_GREEDY_F16_OPERATION_ID: &str = "operation.tail_two_greedy.f16";
pub const TAIL_TWO_GREEDY_F16_CAPABILITY_ID: &str = "capability.operation.tail_two_greedy.f16";

/// Greedy-only, without masks, repetition penalties or sampling processors.
/// For N>=2 projects rows N-2 and N-1; N=1 duplicates the sole row in both
/// slots (a caller consuming one decision uses slot1). Each dot product has
/// the original F16 dense-head semantics: F32 accumulation, then F16 rounding
/// before comparison. Only finite logits are eligible; IEEE total order and
/// the lowest token id break ties, as in last_token_masked_argmax. No eligible
/// value yields u32::MAX. Hidden output copies the two original F16 rows.
///
/// Outputs are independent of prefill length. A provider may reuse one logits
/// row as scratch or fuse the projection and reduction, never exposing an
/// unbounded Tokens x Vocabulary terminal value. Executors requiring any
/// logits processors must select a different operation.
pub fn tail_two_greedy_f16_contract() -> Result<StandardOperationContract, VNextError> {
    let version = ContractVersion::new(1, 0);
    let descriptor = OperationDescriptor {
        id: OperationId::new(TAIL_TWO_GREEDY_F16_OPERATION_ID)?,
        version,
        inputs: vec![
            contiguous_tensor(
                token_hidden_dimensions(),
                [ElementType::F16],
                TensorAccess::Read,
            )?,
            contiguous_tensor(
                vec![
                    DimensionConstraint::Symbol("vocab_size".to_owned()),
                    DimensionConstraint::Symbol("hidden_size".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Read,
            )?,
        ],
        outputs: vec![
            contiguous_tensor(
                vec![DimensionConstraint::Exact(2)],
                [ElementType::U32],
                TensorAccess::Write,
            )?,
            contiguous_tensor(
                vec![
                    DimensionConstraint::Exact(2),
                    DimensionConstraint::Symbol("hidden_size".to_owned()),
                ],
                [ElementType::F16],
                TensorAccess::Write,
            )?,
        ],
        attributes: AttributeSchema::new(BTreeMap::from([
            unsigned_attribute("hidden_size")?,
            unsigned_attribute("vocab_size")?,
        ]))?,
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 16,
            scratch: ResourcePresenceRequirement::Optional,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: OracleSpec::OperationDefined,
        provider: provider_requirement(TAIL_TWO_GREEDY_F16_CAPABILITY_ID, version)?,
        profile_phase: ProfilePhase::Forward,
    };
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}
