//! GatedDelta row scan with two explicit, uncommitted tail-prefix states.
use super::*;

pub const GATED_DELTA_CANDIDATE_SCAN_F16_OPERATION_ID: &str =
    "operation.gated_delta.candidate-scan.f16";
pub const GATED_DELTA_CANDIDATE_SCAN_F16_CAPABILITY_ID: &str =
    "capability.gated_delta.candidate-scan.f16";
pub const GATED_DELTA_CANDIDATE_SCAN_F32_MASTER_OPERATION_ID: &str =
    "operation.gated_delta.candidate-scan.f32-master";
pub const GATED_DELTA_CANDIDATE_SCAN_F32_MASTER_CAPABILITY_ID: &str =
    "capability.gated_delta.candidate-scan.f32-master";

/// Evaluate N >= 1 consecutive rows using ordinary GatedDelta F16 arithmetic.
/// Inputs 0..7 keep the ordinary weight/hidden ABI. Inputs 8 and 9 are the
/// READ-ONLY original convolution (F16) and recurrent (F32) state. Output 0
/// preserves [tokens,hidden]. Outputs 1 and 2 prepend a length-two candidate
/// axis to the respective state shapes: slot 0 is the state after N-1 rows,
/// slot 1 is the state after all N rows. For N=1 slot 0 is the original state;
/// for N=2 the two slots are exactly the first and second prefix states.
///
/// The arithmetic follows serial rows, including each row's F16/F32 rounding.
/// All outputs forbid aliases. Partial outputs on failure are not usable.
/// Both complete candidate states require admission, independent of N. No
/// complete state copy may be hidden in unaccounted provider scratch.
///
/// This operation neither commits a prefix nor advances a causal frontier.
/// A caller needs a separate commit contract to select slot 1 for prefill/
/// ordinary decode, or either slot for two-row verification. The same token
/// scan can therefore serve those paths within one future causal plan.
pub fn gated_delta_candidate_scan_f16_contract() -> Result<StandardOperationContract, VNextError> {
    contract(
        GATED_DELTA_CANDIDATE_SCAN_F16_OPERATION_ID,
        GATED_DELTA_CANDIDATE_SCAN_F16_CAPABILITY_ID,
        ElementType::F16,
    )
}

/// The same bounded-candidate scan contract with the existing F32-master hidden boundary;
/// projected operands/convolution remain F16 and recurrent state remains F32.
pub fn gated_delta_candidate_scan_f32_master_contract(
) -> Result<StandardOperationContract, VNextError> {
    contract(
        GATED_DELTA_CANDIDATE_SCAN_F32_MASTER_OPERATION_ID,
        GATED_DELTA_CANDIDATE_SCAN_F32_MASTER_CAPABILITY_ID,
        ElementType::F32,
    )
}

fn contract(
    operation_id: &str,
    capability_id: &str,
    hidden: ElementType,
) -> Result<StandardOperationContract, VNextError> {
    // Inherit the existing arithmetic, weight ordinals, attributes and row
    // workspace requirement. The new identity declares different state custody.
    let mut descriptor = gated_delta_recurrent_attention_contract_with_hidden(
        operation_id,
        ContractVersion::new(1, 0),
        capability_id,
        hidden,
    )?
    .descriptor;
    let rows = token_hidden_dimensions;
    let convolution = || vec![symbol("qkv_features"), symbol("conv_state_width")];
    let recurrent = || {
        vec![
            symbol("value_heads"),
            symbol("value_head_dim"),
            symbol("key_head_dim"),
        ]
    };
    descriptor.inputs[0] = contiguous_tensor(rows(), [hidden], TensorAccess::Read)?;
    descriptor.inputs[8] =
        contiguous_tensor(convolution(), [ElementType::F16], TensorAccess::Read)?;
    descriptor.inputs[9] = contiguous_tensor(recurrent(), [ElementType::F32], TensorAccess::Read)?;
    let prefix = |shape: Vec<DimensionConstraint>| std::iter::once(exact(2)).chain(shape).collect();
    descriptor.outputs = vec![
        contiguous_tensor(rows(), [hidden], TensorAccess::Write)?,
        contiguous_tensor(
            prefix(convolution()),
            [ElementType::F16],
            TensorAccess::Write,
        )?,
        contiguous_tensor(prefix(recurrent()), [ElementType::F32], TensorAccess::Write)?,
    ];
    // Retain ordinary GDN's optional, accounted address-control workspace.
    // Eager providers may omit it; replay eligibility and fresh state bindings
    // are separately checked by Core and the selected provider. Neither this
    // workspace nor its absence changes candidate storage or numerical types.
    descriptor.validate()?;
    Ok(StandardOperationContract { descriptor })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_scan_preserves_arithmetic_and_declares_read_only_original_state() {
        for (candidate, original) in [
            (
                gated_delta_candidate_scan_f16_contract().unwrap(),
                gated_delta_recurrent_attention_contract().unwrap(),
            ),
            (
                gated_delta_candidate_scan_f32_master_contract().unwrap(),
                gated_delta_recurrent_attention_f32_master_contract().unwrap(),
            ),
        ] {
            let c = candidate.descriptor();
            let o = original.descriptor();
            assert_eq!(c.attributes, o.attributes);
            assert_eq!(c.oracle, o.oracle);
            assert_eq!(c.inputs[1..8], o.inputs[1..8]);
            assert_eq!(c.inputs[0], o.inputs[0]);
            for input in &c.inputs {
                assert_eq!(input.access(), TensorAccess::Read);
            }
            assert_eq!(c.outputs.len(), 3);
            assert_eq!(c.outputs[0].dimensions(), o.outputs[0].dimensions());
            for candidate in &c.outputs[1..] {
                assert_eq!(candidate.dimensions()[0], exact(2));
            }
            for output in &c.outputs {
                assert_eq!(output.access(), TensorAccess::Write);
                assert_eq!(output.alias(), &AliasPolicy::NoAlias);
            }
            assert_eq!(c.resources.scratch, ResourcePresenceRequirement::Required);
            assert_eq!(c.resources, o.resources);
            assert_eq!(c.resources.binding, ResourcePresenceRequirement::Optional);
            for scratch in [false, true] {
                for binding in [false, true] {
                    for persistent in [false, true] {
                        let allowed = c.resources.scratch.accepts(scratch)
                            && c.resources.binding.accepts(binding)
                            && c.resources.persistent.accepts(persistent);
                        assert_eq!(allowed, scratch && !persistent);
                    }
                }
            }
            assert_ne!(c.fingerprint().unwrap(), o.fingerprint().unwrap());
        }
    }
}
