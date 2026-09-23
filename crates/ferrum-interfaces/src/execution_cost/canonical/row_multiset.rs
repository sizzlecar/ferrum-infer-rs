//! New evidence beside V1: neither its hash nor physical execution order changes.
use super::*;
use crate::execution_cost::{HostRowRoleV2, HOST_ROW_MULTISET_FEATURE_SCHEMA_V2};

pub(super) fn static_row(
    row: CanonicalCostRow,
    numeric: CostRowNumericFeatures,
) -> Option<HostRowStaticCostFeaturesV2> {
    let host = row.host_features?;
    if !host.supports_empirical_plain_text_content() {
        return None;
    }
    let role = match row.work {
        ActualRowWork::Prefill { .. } => HostRowRoleV2::Prefill,
        ActualRowWork::Decode { .. } => HostRowRoleV2::Decode,
        _ => return None,
    };
    let mut hash = Sha256::new();
    bytes(&mut hash, b"ferrum.canonical-wave.empirical-row-static.v2");
    number(&mut hash, u64::from(role == HostRowRoleV2::Prefill));
    bytes(&mut hash, &host.policy.categorical_signature);
    number(&mut hash, host.policy.decoder_text_bytes_per_token);
    number(&mut hash, host.policy.decoder_scratch_bytes_per_token);
    number(&mut hash, host.policy.raw_token_bytes_bound);
    number(
        &mut hash,
        u64::from(host.state.generated_tokens_before == 0),
    );
    number(&mut hash, u64::from(row.mask_upload_required));
    number(
        &mut hash,
        u64::from(
            numeric.decoded_prefix_tokens > numeric.generated_tokens_before
                && numeric.decoded_prefix_tokens == numeric.maximum_output_tokens,
        ),
    );
    match row.output {
        CostRowOutput::Prefill { final_logits } => {
            number(&mut hash, u64::from(final_logits));
        }
        CostRowOutput::Decode {
            repetition_penalty_bits,
            ..
        } => {
            number(&mut hash, 2);
            number(&mut hash, u64::from(repetition_penalty_bits));
        }
    }
    Some(HostRowStaticCostFeaturesV2 {
        role,
        categorical_signature: hash.finalize().into(),
    })
}

pub(super) fn finish(
    rows: Vec<HostRowStaticCostFeaturesV2>,
    product: CostProductOutput,
    route: CoreReadbackRoute,
    order: ActualWaveRowOrder,
) -> HostRowMultisetCostFeaturesV2 {
    let mut hash = Sha256::new();
    bytes(&mut hash, b"ferrum.canonical-wave.empirical-row-wave.v2");
    number(&mut hash, u64::from(HOST_ROW_MULTISET_FEATURE_SCHEMA_V2));
    number(
        &mut hash,
        u64::from(product == CostProductOutput::GreedyToken),
    );
    number(
        &mut hash,
        match route {
            CoreReadbackRoute::SubmissionStaged => 0,
            CoreReadbackRoute::HostSynchronized => 1,
            CoreReadbackRoute::NoReadback => 2,
            CoreReadbackRoute::SubmissionFallbackSynchronized => 3,
            CoreReadbackRoute::Unknown => unreachable!("numeric readback checked"),
        },
    );
    number(
        &mut hash,
        u64::from(order == ActualWaveRowOrder::IndependentRows),
    );
    HostRowMultisetCostFeaturesV2 {
        schema_version: HOST_ROW_MULTISET_FEATURE_SCHEMA_V2,
        wave_policy_signature: hash.finalize().into(),
        rows,
    }
}
