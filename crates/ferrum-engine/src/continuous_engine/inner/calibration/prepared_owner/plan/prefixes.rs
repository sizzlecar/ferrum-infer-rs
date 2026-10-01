use super::*;
use ferrum_interfaces::Tokenizer;
use ferrum_types::TokenId;

#[derive(Clone)]
pub(super) struct PrefixPair {
    pub clean: StructuredPrefixSlotV5,
    pub pending: StructuredPrefixSlotV5,
}

pub(super) fn discover(
    tokenizer: &dyn Tokenizer,
    templates: &[AutomaticCostProbeTemplate],
    maximum_output: NonZeroUsize,
    settings: &SloAutomaticCostProbeSettingsV1,
) -> Result<(PrefixPair, CalibrationPrefixTokenDiscoveryAuditV1)> {
    let b = &settings.token_discovery;
    let mut excluded = Vec::<TokenId>::new();
    for template in templates {
        // Excluding every token in each original user stop is conservative.
        // The actual installation still checks complete decoded stop prefixes,
        // model masks and runtime sampler constraints on every forced commit.
        for stop in &template.resolved_request()?.sampling_params.stop_sequences {
            for token in tokenizer.encode(stop, false)? {
                if excluded.len() >= b.maximum_token_ids.get() {
                    return Err(error(
                        "probe stop-token exclusions exceed discovery capacity",
                    ));
                }
                excluded.push(token);
            }
        }
    }
    excluded.sort_unstable();
    excluded.dedup();
    let budget = CalibrationPrefixTokenBudgetV1 {
        maximum_token_ids: b.maximum_token_ids,
        maximum_token_bytes: b.maximum_token_bytes,
        maximum_total_token_bytes: b.maximum_total_token_bytes,
        maximum_prefix_tokens: b.maximum_prefix_tokens,
        maximum_utf8_transitions: b.maximum_utf8_transitions,
        maximum_search_states: b.maximum_search_states,
    };
    let found = match discover_aligned_prefix_tokens(tokenizer, maximum_output, &excluded, budget) {
        CalibrationPrefixTokenDiscoveryV1::Found(v) => v,
        CalibrationPrefixTokenDiscoveryV1::CoverageUnavailable { reason, audit } => {
            return Err(error(format!("automatic cost probe prefix coverage unavailable: {reason:?}; examined={} bytes={} transitions={}",
                audit.token_ids_examined, audit.token_bytes_charged, audit.utf8_transitions)));
        }
    };
    let raw_bound = tokenizer
        .bounded_token_bytes_bound()
        .ok_or_else(|| error("probe tokenizer lost its raw-byte capability"))?
        .get();
    let extra = found
        .clean
        .token_ids
        .len()
        .checked_add(found.pending.token_ids.len())
        .and_then(|n| n.checked_mul(raw_bound))
        .ok_or_else(|| error("probe prefix byte reservation overflow"))?;
    let mut audit = found.audit;
    audit.token_bytes_charged = audit
        .token_bytes_charged
        .checked_add(extra)
        .filter(|n| *n <= b.maximum_total_token_bytes.get())
        .ok_or_else(|| error("probe frozen prefix bytes exceed the same discovery budget"))?;
    let mut scratch = vec![0; raw_bound];
    let mut freeze = |plan: CalibrationPrefixTokensV1| -> Result<StructuredPrefixSlotV5> {
        let mut token_bytes = Vec::with_capacity(plan.token_ids.len());
        for &token in &plan.token_ids {
            let n = tokenizer
                .token_bytes_bounded_into(token, &mut scratch)
                .map_err(|_| error("probe tokenizer prefix read failed"))?
                .ok_or_else(|| error("probe tokenizer prefix token disappeared"))?;
            token_bytes.push(
                scratch
                    .get(..n)
                    .ok_or_else(|| error("probe tokenizer byte bound violated"))?
                    .to_vec(),
            );
        }
        Ok(StructuredPrefixSlotV5 {
            tokenizer_policy_sha256: plan.tokenizer_policy_sha256,
            token_ids: plan.token_ids,
            token_bytes,
        })
    };
    let clean = freeze(found.clean)?;
    let pending = freeze(found.pending)?;
    if !clean
        .expected_pending()
        .map_err(|e| error(format!("probe clean prefix: {e:?}")))?
        .is_empty()
        || pending
            .expected_pending()
            .map_err(|e| error(format!("probe pending prefix: {e:?}")))?
            .is_empty()
    {
        return Err(error(
            "probe tokenizer changed its declared UTF-8 trajectory",
        ));
    }
    Ok((PrefixPair { clean, pending }, audit))
}
