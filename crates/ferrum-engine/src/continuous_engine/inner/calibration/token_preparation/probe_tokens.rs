//! Cold, bounded search of the tokenizer's actual byte surfaces. This discovers
//! declarations only; it cannot install a prefix or grant sample authority.
use super::CalibrationPrefixTokensV1;
use ferrum_interfaces::Tokenizer;
use ferrum_types::TokenId;
use std::{collections::BTreeMap, num::NonZeroUsize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationPrefixTokenBudgetV1 {
    /// IDs examined, including excluded, special and unknown IDs.
    pub maximum_token_ids: NonZeroUsize,
    /// Maximum accepted tokenizer raw-byte bound (and scratch allocation).
    pub maximum_token_bytes: NonZeroUsize,
    /// Actual scan bytes plus conservative original-validator read reservations.
    pub maximum_total_token_bytes: NonZeroUsize,
    pub maximum_prefix_tokens: NonZeroUsize,
    /// Search transitions plus reserved final-validator transitions.
    pub maximum_utf8_transitions: NonZeroUsize,
    /// Total states in both BFS frontiers while one is being replaced.
    pub maximum_search_states: NonZeroUsize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CalibrationPrefixTokenDiscoveryAuditV1 {
    pub token_ids_examined: usize,
    pub token_bytes_charged: usize,
    pub utf8_transitions: usize,
    pub peak_search_states: usize,
    pub vocabulary_scan_complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationPrefixTokenUnavailableReasonV1 {
    MissingTokenizerIdentity,
    MissingRawByteCapability,
    RawByteBoundExceedsBudget,
    InvalidExcludedIds,
    VocabularyExceedsTokenId,
    NoRoomForNormalSuffix,
    TokenIdBudget,
    TokenByteBudget,
    TransitionBudget,
    SearchStateBudget,
    PrefixLengthBudget,
    TokenizerReadFailure,
    TokenizerContractViolation,
    OriginalValidationFailed,
    AllocationFailed,
    /// Search ended within its declarations. This is NOT proof that pending
    /// UTF-8 is unreachable under every token trajectory or tokenizer policy.
    NoValidatedPairWithinSearch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredCalibrationPrefixTokensV1 {
    pub clean: CalibrationPrefixTokensV1,
    pub pending: CalibrationPrefixTokensV1,
    pub audit: CalibrationPrefixTokenDiscoveryAuditV1,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationPrefixTokenDiscoveryV1 {
    Found(DiscoveredCalibrationPrefixTokensV1),
    CoverageUnavailable {
        reason: CalibrationPrefixTokenUnavailableReasonV1,
        audit: CalibrationPrefixTokenDiscoveryAuditV1,
    },
}
use CalibrationPrefixTokenDiscoveryV1::{CoverageUnavailable, Found};
use CalibrationPrefixTokenUnavailableReasonV1 as U;

struct Search {
    budget: CalibrationPrefixTokenBudgetV1,
    audit: CalibrationPrefixTokenDiscoveryAuditV1,
}
impl Search {
    fn unavailable(&self, reason: U) -> CalibrationPrefixTokenDiscoveryV1 {
        CoverageUnavailable {
            reason,
            audit: self.audit,
        }
    }
    fn advance(&mut self, pending: &[u8], bytes: &[u8]) -> Result<Option<Vec<u8>>, U> {
        if self.audit.utf8_transitions == self.budget.maximum_utf8_transitions.get() {
            return Err(U::TransitionBudget);
        }
        self.audit.utf8_transitions += 1;
        Ok(crate::continuous_engine::advance_pending_utf8_fragment(pending, bytes).ok())
    }
    fn charge(&mut self, bytes: usize) -> Result<(), U> {
        let total = self
            .audit
            .token_bytes_charged
            .checked_add(bytes)
            .ok_or(U::TokenByteBudget)?;
        if total > self.budget.maximum_total_token_bytes.get() {
            return Err(U::TokenByteBudget);
        }
        self.audit.token_bytes_charged = total;
        Ok(())
    }
    fn finish(
        &mut self,
        tokenizer: &dyn Tokenizer,
        identity: [u8; 32],
        raw_bound: usize,
        clean: Vec<TokenId>,
        pending: Vec<TokenId>,
        maximum_output: NonZeroUsize,
    ) -> CalibrationPrefixTokenDiscoveryV1 {
        // The existing validator rereads the actual tokenizer. Reserve its full
        // proven bound before calling it, so final verification is also bounded.
        let Some(bytes) = clean
            .len()
            .checked_add(pending.len())
            .and_then(|n| n.checked_mul(raw_bound))
        else {
            return self.unavailable(U::TokenByteBudget);
        };
        if let Err(reason) = self.charge(bytes) {
            return self.unavailable(reason);
        }
        let validations = clean.len() + pending.len();
        if validations
            > self
                .budget
                .maximum_utf8_transitions
                .get()
                .saturating_sub(self.audit.utf8_transitions)
        {
            return self.unavailable(U::TransitionBudget);
        }
        self.audit.utf8_transitions += validations;
        let make = |tokens: Vec<TokenId>| CalibrationPrefixTokensV1 {
            tokenizer_policy_sha256: identity,
            release_generated: tokens.len(),
            token_ids: tokens,
        };
        let clean = make(clean);
        let pending = make(pending);
        let (Ok(c), Ok(p)) = (
            clean.validate_for_tokenizer(tokenizer, maximum_output),
            pending.validate_for_tokenizer(tokenizer, maximum_output),
        ) else {
            return self.unavailable(U::OriginalValidationFailed);
        };
        if !c.expected_pending_bytes().is_empty() || p.expected_pending_bytes().is_empty() {
            return self.unavailable(U::OriginalValidationFailed);
        }
        Found(DiscoveredCalibrationPrefixTokensV1 {
            clean,
            pending,
            audit: self.audit,
        })
    }
}

/// Deterministic token-ID order, then breadth-first UTF-8 state search. Budgets
/// and sorted unique EOS/stop exclusions are fixed by the caller before timing.
/// Both declarations pass the original validator; their lengths may differ.
/// Space is bounded by scan bytes + O(scanned IDs) + O(states * prefix length).
/// No ordinary decode/encode fallback, token-text heuristic, model name or ID.
pub fn discover_prefix_tokens(
    tokenizer: &dyn Tokenizer,
    maximum_output: NonZeroUsize,
    excluded_token_ids: &[TokenId],
    budget: CalibrationPrefixTokenBudgetV1,
) -> CalibrationPrefixTokenDiscoveryV1 {
    discover_prefix_tokens_impl(tokenizer, maximum_output, excluded_token_ids, budget, false)
}

/// Original bounded search with equal token-count release frontiers. This is
/// required for heterogeneous prefix rows in a single declared cohort. It does
/// not pad tokens, reinterpret bytes, or change the legacy discovery entrypoint.
pub(crate) fn discover_aligned_prefix_tokens(
    tokenizer: &dyn Tokenizer,
    maximum_output: NonZeroUsize,
    excluded_token_ids: &[TokenId],
    budget: CalibrationPrefixTokenBudgetV1,
) -> CalibrationPrefixTokenDiscoveryV1 {
    discover_prefix_tokens_impl(tokenizer, maximum_output, excluded_token_ids, budget, true)
}

fn discover_prefix_tokens_impl(
    tokenizer: &dyn Tokenizer,
    maximum_output: NonZeroUsize,
    excluded_token_ids: &[TokenId],
    budget: CalibrationPrefixTokenBudgetV1,
    equal_release: bool,
) -> CalibrationPrefixTokenDiscoveryV1 {
    let mut search = Search {
        budget,
        audit: Default::default(),
    };
    if tokenizer.special_tokens().extra_eos_tokens.len() > budget.maximum_token_ids.get()
        || excluded_token_ids.len() > budget.maximum_token_ids.get()
        || excluded_token_ids.windows(2).any(|p| p[0] >= p[1])
    {
        return search.unavailable(U::InvalidExcludedIds);
    }
    let Some(identity) = tokenizer.host_output_policy_identity() else {
        return search.unavailable(U::MissingTokenizerIdentity);
    };
    let Some(bound) = tokenizer.bounded_token_bytes_bound() else {
        return search.unavailable(U::MissingRawByteCapability);
    };
    let raw_bound = bound.get();
    if raw_bound > budget.maximum_token_bytes.get() {
        return search.unavailable(U::RawByteBoundExceedsBudget);
    }
    let vocabulary = tokenizer.vocab_size();
    if vocabulary > u32::MAX as usize {
        return search.unavailable(U::VocabularyExceedsTokenId);
    }
    let maximum_prefix = budget
        .maximum_prefix_tokens
        .get()
        .min(maximum_output.get() - 1);
    if maximum_prefix == 0 {
        return search.unavailable(U::NoRoomForNormalSuffix);
    }
    let mut scratch = Vec::new();
    if scratch.try_reserve_exact(raw_bound).is_err() {
        return search.unavailable(U::AllocationFailed);
    }
    scratch.resize(raw_bound, 0);
    let mut alphabet = Vec::<(TokenId, Vec<u8>)>::new();
    let mut states = BTreeMap::<Vec<u8>, Vec<TokenId>>::new();
    let mut clean = None;
    let mut pending = None;
    let mut exhausted = U::NoValidatedPairWithinSearch;
    let mut scan_exhausted = false;
    for index in 0..vocabulary {
        if search.audit.token_ids_examined == budget.maximum_token_ids.get() {
            exhausted = U::TokenIdBudget;
            scan_exhausted = true;
            break;
        }
        search.audit.token_ids_examined += 1;
        let token = TokenId::new(index as u32);
        if excluded_token_ids.binary_search(&token).is_ok()
            || tokenizer.is_special_token(token)
            || tokenizer.special_tokens().extra_eos_tokens.contains(&token)
        {
            continue;
        }
        // Check a proven worst case BEFORE invoking the bounded byte producer.
        if raw_bound
            > budget
                .maximum_total_token_bytes
                .get()
                .saturating_sub(search.audit.token_bytes_charged)
        {
            exhausted = U::TokenByteBudget;
            scan_exhausted = true;
            break;
        }
        let written = match tokenizer.token_bytes_bounded_into(token, &mut scratch) {
            Ok(Some(n)) if n <= raw_bound => n,
            Ok(None) => continue,
            Ok(Some(_)) => return search.unavailable(U::TokenizerContractViolation),
            Err(_) => return search.unavailable(U::TokenizerReadFailure),
        };
        if let Err(reason) = search.charge(written) {
            return search.unavailable(reason);
        }
        let bytes = &scratch[..written];
        let next = match search.advance(&[], bytes) {
            Ok(v) => v,
            Err(reason) => return search.unavailable(reason),
        };
        if let Some(next) = next {
            if next.is_empty() && clean.is_none() {
                clean = Some(vec![token]);
            }
            if !next.is_empty() && pending.is_none() {
                pending = Some(vec![token]);
            }
            if clean.is_some() && pending.is_some() {
                search.audit.vocabulary_scan_complete = index + 1 == vocabulary;
                return search.finish(
                    tokenizer,
                    identity,
                    raw_bound,
                    clean.unwrap(),
                    pending.unwrap(),
                    maximum_output,
                );
            }
            if !states.contains_key(&next) {
                if states.len() == budget.maximum_search_states.get() {
                    return search.unavailable(U::SearchStateBudget);
                }
                states.insert(next, vec![token]);
                search.audit.peak_search_states = search.audit.peak_search_states.max(states.len());
            }
        }
        if alphabet.try_reserve(1).is_err() {
            return search.unavailable(U::AllocationFailed);
        }
        alphabet.push((token, bytes.to_vec()));
    }
    search.audit.vocabulary_scan_complete = !scan_exhausted;
    if pending.is_none() {
        return search.unavailable(exhausted);
    }
    // A clean prefix leaves the exact same empty UTF-8 state. Only nonempty
    // states can reveal a clean trajectory absent from the single-token scan.
    for _depth in 2..=maximum_prefix {
        if equal_release {
            // Both candidates must come from this exact BFS depth. Every path
            // is still an original tokenizer trajectory, including a clean
            // path that leaves an empty state before another real token.
            clean = None;
            pending = None;
        }
        let mut next_states = BTreeMap::<Vec<u8>, Vec<TokenId>>::new();
        for (state, path) in &states {
            for (token, bytes) in &alphabet {
                let next = match search.advance(state, bytes) {
                    Ok(Some(v)) => v,
                    Ok(None) => continue,
                    Err(reason) => return search.unavailable(reason),
                };
                let mut tokens = path.clone();
                tokens.push(*token);
                if next.is_empty() && clean.is_none() {
                    clean = Some(tokens.clone());
                }
                if !next.is_empty() && pending.is_none() {
                    pending = Some(tokens.clone());
                }
                if clean.is_some() && pending.is_some() {
                    return search.finish(
                        tokenizer,
                        identity,
                        raw_bound,
                        clean.unwrap(),
                        pending.unwrap(),
                        maximum_output,
                    );
                }
                if !next_states.contains_key(&next) {
                    if states.len().saturating_add(next_states.len())
                        >= budget.maximum_search_states.get()
                    {
                        return search.unavailable(U::SearchStateBudget);
                    }
                    next_states.insert(next, tokens);
                    search.audit.peak_search_states = search
                        .audit
                        .peak_search_states
                        .max(states.len() + next_states.len());
                }
            }
        }
        states = next_states;
        if states.is_empty() {
            break;
        }
    }
    if exhausted == U::NoValidatedPairWithinSearch && !states.is_empty() {
        exhausted = U::PrefixLengthBudget;
    }
    search.unavailable(exhausted)
}

#[cfg(test)]
mod tests;
