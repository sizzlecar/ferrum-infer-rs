//! Weak content evidence preserves actual mask lifetime and exact slot eviction.
use super::*;

pub(super) fn capture(
    ledger: &VNextProductTokenMaskResidency,
    eligible: bool,
    budget: &mut dyn ResourcePlanningBudget,
) -> std::result::Result<ProductTokenMaskResidencySnapshot, U> {
    if ledger.entries.len() > MAX_PRODUCT_TOKEN_MASK_SLOT_CACHE_ENTRIES {
        return Err(U::Capacity);
    }
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(ledger.entries.len())
        .map_err(|_| U::Capacity)?;
    for (&(_, participant_index), entry) in &ledger.entries {
        if !budget.has_budget() {
            return Err(U::BudgetExhausted);
        }
        let identity = match &entry.identity {
            VNextProductTokenMaskSlotIdentity::LaneStable(identity) => identity.as_ref().clone(),
            #[cfg(test)]
            VNextProductTokenMaskSlotIdentity::Test(_) => return Err(U::InvalidInput),
        };
        let content = match &entry.content {
            VNextResidentProductTokenMaskContent::AllValid { vocabulary_size } => {
                ProductTokenMaskContent::AllValid {
                    vocabulary_size: u64::try_from(*vocabulary_size).map_err(|_| U::Capacity)?,
                }
            }
            VNextResidentProductTokenMaskContent::Selection {
                vocabulary_size,
                fingerprint,
                source_len,
                valid_token_mask,
            } => ProductTokenMaskContent::capture_selection(
                u64::try_from(*vocabulary_size).map_err(|_| U::Capacity)?,
                *fingerprint,
                *source_len,
                valid_token_mask,
            )?,
        };
        entries.push(ProductTokenMaskResidencyEntry::with_content(
            identity,
            participant_index,
            content,
        ));
    }
    ProductTokenMaskResidencySnapshot::new(
        eligible,
        MAX_PRODUCT_TOKEN_MASK_SLOT_CACHE_ENTRIES,
        entries,
        budget,
    )
}

/// Use the actual mode/policy selector; do not infer an all-valid mask from a
/// greedy flag or a fingerprint. Evidence borrows source lifetime weakly.
pub(super) fn requested(
    role: &VNextParticipantOutputRole,
    mode: VNextProductOutputMode,
    vocabulary_size: usize,
) -> std::result::Result<ProductTokenMaskContent, U> {
    Ok(
        match VNextProductTokenMaskContent::from_policy(role.logits_policy(), mode, vocabulary_size)
        {
            VNextProductTokenMaskContent::AllValid { vocabulary_size } => {
                ProductTokenMaskContent::AllValid {
                    vocabulary_size: u64::try_from(vocabulary_size).map_err(|_| U::Capacity)?,
                }
            }
            VNextProductTokenMaskContent::Selection {
                vocabulary_size,
                fingerprint,
                valid_token_mask,
            } => ProductTokenMaskContent::selection(
                u64::try_from(vocabulary_size).map_err(|_| U::Capacity)?,
                fingerprint,
                &valid_token_mask,
            ),
        },
    )
}
