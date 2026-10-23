//! Policy accounting is distinct from actual executable token work.
use super::PrefillPhaseCap;
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillPackingMode {
    Refill,
    ReservePolicyChunk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillPacking {
    Refill,
    /// Reserve the live policy quantum through the target's prompt fence,
    /// bounded by the common baseline's cap rather than the new progress cap.
    /// Without a declared quantum, retain request-bounded natural debit.
    ReservePolicyChunk {
        prior_limit: Option<NonZeroUsize>,
    },
}

impl PrefillPacking {
    pub(super) fn after_selection(
        self,
        prior_limit: Option<NonZeroUsize>,
        mode: PrefillPackingMode,
    ) -> Self {
        match (self, mode) {
            (Self::Refill, PrefillPackingMode::ReservePolicyChunk) => {
                Self::ReservePolicyChunk { prior_limit }
            }
            // A later choice cannot forget an existing packing commitment.
            _ => self,
        }
    }

    pub(super) fn reserves_nominal(self) -> bool {
        matches!(self, Self::ReservePolicyChunk { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefillWorkBudget {
    pub(super) execution_tokens: usize,
    pub(super) nominal_charge: usize,
}

impl PrefillWorkBudget {
    pub(super) fn for_cap(
        natural_tokens: usize,
        policy_quantum: Option<usize>,
        cap: Option<NonZeroUsize>,
        packing: PrefillPacking,
    ) -> Self {
        let available =
            policy_quantum.map_or(natural_tokens, |quantum| natural_tokens.min(quantum));
        let execution_tokens = cap.map_or(available, |cap| available.min(cap.get()));
        if execution_tokens == 0 {
            return Self {
                execution_tokens: 0,
                nominal_charge: 0,
            };
        }
        let nominal_charge = match packing {
            PrefillPacking::Refill => execution_tokens,
            PrefillPacking::ReservePolicyChunk { prior_limit } => {
                let nominal = policy_quantum.unwrap_or(natural_tokens);
                prior_limit.map_or(nominal, |cap| nominal.min(cap.get()))
            }
        };
        debug_assert!(execution_tokens <= nominal_charge);
        Self {
            execution_tokens,
            nominal_charge,
        }
    }
}

impl PrefillPhaseCap {
    pub(super) fn selected(
        prior: Option<Self>,
        limit: NonZeroUsize,
        prompt_tokens: usize,
        mode: PrefillPackingMode,
    ) -> Self {
        Self {
            limit: prior.map_or(limit, |old| old.limit.min(limit)),
            prompt_tokens,
            packing: prior
                .map_or(PrefillPacking::Refill, |old| old.packing)
                .after_selection(prior.map(|old| old.limit), mode),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positive(tokens: usize) -> NonZeroUsize {
        NonZeroUsize::new(tokens).unwrap()
    }

    #[test]
    fn nominal_reservation_preserves_prior_cap_and_never_reports_debit_as_work() {
        let old = PrefillPhaseCap::selected(None, positive(32), 256, PrefillPackingMode::Refill);
        let reserved = PrefillPhaseCap::selected(
            Some(old),
            positive(8),
            256,
            PrefillPackingMode::ReservePolicyChunk,
        );
        assert_eq!(
            PrefillWorkBudget::for_cap(128, Some(128), Some(reserved.limit), reserved.packing),
            PrefillWorkBudget {
                execution_tokens: 8,
                nominal_charge: 32
            }
        );
        // Real prompt exhaustion does not release the selected quantum inside
        // this wave; a smaller live budget/ceiling does constrain it.
        assert_eq!(
            PrefillWorkBudget::for_cap(3, Some(128), Some(reserved.limit), reserved.packing),
            PrefillWorkBudget {
                execution_tokens: 3,
                nominal_charge: 32
            }
        );
        assert_eq!(
            PrefillWorkBudget::for_cap(128, Some(3), Some(reserved.limit), reserved.packing),
            PrefillWorkBudget {
                execution_tokens: 3,
                nominal_charge: 3
            }
        );
    }

    #[test]
    fn nominal_reservation_survives_later_refill_action_and_tighter_limit() {
        let reserved = PrefillPhaseCap::selected(
            None,
            positive(32),
            256,
            PrefillPackingMode::ReservePolicyChunk,
        );
        let tighter =
            PrefillPhaseCap::selected(Some(reserved), positive(8), 256, PrefillPackingMode::Refill);
        assert_eq!(tighter.packing, reserved.packing);
        assert_eq!(
            PrefillWorkBudget::for_cap(128, Some(128), Some(tighter.limit), tighter.packing),
            PrefillWorkBudget {
                execution_tokens: 8,
                nominal_charge: 128
            }
        );
        let wider = PrefillPhaseCap::selected(
            Some(tighter),
            positive(64),
            256,
            PrefillPackingMode::ReservePolicyChunk,
        );
        assert_eq!(wider, tighter);
    }

    #[test]
    fn nominal_reservation_without_policy_chunk_preserves_request_bounded_debit() {
        let cap = PrefillPhaseCap::selected(
            None,
            positive(8),
            256,
            PrefillPackingMode::ReservePolicyChunk,
        );
        assert_eq!(
            PrefillWorkBudget::for_cap(3, None, Some(cap.limit), cap.packing),
            PrefillWorkBudget {
                execution_tokens: 3,
                nominal_charge: 3
            }
        );
        assert_eq!(
            PrefillWorkBudget::for_cap(32, None, Some(cap.limit), cap.packing),
            PrefillWorkBudget {
                execution_tokens: 8,
                nominal_charge: 32
            }
        );
        assert_eq!(
            PrefillWorkBudget::for_cap(0, Some(128), Some(cap.limit), cap.packing),
            PrefillWorkBudget {
                execution_tokens: 0,
                nominal_charge: 0
            }
        );
    }
}
