use super::*;
use std::sync::Weak;

pub(super) struct ActivePrefixInput {
    pub(super) request_id: RequestId,
    pub(super) tokens: Vec<u32>,
}

#[derive(Debug)]
pub(super) struct ProtectedPrefix {
    pub(super) request_id: RequestId,
    pub(super) prefix_tokens: usize,
}

pub(super) enum CaptureEviction<C> {
    Evicted(C),
    Protected(Vec<ProtectedPrefix>),
    SharedFallbackProtected(Vec<usize>),
    Unavailable,
}

/// Only token metadata survives this snapshot. No sequence or checkpoint owner
/// is retained across the following capture/maintenance attempt.
pub(super) fn active_prefix_inputs<R: DeviceRuntime>(
    registry: &Mutex<VNextSequenceRegistry<R>>,
    capturing: &Arc<VNextSequence<R>>,
) -> Vec<ActivePrefixInput> {
    let candidates: Vec<(Weak<VNextSequence<R>>, Option<Weak<VNextPrefillSlot<R>>>)> = {
        let registry = registry.lock();
        let mut candidates = registry
            .active
            .values()
            .map(|sequence| (Arc::downgrade(sequence), None))
            .collect::<Vec<_>>();
        // Existing registry -> slot lock order. In particular, never take a
        // tokens or operation lock while holding either of these locks.
        for slot in registry.prefills.values() {
            if slot.cancelled.load(Ordering::Acquire) {
                continue;
            }
            match &*slot.state.lock() {
                VNextPrefillSlotState::Ready(sequence)
                | VNextPrefillSlotState::Executing(sequence) => {
                    candidates.push((Arc::downgrade(sequence), Some(Arc::downgrade(slot))));
                }
                // No admitted sequence/input exists in these states.
                VNextPrefillSlotState::Probing
                | VNextPrefillSlotState::Deferred { .. }
                | VNextPrefillSlotState::Terminal => {}
            }
        }
        candidates
    };
    candidates
        .into_iter()
        .filter_map(|(sequence, slot)| {
            let sequence = sequence.upgrade()?;
            if Arc::ptr_eq(&sequence, capturing)
                || sequence.request_origin != ExecutorRequestOrigin::Product
                || !sequence.active.load(Ordering::Acquire)
            {
                return None;
            }
            // A normally completed prefill can move this same live sequence
            // into registry.active and retire its slot after the snapshot.
            // Slot disappearance alone therefore cannot discard coverage.
            let slot = slot.and_then(|slot| slot.upgrade());
            let count = usize::try_from(sequence.product_prompt_tokens).ok()?;
            if count == 0 {
                return None;
            }
            let tokens = sequence.tokens.lock().get(..count)?.to_vec();
            // Cancellation/retirement racing the metadata copy must not leave
            // an already-terminal input in this snapshot. A later race can at
            // worst skip optional capture; foreground eviction is unrestricted.
            if !sequence.active.load(Ordering::Acquire)
                || slot.is_some_and(|slot| slot.cancelled.load(Ordering::Acquire))
            {
                return None;
            }
            Some(ActivePrefixInput {
                request_id: sequence.request_id().clone(),
                tokens,
            })
        })
        .collect()
}

impl<C> PrefixIndex<C> {
    /// A growth chain is one branch, irrespective of how many snapshots it
    /// contains. A shared fallback must have two actually observed, divergent
    /// product prompts that both permit restoration at this retained boundary.
    /// This is eviction preference only, never a new restore authority or pin.
    pub(super) fn shared_fallbacks(
        &self,
        inputs: &[ActivePrefixInput],
        layout: &SequenceCheckpointLayout,
    ) -> BTreeSet<usize> {
        if layout.input_dependency() != CheckpointInputDependency::ExactTokenPrefix {
            return BTreeSet::new();
        }
        let observed = self
            .entries
            .iter()
            .filter_map(Entry::original_prompt)
            .chain(inputs.iter().map(|input| input.tokens.as_slice()))
            .collect::<Vec<_>>();
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !entry.is_generated_head() && entry.original_prompt().is_some())
            .filter(|(_, entry)| {
                let mut deepest: Option<&[u32]> = None;
                for &input in &observed {
                    if !entry.matches_restore(input, false, &|boundary| {
                        layout.permits_suffix(boundary as u64, input.len() as u64)
                    }) {
                        continue;
                    }
                    match deepest {
                        None => deepest = Some(input),
                        Some(previous) if input.starts_with(previous) => deepest = Some(input),
                        Some(previous) if !previous.starts_with(input) => return true,
                        Some(_) => {}
                    }
                }
                false
            })
            .map(|(index, _)| index)
            .collect()
    }

    pub(super) fn evict_with_active_coverage(
        &mut self,
        purpose: PrefixEvictionPurpose,
        inputs: &[ActivePrefixInput],
        layout: &SequenceCheckpointLayout,
    ) -> CaptureEviction<C> {
        if matches!(purpose, PrefixEvictionPurpose::Foreground) {
            return self
                .evict_for_layout(purpose, layout)
                .map_or(CaptureEviction::Unavailable, CaptureEviction::Evicted);
        }
        let entire_input = layout.input_dependency() == CheckpointInputDependency::EntireTokenInput;
        let mut protected = BTreeSet::new();
        let mut coverage = Vec::new();
        for input in inputs {
            // Identical eligibility and tie ordering to longest(). Protect the
            // deepest legal checkpoint, even if a shorter common root exists.
            if let Some((index, entry)) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| {
                    entry.matches_restore(&input.tokens, entire_input, &|boundary| {
                        layout.permits_suffix(boundary as u64, input.tokens.len() as u64)
                    })
                })
                .max_by_key(|(_, entry)| entry.prefix.len())
            {
                protected.insert(index);
                coverage.push(ProtectedPrefix {
                    request_id: input.request_id.clone(),
                    prefix_tokens: entry.prefix.len(),
                });
            }
        }
        let shared = self.shared_fallbacks(inputs, layout);
        if matches!(purpose, PrefixEvictionPurpose::PromptCapture) {
            protected.extend(&shared);
        }
        if let Some(checkpoint) = self.evict_unprotected(purpose, &protected, &shared) {
            return CaptureEviction::Evicted(checkpoint);
        }
        if matches!(purpose, PrefixEvictionPurpose::PromptCapture) && !shared.is_empty() {
            return CaptureEviction::SharedFallbackProtected(
                shared
                    .iter()
                    .map(|index| self.entries[*index].prefix.len())
                    .collect(),
            );
        }
        if protected.iter().any(|index| {
            !matches!(purpose, PrefixEvictionPurpose::GeneratedCapture)
                || self.entries[*index].is_generated_head()
        }) {
            CaptureEviction::Protected(coverage)
        } else {
            CaptureEviction::Unavailable
        }
    }
}
