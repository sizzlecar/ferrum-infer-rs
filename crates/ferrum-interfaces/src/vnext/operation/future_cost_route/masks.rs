//! Bounded numeric copy of the product's slot-local token-mask ledger.
//! A successful simulated wave publishes exactly the entries uploaded by that
//! wave. It never manufactures residency from a bucket name or a request ID.
use super::{core::poll, ExecutionCostRouteUnknown as U};
use crate::execution_cost::MAX_COST_ROWS;
use crate::vnext::{
    AllocationLifetime, LaneStableArenaSlotIdentity, NodeId, ResourcePlanningBudget,
};
use std::sync::{Arc, Weak};

/// Read-only content evidence. Weak references deliberately do not extend the
/// actual ledger's source lifetime: doing so could change a real cache miss
/// into a hit. A source which expires after capture invalidates the view.
#[derive(Debug, Clone)]
pub enum ProductTokenMaskContent {
    AllValid {
        vocabulary_size: u64,
    },
    Selection(ProductTokenMaskSelection),
    /// A selection source was already dead when the live ledger was read.
    ExpiredSelection,
    /// Legacy opaque selection evidence cannot establish a selection hit/miss.
    UnavailableSelection,
}

#[derive(Debug, Clone)]
pub struct ProductTokenMaskSelection {
    vocabulary_size: u64,
    fingerprint: u64,
    source_len: usize,
    source: Weak<[i8]>,
}

impl PartialEq for ProductTokenMaskSelection {
    fn eq(&self, other: &Self) -> bool {
        self.vocabulary_size == other.vocabulary_size
            && self.fingerprint == other.fingerprint
            && self.source_len == other.source_len
            && Weak::ptr_eq(&self.source, &other.source)
    }
}
impl Eq for ProductTokenMaskSelection {}

impl PartialEq for ProductTokenMaskContent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::AllValid { vocabulary_size: a }, Self::AllValid { vocabulary_size: b }) => {
                a == b
            }
            (Self::Selection(a), Self::Selection(b)) => a == b,
            (Self::ExpiredSelection, Self::ExpiredSelection)
            | (Self::UnavailableSelection, Self::UnavailableSelection) => true,
            _ => false,
        }
    }
}
impl Eq for ProductTokenMaskContent {}

/// The actual executor uses the unchanged Arc/full-slice fast path. A planning
/// read checks the identical raw contents in bounded, interruptible chunks.
/// Normalized equality is insufficient: the real ledger also compares source
/// length and every original i8, even beyond the uploaded vocabulary range.
pub fn selection_mask_bytes_match(
    resident: &Arc<[i8]>,
    requested: &Arc<[i8]>,
    budget: Option<&mut dyn ResourcePlanningBudget>,
) -> Result<bool, U> {
    let Some(budget) = budget else {
        return Ok(Arc::ptr_eq(resident, requested) || resident.as_ref() == requested.as_ref());
    };
    poll(budget)?;
    if resident.len() != requested.len() {
        return Ok(false);
    }
    if Arc::ptr_eq(resident, requested) {
        return Ok(true);
    }
    for (left, right) in resident.chunks(1024).zip(requested.chunks(1024)) {
        poll(budget)?;
        if left != right {
            return Ok(false);
        }
    }
    poll(budget)?;
    Ok(true)
}

impl ProductTokenMaskContent {
    pub fn selection(vocabulary_size: u64, fingerprint: u64, source: &Arc<[i8]>) -> Self {
        Self::Selection(ProductTokenMaskSelection {
            vocabulary_size,
            fingerprint,
            source_len: source.len(),
            source: Arc::downgrade(source),
        })
    }

    /// Capture the actual weak ledger entry without retaining its source.
    pub fn capture_selection(
        vocabulary_size: u64,
        fingerprint: u64,
        source_len: usize,
        source: &Weak<[i8]>,
    ) -> Result<Self, U> {
        if vocabulary_size == 0 {
            return Err(U::InvalidInput);
        }
        let Some(live) = source.upgrade() else {
            return Ok(Self::ExpiredSelection);
        };
        if live.len() != source_len {
            return Err(U::InvalidInput);
        }
        Ok(Self::Selection(ProductTokenMaskSelection {
            vocabulary_size,
            fingerprint,
            source_len,
            source: Weak::clone(source),
        }))
    }

    fn vocabulary_size(&self) -> Option<u64> {
        match self {
            Self::AllValid { vocabulary_size } => Some(*vocabulary_size),
            Self::Selection(selection) => Some(selection.vocabulary_size),
            Self::ExpiredSelection | Self::UnavailableSelection => None,
        }
    }

    fn matches(
        &self,
        requested: &Self,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<bool, U> {
        poll(budget)?;
        match (self, requested) {
            (Self::AllValid { vocabulary_size: a }, Self::AllValid { vocabulary_size: b }) => {
                Ok(a == b)
            }
            (Self::Selection(a), Self::Selection(b)) => {
                // A captured live source must not silently become an upload.
                let left = a.source.upgrade().ok_or(U::StaleView)?;
                let right = b.source.upgrade().ok_or(U::StaleView)?;
                if left.len() != a.source_len || right.len() != b.source_len {
                    return Err(U::InvalidInput);
                }
                if a.vocabulary_size != b.vocabulary_size
                    || a.fingerprint != b.fingerprint
                    || a.source_len != b.source_len
                {
                    return Ok(false);
                }
                selection_mask_bytes_match(&left, &right, Some(budget))
            }
            (Self::UnavailableSelection, Self::Selection(_)) => Err(U::OutputBranch),
            (_, Self::ExpiredSelection | Self::UnavailableSelection) => Err(U::InvalidInput),
            _ => Ok(false),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductTokenMaskResidencyEntry {
    identity: LaneStableArenaSlotIdentity,
    participant_index: usize,
    content: ProductTokenMaskContent,
}
impl ProductTokenMaskResidencyEntry {
    pub fn new(
        identity: LaneStableArenaSlotIdentity,
        participant_index: usize,
        all_valid_vocabulary_size: Option<u64>,
    ) -> Self {
        Self {
            identity,
            participant_index,
            content: all_valid_vocabulary_size.map_or(
                ProductTokenMaskContent::UnavailableSelection,
                |vocabulary_size| ProductTokenMaskContent::AllValid { vocabulary_size },
            ),
        }
    }
    pub fn with_content(
        identity: LaneStableArenaSlotIdentity,
        participant_index: usize,
        content: ProductTokenMaskContent,
    ) -> Self {
        Self {
            identity,
            participant_index,
            content,
        }
    }
    fn key(&self) -> (u64, usize) {
        (self.identity.slot_id(), self.participant_index)
    }
}

/// The producer must copy its actual ledger while retaining the read lock
/// across the resource/lane snapshot. This value holds identities, never leases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductTokenMaskResidencySnapshot {
    eligible: bool,
    maximum_entries: usize,
    entries: Vec<ProductTokenMaskResidencyEntry>,
}
impl ProductTokenMaskResidencySnapshot {
    pub(super) fn same_future_state(
        &self,
        other: &Self,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<bool, U> {
        poll(budget)?;
        if self.eligible != other.eligible || self.maximum_entries != other.maximum_entries {
            return Ok(false);
        }
        // Entry equality includes slot identity and the weak source pointer,
        // length and fingerprint. Do not upgrade sources or infer equality from
        // normalized mask bytes: both would change the residency contract.
        super::state_equivalence::same_values(&self.entries, &other.entries, budget)
    }

    pub fn new(
        eligible: bool,
        maximum_entries: usize,
        mut entries: Vec<ProductTokenMaskResidencyEntry>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<Self, U> {
        poll(budget)?;
        if maximum_entries == 0
            || maximum_entries > MAX_COST_ROWS
            || entries.len() > maximum_entries
        {
            return Err(U::Capacity);
        }
        for entry in &entries {
            poll(budget)?;
            if entry.identity.lifetime() != AllocationLifetime::Step
                || entry.participant_index >= MAX_COST_ROWS
                || entry.content.vocabulary_size() == Some(0)
            {
                return Err(U::InvalidInput);
            }
        }
        entries.sort_unstable_by_key(ProductTokenMaskResidencyEntry::key);
        poll(budget)?;
        if entries
            .windows(2)
            .any(|pair| pair[0].key() == pair[1].key())
        {
            return Err(U::InvalidInput);
        }
        Ok(Self {
            eligible,
            maximum_entries,
            entries,
        })
    }

    #[cfg(test)]
    pub(crate) fn project_uploads(
        &mut self,
        selected_step: Option<&LaneStableArenaSlotIdentity>,
        vocabulary_size: u64,
        participants: usize,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<Vec<bool>, U> {
        if vocabulary_size == 0 || participants == 0 || participants > MAX_COST_ROWS {
            return Err(U::InvalidInput);
        }
        let contents = vec![ProductTokenMaskContent::AllValid { vocabulary_size }; participants];
        self.project_contents(selected_step, vocabulary_size, &contents, budget)
    }

    pub(crate) fn project_contents(
        &mut self,
        selected_step: Option<&LaneStableArenaSlotIdentity>,
        vocabulary_size: u64,
        contents: &[ProductTokenMaskContent],
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<Vec<bool>, U> {
        poll(budget)?;
        let participants = contents.len();
        if vocabulary_size == 0 || participants == 0 || participants > MAX_COST_ROWS {
            return Err(U::InvalidInput);
        }
        for content in contents {
            poll(budget)?;
            if content.vocabulary_size() != Some(vocabulary_size) {
                return Err(U::InvalidInput);
            }
            if let ProductTokenMaskContent::Selection(selection) = content {
                let source = selection.source.upgrade().ok_or(U::StaleView)?;
                if source.len() != selection.source_len {
                    return Err(U::InvalidInput);
                }
            }
        }
        let mut uploads = vec![true; participants];
        let Some(identity) = selected_step.filter(|_| self.eligible) else {
            return Ok(uploads);
        };
        if identity.lifetime() != AllocationLifetime::Step {
            return Err(U::InvalidInput);
        }
        // Actual prepare removes a stale/mismatching entry before publish.
        for (index, upload) in uploads.iter_mut().enumerate() {
            poll(budget)?;
            let key = (identity.slot_id(), index);
            if let Ok(position) = self.entries.binary_search_by_key(&key, |entry| entry.key()) {
                let entry = &self.entries[position];
                if &entry.identity == identity && entry.content.matches(&contents[index], budget)? {
                    *upload = false;
                } else {
                    self.entries.remove(position);
                }
            }
        }
        let added = uploads.iter().filter(|&&upload| upload).count();
        if added == 0 {
            return Ok(uploads);
        }
        if added > self.maximum_entries {
            self.entries.clear();
            return Ok(uploads);
        }
        if self.entries.len().checked_add(added).ok_or(U::Capacity)? > self.maximum_entries {
            self.entries.clear();
        }
        self.entries
            .try_reserve_exact(added)
            .map_err(|_| U::Capacity)?;
        for (participant_index, &upload) in uploads.iter().enumerate() {
            poll(budget)?;
            if upload {
                self.entries
                    .push(ProductTokenMaskResidencyEntry::with_content(
                        identity.clone(),
                        participant_index,
                        contents[participant_index].clone(),
                    ));
            }
        }
        self.entries
            .sort_unstable_by_key(ProductTokenMaskResidencyEntry::key);
        poll(budget)?;
        Ok(uploads)
    }
}

/// Identifies the product's full-vocabulary mask input among ordinary uploads.
/// The core checks exactly one matching upload per row before filtering it.
#[derive(Debug, Clone, Copy)]
pub struct EagerCoreTokenMaskInput<'a> {
    pub node_id: &'a NodeId,
    pub input_ordinal: u32,
    pub vocabulary_size: u64,
    /// Exact per-physical-row contents selected by the actual output-mode rule.
    pub contents: &'a [ProductTokenMaskContent],
}
