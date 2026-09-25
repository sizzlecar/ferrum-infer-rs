//! Ordered bounded arrivals, not an engine admission or execution permit.
use super::*;

pub(super) struct ActiveOwner<K> {
    pub id: K,
    pub source_index: usize,
    pub ordinal: usize,
    pub evidenced: bool,
}

pub(super) struct AdmissionWindow<'a, K> {
    prompts: &'a [usize],
    limit: usize,
    next: usize,
    active: Vec<ActiveOwner<K>>,
}

impl<'a, K: PartialEq> AdmissionWindow<'a, K> {
    pub fn new(prompts: &'a [usize], limit: usize) -> Result<Self> {
        if prompts.is_empty() || limit == 0 || limit > 256 {
            return Err(FerrumError::config(
                "invalid bounded calibration arrival window",
            ));
        }
        Ok(Self {
            prompts,
            limit,
            next: 0,
            active: Vec::with_capacity(limit.min(prompts.len())),
        })
    }

    pub fn next_prompt(&self) -> Option<(usize, usize)> {
        (self.active.len() < self.limit)
            .then(|| self.prompts.get(self.next).map(|&index| (self.next, index)))
            .flatten()
    }

    /// Called only after add_request succeeds. One owner can never occupy two
    /// window positions, including when the source prompt index is repeated.
    pub fn admitted(&mut self, ordinal: usize, id: K) -> Result<()> {
        let Some((expected, source_index)) = self.next_prompt() else {
            return Err(FerrumError::internal("calibration arrival window is full"));
        };
        if ordinal != expected || self.active.iter().any(|owner| owner.id == id) {
            return Err(FerrumError::internal(
                "calibration arrival identity mismatch",
            ));
        }
        self.active.push(ActiveOwner {
            id,
            source_index,
            ordinal,
            evidenced: false,
        });
        self.next += 1;
        Ok(())
    }

    /// The caller must have consumed terminal wire AND successful completion.
    /// A terminal frontier or finished GPU wave alone must not refill the slot.
    pub fn completed(&mut self, id: &K) -> Result<ActiveOwner<K>> {
        let index = self
            .active
            .iter()
            .position(|owner| &owner.id == id)
            .ok_or_else(|| FerrumError::internal("unowned calibration completion"))?;
        Ok(self.active.remove(index))
    }

    pub fn owner_mut(&mut self, id: &K) -> Option<&mut ActiveOwner<K>> {
        self.active.iter_mut().find(|owner| &owner.id == id)
    }

    pub fn active(&self) -> &[ActiveOwner<K>] {
        &self.active
    }

    pub fn drained(&self) -> bool {
        self.next == self.prompts.len() && self.active.is_empty()
    }
}

#[cfg(test)]
mod tests;
