//! One process-local authority for both V2 catalog and feedback replacement.
//! It never supplies cost values or renews an imported model's source clock.
use ferrum_types::FerrumError;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

pub(super) struct Epoch {
    last: AtomicU64,
    pub(super) gate: Arc<AtomicU64>,
}
#[derive(Clone)]
pub(super) struct View {
    pub epoch: u64,
    pub authority: Arc<Epoch>,
}
impl Epoch {
    pub fn new(initial: u64) -> Arc<Self> {
        Arc::new(Self {
            last: AtomicU64::new(initial),
            gate: Arc::new(AtomicU64::new(initial)),
        })
    }
    pub fn reserve(&self) -> Result<u64, FerrumError> {
        self.last
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
            .map(|old| old + 1)
            .map_err(|_| FerrumError::config("structured runtime epoch exhausted"))
    }
    pub fn view(self: &Arc<Self>, epoch: u64) -> View {
        View {
            epoch,
            authority: self.clone(),
        }
    }
    pub fn activate(&self, epoch: u64) {
        self.gate.store(epoch, Ordering::Release);
    }
}
impl View {
    pub fn initial() -> Self {
        Epoch::new(1).view(1)
    }
    pub fn current(&self) -> bool {
        self.epoch != 0 && self.authority.gate.load(Ordering::Acquire) == self.epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_epoch_exhaustion_keeps_existing_authority_without_aba() {
        let epoch = Epoch::new(u64::MAX - 1);
        let old = epoch.view(u64::MAX - 1);
        let last = epoch.reserve().unwrap();
        assert_eq!(last, u64::MAX);
        assert!(old.current());
        epoch.activate(last);
        assert!(!old.current());
        assert!(epoch.reserve().is_err());
        assert!(epoch.view(last).current());
    }
}
