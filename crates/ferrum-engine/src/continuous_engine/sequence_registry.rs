//! The global lock is an index only. Mutable request state has one owner.
use super::*;

pub(super) struct SequenceHandle {
    request_id: RequestId,
    state: Mutex<Option<SequenceState>>,
}

impl SequenceHandle {
    fn new(request_id: RequestId, state: SequenceState) -> Arc<Self> {
        Arc::new_cyclic(|weak| {
            if let Some(sender) = &state.stream_sender {
                sender.bind_sequence(weak.clone());
            }
            Self {
                request_id,
                state: Mutex::new(Some(state)),
            }
        })
    }

    pub(super) fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    /// None means the original state was handed off. An old handle never
    /// acquires authority over a replacement using the same RequestId.
    pub(super) fn lock(&self) -> parking_lot::MutexGuard<'_, Option<SequenceState>> {
        self.state.lock()
    }
}

#[derive(Default)]
pub(super) struct SequenceRegistry {
    index: RwLock<HashMap<RequestId, Arc<SequenceHandle>>>,
}

impl SequenceRegistry {
    pub(super) fn get(&self, id: &RequestId) -> Option<Arc<SequenceHandle>> {
        self.index.read().get(id).cloned()
    }

    pub(super) fn snapshot(&self) -> Vec<Arc<SequenceHandle>> {
        self.index.read().values().cloned().collect()
    }

    pub(super) fn len(&self) -> usize {
        self.index.read().len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.index.read().is_empty()
    }

    pub(super) fn is_current(&self, handle: &Arc<SequenceHandle>) -> bool {
        self.index
            .read()
            .get(handle.request_id())
            .is_some_and(|current| Arc::ptr_eq(current, handle))
    }

    /// Caller holds the original request lock and has already transferred its
    /// scheduler/physical obligations. Lock order is request -> brief index;
    /// no index holder may wait for a request lock.
    pub(super) fn remove_exact(&self, handle: &Arc<SequenceHandle>) -> bool {
        let mut index = self.index.write();
        if index
            .get(handle.request_id())
            .is_some_and(|current| Arc::ptr_eq(current, handle))
        {
            index.remove(handle.request_id());
            true
        } else {
            false
        }
    }

    /// The closure only constructs fresh state and synchronously submits its
    /// scheduler request. It must not decode, await, or acquire a request lock.
    pub(super) fn publish(
        &self,
        id: RequestId,
        create: impl FnOnce() -> Result<SequenceState>,
    ) -> Result<Arc<SequenceHandle>> {
        let mut index = self.index.write();
        if index.contains_key(&id) {
            return Err(FerrumError::already_exists(format!(
                "request {id} is already active"
            )));
        }
        let handle = SequenceHandle::new(id.clone(), create()?);
        index.insert(id, Arc::clone(&handle));
        Ok(handle)
    }

    /// Short index transaction for an already captured cohort. The callback
    /// may synchronously update scheduler membership, but must never acquire
    /// a request lock or run product processing. A replacement cannot appear
    /// between the exact handle check and a scheduler API keyed by RequestId.
    pub(super) fn with_current<'a, T>(
        &self,
        handles: impl IntoIterator<Item = &'a Arc<SequenceHandle>>,
        action: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let index = self.index.read();
        if handles.into_iter().any(|handle| {
            !index
                .get(handle.request_id())
                .is_some_and(|current| Arc::ptr_eq(current, handle))
        }) {
            return Err(FerrumError::scheduler(
                "readiness cohort no longer belongs to its original requests",
            ));
        }
        action()
    }

    #[cfg(test)]
    pub(super) fn insert(
        &self,
        id: RequestId,
        mut state: SequenceState,
    ) -> Option<Arc<SequenceHandle>> {
        if let Some(sender) = &state.stream_sender {
            state
                .bind_output_session(sender.output_session().clone())
                .expect("fixture content fits real output budget");
        }
        self.index
            .write()
            .insert(id.clone(), SequenceHandle::new(id, state))
    }

    #[cfg(test)]
    pub(super) fn index_is_unlocked(&self) -> bool {
        self.index.try_write().is_some()
    }

    #[cfg(test)]
    pub(super) fn take(&self, id: &RequestId) -> Option<SequenceState> {
        let handle = self.get(id)?;
        let mut state = handle.lock();
        let taken = state.take();
        self.remove_exact(&handle);
        taken
    }
}
