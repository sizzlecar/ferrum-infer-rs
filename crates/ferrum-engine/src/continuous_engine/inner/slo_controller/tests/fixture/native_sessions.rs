//! Original logical owners are replaced only after their old request completes.
//! Recycling a request ID alone must never turn a retired restore target fresh.
use super::*;

pub(super) struct Sessions(Mutex<Vec<Arc<vnext::SequenceSession<contract::TestRuntime>>>>);
impl Sessions {
    pub(super) fn new(sessions: Vec<Arc<vnext::SequenceSession<contract::TestRuntime>>>) -> Self {
        Self(Mutex::new(sessions))
    }
    pub(super) fn len(&self) -> usize {
        self.0.lock().len()
    }
    pub(super) fn get(
        &self,
        index: usize,
    ) -> Option<Arc<vnext::SequenceSession<contract::TestRuntime>>> {
        self.0.lock().get(index).cloned()
    }
    pub(super) fn snapshot(&self) -> Vec<Arc<vnext::SequenceSession<contract::TestRuntime>>> {
        self.0.lock().clone()
    }
    pub(super) fn clear(&mut self) {
        self.0.get_mut().clear();
    }
    fn replace_after_completion(
        &self,
        index: usize,
        make: impl FnOnce() -> Arc<vnext::SequenceSession<contract::TestRuntime>>,
    ) {
        let mut slots = self.0.lock();
        // Release the previous slot before allocating its replacement. Any
        // real outstanding owner keeps its own lease and original pool charge.
        drop(slots.remove(index));
        slots.insert(index, make());
    }
}
impl CoreEvidence {
    fn replace_checkpoint_session(&self, index: usize, work: vnext::TokenSpanWork) {
        let serial = self
            .session_serial
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .expect("fixture session identities exhausted");
        let resources = &self.fixture.as_ref().unwrap().plan_resources;
        self.sessions.replace_after_completion(index, || {
            contract::logical_resources_with_work(
                resources,
                &format!("run.controller.recycled.{index}.{serial}"),
                &format!("request.controller.recycled.{index}.{serial}"),
                work,
            )
            .open_session()
            .unwrap()
        });
    }
}

impl ControlledExecutor {
    pub(super) fn admit_checkpoint_native_session(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<()> {
        let Some(prefix) = self.evidence.prefix.as_ref() else {
            return Ok(());
        };
        let tokens = input
            .input_tokens
            .iter()
            .map(|token| token.get())
            .collect::<Vec<_>>();
        let work = vnext::TokenSpanWork::from_token_ids_with_fit(
            &tokens,
            0..tokens.len(),
            input.maximum_sequence_tokens,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let mut bindings = self.session_bindings.lock();
        let mut admissions = prefix.admitted.lock();
        if bindings.iter().any(|id| id == input.request_id) {
            return if admissions.get(input.request_id) == Some(&work) {
                Ok(())
            } else {
                Err(FerrumError::backend(
                    "fixture repeat admission changes original token authority",
                ))
            };
        }
        let mut completed = self.completed_bindings.lock();
        let index = if bindings.len() < self.evidence.sessions.len() {
            bindings.len()
        } else if self.recycle_completed_bindings.load(Ordering::Acquire) {
            let caches = self.produced_caches.lock();
            bindings
                .iter()
                .enumerate()
                .find_map(|(index, old)| {
                    let cache_id = format!("mock_{old}");
                    (completed.contains(old)
                        && caches
                            .iter()
                            .filter_map(std::sync::Weak::upgrade)
                            .all(|cache| cache.cache_id() != cache_id))
                    .then_some(index)
                })
                .ok_or_else(|| {
                    FerrumError::backend("fixture has no completed native slot to recycle")
                })?
        } else {
            return Err(FerrumError::backend(
                "checkpoint fixture declared session capacity exhausted",
            ));
        };
        self.evidence
            .replace_checkpoint_session(index, work.clone());
        if index == bindings.len() {
            bindings.push(input.request_id.clone());
        } else {
            admissions.remove(&bindings[index]);
            completed.remove(&bindings[index]);
            bindings[index] = input.request_id.clone();
        }
        admissions.insert(input.request_id.clone(), work);
        Ok(())
    }
}
