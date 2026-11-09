//! Select one source policy before any materialization or initialization.
use super::*;
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Policy {
    Fresh,
    Reusable,
    Streaming,
}

enum State {
    Unselected,
    Selected(Policy),
    Failed,
}

pub(super) struct Selection(Mutex<State>);

impl Selection {
    pub(super) fn new() -> Self {
        Self(Mutex::new(State::Unselected))
    }

    pub(super) fn prepare(
        &self,
        policy: Policy,
        prepare: impl FnOnce() -> Result<(), VNextError>,
    ) -> Result<(), VNextError> {
        let mut state = self
            .0
            .try_lock()
            .map_err(|_| invalid("source policy is busy or poisoned"))?;
        if !matches!(*state, State::Unselected) {
            return Err(invalid("source backing must be selected once before use"));
        }
        // Failed preparation cannot silently recover through the fresh route.
        *state = State::Failed;
        prepare()?;
        *state = State::Selected(policy);
        Ok(())
    }

    pub(super) fn get(&self, seal_default: bool) -> Result<Policy, VNextError> {
        let mut state = self
            .0
            .try_lock()
            .map_err(|_| invalid("source policy is busy or poisoned"))?;
        match *state {
            State::Unselected => {
                if seal_default {
                    *state = State::Selected(Policy::Fresh);
                }
                Ok(Policy::Fresh)
            }
            State::Selected(policy) => Ok(policy),
            State::Failed => Err(invalid("source backing preparation failed")),
        }
    }
}
