//! Server-local finite observer obligations; no task/record list or I/O lock.
use std::sync::{Arc, Mutex};
use tokio::sync::watch;

#[derive(Default)]
struct State {
    active: usize,
    sealed: bool,
    first_error: Option<String>,
}

pub(in crate::axum_server) struct EvidenceObservers {
    state: Mutex<State>,
    changed: watch::Sender<()>,
}
impl Default for EvidenceObservers {
    fn default() -> Self {
        Self {
            state: Mutex::new(State::default()),
            changed: watch::channel(()).0,
        }
    }
}
impl EvidenceObservers {
    pub(super) fn register(self: &Arc<Self>) -> Result<Ticket, &'static str> {
        let mut state = self.state.lock().unwrap();
        if state.sealed {
            return Err("credited evidence admission is closed for server shutdown");
        }
        state.active = state
            .active
            .checked_add(1)
            .ok_or("observer count overflow")?;
        Ok(Ticket {
            owner: Arc::clone(self),
            armed: false,
            finished: false,
        })
    }
    pub(in crate::axum_server) fn seal(&self) {
        self.state.lock().unwrap().sealed = true;
    }
    pub(in crate::axum_server) async fn drain(&self) -> Result<(), String> {
        // Subscribe before checking to avoid a completion/check lost wake.
        let mut changed = self.changed.subscribe();
        loop {
            {
                let state = self.state.lock().unwrap();
                if state.active == 0 {
                    return state.first_error.clone().map_or(Ok(()), Err);
                }
            }
            // The Sender belongs to self and remains alive throughout this call.
            changed.changed().await.expect("observer notifier alive");
        }
    }
    fn complete(&self, result: Result<(), String>) {
        if let Err(error) = &result {
            tracing::warn!(%error, "credited evidence observation failed");
        }
        {
            let mut state = self.state.lock().unwrap();
            if let Err(error) = result {
                state.first_error.get_or_insert(error);
            }
            state.active = state.active.checked_sub(1).expect("registered observer");
        }
        self.changed.send_replace(());
    }
}

pub(super) struct Ticket {
    owner: Arc<EvidenceObservers>,
    armed: bool,
    finished: bool,
}
impl Ticket {
    pub(super) fn arm(&mut self) {
        self.armed = true;
    }
    pub(super) fn finish(mut self, result: Result<(), String>) {
        self.finished = true;
        self.owner.complete(result);
    }
}
impl Drop for Ticket {
    fn drop(&mut self) {
        if !self.finished {
            let result = if self.armed {
                Err("credited evidence observer cancelled or panicked before completion".into())
            } else {
                Ok(())
            };
            self.owner.complete(result);
        }
    }
}

#[cfg(test)]
mod tests;
