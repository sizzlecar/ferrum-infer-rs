//! Retained logical content has a different lifetime from queued wire bytes.

use crate::{exhausted, OutputSession};
use ferrum_types::{FerrumError, InferenceResponse, Result};
use std::sync::Arc;
use tokio::sync::OwnedSemaphorePermit;

mod cost;
pub use cost::semantic_cost;

/// Unique ownership of retained content credit. Growing never waits: a history
/// cannot wait for capacity that its own retained prefix is holding.
#[derive(Debug)]
pub struct OutputSemanticLease {
    session: OutputSession,
    global: Option<OwnedSemaphorePermit>,
    local: Option<OwnedSemaphorePermit>,
}

impl Drop for OutputSemanticLease {
    fn drop(&mut self) {
        let released = self.global.is_some() || self.local.is_some();
        drop(self.global.take());
        drop(self.local.take());
        if released {
            self.session.0.budget.capacity_changed.notify();
        }
    }
}

impl OutputSession {
    pub fn semantic_usage(&self) -> usize {
        self.config().semantic_bytes_per_session - self.0.semantic.available_permits()
    }

    pub fn try_reserve_semantic(&self, bytes: usize) -> Result<OutputSemanticLease> {
        if bytes > self.config().semantic_bytes_per_session {
            return Err(exhausted(
                "retained output exceeds the per-session content limit",
            ));
        }
        let count = u32::try_from(bytes)
            .map_err(|_| exhausted("retained output size exceeds addressable credit"))?;
        // No suspension point or retained partial grant: failure of the second
        // pool immediately drops the first permit, preserving the old owner.
        let local = self
            .0
            .semantic
            .clone()
            .try_acquire_many_owned(count)
            .map_err(|_| exhausted("retained output session content limit reached"))?;
        let global = self
            .0
            .budget
            .semantic
            .clone()
            .try_acquire_many_owned(count)
            .map_err(|_| exhausted("shared retained output content limit reached"))?;
        Ok(OutputSemanticLease {
            session: self.clone(),
            global: Some(global),
            local: Some(local),
        })
    }
}

impl OutputSemanticLease {
    pub fn bytes(&self) -> usize {
        self.global
            .as_ref()
            .map_or(0, OwnedSemaphorePermit::num_permits)
    }

    pub fn output_session(&self) -> &OutputSession {
        &self.session
    }

    pub fn try_grow(&mut self, additional: usize) -> Result<()> {
        if additional == 0 {
            return Ok(());
        }
        self.bytes()
            .checked_add(additional)
            .ok_or_else(|| exhausted("retained output size overflow"))?;
        let mut grant = self.session.try_reserve_semantic(additional)?;
        merge_permit(&mut self.global, grant.global.take());
        merge_permit(&mut self.local, grant.local.take());
        Ok(())
    }

    /// Resize only as the owner's actual retained content changes.
    pub fn try_resize(&mut self, bytes: usize) -> Result<()> {
        let current = self.bytes();
        if bytes > current {
            self.try_grow(bytes - current)
        } else {
            self.shrink_to(bytes)
        }
    }

    pub fn shrink_to(&mut self, bytes: usize) -> Result<()> {
        let release = self
            .bytes()
            .checked_sub(bytes)
            .ok_or_else(|| FerrumError::internal("cannot grow retained output by shrinking"))?;
        drop(self.split_off(release)?);
        Ok(())
    }

    /// Move exactly `bytes` of credit to a new unique content owner.
    pub fn split_off(&mut self, bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Ok(Self {
                session: self.session.clone(),
                global: None,
                local: None,
            });
        }
        if bytes > self.bytes()
            || bytes
                > self
                    .local
                    .as_ref()
                    .map_or(0, OwnedSemaphorePermit::num_permits)
        {
            return Err(FerrumError::internal(
                "retained output split exceeds owned credit",
            ));
        }
        let global = self
            .global
            .as_mut()
            .and_then(|permit| permit.split(bytes))
            .ok_or_else(|| FerrumError::internal("retained output global split failed"))?;
        let Some(local) = self.local.as_mut().and_then(|permit| permit.split(bytes)) else {
            // Preserve both original pools even if their internal accounting
            // ever disagrees; no content is authorized by a partial transfer.
            merge_permit(&mut self.global, Some(global));
            return Err(FerrumError::internal(
                "retained output session split failed",
            ));
        };
        // Keep the grant non-cloneable; all successors retain the same session.
        Ok(Self {
            session: self.session.clone(),
            global: Some(global),
            local: Some(local),
        })
    }

    /// Transfer all credit from an owner whose enclosing type has Drop. The
    /// original remains a valid empty owner of the same session; no reservation,
    /// failure or replacement budget is needed to perform the handoff.
    pub fn take_all(&mut self) -> Self {
        Self {
            session: self.session.clone(),
            global: self.global.take(),
            local: self.local.take(),
        }
    }
}

fn merge_permit(
    current: &mut Option<OwnedSemaphorePermit>,
    incoming: Option<OwnedSemaphorePermit>,
) {
    if let Some(incoming) = incoming {
        match current {
            Some(current) => current.merge(incoming),
            None => *current = Some(incoming),
        }
    }
}

#[derive(Debug)]
struct InferenceOutputInner {
    response: InferenceResponse,
    lease: OutputSemanticLease,
}

/// Immutable completion content. Cloning shares the same allocation and quota;
/// the final consumer, rather than model completion, releases its reservation.
#[derive(Debug, Clone)]
pub struct InferenceOutput(Arc<InferenceOutputInner>);

impl InferenceOutput {
    pub fn new(response: InferenceResponse, mut lease: OutputSemanticLease) -> Result<Self> {
        lease.try_resize(semantic_cost(&response)?)?;
        Ok(Self(Arc::new(InferenceOutputInner { response, lease })))
    }

    pub fn response(&self) -> &InferenceResponse {
        &self.0.response
    }

    pub fn output_session(&self) -> &OutputSession {
        self.0.lease.output_session()
    }
}

#[cfg(test)]
mod tests;
