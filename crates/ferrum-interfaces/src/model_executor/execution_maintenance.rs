//! One-use continuation of a real unsubmitted capacity deferral.
use super::*;
use crate::execution_cost::HostSubmissionRejection;
use std::any::Any;

/// Backend-owned maintenance context, never allocation or execution permission.
/// Dropping this value abandons maintenance. It cannot be cloned or serialized.
pub struct ExecutorExecutionMaintenanceTicket {
    stage: ExecutorExecutionCapacityStage,
    request_ids: Vec<RequestId>,
    context: Box<dyn Any + Send + Sync>,
}

impl std::fmt::Debug for ExecutorExecutionMaintenanceTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutorExecutionMaintenanceTicket")
            .field("stage", &self.stage)
            .field("request_ids", &self.request_ids)
            .finish_non_exhaustive()
    }
}

impl ExecutorExecutionMaintenanceTicket {
    /// Executors supply their own sealed context from an actual failed acquire.
    /// The receiving executor must validate both its type and runtime/owner
    /// identity; this envelope alone authorizes no maintenance.
    pub fn from_backend<T: Any + Send + Sync>(
        stage: ExecutorExecutionCapacityStage,
        request_ids: Vec<RequestId>,
        context: T,
    ) -> Result<Self> {
        if request_ids.is_empty()
            || request_ids.iter().collect::<HashSet<_>>().len() != request_ids.len()
        {
            return Err(FerrumError::request_validation(
                "execution maintenance requires a nonempty unique owner cohort",
            ));
        }
        Ok(Self {
            stage,
            request_ids,
            context: Box::new(context),
        })
    }

    pub const fn stage(&self) -> ExecutorExecutionCapacityStage {
        self.stage
    }
    pub fn request_ids(&self) -> &[RequestId] {
        &self.request_ids
    }

    /// Consumes the envelope even on a type mismatch. It cannot be replayed.
    pub fn into_backend<T: Any + Send + Sync>(self) -> Option<T> {
        self.context.downcast::<T>().ok().map(|value| *value)
    }
}

#[derive(Debug)]
pub enum ExecutorExecutionMaintenanceOutcome {
    /// The next attempt must capture new resource/work evidence. A changed
    /// epoch without growth is not misreported as a physical allocation.
    Recapture {
        observed: ExecutorAdmissionEpochs,
        progress: Option<ExecutorExecutionMaintenanceProgress>,
    },
    Wait(ExecutorExecutionCapacityDeferral),
    Rejected(HostSubmissionRejection),
    Unsupported,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Context(Arc<AtomicUsize>);
    impl Drop for Context {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn execution_maintenance_ticket_drop_and_wrong_backend_release_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let ticket = || {
            ExecutorExecutionMaintenanceTicket::from_backend(
                ExecutorExecutionCapacityStage::StepAdmission,
                vec![RequestId::new()],
                Context(Arc::clone(&drops)),
            )
            .unwrap()
        };
        drop(ticket());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(ticket().into_backend::<u64>().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        let held = ticket().into_backend::<Context>().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        drop(held);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn execution_maintenance_ticket_rejects_empty_or_duplicate_owner_cohorts() {
        let stage = ExecutorExecutionCapacityStage::SubmissionWave;
        assert!(ExecutorExecutionMaintenanceTicket::from_backend(stage, vec![], ()).is_err());
        let id = RequestId::new();
        assert!(
            ExecutorExecutionMaintenanceTicket::from_backend(stage, vec![id.clone(), id], ())
                .is_err()
        );
    }
}
