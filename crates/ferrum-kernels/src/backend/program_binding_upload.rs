//! Submission-side accounting for typed binding preludes. A successful copy
//! or scatter launch means the driver accepted that API call; neither is a
//! device completion.

use std::sync::Mutex;

use ferrum_interfaces::vnext::DeviceProgramBindingUploadSnapshot;

#[derive(Default)]
pub(crate) struct ProgramBindingUploadCounters(Mutex<DeviceProgramBindingUploadSnapshot>);

impl ProgramBindingUploadCounters {
    pub(crate) fn snapshot(&self) -> DeviceProgramBindingUploadSnapshot {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn attempt(
        &self,
        planned: DeviceProgramBindingUploadSnapshot,
    ) -> ProgramBindingUploadAttempt<'_> {
        ProgramBindingUploadAttempt {
            counters: self,
            stats: DeviceProgramBindingUploadSnapshot {
                attempted_preludes: 1,
                failed_preludes: 1,
                compact_scatter_preludes: planned.compact_scatter_preludes,
                compact_scatter_sparse_fallback_preludes: planned
                    .compact_scatter_sparse_fallback_preludes,
                live_payload_bytes: planned.live_payload_bytes,
                planned_upload_bytes: planned.planned_upload_bytes,
                logical_arena_bytes: planned.logical_arena_bytes,
                physical_arena_bytes: planned.physical_arena_bytes,
                ..Default::default()
            },
        }
    }
}

pub(crate) struct ProgramBindingUploadAttempt<'a> {
    counters: &'a ProgramBindingUploadCounters,
    stats: DeviceProgramBindingUploadSnapshot,
}

impl ProgramBindingUploadAttempt<'_> {
    pub(crate) fn copied(&mut self, bytes: u64, is_2d: bool) {
        self.stats.successful_upload_bytes =
            self.stats.successful_upload_bytes.saturating_add(bytes);
        if is_2d {
            self.stats.successful_2d_copies = self.stats.successful_2d_copies.saturating_add(1);
        } else {
            self.stats.successful_1d_copies = self.stats.successful_1d_copies.saturating_add(1);
        }
    }

    pub(crate) fn succeeded(&mut self) {
        self.stats.succeeded_preludes = 1;
        self.stats.failed_preludes = 0;
    }

    pub(crate) fn scatter_dispatched(&mut self) {
        self.stats.successful_scatter_dispatches =
            self.stats.successful_scatter_dispatches.saturating_add(1);
    }
}

impl Drop for ProgramBindingUploadAttempt<'_> {
    fn drop(&mut self) {
        let mut total = self
            .counters
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // One aggregation per attempted prelude, including ordinary errors and
        // unwinding. Saturation is diagnostic only and cannot affect execution.
        macro_rules! add {
            ($($field:ident),+ $(,)?) => { $(total.$field = total.$field.saturating_add(self.stats.$field);)+ };
        }
        add!(
            attempted_preludes,
            succeeded_preludes,
            failed_preludes,
            compact_scatter_preludes,
            compact_scatter_sparse_fallback_preludes,
            live_payload_bytes,
            planned_upload_bytes,
            logical_arena_bytes,
            physical_arena_bytes,
            successful_upload_bytes,
            successful_1d_copies,
            successful_2d_copies,
            successful_scatter_dispatches
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> DeviceProgramBindingUploadSnapshot {
        DeviceProgramBindingUploadSnapshot {
            live_payload_bytes: 24,
            planned_upload_bytes: 64,
            logical_arena_bytes: 64,
            physical_arena_bytes: 128,
            ..Default::default()
        }
    }

    #[test]
    fn accounting_preserves_partial_uploads_and_never_counts_an_unattempted_plan() {
        let counters = ProgramBindingUploadCounters::default();
        let planned = plan();
        assert_eq!(
            counters.snapshot(),
            DeviceProgramBindingUploadSnapshot::default()
        );
        {
            let mut attempt = counters.attempt(planned);
            attempt.copied(16, false);
            // The next driver call failed: leave this attempt unsuccessful.
        }
        {
            let mut attempt = counters.attempt(planned);
            attempt.copied(64, true);
            attempt.succeeded();
        }
        {
            // Failure before any successful API call still records an attempt,
            // not uploaded bytes or a successful copy.
            let _attempt = counters.attempt(planned);
        }
        let actual = counters.snapshot();
        assert_eq!(actual.attempted_preludes, 3);
        assert_eq!(actual.succeeded_preludes, 1);
        assert_eq!(actual.failed_preludes, 2);
        assert_eq!(actual.live_payload_bytes, 72);
        assert_eq!(actual.planned_upload_bytes, 192);
        assert_eq!(actual.logical_arena_bytes, 192);
        assert_eq!(actual.physical_arena_bytes, 384);
        assert_eq!(actual.successful_upload_bytes, 80);
        assert_eq!(
            (actual.successful_1d_copies, actual.successful_2d_copies),
            (1, 1)
        );
    }

    #[test]
    fn accounting_flushes_completed_calls_during_unwind() {
        let counters = ProgramBindingUploadCounters::default();
        let result = std::panic::catch_unwind(|| {
            let mut attempt = counters.attempt(plan());
            attempt.copied(8, true);
            panic!("test provider unwind");
        });
        assert!(result.is_err());
        let actual = counters.snapshot();
        assert_eq!((actual.attempted_preludes, actual.failed_preludes), (1, 1));
        assert_eq!(actual.successful_upload_bytes, 8);
        assert_eq!(actual.successful_2d_copies, 1);
    }

    #[test]
    fn compact_scatter_accounting_distinguishes_copy_acceptance_from_scatter_failure() {
        let counters = ProgramBindingUploadCounters::default();
        let compact = DeviceProgramBindingUploadSnapshot {
            compact_scatter_preludes: 1,
            ..plan()
        };
        // Merely constructing a route plan never records an enqueue attempt.
        let baseline = counters.snapshot();
        assert_eq!(baseline, DeviceProgramBindingUploadSnapshot::default());
        {
            let mut attempt = counters.attempt(compact);
            attempt.copied(64, false);
            // The following scatter launch failed; no dispatch was accepted.
        }
        let failed = counters.snapshot().checked_since(baseline).unwrap();
        assert_eq!(failed.attempted_preludes, 1);
        assert_eq!(failed.failed_preludes, 1);
        assert_eq!(failed.succeeded_preludes, 0);
        assert_eq!(failed.compact_scatter_preludes, 1);
        assert_eq!(failed.compact_scatter_sparse_fallback_preludes, 0);
        assert_eq!(failed.successful_upload_bytes, 64);
        assert_eq!(failed.successful_1d_copies, 1);
        assert_eq!(failed.successful_scatter_dispatches, 0);
        assert_eq!(failed.live_payload_bytes, compact.live_payload_bytes);
        assert_eq!(failed.planned_upload_bytes, compact.planned_upload_bytes);

        let result = std::panic::catch_unwind(|| {
            let mut attempt = counters.attempt(compact);
            attempt.copied(64, false);
            attempt.scatter_dispatched();
            // An accepted launch is still observed if later host work unwinds.
            panic!("failure after accepted scatter");
        });
        assert!(result.is_err());
        let accepted = counters.snapshot().checked_since(failed).unwrap();
        assert_eq!(accepted.failed_preludes, 1);
        assert_eq!(accepted.succeeded_preludes, 0);
        assert_eq!(accepted.compact_scatter_preludes, 1);
        assert_eq!(accepted.successful_scatter_dispatches, 1);
    }

    #[test]
    fn compact_scatter_accounting_records_the_submitted_route_and_keeps_sparse_independent() {
        let counters = ProgramBindingUploadCounters::default();
        {
            let mut attempt = counters.attempt(DeviceProgramBindingUploadSnapshot {
                compact_scatter_preludes: 1,
                ..plan()
            });
            attempt.copied(64, false);
            attempt.scatter_dispatched();
            attempt.succeeded();
        }
        let before_fallback = counters.snapshot();
        {
            let mut attempt = counters.attempt(DeviceProgramBindingUploadSnapshot {
                compact_scatter_sparse_fallback_preludes: 1,
                ..plan()
            });
            attempt.copied(64, true);
            attempt.succeeded();
        }
        let fallback = counters.snapshot().checked_since(before_fallback).unwrap();
        assert_eq!(fallback.compact_scatter_preludes, 0);
        assert_eq!(fallback.compact_scatter_sparse_fallback_preludes, 1);
        assert_eq!(fallback.successful_scatter_dispatches, 0);
        assert_eq!(fallback.successful_2d_copies, 1);
        assert_eq!(fallback.succeeded_preludes, 1);
        let before_sparse = counters.snapshot();
        {
            let mut attempt = counters.attempt(plan());
            attempt.copied(64, true);
            attempt.succeeded();
        }
        let sparse = counters.snapshot().checked_since(before_sparse).unwrap();
        assert_eq!(sparse.compact_scatter_preludes, 0);
        assert_eq!(sparse.compact_scatter_sparse_fallback_preludes, 0);
        assert_eq!(sparse.successful_scatter_dispatches, 0);
        assert_eq!(sparse.succeeded_preludes, 1);
        let total = counters.snapshot();
        assert_eq!(total.attempted_preludes, 3);
        assert_eq!(total.succeeded_preludes, 3);
        assert_eq!(total.compact_scatter_preludes, 1);
        assert_eq!(total.compact_scatter_sparse_fallback_preludes, 1);
        assert_eq!(total.successful_scatter_dispatches, 1);
    }
}
