//! Submission-side accounting for typed binding preludes. A successful copy
//! means the driver accepted that API call; it is not a device completion.

use std::sync::Mutex;

use ferrum_interfaces::vnext::DeviceProgramBindingUploadSnapshot;

#[derive(Default)]
pub(crate) struct ProgramBindingUploadCounters(Mutex<DeviceProgramBindingUploadSnapshot>);

impl ProgramBindingUploadCounters {
    pub(crate) fn pinned_staged(&self, allocated_bytes: u64, reused: bool, staged_bytes: u64) {
        let mut total = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        total.pinned_upload_batches = total.pinned_upload_batches.saturating_add(1);
        total.pinned_upload_bytes = total.pinned_upload_bytes.saturating_add(staged_bytes);
        total.pinned_upload_allocation_bytes = total
            .pinned_upload_allocation_bytes
            .saturating_add(allocated_bytes);
        total.pinned_upload_reuse_batches = total
            .pinned_upload_reuse_batches
            .saturating_add(u64::from(reused));
    }

    pub(crate) fn pinned_fallback(&self) {
        let mut total = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        total.pinned_upload_fallback_batches =
            total.pinned_upload_fallback_batches.saturating_add(1);
    }

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
            live_payload_bytes,
            planned_upload_bytes,
            logical_arena_bytes,
            physical_arena_bytes,
            successful_upload_bytes,
            successful_1d_copies,
            successful_2d_copies
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
    fn pinned_staging_does_not_fabricate_successful_device_uploads() {
        let counters = ProgramBindingUploadCounters::default();
        counters.pinned_staged(64, false, 24);
        {
            let mut attempt = counters.attempt(plan());
            attempt.copied(8, true);
        }
        counters.pinned_staged(0, true, 16);
        counters.pinned_fallback();
        let actual = counters.snapshot();
        assert_eq!(actual.pinned_upload_batches, 2);
        assert_eq!(actual.pinned_upload_bytes, 40);
        assert_eq!(actual.pinned_upload_allocation_bytes, 64);
        assert_eq!(actual.pinned_upload_reuse_batches, 1);
        assert_eq!(actual.pinned_upload_fallback_batches, 1);
        assert_eq!(actual.successful_upload_bytes, 8);
        assert_eq!((actual.succeeded_preludes, actual.failed_preludes), (0, 1));
    }
}
