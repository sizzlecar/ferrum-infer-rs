//! Bounded reusable planning evidence, never a runtime admission grant.
use super::*;
/// The generator's capacity shape for each homogeneous row, independent of
/// batch size and seed. The worker owns one generator, so immutable model,
/// operator, precision and runtime settings cannot cross this cache boundary.
/// Mutable implementations must include their changing configuration in `shape`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacitySearchContext {
    pub shape: [u8; 32],
    pub capacity_bytes: u64,
}

/// One entry bounds metadata even when clients send unbounded shape variants.
/// It retains neither requests, prompts, images nor device allocations.
#[derive(Default)]
pub(crate) struct CapacitySearchHint(Option<Boundary>);

struct Boundary {
    shape: [u8; 32],
    adapter: Option<Arc<AdapterDomainKey>>,
    rejected: Vec<RejectedBatch>,
    largest_fit: usize,
    resident_rejected: Vec<RejectedBatch>,
}

struct RejectedBatch {
    batch_size: usize,
    required_bytes: u64,
}

impl CapacitySearchHint {
    pub(crate) fn known_fit(
        &self,
        context: Option<CapacitySearchContext>,
        adapter: &Option<Arc<AdapterDomainKey>>,
    ) -> Option<usize> {
        let context = context?;
        self.0
            .as_ref()
            .filter(|b| b.shape == context.shape && b.adapter == *adapter)
            .map(|b| b.largest_fit)
            .filter(|count| *count > 0)
    }

    /// One expansion between the known fitting prefix and the available queue.
    /// A first fitting plan starts real work; later execution windows continue
    /// exploration. This is a probe choice, never a physical admission grant.
    pub(crate) fn next_probe(
        &self,
        context: Option<CapacitySearchContext>,
        adapter: &Option<Arc<AdapterDomainKey>>,
        upper: usize,
    ) -> usize {
        match self.known_fit(context, adapter) {
            Some(fit) if fit < upper => fit + ((upper - fit) / 2).max(1),
            _ => upper,
        }
    }

    pub(crate) fn remember_fit(
        &mut self,
        context: Option<CapacitySearchContext>,
        adapter: &Option<Arc<AdapterDomainKey>>,
        count: usize,
    ) {
        let Some(context) = context else { return };
        if self
            .0
            .as_ref()
            .is_none_or(|b| b.shape != context.shape || b.adapter != *adapter)
        {
            self.0 = Some(Boundary {
                shape: context.shape,
                adapter: adapter.clone(),
                rejected: Vec::new(),
                largest_fit: 0,
                resident_rejected: Vec::new(),
            });
        }
        let b = self.0.as_mut().unwrap();
        b.largest_fit = b.largest_fit.max(count);
    }

    pub(crate) fn upper(
        &self,
        context: Option<CapacitySearchContext>,
        adapter: &Option<Arc<AdapterDomainKey>>,
        queued: usize,
    ) -> usize {
        let (Some(context), Some(boundary)) = (context, &self.0) else {
            return queued;
        };
        if boundary.shape != context.shape || boundary.adapter != *adapter {
            return queued;
        }
        // Preserve independent failure bounds when physical capacity changes.
        // A smaller available ceiling must not erase what a previous larger
        // ceiling proved. Reaching a bound permits fresh search, never execution.
        // Prefix exclusion inherits the same-shape monotonicity assumption of
        // the scheduler's binary search; it does not prove arbitrary native
        // workspace choices monotonic or establish a globally optimal batch.
        boundary
            .rejected
            .iter()
            .filter(|bound| bound.required_bytes > context.capacity_bytes)
            .map(|bound| bound.batch_size - 1)
            .fold(queued, usize::min)
    }

    /// A soft boundary, distinct from the staged plan's hard capacity proof.
    /// Zero means even one row needs stage swapping; it must not reject work.
    pub(crate) fn resident_upper(
        &self,
        context: Option<CapacitySearchContext>,
        adapter: &Option<Arc<AdapterDomainKey>>,
        queued: usize,
    ) -> usize {
        let (Some(context), Some(boundary)) = (context, &self.0) else {
            return queued;
        };
        if boundary.shape != context.shape || boundary.adapter != *adapter {
            return queued;
        }
        boundary
            .resident_rejected
            .iter()
            .filter(|b| b.required_bytes > context.capacity_bytes)
            .map(|b| b.batch_size.saturating_sub(1))
            .fold(queued, usize::min)
    }

    pub(crate) fn remember_residency(
        &mut self,
        context: Option<CapacitySearchContext>,
        adapter: &Option<Arc<AdapterDomainKey>>,
        count: usize,
        report: DeviceCapacityReport,
    ) {
        if !report.complete_batch || report.fits() || count == 0 {
            return;
        }
        self.remember_fit(context, adapter, count);
        if context.is_none() {
            return;
        }
        let bounds = &mut self.0.as_mut().unwrap().resident_rejected;
        if bounds
            .iter()
            .any(|b| b.batch_size <= count && b.required_bytes >= report.required_bytes)
        {
            return;
        }
        bounds.retain(|b| !(count <= b.batch_size && report.required_bytes >= b.required_bytes));
        bounds.push(RejectedBatch {
            batch_size: count,
            required_bytes: report.required_bytes,
        });
    }

    pub(crate) fn remember(
        &mut self,
        context: Option<CapacitySearchContext>,
        adapter: &Option<Arc<AdapterDomainKey>>,
        count: usize,
        report: DeviceCapacityReport,
        maximum_batch: usize,
    ) {
        let Some(context) = context else {
            return;
        };
        // Only a typed fresh-plan failure reaches this method. A single-stage
        // lower bound proves failure too. Success, short queues and Host limits
        // are not device bounds. Distinct counts cannot exceed maximum_batch.
        if count <= 1 || count > maximum_batch || report.required_bytes <= report.capacity_bytes {
            return;
        }
        if self
            .0
            .as_ref()
            .is_none_or(|old| old.shape != context.shape || old.adapter != *adapter)
        {
            self.0 = Some(Boundary {
                shape: context.shape,
                adapter: adapter.clone(),
                rejected: Vec::new(),
                largest_fit: 0,
                resident_rejected: Vec::new(),
            });
        }
        let bounds = &mut self.0.as_mut().unwrap().rejected;
        // A smaller rejected batch with an equal or stronger byte lower bound
        // already excludes every candidate that the new observation excludes.
        if bounds
            .iter()
            .any(|old| old.batch_size <= count && old.required_bytes >= report.required_bytes)
        {
            return;
        }
        bounds.retain(|old| {
            !(count <= old.batch_size && report.required_bytes >= old.required_bytes)
        });
        bounds.push(RejectedBatch {
            batch_size: count,
            required_bytes: report.required_bytes,
        });
    }
}
