//! Physical phases execute serially. Retained populations add; their temporary
//! workspaces share one reservation. Prepared source8 uses this ledger only for
//! the identified estimator; its original modes keep their existing accounting.
use super::*;

impl StructuredServiceCollectorV7 {
    pub(super) fn observe_reservation(&self, _bytes: usize) {
        #[cfg(test)]
        self.reserved_peak
            .fetch_max(_bytes, std::sync::atomic::Ordering::Relaxed);
    }
    /// Requested charge, including proposals refused before allocation.
    #[cfg(test)]
    pub fn reserved_peak_for_tests(&self) -> usize {
        self.reserved_peak
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    pub(super) fn observe_accepted_reservation(&self, _bytes: usize) {
        #[cfg(test)]
        self.accepted_reservation_peak
            .fetch_max(_bytes, std::sync::atomic::Ordering::Relaxed);
    }
    /// Successful conservative charge, including reserved transient workspace.
    /// This is a bounded allocation allowance, not process RSS.
    #[cfg(test)]
    pub fn accepted_reservation_peak_for_tests(&self) -> usize {
        self.accepted_reservation_peak
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    pub(super) fn serial_physical_workspace(&self) -> bool {
        (self.header.source_kind == PopulationSource::OwnerBlocksV7 || self.identified_envelope())
            && self.header.declaration.domain_policy
                == StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
    }

    fn identified_envelope(&self) -> bool {
        self.header
            .declaration
            .nonnegative_envelope
            .as_ref()
            .is_some_and(|contract| {
                matches!(
                    contract.planning_estimator,
                    NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2
                        | NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1
                        | NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1
                )
            })
    }

    pub(super) fn physical_workspace(
        &self,
        owner: &Owner,
        extra: Option<&StructuredInputV2>,
    ) -> Result<usize, CostProfileError> {
        let overflow = || CostProfileError::Limit("source7 physical workspace overflow");
        let n = owner
            .samples
            .len()
            .checked_add(usize::from(extra.is_some()))
            .ok_or_else(overflow)?;
        if n == 0 {
            return Ok(0);
        }
        // physical_envelope::axes/membership_axes convert basis one-for-one;
        // this is the projected physical width, including completion axes.
        let d = owner
            .sample_axes
            .max(extra.map_or(0, |v| v.regression_axes().len()));
        n.checked_mul(d)
            .and_then(|v| v.checked_mul(128))
            .and_then(|v| v.checked_add(n.checked_mul(1024)?))
            .and_then(|v| v.checked_add(d.checked_mul(256)?))
            .ok_or_else(overflow)
    }

    /// Reserve every owner's resulting model alongside all old populations.
    /// Unlike scratch these results can survive the next owner's transition.
    pub(super) fn physical_transition(
        &self,
        owner: &Owner,
        extra: Option<&StructuredInputV2>,
    ) -> Result<usize, CostProfileError> {
        let overflow = || CostProfileError::Limit("source7 transition retained overflow");
        let n = owner
            .samples
            .len()
            .checked_add(usize::from(extra.is_some()))
            .ok_or_else(overflow)?;
        if n == 0 {
            return Ok(0);
        }
        let d = owner
            .sample_axes
            .max(extra.map_or(0, |v| v.regression_axes().len()));
        let sample = owner.maximum_sample_bytes.max(
            extra
                .map_or(Some(0), |v| {
                    v.retained_bytes_with_shared_universe(self.generation_universe.as_ref())
                })
                .ok_or_else(overflow)?,
        );
        // Allocation inventory, all upper bounds include simultaneous arrays:
        // - Fit: integer outer rows, converted float rows, normalized rows,
        //   QR directions (rank<=n), pivot/residual, row/vector headers.
        // - Readiness V2: outer converted rows plus residual cache, directions,
        //   norms and three d-vectors; V1 fits inside that same reservation.
        // 128*n*d allows sixteen 8-byte matrices (including Vec growth), more
        // than either phase's live matrices; no d*d Gram matrix is allocated.
        // 1024*n covers row headers, borrowed views, norms/residuals, call-ID
        // growth and the certificate container. 512*d covers NNLS coefficients,
        // maxima, coverage and all eight u128 query certificates simultaneously.
        // Four full inputs cover exemplar/query and support clones; twice the
        // old model covers its reallocating call history and transition copies.
        // Samples themselves and every other owner's state remain in SUM.
        let context = owner
            .scope
            .retained_heap_bytes()
            .and_then(|v| {
                v.checked_add(
                    owner
                        .contract
                        .input_target
                        .as_ref()
                        .map_or(0, OwnerInputTargetV1::retained_heap_bytes),
                )
            })
            .and_then(|v| {
                v.checked_add(
                    owner
                        .input_target
                        .as_ref()
                        .map_or(0, OwnerInputTargetV1::retained_heap_bytes),
                )
            })
            .ok_or_else(overflow)?;
        let identified = if self.identified_envelope() {
            let rank = n.min(d).min(self.header.declaration.settings.max_rank);
            // Per-owner SUM, including close_block's simultaneously returned
            // certificate and canonical JSON: model anchors/map (16*r*d),
            // certificate clone (8*r*d), Value cells and growing byte buffer.
            // Replay may also retain the original record. 128*r*d covers these
            // coexisting allocations; geometry scratch remains in serial MAX.
            rank.checked_mul(d)
                .and_then(|v| v.checked_mul(128))
                .and_then(|v| v.checked_add(rank.checked_mul(256)?))
                .and_then(|v| v.checked_add(d.checked_mul(128)?))
                .ok_or_else(overflow)?
        } else {
            0
        };
        // Joint banks retain at most one complete tuple per original member,
        // plus R/Q counts. This result survives the serial transition workspace.
        let joint_bank = if owner
            .contract
            .nonnegative_envelope
            .as_ref()
            .is_some_and(|c| {
                c.planning_estimator == NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1
            }) {
            n.checked_mul(d)
                .and_then(|v| v.checked_mul(2))
                .and_then(|v| v.checked_add(n.checked_mul(256)?))
                .ok_or_else(overflow)?
        } else {
            0
        };
        n.checked_mul(1024)
            .and_then(|v| v.checked_add(joint_bank))
            .and_then(|v| v.checked_add(identified))
            .and_then(|v| v.checked_add(d.checked_mul(512)?))
            .and_then(|v| v.checked_add(sample.checked_mul(4)?))
            .and_then(|v| v.checked_add(owner.state.retained()?.checked_mul(2)?))
            .and_then(|v| v.checked_add(context.checked_mul(4)?))
            .and_then(|v| v.checked_add(4 * std::mem::size_of::<QualifiedStructuredModelV2>()))
            .ok_or_else(overflow)
    }

    pub(super) fn capacity_error(
        &self,
        reason: &'static str,
        retained: usize,
        transient: usize,
    ) -> CostProfileError {
        if !self
            .capacity_reported
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            // Fixed stack payload, once per failed collector; never per token.
            let mut owners = [(0u64, None, 0usize, 0usize); 8];
            for (slot, owner) in owners.iter_mut().zip(&self.owners) {
                *slot = (
                    owner.contract.owner_attempt_id,
                    owner.state.phase(),
                    owner.samples.len(),
                    owner.sample_axes,
                );
            }
            tracing::warn!(target: "ferrum_scheduler::structured_owner_diagnostics",
                event = "structured_source7_capacity_v1", reason,
                current_owned_bytes = self.persistent_bytes.saturating_sub(self.transition_bytes),
                current_transition_reservation_bytes = self.transition_bytes,
                required_retained_and_transition_bytes = retained, required_transient_bytes = transient,
                limit_bytes = self.header.declaration.maximum_retained_numeric_bytes,
                source_kind = ?self.header.source_kind,
                block = self.block, offered = self.offered, owner_count = self.owners.len(),
                owners = ?owners, "Original source capacity exhausted");
        }
        CostProfileError::Limit(reason)
    }

    pub(super) fn reserve_physical_sample(
        &mut self,
        index: usize,
        input: &StructuredInputV2,
    ) -> Result<(), CostProfileError> {
        self.check_retained()?;
        let overflow = || CostProfileError::Limit("source7 sample retained overflow");
        let owner = &self.owners[index];
        let payload = input
            .retained_bytes_with_shared_universe(self.generation_universe.as_ref())
            .ok_or_else(overflow)?;
        let heap = payload
            .checked_sub(std::mem::size_of::<StructuredInputV2>())
            .ok_or_else(overflow)?;
        let old_capacity = owner.samples.capacity();
        let capacity = if owner.samples.len() == old_capacity {
            old_capacity
                .checked_mul(2)
                .map(|n| n.max(4))
                .ok_or_else(overflow)?
        } else {
            old_capacity
        };
        // A growing Vec may allocate the complete new buffer before releasing
        // the old allocation; reserve that peak, not only its final delta.
        let allocation = if capacity == old_capacity {
            0
        } else {
            capacity
                .checked_mul(std::mem::size_of::<StructuredNumericObservationV2>())
                .ok_or_else(overflow)?
        };
        let persistent = self
            .persistent_bytes
            .checked_add(heap)
            .and_then(|n| n.checked_add(allocation))
            .ok_or_else(overflow)?;
        let persistent = persistent
            .checked_sub(self.physical_transition(owner, None)?)
            .and_then(|v| v.checked_add(self.physical_transition(owner, Some(input)).ok()?))
            .ok_or_else(overflow)?;
        let mut transient = 0;
        for (i, owner) in self.owners.iter().enumerate() {
            transient =
                transient.max(self.physical_workspace(owner, (i == index).then_some(input))?);
        }
        self.observe_reservation(persistent.checked_add(transient).ok_or_else(overflow)?);
        if persistent
            .checked_add(transient)
            .is_none_or(|n| n > self.header.declaration.maximum_retained_numeric_bytes)
        {
            return Err(self.capacity_error(
                "source7 shared workspace capacity",
                persistent,
                transient,
            ));
        }
        self.observe_accepted_reservation(persistent.checked_add(transient).ok_or_else(overflow)?);
        let owner = &mut self.owners[index];
        if capacity != old_capacity {
            owner
                .samples
                .try_reserve_exact(capacity - owner.samples.len())
                .map_err(|_| CostProfileError::Limit("source7 sample allocation"))?;
            if owner.samples.capacity() > capacity {
                return Err(CostProfileError::Limit(
                    "source7 sample allocation capacity",
                ));
            }
        }
        owner.sample_heap_bytes = owner
            .sample_heap_bytes
            .checked_add(heap)
            .ok_or_else(overflow)?;
        owner.maximum_sample_bytes = owner.maximum_sample_bytes.max(payload);
        owner.sample_axes = owner.sample_axes.max(input.regression_axes().len());
        Ok(())
    }

    pub(super) fn reserve_original_call(&mut self) -> Result<(), CostProfileError> {
        if !self.serial_physical_workspace() || self.calls.len() < self.calls.capacity() {
            return Ok(());
        }
        self.check_retained()?;
        let capacity = self
            .calls
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(4))
            .ok_or(CostProfileError::Limit("source7 calls overflow"))?;
        let bytes = capacity
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(CostProfileError::Limit("source7 calls overflow"))?;
        let retained = self
            .persistent_bytes
            .checked_add(bytes)
            .ok_or(CostProfileError::Limit("source7 calls overflow"))?;
        let transient = self.numeric_bytes - self.persistent_bytes;
        let total = retained
            .checked_add(transient)
            .ok_or(CostProfileError::Limit("source7 calls overflow"))?;
        self.observe_reservation(total);
        if total > self.header.declaration.maximum_retained_numeric_bytes {
            return Err(self.capacity_error("source7 calls capacity", retained, transient));
        }
        self.observe_accepted_reservation(total);
        self.calls
            .try_reserve_exact(capacity - self.calls.len())
            .map_err(|_| CostProfileError::Limit("source7 calls allocation"))?;
        if self.calls.capacity() > capacity {
            return Err(CostProfileError::Limit("source7 calls allocation capacity"));
        }
        Ok(())
    }
}
