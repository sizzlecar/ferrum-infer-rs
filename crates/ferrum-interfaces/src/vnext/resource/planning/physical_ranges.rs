//! Fenced numeric copies of actual resident addresses. No buffer or lease is
//! retained. Runtimes must explicitly provide stable numeric allocation evidence.
use super::*;
use crate::vnext::DeviceCostBufferRange;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PhysicalRanges {
    static_ranges: Arc<BTreeMap<ResourceId, (u64, DeviceCostBufferRange)>>,
    pub(super) chunks: BTreeMap<BackingChunkIdentity, DeviceCostBufferRange>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResourceCostRangeProof {
    static_ranges: Arc<BTreeMap<ResourceId, (u64, DeviceCostBufferRange)>>,
    dynamic_ranges: BTreeMap<ResourceId, DeviceCostBufferRange>,
    reusable_scopes: BTreeMap<ResourceId, crate::vnext::DeviceReusableAddressScope>,
    sequence_ranges: Vec<BTreeMap<ResourceId, Vec<DeviceCostBufferRange>>>,
}

impl ResourceCostRangeProof {
    pub(crate) fn reusable_scope(
        &self,
        resource: &ResourceId,
    ) -> Option<crate::vnext::DeviceReusableAddressScope> {
        self.reusable_scopes.get(resource).copied()
    }
    pub(super) fn record_workspace_scope(
        &mut self,
        resource: &ResourceId,
        lane: ExecutionLaneId,
    ) -> Result<(), ResourcePlanningUnknown> {
        let scope = crate::vnext::DeviceReusableAddressScope::ExecutionLane(lane);
        if self
            .reusable_scopes
            .insert(resource.clone(), scope)
            .is_some_and(|old| old != scope)
        {
            return Err(ResourcePlanningUnknown::InvalidDemand);
        }
        Ok(())
    }
    pub(crate) fn sequence(
        &self,
        participant: usize,
        resource: &ResourceId,
    ) -> Option<&[DeviceCostBufferRange]> {
        self.sequence_ranges
            .get(participant)?
            .get(resource)
            .map(Vec::as_slice)
    }
    pub(crate) fn get(&self, resource: &ResourceId) -> Option<DeviceCostBufferRange> {
        self.dynamic_ranges
            .get(resource)
            .copied()
            .or_else(|| self.static_ranges.get(resource).map(|(_, range)| *range))
    }
}

pub(super) fn capture_static<R: DeviceRuntime>(
    resources: &PlanRuntimeResources<R>,
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Option<PhysicalRanges>, ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    if !resources.runtime.supports_cost_buffer_ranges() {
        return Ok(None);
    }
    let mut ranges = BTreeMap::new();
    if let PlanRuntimeStatic::Static(source) = &resources.static_resources {
        let lease = source.lease.as_ref().ok_or(U::StaleIdentity)?;
        if source.finalized || lease.slots.len() > limits.maximum_descriptors {
            return Err(U::LimitExceeded);
        }
        for slot in &lease.slots {
            poll(budget)?;
            let entry = &slot.entry;
            let descriptor = slot.descriptor.as_ref().ok_or(U::StaleIdentity)?;
            let buffer = slot.buffer.as_ref().ok_or(U::StaleIdentity)?;
            if entry.state() != ResourceLeaseState::Active
                || entry.generation() == 0
                || slot.actual_generation != Some(entry.generation())
                || slot.actual_resource_id.as_ref() != Some(entry.resource_id())
                || descriptor.resource_id != *entry.resource_id()
                || descriptor.size_bytes != entry.size_bytes()
            {
                return Err(U::StaleIdentity);
            }
            let Some(range) = resources.runtime.cost_buffer_range(buffer) else {
                // A segmented static arena need not have one physical range.
                // Keep independent resident dynamic proofs; this missing static
                // binding itself remains unavailable to its provider query.
                continue;
            };
            if range.length() != descriptor.size_bytes {
                return Err(U::InvalidDemand);
            }
            if ranges
                .insert(entry.resource_id().clone(), (entry.generation(), range))
                .is_some()
            {
                return Err(U::InvalidDemand);
            }
        }
    }
    Ok(Some(PhysicalRanges {
        static_ranges: Arc::new(ranges),
        chunks: BTreeMap::new(),
    }))
}

impl PhysicalRanges {
    pub(super) fn proof(&self) -> ResourceCostRangeProof {
        ResourceCostRangeProof {
            static_ranges: Arc::clone(&self.static_ranges),
            dynamic_ranges: BTreeMap::new(),
            reusable_scopes: BTreeMap::new(),
            sequence_ranges: Vec::new(),
        }
    }

    pub(super) fn insert_sequence_ranges(
        &self,
        proof: &mut ResourceCostRangeProof,
        sequences: &sequence_ranges::SequenceRanges,
        rows: &[ResourcePlanningRow],
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<(), ResourcePlanningUnknown> {
        use ResourcePlanningUnknown as U;
        let mut selected = Vec::with_capacity(rows.len());
        for row in rows {
            poll(budget)?;
            let source = sequences
                .get(row.participant_index)
                .ok_or(U::InvalidInput)?;
            let mut resources = BTreeMap::new();
            for (resource, segments) in source.iter() {
                let mut ranges = Vec::with_capacity(segments.len());
                for segment in segments.iter() {
                    poll(budget)?;
                    let base = self.chunks.get(segment.chunk()).ok_or(U::StaleIdentity)?;
                    ranges.push(
                        base.slice(segment.offset_bytes(), segment.length_bytes())
                            .ok_or(U::InvalidDemand)?,
                    );
                }
                resources.insert(resource.clone(), ranges);
            }
            selected.push(resources);
        }
        proof.sequence_ranges = selected;
        Ok(())
    }

    pub(super) fn insert_projection(
        &self,
        proof: &mut ResourceCostRangeProof,
        resource: &ResourceId,
        segments: &[BackingSegment],
        offset: u64,
        length: u64,
    ) -> Result<(), ResourcePlanningUnknown> {
        use ResourcePlanningUnknown as U;
        // Only contiguous retained windows can support the native packed-head
        // predicate. Omitted paged ranges remain Unknown to the provider.
        let [segment] = segments else {
            return Ok(());
        };
        let base = self.chunks.get(segment.chunk()).ok_or(U::StaleIdentity)?;
        let allocation = base
            .slice(segment.offset_bytes(), segment.length_bytes())
            .ok_or(U::InvalidDemand)?;
        let range = allocation.slice(offset, length).ok_or(U::InvalidDemand)?;
        if proof.static_ranges.contains_key(resource)
            || proof
                .dynamic_ranges
                .insert(resource.clone(), range)
                .is_some()
        {
            return Err(U::InvalidDemand);
        }
        Ok(())
    }

    pub(super) fn insert_transient(
        &self,
        proof: &mut ResourceCostRangeProof,
        requests: &[EvaluatedBackingRequest<'_>],
        allocated: &[(usize, Vec<BackingSegment>)],
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<(), ResourcePlanningUnknown> {
        let mut ordered = requests.iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| {
            a.domain
                .pool_id()
                .cmp(b.domain.pool_id())
                .then_with(|| b.capacity_size_bytes.cmp(&a.capacity_size_bytes))
                .then_with(|| a.claim_identity.cmp(&b.claim_identity))
        });
        if ordered.len() != allocated.len() {
            return Err(ResourcePlanningUnknown::InvalidDemand);
        }
        for (request, (_, segments)) in ordered.into_iter().zip(allocated) {
            for projection in &request.projections {
                poll(budget)?;
                self.insert_projection(
                    proof,
                    projection.descriptor.base_resource_id(),
                    segments,
                    projection.physical_offset_bytes,
                    projection.capacity_size_bytes,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn resource(value: &str) -> ResourceId {
        ResourceId::new(value).unwrap()
    }
    fn segment(generation: u64, offset: u64, length: u64) -> BackingSegment {
        BackingSegment::from_chunk(
            &serde_json::from_value(serde_json::json!(format!(
                "dynamic-pool/sha256/{}",
                "a".repeat(64)
            )))
            .unwrap(),
            1,
            generation,
            offset,
            length,
        )
        .unwrap()
    }
    fn capture() -> PhysicalRanges {
        PhysicalRanges {
            static_ranges: Arc::new(BTreeMap::new()),
            chunks: BTreeMap::from([(
                segment(2, 0, 256).chunk().clone(),
                DeviceCostBufferRange::new(4096, 256).unwrap(),
            )]),
        }
    }
    #[test]
    fn selected_sequence_ranges_preserve_row_order_aliases_and_chunk_generation() {
        let view = capture();
        let resource = resource("kv");
        let sequences = vec![
            Arc::new(BTreeMap::from([(
                resource.clone(),
                Arc::new(vec![segment(2, 0, 64)]),
            )])),
            Arc::new(BTreeMap::from([(
                resource.clone(),
                Arc::new(vec![segment(2, 32, 64)]),
            )])),
        ];
        let rows = [
            ResourcePlanningRow {
                participant_index: 1,
                start_token: 7,
                token_count: 1,
            },
            ResourcePlanningRow {
                participant_index: 0,
                start_token: 3,
                token_count: 1,
            },
        ];
        let mut proof = view.proof();
        view.insert_sequence_ranges(&mut proof, &sequences, &rows, &mut || true)
            .unwrap();
        let first = proof.sequence(0, &resource).unwrap()[0];
        let second = proof.sequence(1, &resource).unwrap()[0];
        assert_eq!(first.start(), 4128);
        assert_eq!(second.start(), 4096);
        assert!(
            first.overlaps(second),
            "different owners never imply disjoint storage"
        );
        assert!(proof.sequence(2, &resource).is_none());
        let stale = vec![Arc::new(BTreeMap::from([(
            resource,
            Arc::new(vec![segment(1, 0, 64)]),
        )]))];
        assert_eq!(
            view.insert_sequence_ranges(&mut proof, &stale, &rows[1..], &mut || true),
            Err(ResourcePlanningUnknown::StaleIdentity)
        );
        assert_eq!(
            view.insert_sequence_ranges(&mut proof, &sequences, &rows, &mut || false),
            Err(ResourcePlanningUnknown::BudgetExhausted)
        );
    }
    #[test]
    fn projected_claim_offset_preserves_real_aliases_and_exact_contiguous_span() {
        let view = capture();
        let mut proof = view.proof();
        let claim = segment(2, 32, 128);
        view.insert_projection(&mut proof, &resource("a"), &[claim.clone()], 16, 32)
            .unwrap();
        view.insert_projection(&mut proof, &resource("b"), &[claim], 32, 32)
            .unwrap();
        let a = proof.get(&resource("a")).unwrap();
        assert_eq!((a.start(), a.length()), (4144, 32));
        assert!(a.overlaps(proof.get(&resource("b")).unwrap()));
    }
    #[test]
    fn generation_escape_duplicate_and_paged_windows_never_mint_contiguous_evidence() {
        let view = capture();
        let mut proof = view.proof();
        assert_eq!(
            view.insert_projection(&mut proof, &resource("stale"), &[segment(1, 0, 64)], 0, 32),
            Err(ResourcePlanningUnknown::StaleIdentity)
        );
        assert_eq!(
            view.insert_projection(
                &mut proof,
                &resource("escape"),
                &[segment(2, 0, 64)],
                32,
                33
            ),
            Err(ResourcePlanningUnknown::InvalidDemand)
        );
        view.insert_projection(
            &mut proof,
            &resource("paged"),
            &[segment(2, 0, 32), segment(2, 64, 32)],
            0,
            64,
        )
        .unwrap();
        assert!(proof.get(&resource("paged")).is_none());
        view.insert_projection(&mut proof, &resource("one"), &[segment(2, 0, 64)], 0, 32)
            .unwrap();
        assert_eq!(
            view.insert_projection(&mut proof, &resource("one"), &[segment(2, 0, 64)], 0, 32),
            Err(ResourcePlanningUnknown::InvalidDemand)
        );
    }
}
