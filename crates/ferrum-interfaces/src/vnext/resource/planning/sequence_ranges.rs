//! Numeric sequence windows, copied under the same session/backing/pool bracket.
//! These segments own no lease or buffer. Future growth uses the actual allocator
//! projection; participant identity alone never implies physical disjointness.
//! Windows retain committed physical capacity, including pitched prefix slots.
//! Semantic demands and initialized copy regions remain independently checked.
use super::*;

pub(super) type SequenceRanges = Vec<Arc<BTreeMap<ResourceId, Arc<Vec<BackingSegment>>>>>;

pub(super) fn capture(
    slices: &[LogicalBackingSliceAuthority],
    maximum: usize,
    visited: &mut usize,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Arc<BTreeMap<ResourceId, Arc<Vec<BackingSegment>>>>, ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    let mut result: BTreeMap<ResourceId, Arc<Vec<BackingSegment>>> = BTreeMap::new();
    for slice in slices {
        poll(budget)?;
        let evidence = slice.evidence();
        let lease = &slice.segment_lease;
        if lease.released
            || lease.owner_instance_id != evidence.pool_instance_id()
            || lease.segment_generation != evidence.segment_generation()
            || lease.claim_identity != *evidence.physical_claim_identity()
            || lease.size_bytes != evidence.physical_size_bytes()
        {
            return Err(U::StaleIdentity);
        }
        let matches = lease
            .segments
            .range_matches_with_poll(
                evidence.physical_offset_bytes(),
                evidence.capacity_size_bytes(),
                evidence.segments(),
                || budget.has_budget(),
            )
            .map_err(|_| U::InvalidDemand)?
            .ok_or(U::BudgetExhausted)?;
        if !matches {
            return Err(U::InvalidDemand);
        }
        append(
            &mut result,
            evidence.resource_id(),
            evidence.segments(),
            0,
            evidence.capacity_size_bytes(),
            maximum,
            visited,
            budget,
        )?;
    }
    Ok(Arc::new(result))
}

fn append(
    target: &mut BTreeMap<ResourceId, Arc<Vec<BackingSegment>>>,
    resource: &ResourceId,
    segments: &[BackingSegment],
    offset: u64,
    length: u64,
    maximum: usize,
    visited: &mut usize,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    if length == 0 {
        return Ok(());
    }
    let mut remaining = length;
    let mut skip = offset;
    let output = Arc::make_mut(target.entry(resource.clone()).or_default());
    for segment in segments {
        poll(budget)?;
        if skip >= segment.length_bytes() {
            skip -= segment.length_bytes();
            continue;
        }
        let take = remaining.min(segment.length_bytes() - skip);
        *visited = visited.checked_add(1).ok_or(U::LimitExceeded)?;
        if *visited > maximum {
            return Err(U::LimitExceeded);
        }
        output.push(
            BackingSegment::new(
                segment.chunk().clone(),
                segment
                    .offset_bytes()
                    .checked_add(skip)
                    .ok_or(U::InvalidDemand)?,
                take,
            )
            .map_err(|_| U::InvalidDemand)?,
        );
        remaining -= take;
        skip = 0;
        if remaining == 0 {
            return Ok(());
        }
    }
    Err(U::InvalidDemand)
}

pub(super) fn extend(
    target: &mut Arc<BTreeMap<ResourceId, Arc<Vec<BackingSegment>>>>,
    requests: &[EvaluatedBackingRequest<'_>],
    allocated: &[(usize, Vec<BackingSegment>)],
    maximum: usize,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    use ResourcePlanningUnknown as U;
    // An empty extension changes no range. In particular, do not detach the
    // captured Arc and copy every resource merely because tokens advanced.
    // Allocation/request disagreement remains invalid, including this case.
    if requests.is_empty() {
        return if allocated.is_empty() {
            Ok(())
        } else {
            Err(U::InvalidDemand)
        };
    }
    let mut ordered: Vec<_> = requests.iter().collect();
    ordered.sort_by(|a, b| {
        a.domain
            .pool_id()
            .cmp(b.domain.pool_id())
            .then_with(|| b.capacity_size_bytes.cmp(&a.capacity_size_bytes))
            .then_with(|| a.claim_identity.cmp(&b.claim_identity))
    });
    if ordered.len() != allocated.len() {
        return Err(U::InvalidDemand);
    }
    let target = Arc::make_mut(target);
    let mut count = target
        .values()
        .try_fold(0_usize, |n, v| n.checked_add(v.len()))
        .ok_or(U::LimitExceeded)?;
    for (request, (_, segments)) in ordered.into_iter().zip(allocated) {
        for projection in &request.projections {
            poll(budget)?;
            if !matches!(
                projection.descriptor.storage().profile().view(),
                DynamicStorageView::PagedRegions { .. }
            ) {
                return Err(U::Unsupported);
            }
            append(
                target,
                projection.descriptor.base_resource_id(),
                segments,
                projection.physical_offset_bytes,
                projection.capacity_size_bytes,
                maximum,
                &mut count,
                budget,
            )?;
        }
    }
    Ok(())
}

pub(super) fn check_bound(
    ranges: &SequenceRanges,
    maximum: usize,
) -> Result<(), ResourcePlanningUnknown> {
    let count = ranges
        .iter()
        .flat_map(|row| row.values())
        .try_fold(0_usize, |n, v| n.checked_add(v.len()));
    if count.is_none_or(|n| n > maximum) {
        Err(ResourcePlanningUnknown::LimitExceeded)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn resource() -> ResourceId {
        ResourceId::new("sequence.kv").unwrap()
    }
    fn segment(offset: u64, length: u64) -> BackingSegment {
        BackingSegment::from_chunk(
            &serde_json::from_value(serde_json::json!(format!(
                "dynamic-pool/sha256/{}",
                "a".repeat(64)
            )))
            .unwrap(),
            3,
            7,
            offset,
            length,
        )
        .unwrap()
    }

    #[test]
    fn sequence_empty_extension_preserves_shared_inventory_and_rejects_phantom_allocation() {
        let original = Arc::new(BTreeMap::from([(
            resource(),
            Arc::new(vec![segment(0, 64), segment(128, 64)]),
        )]));
        let mut branch = Arc::clone(&original);
        extend(&mut branch, &[], &[], 8, &mut || true).unwrap();
        assert!(Arc::ptr_eq(&branch, &original));
        assert_eq!(
            branch.get(&resource()).unwrap().as_ref(),
            &vec![segment(0, 64), segment(128, 64)]
        );
        assert_eq!(
            extend(
                &mut branch,
                &[],
                &[(0, vec![segment(256, 64)])],
                8,
                &mut || true
            ),
            Err(ResourcePlanningUnknown::InvalidDemand)
        );
        assert!(Arc::ptr_eq(&branch, &original));
        assert_eq!(
            check_bound(&vec![original, branch], 3),
            Err(ResourcePlanningUnknown::LimitExceeded)
        );
    }

    #[test]
    fn sequence_numeric_windows_keep_logical_order_offsets_and_growth() {
        let mut map = BTreeMap::new();
        let mut count = 0;
        append(
            &mut map,
            &resource(),
            &[segment(128, 128), segment(512, 128)],
            64,
            128,
            4,
            &mut count,
            &mut || true,
        )
        .unwrap();
        assert_eq!(
            map[&resource()].as_slice(),
            &[segment(192, 64), segment(512, 64)]
        );
        let original = Arc::new(map);
        let mut future = Arc::clone(&original);
        append(
            Arc::make_mut(&mut future),
            &resource(),
            &[segment(1024, 128)],
            0,
            128,
            4,
            &mut count,
            &mut || true,
        )
        .unwrap();
        assert_eq!(
            original[&resource()].len(),
            2,
            "rollout cannot mutate the captured snapshot"
        );
        assert_eq!(
            future[&resource()].as_slice(),
            &[segment(192, 64), segment(512, 64), segment(1024, 128)]
        );
        assert!(check_bound(&vec![Arc::clone(&original), future], 4).is_err());
    }
    #[test]
    fn sequence_numeric_windows_reject_escape_capacity_and_budget() {
        let call = |offset, length, maximum, budget: &mut dyn ResourcePlanningBudget| {
            append(
                &mut BTreeMap::new(),
                &resource(),
                &[segment(128, 128)],
                offset,
                length,
                maximum,
                &mut 0,
                budget,
            )
        };
        assert_eq!(
            call(64, 65, 1, &mut || true),
            Err(ResourcePlanningUnknown::InvalidDemand)
        );
        assert_eq!(
            call(0, 1, 0, &mut || true),
            Err(ResourcePlanningUnknown::LimitExceeded)
        );
        assert_eq!(
            call(0, 1, 1, &mut || false),
            Err(ResourcePlanningUnknown::BudgetExhausted)
        );
    }
}
