//! The actual initialization order, shared with its bounded numeric snapshot.
//! Borrowed cells and slices never escape capture or grant execution authority.
use super::*;
use crate::vnext::BackingSegment;

pub(in crate::vnext::resource) struct InitializationOrder<'a> {
    groups: BTreeMap<
        &'a str,
        (
            BatchParticipantAuthority,
            &'a Arc<BackingInitializationCell>,
            Vec<&'a LogicalBackingSliceAuthority>,
        ),
    >,
}

impl<'a> InitializationOrder<'a> {
    pub(in crate::vnext::resource) fn new() -> Self {
        Self {
            groups: BTreeMap::new(),
        }
    }

    pub(in crate::vnext::resource) fn insert(
        &mut self,
        owner: BatchParticipantAuthority,
        authority: &'a LogicalBackingSliceAuthority,
    ) -> Result<(), VNextError> {
        let cell = authority.initialization_cell().ok_or_else(|| {
            invalid_resource("zero-initialized backing slice has no initialization authority")
        })?;
        let entry = self
            .groups
            .entry(cell.target_fingerprint())
            .or_insert_with(|| (owner, cell, Vec::new()));
        if !Arc::ptr_eq(entry.1, cell) || entry.0 != owner {
            return Err(invalid_resource(
                "distinct backing initialization authorities share a target fingerprint",
            ));
        }
        if !entry
            .2
            .iter()
            .any(|existing| existing.evidence() == authority.evidence())
        {
            entry.2.push(authority);
        }
        Ok(())
    }

    pub(in crate::vnext::resource) fn finish(
        self,
    ) -> impl Iterator<
        Item = (
            BatchParticipantAuthority,
            &'a Arc<BackingInitializationCell>,
            Vec<&'a LogicalBackingSliceAuthority>,
        ),
    > {
        self.groups.into_values().map(|(owner, cell, mut slices)| {
            slices.sort_by(|left, right| {
                left.resource_id().cmp(right.resource_id()).then_with(|| {
                    left.evidence()
                        .physical_offset_bytes()
                        .cmp(&right.evidence().physical_offset_bytes())
                })
            });
            (owner, cell, slices)
        })
    }
}

/// Deduplication preserves first occurrence in the ordered slices and each
/// slice's original segment order; sorting ranges or byte lengths is different.
pub(in crate::vnext::resource) struct InitializationRanges(BTreeSet<(u32, u64, u64, u64)>);

impl InitializationRanges {
    pub(in crate::vnext::resource) fn new() -> Self {
        Self(BTreeSet::new())
    }

    pub(in crate::vnext::resource) fn insert(&mut self, segment: &BackingSegment) -> bool {
        self.0.insert((
            segment.chunk_ordinal(),
            segment.chunk_generation(),
            segment.offset_bytes(),
            segment.length_bytes(),
        ))
    }
}
