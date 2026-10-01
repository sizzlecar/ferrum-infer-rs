//! Stream-owned, bounded passive inventory prepared by the real cache writer.
//! A root contains only numeric metadata; no executable or resource is pinned.
use super::*;
use crate::vnext::device::observation::{
    DeviceCostCatalogPayloadLease, DeviceCostCatalogReservationError,
};
use crate::vnext::{DeviceObservationTemplateBudget, ExecutionLaneId};
use std::{mem::size_of, sync::Arc};

/// A quota refusal records this build's own charged bytes plus the refused
/// allocation at that instant. It excludes other owners and survives rollback.
/// It is a lower bound for retry admission, not a promise the build will fit.
#[derive(Debug)]
pub enum DevicePreparedCostGraphCatalogBuildError {
    Capacity {
        required_exclusive_peak_bytes: usize,
    },
    Rejected(VNextError),
}
impl From<VNextError> for DevicePreparedCostGraphCatalogBuildError {
    fn from(error: VNextError) -> Self {
        Self::Rejected(error)
    }
}
impl From<DeviceCostCatalogReservationError> for DevicePreparedCostGraphCatalogBuildError {
    fn from(error: DeviceCostCatalogReservationError) -> Self {
        match error {
            DeviceCostCatalogReservationError::Capacity {
                required_exclusive_peak_bytes,
            } => Self::Capacity {
                required_exclusive_peak_bytes,
            },
            DeviceCostCatalogReservationError::Invalid => Self::Rejected(invalid(
                "prepared graph catalog reservation overflow or invalid lease",
            )),
        }
    }
}
impl std::fmt::Display for DevicePreparedCostGraphCatalogBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity { required_exclusive_peak_bytes } => write!(f,
                "prepared graph catalog requires at least {required_exclusive_peak_bytes} exclusive metadata bytes"),
            Self::Rejected(error) => std::fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for DevicePreparedCostGraphCatalogBuildError {}

#[derive(Debug)]
struct Owner {
    runtime_instance: u64,
    stream_instance: u64,
    runtime_fingerprint: Box<str>,
    lane: ExecutionLaneId,
}

/// One real stream, bound once by ExecutionLane creation. Only a cache writer
/// owns this mutable source. Generations use private pointer identity, so a
/// recycled/wrapped integer cannot make an older root current again.
#[derive(Debug)]
pub struct DeviceCostGraphCatalogSource {
    owner: Arc<Owner>,
    generation: Arc<()>,
}
impl DeviceCostGraphCatalogSource {
    pub fn new(
        runtime_instance: u64,
        stream_instance: u64,
        runtime_fingerprint: &str,
        lane: ExecutionLaneId,
    ) -> Result<Self, VNextError> {
        if runtime_instance == 0
            || stream_instance == 0
            || runtime_fingerprint.len() != 64
            || !runtime_fingerprint
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(invalid(
                "prepared graph catalog requires a bound real stream owner",
            ));
        }
        Ok(Self {
            owner: Arc::new(Owner {
                runtime_instance,
                stream_instance,
                runtime_fingerprint: runtime_fingerprint.into(),
                lane,
            }),
            generation: Arc::new(()),
        })
    }
    pub fn matches_owner(
        &self,
        runtime_instance: u64,
        stream_instance: u64,
        fingerprint: &str,
    ) -> bool {
        self.owner.runtime_instance == runtime_instance
            && self.owner.stream_instance == stream_instance
            && self.owner.runtime_fingerprint.as_ref() == fingerprint
    }
    pub fn lane_id(&self) -> ExecutionLaneId {
        self.owner.lane
    }
    pub fn invalidate(&mut self) {
        self.generation = Arc::new(());
    }

    pub fn begin(
        &self,
        state: DeviceCostGraphStreamState,
        budget: &Arc<DeviceObservationTemplateBudget>,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<DevicePreparedCostGraphCatalogBuilder, DevicePreparedCostGraphCatalogBuildError>
    {
        poll()?;
        let count = usize::try_from(state.resident_programs)
            .map_err(|_| invalid("prepared graph program count exceeds host capacity"))?;
        let slots = count
            .checked_mul(size_of::<DeviceCostGraphProgram>())
            .ok_or_else(capacity)?;
        // Both the builder's Vec and a possible finish-to-Box allocation are
        // reserved before either can exist. The temporary half is released
        // only after conversion. Owned ID strings/arrays are charged per push.
        let fixed = size_of::<DevicePreparedCostGraphCatalog>()
            .checked_add(size_of::<DevicePreparedCostGraphCatalogBuilder>())
            .and_then(|n| n.checked_add(size_of::<Owner>() + self.owner.runtime_fingerprint.len()))
            .and_then(|n| n.checked_add(6 * size_of::<usize>()))
            .and_then(|n| n.checked_add(slots.checked_mul(2)?))
            .ok_or_else(capacity)?;
        let lease = budget.reserve_cost_catalog(fixed)?;
        let mut programs = Vec::new();
        programs.try_reserve_exact(count).map_err(|_| capacity())?;
        if programs.capacity() != count {
            return Err(
                invalid("prepared graph catalog allocation exceeded its reservation").into(),
            );
        }
        poll()?;
        Ok(DevicePreparedCostGraphCatalogBuilder {
            owner: Arc::clone(&self.owner),
            generation: Arc::clone(&self.generation),
            state,
            programs,
            expected: count,
            nodes: 0,
            commands: 0,
            temporary: slots,
            lease,
            failed: false,
        })
    }
}

#[derive(Debug)]
pub enum DevicePreparedCostGraphCatalogAvailability {
    /// This optional producer is not implemented. The old bounded catalog
    /// producer remains available; graph-unsupported runtimes keep eager.
    Unsupported,
    /// A supported producer has no complete current root. Never lazy-build or
    /// fall back to an older numeric root when a writer left it dirty.
    Unprepared,
    Ready(Arc<DevicePreparedCostGraphCatalog>),
}

#[derive(Debug)]
pub struct DevicePreparedCostGraphCatalog {
    catalog: DeviceCostGraphCatalog,
    owner: Arc<Owner>,
    generation: Arc<()>,
    // Field order keeps the owned metadata alive under charge until dropped.
    _lease: DeviceCostCatalogPayloadLease,
}
impl DevicePreparedCostGraphCatalog {
    pub fn catalog(&self) -> &DeviceCostGraphCatalog {
        &self.catalog
    }
    pub fn is_current(
        &self,
        source: &DeviceCostGraphCatalogSource,
        state: DeviceCostGraphStreamState,
    ) -> bool {
        Arc::ptr_eq(&self.owner, &source.owner)
            && Arc::ptr_eq(&self.generation, &source.generation)
            && self.catalog.stream_state == state
    }
    pub fn matches_runtime_lane(&self, fingerprint: &str, lane: ExecutionLaneId) -> bool {
        self.owner.runtime_fingerprint.as_ref() == fingerprint && self.owner.lane == lane
    }
    pub fn same_generation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner)
            && Arc::ptr_eq(&self.generation, &other.generation)
            && self.catalog.stream_state == other.catalog.stream_state
    }
    pub fn lookup(
        &self,
        id: &DeviceReusableExecutionProgramId,
        selected: DeviceCostGraphCatalogLimits,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<Option<&DeviceCostGraphProgram>, VNextError> {
        poll()?;
        if !self.matches_runtime_lane(id.runtime_implementation_fingerprint(), id.lane_id()) {
            return Err(invalid(
                "prepared graph lookup differs from its bound stream lane",
            ));
        }
        lookup(&self.catalog, id, selected, poll)
    }
}

/// Capture-private compatibility wrapper. A partial record is never promoted
/// into this complete inventory. Prepared roots compare writer generation;
/// legacy runtimes retain their existing complete-value comparison.
#[derive(Debug, Clone)]
pub(crate) enum DeviceCostGraphCatalogSnapshot {
    Legacy(Arc<DeviceCostGraphCatalog>),
    Prepared(Arc<DevicePreparedCostGraphCatalog>),
}
impl DeviceCostGraphCatalogSnapshot {
    pub(crate) fn catalog(&self) -> &DeviceCostGraphCatalog {
        match self {
            Self::Legacy(c) => c,
            Self::Prepared(c) => c.catalog(),
        }
    }
    pub(crate) fn lookup(
        &self,
        id: &DeviceReusableExecutionProgramId,
        selected: DeviceCostGraphCatalogLimits,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<Option<&DeviceCostGraphProgram>, VNextError> {
        match self {
            Self::Legacy(c) => lookup(c, id, selected, poll),
            Self::Prepared(c) => c.lookup(id, selected, poll),
        }
    }
}
impl PartialEq for DeviceCostGraphCatalogSnapshot {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Legacy(a), Self::Legacy(b)) => a == b,
            (Self::Prepared(a), Self::Prepared(b)) => a.same_generation(b),
            _ => false,
        }
    }
}

fn lookup<'a>(
    catalog: &'a DeviceCostGraphCatalog,
    id: &DeviceReusableExecutionProgramId,
    selected: DeviceCostGraphCatalogLimits,
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
) -> Result<Option<&'a DeviceCostGraphProgram>, VNextError> {
    poll()?;
    let found = catalog
        .programs
        .binary_search_by(|p| p.program.program_id().cmp(id));
    let result = if let Ok(index) = found {
        let program = &catalog.programs[index];
        if program.program.node_count() as usize > selected.maximum_nodes {
            return Err(invalid("selected graph program node limit exceeded"));
        }
        let mut commands = 0usize;
        for segment in &program.uploaded_segments {
            poll()?;
            commands = commands
                .checked_add(segment.logical_commands.len())
                .filter(|n| *n <= selected.maximum_logical_commands)
                .ok_or_else(|| invalid("selected graph logical command limit exceeded"))?;
        }
        Some(program)
    } else {
        None
    };
    // Absence must obey the same original query deadline as a hit.
    poll()?;
    Ok(result)
}

fn capacity() -> VNextError {
    invalid("prepared graph catalog metadata allocation overflows or is unavailable")
}

/// A poisoned build cannot publish a truncated directory. Every new owned
/// allocation is charged to the installed shared CPU ledger before copying.
pub struct DevicePreparedCostGraphCatalogBuilder {
    owner: Arc<Owner>,
    generation: Arc<()>,
    state: DeviceCostGraphStreamState,
    programs: Vec<DeviceCostGraphProgram>,
    expected: usize,
    nodes: usize,
    commands: usize,
    temporary: usize,
    lease: DeviceCostCatalogPayloadLease,
    failed: bool,
}
impl DevicePreparedCostGraphCatalogBuilder {
    pub fn push_program(
        &mut self,
        program: &DeviceReusableExecutionProgram,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<(), DevicePreparedCostGraphCatalogBuildError> {
        let failed = self.failed;
        self.failed = true;
        poll()?;
        if failed
            || self.programs.len() >= self.expected
            || program.program_id().lane_id() != self.owner.lane
            || program.program_id().runtime_implementation_fingerprint()
                != self.owner.runtime_fingerprint.as_ref()
        {
            return Err(invalid(
                "prepared graph program differs from the bound owner or population",
            )
            .into());
        }
        let nodes = self
            .nodes
            .checked_add(program.node_count() as usize)
            .ok_or_else(capacity)?;
        let id = program.program_id();
        let strings = [
            id.plan_hash().as_str(),
            id.runtime_implementation_fingerprint(),
            id.bucket_id().as_str(),
            id.program_binding_layout_fingerprint(),
            id.lane_stable_layout_fingerprint(),
        ]
        .into_iter()
        .try_fold(0usize, |n, text| n.checked_add(text.len()))
        .ok_or_else(capacity)?;
        let arrays = program
            .eager_boundary_node_indices
            .len()
            .checked_mul(size_of::<u32>())
            .and_then(|n| {
                n.checked_add(
                    program
                        .segments
                        .len()
                        .checked_mul(size_of::<DeviceReusableExecutionSegment>())?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    program
                        .per_wave_binding_node_indices
                        .len()
                        .checked_mul(size_of::<u32>())?,
                )
            })
            .and_then(|n| {
                n.checked_add(program.gaps.len().checked_mul(size_of::<
                    super::super::super::DeviceReusableExecutionProgramGap,
                >())?)
            })
            .ok_or_else(capacity)?;
        let uploaded = program
            .segments
            .len()
            .checked_mul(size_of::<DeviceCostGraphUploadedSegment>())
            .ok_or_else(capacity)?;
        let additional = strings
            .checked_add(arrays.checked_mul(2).ok_or_else(capacity)?)
            .and_then(|n| n.checked_add(uploaded))
            .ok_or_else(capacity)?;
        self.lease.grow(additional)?;
        let mut uploaded_segments = Vec::new();
        uploaded_segments
            .try_reserve_exact(program.segments.len())
            .map_err(|_| capacity())?;
        if uploaded_segments.capacity() != program.segments.len() {
            return Err(
                invalid("prepared graph segment allocation exceeded its reservation").into(),
            );
        }
        let copied = DeviceReusableExecutionProgram {
            program_id: id.clone(),
            node_count: program.node_count,
            eager_boundary_node_indices: copy_prepared_rows(
                &program.eager_boundary_node_indices,
                poll,
            )?,
            segments: copy_prepared_rows(&program.segments, poll)?,
            per_wave_binding_node_indices: copy_prepared_rows(
                &program.per_wave_binding_node_indices,
                poll,
            )?,
            gaps: copy_prepared_rows(&program.gaps, poll)?,
        };
        self.lease.release_temporary(arrays);
        self.programs.push(DeviceCostGraphProgram {
            program: copied,
            uploaded_segments,
        });
        self.nodes = nodes;
        self.failed = false;
        Ok(())
    }

    pub fn push_uploaded_segment(
        &mut self,
        segment: &DeviceReusableExecutionSegment,
        fingerprint: &str,
        logical: &[DeviceReplayedLogicalCommandAttribution],
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<(), DevicePreparedCostGraphCatalogBuildError> {
        let failed = self.failed;
        self.failed = true;
        poll()?;
        let current = self
            .programs
            .last_mut()
            .ok_or_else(|| invalid("prepared graph segment has no current program"))?;
        if failed
            || current.program.segments().get(segment.ordinal() as usize) != Some(segment)
            || current
                .uploaded_segments
                .last()
                .is_some_and(|p| p.segment.ordinal() >= segment.ordinal())
            || fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || logical.len() != segment.logical_command_count() as usize
            || segment
                .start_node_index()
                .checked_add(segment.logical_command_count())
                != Some(segment.end_node_index())
            || self.state.resident_executables == 0
        {
            return Err(invalid(
                "prepared graph uploaded segment differs from its sealed descriptor",
            )
            .into());
        }
        let commands = self
            .commands
            .checked_add(logical.len())
            .ok_or_else(capacity)?;
        let rows_bytes = logical
            .len()
            .checked_mul(size_of::<DeviceReplayedLogicalCommandAttribution>())
            .ok_or_else(capacity)?;
        let bytes = rows_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(fingerprint.len()))
            .ok_or_else(capacity)?;
        self.lease.grow(bytes)?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(logical.len())
            .map_err(|_| capacity())?;
        if rows.capacity() != logical.len() {
            return Err(invalid("prepared graph row allocation exceeded its reservation").into());
        }
        for (ordinal, row) in logical.iter().enumerate() {
            poll()?;
            // Only passive templates are copied, never captured sample/work
            // tables or their potentially shared dynamic algorithm payload.
            if row.statistical_evidence.is_some()
                || u32::try_from(ordinal).ok() != Some(row.logical_command_ordinal())
                || segment
                    .start_node_index()
                    .checked_add(row.logical_command_ordinal())
                    != Some(row.node_index())
            {
                return Err(invalid(
                    "prepared graph logical rows are incomplete or carry live sample work",
                )
                .into());
            }
            rows.push(row.clone());
        }
        let logical_commands = rows.into_boxed_slice();
        self.lease.release_temporary(rows_bytes);
        current
            .uploaded_segments
            .push(DeviceCostGraphUploadedSegment {
                segment: segment.clone(),
                reusable_executable_fingerprint: fingerprint.to_owned(),
                logical_commands,
            });
        self.commands = commands;
        self.failed = false;
        Ok(())
    }

    pub fn finish(
        mut self,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<Arc<DevicePreparedCostGraphCatalog>, DevicePreparedCostGraphCatalogBuildError> {
        poll()?;
        if self.failed || self.programs.len() != self.expected {
            return Err(invalid(
                "prepared graph catalog omits programs or contains rejected metadata",
            )
            .into());
        }
        sort_programs(&mut self.programs, poll)?;
        for pair in self.programs.windows(2) {
            poll()?;
            if pair[0].program.program_id() == pair[1].program.program_id() {
                return Err(invalid(
                    "prepared graph catalog contains duplicate program identities",
                )
                .into());
            }
        }
        let programs = self.programs.into_boxed_slice();
        poll()?;
        // All temporary allocation conversions completed before releasing
        // their conservative construction allowance.
        self.lease.release_temporary(self.temporary);
        Ok(Arc::new(DevicePreparedCostGraphCatalog {
            catalog: DeviceCostGraphCatalog {
                stream_state: self.state,
                programs,
                node_count: self.nodes,
                logical_command_count: self.commands,
            },
            owner: self.owner,
            generation: self.generation,
            _lease: self.lease,
        }))
    }
}

fn copy_prepared_rows<T: Clone>(
    input: &[T],
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
) -> Result<Box<[T]>, VNextError> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(input.len())
        .map_err(|_| capacity())?;
    if rows.capacity() != input.len() {
        return Err(invalid(
            "prepared graph metadata allocation exceeded its reservation",
        ));
    }
    for row in input {
        poll()?;
        rows.push(row.clone());
    }
    Ok(rows.into_boxed_slice())
}

// In-place heapsort keeps the original canonical full-ID order without a
// second index/tree allocation and can stop promptly on the caller's budget.
fn sort_programs(
    values: &mut [DeviceCostGraphProgram],
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
) -> Result<(), VNextError> {
    fn sift(
        values: &mut [DeviceCostGraphProgram],
        mut root: usize,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<(), VNextError> {
        while let Some(mut child) = root
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .filter(|n| *n < values.len())
        {
            poll()?;
            if child + 1 < values.len()
                && values[child].program.program_id() < values[child + 1].program.program_id()
            {
                child += 1;
            }
            if values[root].program.program_id() >= values[child].program.program_id() {
                break;
            }
            values.swap(root, child);
            root = child;
        }
        Ok(())
    }
    for root in (0..values.len() / 2).rev() {
        sift(values, root, poll)?;
    }
    for end in (1..values.len()).rev() {
        poll()?;
        values.swap(0, end);
        sift(&mut values[..end], 0, poll)?;
    }
    poll()
}

#[cfg(test)]
mod tests;
