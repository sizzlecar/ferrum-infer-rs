//! Numeric inventory of a quiescent stream, not executable ownership. A program
//! descriptor may be present with only a subset of its segments uploaded. A
//! consumer must find every segment it intends to replay; counts cannot fill a
//! missing segment or authenticate its logical work.
use super::super::{
    DeviceReplayedLogicalCommandAttribution, DeviceReusableExecutionProgram,
    DeviceReusableExecutionProgramId, DeviceReusableExecutionSegment, VNextError,
};
use super::DeviceCostGraphStreamState;
use serde::Serialize;
use std::collections::BTreeMap;

fn invalid(reason: &'static str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.to_owned(),
    }
}

/// Aggregate allocation limits, applied before copying borrowed live metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCostGraphCatalogLimits {
    maximum_programs: usize,
    maximum_nodes: usize,
    maximum_logical_commands: usize,
}
impl DeviceCostGraphCatalogLimits {
    pub fn new(programs: usize, nodes: usize, commands: usize) -> Result<Self, VNextError> {
        if programs == 0 || nodes == 0 || commands == 0 {
            return Err(invalid("graph catalog limits must be nonzero"));
        }
        Ok(Self {
            maximum_programs: programs,
            maximum_nodes: nodes,
            maximum_logical_commands: commands,
        })
    }
    pub const fn maximum_programs(self) -> usize {
        self.maximum_programs
    }
    pub const fn maximum_nodes(self) -> usize {
        self.maximum_nodes
    }
    pub const fn maximum_logical_commands(self) -> usize {
        self.maximum_logical_commands
    }
}

/// No physical command index exists until a particular wave chooses a replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceCostGraphUploadedSegment {
    segment: DeviceReusableExecutionSegment,
    reusable_executable_fingerprint: String,
    logical_commands: Box<[DeviceReplayedLogicalCommandAttribution]>,
}
impl DeviceCostGraphUploadedSegment {
    pub fn segment(&self) -> &DeviceReusableExecutionSegment {
        &self.segment
    }
    pub fn reusable_executable_fingerprint(&self) -> &str {
        &self.reusable_executable_fingerprint
    }
    pub fn logical_commands(&self) -> &[DeviceReplayedLogicalCommandAttribution] {
        &self.logical_commands
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceCostGraphProgram {
    program: DeviceReusableExecutionProgram,
    uploaded_segments: Vec<DeviceCostGraphUploadedSegment>,
}
impl DeviceCostGraphProgram {
    pub fn program(&self) -> &DeviceReusableExecutionProgram {
        &self.program
    }
    pub fn uploaded_segments(&self) -> &[DeviceCostGraphUploadedSegment] {
        &self.uploaded_segments
    }
}

/// Canonically ordered value data. Equality includes original program IDs,
/// full descriptors, uploaded executable fingerprints and every logical row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceCostGraphCatalog {
    stream_state: DeviceCostGraphStreamState,
    programs: Box<[DeviceCostGraphProgram]>,
    node_count: usize,
    logical_command_count: usize,
}
impl DeviceCostGraphCatalog {
    pub const fn stream_state(&self) -> DeviceCostGraphStreamState {
        self.stream_state
    }
    pub fn programs(&self) -> &[DeviceCostGraphProgram] {
        &self.programs
    }
    pub fn fits(&self, limits: DeviceCostGraphCatalogLimits) -> bool {
        self.programs.len() <= limits.maximum_programs
            && self.node_count <= limits.maximum_nodes
            && self.logical_command_count <= limits.maximum_logical_commands
    }
}

/// Push one borrowed descriptor, then its uploaded segments in ordinal order.
/// Program input order is arbitrary. A failed operation poisons this builder:
/// callers cannot ignore a rejected row and publish a truncated inventory.
pub struct DeviceCostGraphCatalogBuilder {
    state: DeviceCostGraphStreamState,
    limits: DeviceCostGraphCatalogLimits,
    programs: BTreeMap<DeviceReusableExecutionProgramId, DeviceCostGraphProgram>,
    current: Option<DeviceReusableExecutionProgramId>,
    nodes: usize,
    commands: usize,
    failed: bool,
}
impl DeviceCostGraphCatalogBuilder {
    pub fn new(
        state: DeviceCostGraphStreamState,
        limits: DeviceCostGraphCatalogLimits,
    ) -> Result<Self, VNextError> {
        if state.resident_programs > limits.maximum_programs as u64 {
            return Err(invalid("graph catalog program limit exceeded"));
        }
        Ok(Self {
            state,
            limits,
            programs: BTreeMap::new(),
            current: None,
            nodes: 0,
            commands: 0,
            failed: false,
        })
    }
    pub fn push_program(
        &mut self,
        program: &DeviceReusableExecutionProgram,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<(), VNextError> {
        let was_failed = self.failed;
        self.failed = true;
        poll()?;
        if was_failed
            || self.programs.len() >= self.limits.maximum_programs
            || self.programs.contains_key(program.program_id())
        {
            return Err(invalid(
                "graph catalog contains a duplicate program or exceeds limits",
            ));
        }
        let nodes = self
            .nodes
            .checked_add(program.node_count() as usize)
            .filter(|n| *n <= self.limits.maximum_nodes)
            .ok_or_else(|| invalid("graph catalog node limit exceeded"))?;
        // The original descriptor is validated and immutable. Copy its bounded
        // arrays with cancellation points rather than one uninterruptible clone.
        let copied = DeviceReusableExecutionProgram {
            program_id: program.program_id.clone(),
            node_count: program.node_count,
            eager_boundary_node_indices: copy_rows(&program.eager_boundary_node_indices, poll)?
                .into_boxed_slice(),
            segments: copy_rows(&program.segments, poll)?.into_boxed_slice(),
            per_wave_binding_node_indices: copy_rows(&program.per_wave_binding_node_indices, poll)?
                .into_boxed_slice(),
            gaps: copy_rows(&program.gaps, poll)?.into_boxed_slice(),
        };
        let id = copied.program_id.clone();
        self.programs.insert(
            id.clone(),
            DeviceCostGraphProgram {
                program: copied,
                uploaded_segments: Vec::new(),
            },
        );
        self.current = Some(id);
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
    ) -> Result<(), VNextError> {
        let was_failed = self.failed;
        self.failed = true;
        poll()?;
        let current = self
            .current
            .as_ref()
            .and_then(|id| self.programs.get_mut(id))
            .ok_or_else(|| invalid("graph segment has no current program"))?;
        if was_failed
            || current.program.segments().get(segment.ordinal() as usize) != Some(segment)
            || current
                .uploaded_segments
                .last()
                .is_some_and(|last| last.segment.ordinal() >= segment.ordinal())
            || fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || logical.len() != segment.logical_command_count() as usize
            || segment
                .start_node_index()
                .checked_add(segment.logical_command_count())
                != Some(segment.end_node_index())
        {
            return Err(invalid(
                "graph uploaded segment differs from its sealed descriptor",
            ));
        }
        let commands = self
            .commands
            .checked_add(logical.len())
            .filter(|n| *n <= self.limits.maximum_logical_commands)
            .ok_or_else(|| invalid("graph catalog logical command limit exceeded"))?;
        let mut rows = Vec::with_capacity(logical.len());
        for (ordinal, row) in logical.iter().enumerate() {
            poll()?;
            if u32::try_from(ordinal).ok() != Some(row.logical_command_ordinal())
                || segment
                    .start_node_index()
                    .checked_add(row.logical_command_ordinal())
                    != Some(row.node_index())
            {
                return Err(invalid("graph logical rows are incomplete or out of order"));
            }
            rows.push(row.clone());
        }
        current
            .uploaded_segments
            .push(DeviceCostGraphUploadedSegment {
                segment: segment.clone(),
                reusable_executable_fingerprint: fingerprint.to_owned(),
                logical_commands: rows.into_boxed_slice(),
            });
        self.commands = commands;
        self.failed = false;
        Ok(())
    }
    pub fn finish(
        self,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<DeviceCostGraphCatalog, VNextError> {
        poll()?;
        if self.failed || self.programs.len() as u64 != self.state.resident_programs {
            return Err(invalid(
                "graph catalog omits resident programs or contains rejected metadata",
            ));
        }
        let mut programs = Vec::with_capacity(self.programs.len());
        for (_, program) in self.programs {
            poll()?;
            // Entries may be shared between program descriptors. Do not infer
            // a one-to-one relation from executable/program/segment counts.
            if self.state.resident_executables == 0 && !program.uploaded_segments.is_empty() {
                return Err(invalid("uploaded graph segment has no resident executable"));
            }
            programs.push(program);
        }
        Ok(DeviceCostGraphCatalog {
            stream_state: self.state,
            programs: programs.into_boxed_slice(),
            node_count: self.nodes,
            logical_command_count: self.commands,
        })
    }
}
fn copy_rows<T: Clone>(
    rows: &[T],
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
) -> Result<Vec<T>, VNextError> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        poll()?;
        out.push(row.clone());
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
