//! Immutable catalog index retaining its actual lane-snapshot provenance.
use super::*;
use crate::vnext::DeviceReusableExecutionProgramId;

#[derive(Debug, Clone)]
pub struct IndexedExecutionLaneReusableCatalog {
    lane_id: Option<ExecutionLaneId>,
    epoch: u64,
    programs: BTreeMap<DeviceReusableExecutionProgramId, DeviceReusableExecutionProgram>,
    observed: bool,
}
impl ExecutionLaneReusableExecutionCatalog {
    /// Index one complete, privately produced lane snapshot without cloning its
    /// program payload. Duplicate IDs invalidate the snapshot.
    pub fn into_index(self) -> Result<IndexedExecutionLaneReusableCatalog, VNextError> {
        let mut programs = BTreeMap::new();
        for program in self.programs {
            if programs
                .insert(program.program_id().clone(), program)
                .is_some()
            {
                return Err(invalid_completion(
                    "duplicate program in sealed lane catalog",
                ));
            }
        }
        Ok(IndexedExecutionLaneReusableCatalog {
            lane_id: Some(self.lane_id),
            epoch: self.epoch,
            programs,
            observed: true,
        })
    }
}
impl IndexedExecutionLaneReusableCatalog {
    /// Existing executors install an empty placeholder when reuse is disabled.
    /// It is not evidence that a particular program was absent from a catalog.
    pub fn unobserved_empty(epoch: u64) -> Self {
        Self {
            lane_id: None,
            epoch,
            programs: BTreeMap::new(),
            observed: false,
        }
    }
    pub(crate) const fn lane_id(&self) -> Option<ExecutionLaneId> {
        self.lane_id
    }
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn programs(
        &self,
    ) -> &BTreeMap<DeviceReusableExecutionProgramId, DeviceReusableExecutionProgram> {
        &self.programs
    }
    pub(crate) const fn is_observed(&self) -> bool {
        self.observed
    }
}
