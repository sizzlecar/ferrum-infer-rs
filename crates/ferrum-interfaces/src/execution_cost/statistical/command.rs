use super::*;

/// Complete selected sub-work of ONE actual or projected physical command.
/// Private fields prevent independent mutation of its family/work/counts.
/// Only serialization is provided: old profiles cannot deserialize/invent it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectedCommandCostEvidenceV1 {
    schema_version: u32,
    family_signature: [u8; 32],
    token_count: u64,
    compute_dispatches: u64,
    transfer_commands: u64,
    work: DeviceNumericWorkV1,
}
impl SelectedCommandCostEvidenceV1 {
    pub fn family_signature(&self) -> &[u8; 32] {
        &self.family_signature
    }
    pub fn work(&self) -> DeviceNumericWorkV1 {
        self.work
    }
    pub fn validate_command(
        &self,
        tokens: u64,
        compute: u64,
        transfers: u64,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.schema_version != STATISTICAL_ROUTE_WORK_SCHEMA_V1
            || self.token_count != tokens
            || self.compute_dispatches != compute
            || self.transfer_commands != transfers
        {
            Err(StatisticalEvidenceUnknown::CommandMismatch)
        } else {
            Ok(())
        }
    }
}

/// Every push denotes a selected real dispatch/copy. Partial evidence may not
/// be completed by defaulting unknown kernels to zero work. Errors are sticky.
pub struct SelectedCommandCostBuilderV1 {
    family: Sha256,
    tokens: u64,
    compute: u64,
    transfers: u64,
    work: DeviceNumericWorkV1,
    failure: Option<StatisticalEvidenceUnknown>,
}
impl SelectedCommandCostBuilderV1 {
    pub fn new(token_count: u64) -> Self {
        let mut family = Sha256::new();
        bytes(&mut family, b"ferrum.selected-command-cost.v1");
        Self {
            family,
            tokens: token_count,
            compute: 0,
            transfers: 0,
            work: DeviceNumericWorkV1::default(),
            failure: None,
        }
    }
    fn guard(
        &mut self,
        apply: impl FnOnce(&mut Self) -> Result<(), StatisticalEvidenceUnknown>,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = apply(self);
        if let Err(error) = result {
            self.failure = Some(error);
        }
        result
    }
    fn check_limit(&self) -> Result<(), StatisticalEvidenceUnknown> {
        if self.compute + self.transfers >= MAX_COST_COMMANDS as u64 {
            Err(StatisticalEvidenceUnknown::Capacity)
        } else {
            Ok(())
        }
    }
    pub fn kernel(
        &mut self,
        algorithm: SelectedAlgorithmClassV1,
        work: KernelNumericWorkV1,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        self.guard(|this| {
            this.check_limit()?;
            if work.logical_units == 0
                || work.padded_units < work.logical_units
                || work.inner_units_per_logical_unit == 0
                || work.grid.contains(&0)
            {
                return Err(StatisticalEvidenceUnknown::InvalidWork);
            }
            let inner_work_units = work
                .logical_units
                .checked_mul(work.inner_units_per_logical_unit)
                .ok_or(StatisticalEvidenceUnknown::Overflow)?;
            let grid_blocks = work
                .grid
                .iter()
                .try_fold(1_u64, |a, &b| a.checked_mul(u64::from(b)))
                .ok_or(StatisticalEvidenceUnknown::Overflow)?;
            let next = this.work.checked_add(DeviceNumericWorkV1 {
                logical_units: work.logical_units,
                padded_units: work.padded_units,
                inner_work_units,
                grid_blocks,
                peak_scratch_bytes: work.scratch_bytes,
                staged_weight_bytes: work.staged_weight_bytes,
                ..Default::default()
            })?;
            number(&mut this.family, 0);
            this.family.update(algorithm.0);
            this.compute += 1;
            this.work = next;
            Ok(())
        })
    }
    /// Descriptor commits to alignment/striding/copy mechanism when those
    /// change its selected route. A generic "copy" label is not sufficient.
    pub fn transfer(
        &mut self,
        algorithm: SelectedAlgorithmClassV1,
        kind: StatisticalTransferKindV1,
        count_bytes: u64,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        self.guard(|this| {
            this.check_limit()?;
            if count_bytes == 0 {
                return Err(StatisticalEvidenceUnknown::InvalidWork);
            }
            let mut work = DeviceNumericWorkV1::default();
            let tag = match kind {
                StatisticalTransferKindV1::HostToDevice => {
                    work.host_to_device_bytes = count_bytes;
                    0
                }
                StatisticalTransferKindV1::DeviceToHost => {
                    work.device_to_host_bytes = count_bytes;
                    1
                }
                StatisticalTransferKindV1::DeviceToDevice => {
                    work.device_to_device_bytes = count_bytes;
                    2
                }
                StatisticalTransferKindV1::Fill => {
                    work.fill_bytes = count_bytes;
                    3
                }
            };
            let next = this.work.checked_add(work)?;
            number(&mut this.family, 1);
            this.family.update(algorithm.0);
            number(&mut this.family, tag);
            this.transfers += 1;
            this.work = next;
            Ok(())
        })
    }
    pub fn finish(self) -> Result<SelectedCommandCostEvidenceV1, StatisticalEvidenceUnknown> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if self.compute + self.transfers == 0 {
            return Err(StatisticalEvidenceUnknown::MissingProducer);
        }
        Ok(SelectedCommandCostEvidenceV1 {
            schema_version: STATISTICAL_ROUTE_WORK_SCHEMA_V1,
            family_signature: self.family.finalize().into(),
            token_count: self.tokens,
            compute_dispatches: self.compute,
            transfer_commands: self.transfers,
            work: self.work,
        })
    }
}
