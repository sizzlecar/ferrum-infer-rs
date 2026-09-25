use super::*;

/// Complete selected sub-work of ONE actual or projected physical command.
/// Private fields prevent independent mutation of its family/work/counts.
/// Only serialization is provided: old profiles cannot deserialize/invent it.
#[derive(Debug, Clone, Serialize)]
pub struct SelectedCommandCostEvidenceV1 {
    schema_version: u32,
    family_signature: [u8; 32],
    token_count: u64,
    compute_dispatches: u64,
    transfer_commands: u64,
    work: DeviceNumericWorkV1,
    /// The old wire never exports a new empirical family implicitly.
    #[serde(skip)]
    independent_attention_family_v2: Option<[u8; 32]>,
    /// Same-push numeric assignment, unavailable from the old aggregate hash.
    #[serde(skip)]
    algorithm_assignment_signature: Option<[u8; 32]>,
    #[serde(skip)]
    algorithm_work:
        Option<Result<std::sync::Arc<SelectedAlgorithmWorkEvidenceV1>, StatisticalEvidenceUnknown>>,
}
impl PartialEq for SelectedCommandCostEvidenceV1 {
    fn eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.family_signature == other.family_signature
            && self.token_count == other.token_count
            && self.compute_dispatches == other.compute_dispatches
            && self.transfer_commands == other.transfer_commands
            && self.work == other.work
    }
}
impl Eq for SelectedCommandCostEvidenceV1 {}
impl SelectedCommandCostEvidenceV1 {
    pub fn algorithm_work(
        &self,
    ) -> Option<Result<&SelectedAlgorithmWorkEvidenceV1, StatisticalEvidenceUnknown>> {
        self.algorithm_work
            .as_ref()
            .map(|value| value.as_deref().map_err(|error| *error))
    }
    pub(super) fn token_count(&self) -> u64 {
        self.token_count
    }
    pub(super) fn algorithm_work_binding(&self) -> Result<[u8; 32], StatisticalEvidenceUnknown> {
        let assignment = self
            .algorithm_assignment_signature
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)?;
        let mut hash = Sha256::new();
        bytes(&mut hash, b"ferrum.algorithm-work-command-binding.v1");
        hash.update(self.family_signature);
        hash.update(assignment);
        for n in [
            u64::from(self.schema_version),
            self.token_count,
            self.compute_dispatches,
            self.transfer_commands,
            self.work.logical_units,
            self.work.padded_units,
            self.work.inner_work_units,
            self.work.grid_blocks,
            self.work.peak_scratch_bytes,
            self.work.staged_weight_bytes,
            self.work.host_to_device_bytes,
            self.work.device_to_host_bytes,
            self.work.device_to_device_bytes,
            self.work.fill_bytes,
        ] {
            number(&mut hash, n);
        }
        Ok(hash.finalize().into())
    }
    pub fn independent_attention_family_v2(&self) -> Option<&[u8; 32]> {
        self.independent_attention_family_v2.as_ref()
    }
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
    independent: super::independent_rows::IndependentRowsDigest,
    tokens: u64,
    compute: u64,
    transfers: u64,
    work: DeviceNumericWorkV1,
    failure: Option<StatisticalEvidenceUnknown>,
    algorithm_work: Option<super::algorithm_work::AlgorithmWorkAccumulator>,
}
impl SelectedCommandCostBuilderV1 {
    pub fn new(token_count: u64) -> Self {
        let mut family = Sha256::new();
        bytes(&mut family, b"ferrum.selected-command-cost.v1");
        Self {
            family,
            independent: super::independent_rows::IndependentRowsDigest::Ordered,
            tokens: token_count,
            compute: 0,
            transfers: 0,
            work: DeviceNumericWorkV1::default(),
            failure: None,
            algorithm_work: None,
        }
    }
    /// Collect sparse work by selected algorithm while retaining the legacy
    /// evidence unchanged. The default constructor does not allocate this table.
    pub fn new_with_algorithm_work(token_count: u64) -> Self {
        let mut builder = Self::new(token_count);
        builder.algorithm_work = Some(super::algorithm_work::AlgorithmWorkAccumulator::new());
        builder
    }
    /// Declare one compute-only group of complete, independent attention row
    /// blocks. The provider must establish actual/future address independence
    /// and the same packed decode selector before calling. This changes only
    /// the optional V2 empirical digest; V1, numeric work and execution stay
    /// ordered. No independence is inferred from owner IDs or kernel labels.
    pub fn independent_attention_rows_v2<T>(
        &mut self,
        rows: impl ExactSizeIterator<Item = T>,
        mut emit: impl FnMut(&mut Self, T) -> Result<(), StatisticalEvidenceUnknown>,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        self.guard(|this| {
            this.independent.begin(&this.family, rows.len());
            for row in rows {
                this.independent.begin_row();
                emit(this, row)?;
                this.independent.end_row();
            }
            this.independent.end();
            Ok(())
        })
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
            let numeric = DeviceNumericWorkV1 {
                logical_units: work.logical_units,
                padded_units: work.padded_units,
                inner_work_units,
                grid_blocks,
                peak_scratch_bytes: work.scratch_bytes,
                staged_weight_bytes: work.staged_weight_bytes,
                ..Default::default()
            };
            let next = this.work.checked_add(numeric)?;
            number(&mut this.family, 0);
            this.family.update(algorithm.0);
            this.independent.kernel(algorithm);
            this.compute += 1;
            this.work = next;
            if let Some(capture) = &mut this.algorithm_work {
                capture.observe(algorithm, AlgorithmWorkKindV1::Kernel, numeric);
            }
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
            this.independent.transfer(algorithm, tag);
            this.transfers += 1;
            this.work = next;
            if let Some(capture) = &mut this.algorithm_work {
                capture.observe(algorithm, kind.into(), work);
            }
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
        let family_signature = self.family.finalize().into();
        let independent_attention_family_v2 = self.independent.finish(family_signature);
        let mut value = SelectedCommandCostEvidenceV1 {
            schema_version: STATISTICAL_ROUTE_WORK_SCHEMA_V1,
            family_signature,
            independent_attention_family_v2,
            token_count: self.tokens,
            compute_dispatches: self.compute,
            transfer_commands: self.transfers,
            work: self.work,
            algorithm_assignment_signature: self
                .algorithm_work
                .as_ref()
                .and_then(|capture| capture.assignment_signature()),
            algorithm_work: None,
        };
        value.algorithm_work = self
            .algorithm_work
            .map(|capture| capture.finish(&value).map(std::sync::Arc::new));
        Ok(value)
    }
}
