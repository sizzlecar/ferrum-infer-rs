use super::*;

/// Associated with the same exact completed/proposed shape, never executable
/// permission. Private construction, serialize-only; profile1--5 cannot infer it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StatisticalWaveEvidenceV1 {
    pub(super) schema_version: u32,
    pub(super) exact_binding: [u8; 32],
    pub(super) family_signature: [u8; 32],
    pub(super) physical_commands: u32,
    pub(super) work: DeviceNumericWorkV1,
}
impl StatisticalWaveEvidenceV1 {
    pub fn family_signature(&self) -> &[u8; 32] {
        &self.family_signature
    }
    pub fn work(&self) -> DeviceNumericWorkV1 {
        self.work
    }
    pub fn validate_exact(
        &self,
        shape: &CanonicalWaveCostShape,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.schema_version != STATISTICAL_ROUTE_WORK_SCHEMA_V1
            || self.physical_commands == 0
            || self.physical_commands as usize > MAX_COST_COMMANDS
            || self.exact_binding != exact_binding(shape)?
        {
            return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
        }
        Ok(())
    }
}

/// The optional sidecar cannot change old exact validation, hash or equality.
/// Missing/bad statistics remain explicit Unknown beside a valid exact shape.
#[derive(Debug, Clone)]
pub struct CanonicalStatisticalWave {
    pub exact: CanonicalWaveCostShape,
    pub statistical: Result<StatisticalWaveEvidenceV1, StatisticalEvidenceUnknown>,
}

pub(in crate::execution_cost) struct StatisticalWaveAccumulator {
    family: Sha256,
    count: usize,
    work: DeviceNumericWorkV1,
    failure: Option<StatisticalEvidenceUnknown>,
}
impl StatisticalWaveAccumulator {
    pub(in crate::execution_cost) fn new() -> Self {
        let mut family = Sha256::new();
        bytes(&mut family, b"ferrum.statistical-wave-family.v1");
        Self {
            family,
            count: 0,
            work: Default::default(),
            failure: None,
        }
    }
    pub(in crate::execution_cost) fn observe(&mut self, command: CostPhysicalCommand<'_>) {
        if self.failure.is_some() {
            return;
        }
        if let Err(error) = self.append(command) {
            self.failure = Some(error);
        }
    }
    fn append(
        &mut self,
        command: CostPhysicalCommand<'_>,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.count == MAX_COST_COMMANDS {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        // Replay needs sealed logical sub-work evidence. Do not infer it from
        // one physical graph-launch command or reuse an eager classification.
        if command.path != CostCommandPath::Eager || command.reusable_graph_node_count.is_some() {
            return Err(StatisticalEvidenceUnknown::UnsupportedReplay);
        }
        let evidence = command
            .statistical_evidence
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)?;
        evidence.validate_command(
            command.token_count,
            command.compute_dispatch_count,
            command.transfer_command_count,
        )?;
        let next = self.work.checked_add(evidence.work())?;
        number(&mut self.family, u64::from(command.command_index));
        number(
            &mut self.family,
            command.node_index.map_or(u64::MAX, u64::from),
        );
        number(&mut self.family, command.command_phase as u64);
        bytes(&mut self.family, command.native_op_id.as_bytes());
        bytes(&mut self.family, command.batching_form.as_bytes());
        match command.provider {
            Some(provider) => {
                number(&mut self.family, 1);
                bytes(&mut self.family, provider.provider_id.as_bytes());
                bytes(
                    &mut self.family,
                    provider.implementation_fingerprint.as_bytes(),
                );
                bytes(&mut self.family, provider.operation_fingerprint.as_bytes());
            }
            None => number(&mut self.family, 0),
        }
        number(&mut self.family, u64::from(command.participant_start));
        number(&mut self.family, u64::from(command.participant_count));
        number(&mut self.family, command.compute_dispatch_count);
        number(&mut self.family, command.transfer_command_count);
        self.family.update(evidence.family_signature());
        self.work = next;
        self.count += 1;
        Ok(())
    }
    pub(in crate::execution_cost) fn finish(
        mut self,
        shape: &CanonicalWaveCostShape,
    ) -> Result<StatisticalWaveEvidenceV1, StatisticalEvidenceUnknown> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        if self.count == 0 {
            return Err(StatisticalEvidenceUnknown::MissingProducer);
        }
        if shape.path != ActualWavePath::PlanRuntime
            || shape.graph != ActualWaveGraphState::Disabled
        {
            return Err(StatisticalEvidenceUnknown::UnsupportedWave);
        }
        let host = shape
            .row_multiset_features
            .as_ref()
            .ok_or(StatisticalEvidenceUnknown::MissingHostDomain)?;
        host.validate(shape.rows.len())
            .map_err(|_| StatisticalEvidenceUnknown::MissingHostDomain)?;
        number(&mut self.family, shape.kind as u64);
        number(&mut self.family, shape.path as u64);
        number(&mut self.family, shape.graph as u64);
        number(&mut self.family, shape.row_order as u64);
        number(&mut self.family, shape.recurrent_state_bytes);
        self.family.update(host.wave_policy_signature);
        // Initial revision preserves the full physical static role pattern.
        // No assumption of owner/role permutation or cross-width invariance.
        for (row, work) in host.rows.iter().zip(&shape.rows) {
            if !row.role.matches_work(*work) {
                return Err(StatisticalEvidenceUnknown::MissingHostDomain);
            }
            number(&mut self.family, row.role as u64);
            self.family.update(row.categorical_signature);
        }
        Ok(StatisticalWaveEvidenceV1 {
            schema_version: STATISTICAL_ROUTE_WORK_SCHEMA_V1,
            exact_binding: exact_binding(shape)?,
            family_signature: self.family.finalize().into(),
            physical_commands: self.count as u32,
            work: self.work,
        })
    }
}

fn exact_binding(shape: &CanonicalWaveCostShape) -> Result<[u8; 32], StatisticalEvidenceUnknown> {
    exact_binding_parts(
        shape.kind,
        shape.path,
        shape.graph,
        shape.row_order,
        shape.provider_signature,
        shape.output_policy_signature,
        shape.recurrent_state_bytes,
        shape.numeric_features.as_ref(),
        shape.row_multiset_features.as_ref(),
        shape.rows.iter().copied(),
    )
}

impl StatisticalWaveEvidenceV1 {
    /// Same validation over the actual receipt without allocating a second row
    /// trace or erasing owner/generation correlation in the enclosing receipt.
    pub fn validate_actual(
        &self,
        shape: &ActualWaveShape,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if shape.restore_bytes != 0 || shape.maintenance_bytes != 0 || shape.maintenance_units != 0
        {
            return Err(StatisticalEvidenceUnknown::UnsupportedWave);
        }
        let digest = exact_binding_parts(
            shape.kind,
            shape.path,
            shape.graph,
            shape.row_order,
            shape.provider_signature,
            shape.output_policy_signature,
            shape.recurrent_state_bytes,
            shape.numeric_features.as_ref(),
            shape.row_multiset_features.as_ref(),
            shape.rows.iter().map(|row| row.work),
        )?;
        if self.schema_version != STATISTICAL_ROUTE_WORK_SCHEMA_V1
            || self.physical_commands == 0
            || self.physical_commands as usize > MAX_COST_COMMANDS
            || self.exact_binding != digest
        {
            return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_binding_parts(
    kind: ActualWaveKind,
    path: ActualWavePath,
    graph: ActualWaveGraphState,
    order: ActualWaveRowOrder,
    provider: [u8; 32],
    output: [u8; 32],
    recurrent: u64,
    numeric: Option<&CanonicalWaveCostFeatures>,
    host: Option<&HostRowMultisetCostFeaturesV2>,
    rows: impl ExactSizeIterator<Item = ActualRowWork>,
) -> Result<[u8; 32], StatisticalEvidenceUnknown> {
    let count = rows.len();
    if count == 0 || count > MAX_COST_ROWS {
        return Err(StatisticalEvidenceUnknown::Capacity);
    }
    let numeric = numeric.ok_or(StatisticalEvidenceUnknown::MissingHostDomain)?;
    numeric
        .validate(count)
        .map_err(|_| StatisticalEvidenceUnknown::InvalidWork)?;
    let host = host.ok_or(StatisticalEvidenceUnknown::MissingHostDomain)?;
    host.validate(count)
        .map_err(|_| StatisticalEvidenceUnknown::MissingHostDomain)?;
    let mut hash = Sha256::new();
    bytes(&mut hash, b"ferrum.statistical-wave-exact-binding.v1");
    hash.update(provider);
    hash.update(output);
    number(&mut hash, kind as u64);
    number(&mut hash, path as u64);
    number(&mut hash, graph as u64);
    number(&mut hash, order as u64);
    number(&mut hash, recurrent);
    number(&mut hash, u64::from(numeric.schema_version));
    hash.update(numeric.output_policy_signature);
    hash.update(host.wave_policy_signature);
    for ((work, row), static_row) in rows.zip(&numeric.rows).zip(&host.rows) {
        if !static_row.role.matches_work(work) {
            return Err(StatisticalEvidenceUnknown::MissingHostDomain);
        }
        number(&mut hash, static_row.role as u64);
        hash.update(static_row.categorical_signature);
        match work {
            ActualRowWork::Decode { kv_tokens } => {
                number(&mut hash, 0);
                number(&mut hash, u64::from(kv_tokens));
            }
            ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => {
                number(&mut hash, 1);
                for n in [offset, count, total_prompt_tokens] {
                    number(&mut hash, u64::from(n));
                }
            }
            _ => return Err(StatisticalEvidenceUnknown::UnsupportedWave),
        }
        for n in [
            row.generated_tokens_before,
            row.maximum_output_tokens,
            row.sampling_history_tokens,
            row.repetition_tokens,
            row.decoded_prefix_tokens,
            row.decoded_text_bytes_bound,
            row.decode_scratch_bytes_bound,
        ] {
            number(&mut hash, n);
        }
    }
    Ok(hash.finalize().into())
}
