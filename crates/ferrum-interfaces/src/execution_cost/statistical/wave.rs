use super::*;

/// Associated with the same exact completed/proposed shape, never executable
/// permission. Private construction, serialize-only; profile1--5 cannot infer it.
#[derive(Debug, Clone, Serialize)]
pub struct StatisticalWaveEvidenceV1 {
    pub(super) schema_version: u32,
    pub(super) exact_binding: [u8; 32],
    pub(super) family_signature: [u8; 32],
    pub(super) physical_commands: u32,
    pub(super) work: DeviceNumericWorkV1,
    #[serde(skip)]
    pub(super) independent_attention_v2: Option<IndependentAttentionWaveEvidenceV2>,
    #[serde(skip)]
    pub(super) structured_capture: Option<
        Result<std::sync::Arc<UnsettledStructuredWaveEvidenceV1>, StatisticalEvidenceUnknown>,
    >,
}
impl PartialEq for StatisticalWaveEvidenceV1 {
    fn eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.exact_binding == other.exact_binding
            && self.family_signature == other.family_signature
            && self.physical_commands == other.physical_commands
            && self.work == other.work
    }
}
impl Eq for StatisticalWaveEvidenceV1 {}

/// Separately versioned passive hypothesis. V1 wire/profile import cannot
/// synthesize this record from its irreversible ordered digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndependentAttentionWaveEvidenceV2 {
    pub(super) schema_version: u32,
    pub(super) exact_binding: [u8; 32],
    pub(super) family_signature: [u8; 32],
    pub(super) physical_commands: u32,
    pub(super) work: DeviceNumericWorkV1,
}
impl IndependentAttentionWaveEvidenceV2 {
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
        if self.schema_version != 2
            || self.physical_commands == 0
            || self.physical_commands as usize > MAX_COST_COMMANDS
            || self.exact_binding != exact_binding(shape)?
        {
            return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
        }
        Ok(())
    }
}
impl IndependentAttentionWaveEvidenceV2 {
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
        if self.schema_version != 2
            || self.physical_commands == 0
            || self.physical_commands as usize > MAX_COST_COMMANDS
            || self.exact_binding != digest
        {
            return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
        }
        Ok(())
    }
}
impl StatisticalWaveEvidenceV1 {
    pub fn structured_capture(
        &self,
    ) -> Option<
        Result<&std::sync::Arc<UnsettledStructuredWaveEvidenceV1>, StatisticalEvidenceUnknown>,
    > {
        self.structured_capture
            .as_ref()
            .map(|value| value.as_ref().map_err(|error| *error))
    }
    pub fn structured_physical_host_capacity(&self) -> usize {
        self.structured_capture()
            .and_then(Result::ok)
            .map_or(0, |value| value.physical_host_capacity())
    }
    /// Conservative retention units, including opt-in auxiliary table bytes.
    /// Use structured_physical_host_capacity for physical-row limits.
    pub fn structured_retained_rows(&self) -> usize {
        self.structured_capture()
            .and_then(Result::ok)
            .map_or(0, |value| value.retained_rows())
    }
    pub(in crate::execution_cost) fn attach_structured_capture(
        mut self,
        value: Result<UnsettledStructuredWaveEvidenceV1, StatisticalEvidenceUnknown>,
        exact: &CanonicalWaveCostShape,
    ) -> Self {
        self.structured_capture = Some(value.and_then(|value| {
            self.validate_exact(exact)?;
            value.validate_exact(exact)?;
            if self.physical_commands != value.device().physical_commands()
                || self.work != value.device().aggregate_work()
            {
                return Err(StatisticalEvidenceUnknown::CommandMismatch);
            }
            Ok(std::sync::Arc::new(value))
        }));
        self
    }
    /// Explicit new-profile import only. Both independently serialized records
    /// must bind the same exact shape and identical actual numeric work/counts.
    /// The old V1 import never calls this and never gains a V2 family implicitly.
    pub fn with_independent_attention_v2(
        mut self,
        value: IndependentAttentionWaveEvidenceV2,
        exact: &CanonicalWaveCostShape,
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        self.validate_exact(exact)?;
        value.validate_exact(exact)?;
        if self.physical_commands != value.physical_commands || self.work != value.work {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        self.independent_attention_v2 = Some(value);
        Ok(self)
    }
    pub fn independent_attention_v2(&self) -> Option<&IndependentAttentionWaveEvidenceV2> {
        self.independent_attention_v2.as_ref()
    }
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
    independent_family: Option<Sha256>,
    count: usize,
    work: DeviceNumericWorkV1,
    failure: Option<StatisticalEvidenceUnknown>,
    algorithm_work: Option<super::wave_algorithm_work::WaveAlgorithmAccumulator>,
    replay: Option<super::wave_replay::ReplayWaveAccumulator>,
}
impl StatisticalWaveAccumulator {
    pub(in crate::execution_cost) fn new() -> Self {
        let mut family = Sha256::new();
        bytes(&mut family, b"ferrum.statistical-wave-family.v1");
        let mut independent_family = Sha256::new();
        bytes(
            &mut independent_family,
            b"ferrum.statistical-wave.independent-attention.v2",
        );
        Self {
            family,
            independent_family: Some(independent_family),
            count: 0,
            work: Default::default(),
            failure: None,
            algorithm_work: None,
            replay: None,
        }
    }
    pub(in crate::execution_cost) fn capture_algorithm_work(&mut self) {
        self.algorithm_work = Some(super::wave_algorithm_work::WaveAlgorithmAccumulator::new());
        self.replay = Some(super::wave_replay::ReplayWaveAccumulator::new());
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
        if command.path == CostCommandPath::Replayed {
            self.replay
                .as_mut()
                .ok_or(StatisticalEvidenceUnknown::UnsupportedReplay)?
                .physical(command)?;
            self.algorithm_work
                .as_mut()
                .ok_or(StatisticalEvidenceUnknown::MissingProducer)?
                .observe_replay(command);
            bytes(&mut self.family, b"ferrum.physical-replay.v1");
            append_command_identity(&mut self.family, command);
            if let Some(hash) = &mut self.independent_family {
                bytes(hash, b"ferrum.physical-replay.v1");
                append_command_identity(hash, command);
            }
            self.count += 1;
            return Ok(());
        }
        if command.reusable_graph_node_count.is_some() {
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
        if let Some(capture) = &mut self.algorithm_work {
            capture.observe(command);
        }
        append_command_identity(&mut self.family, command);
        self.family.update(evidence.family_signature());
        match (
            self.independent_family.as_mut(),
            evidence.independent_attention_family_v2(),
        ) {
            (Some(hash), Some(family)) => {
                // Reuse fixed command digests and numeric work. Do not walk
                // kernel traces or reparse/rebind the exact canonical shape.
                append_command_identity(hash, command);
                hash.update(family);
            }
            _ => self.independent_family = None,
        }
        self.work = next;
        self.count += 1;
        Ok(())
    }
    pub(in crate::execution_cost) fn replay_segment(
        &mut self,
        physical: u32,
        fingerprint: &str,
        count: usize,
    ) {
        if self.failure.is_some() {
            return;
        }
        let result = self
            .replay
            .as_mut()
            .ok_or(StatisticalEvidenceUnknown::UnsupportedReplay)
            .and_then(|replay| replay.segment(physical, fingerprint, count));
        if let Err(error) = result {
            self.failure = Some(error);
            return;
        }
        if let Some(capture) = &mut self.algorithm_work {
            capture.replay_segment(physical, fingerprint, count);
        }
        // The instance fingerprint belongs only to resident/assignment binding.
        // The numerical model may share topology across legitimate instances.
        bytes(&mut self.family, b"ferrum.logical-segment.v1");
        number(&mut self.family, u64::from(physical));
        number(&mut self.family, count as u64);
        if let Some(hash) = &mut self.independent_family {
            bytes(hash, b"ferrum.logical-segment.v1");
            number(hash, u64::from(physical));
            number(hash, count as u64);
        }
    }
    pub(in crate::execution_cost) fn logical_command(&mut self, command: CostLogicalCommand<'_>) {
        if self.failure.is_some() {
            return;
        }
        if let Err(error) = self.append_logical(command) {
            self.failure = Some(error);
        }
    }
    fn append_logical(
        &mut self,
        command: CostLogicalCommand<'_>,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        let physical = self
            .replay
            .as_mut()
            .ok_or(StatisticalEvidenceUnknown::UnsupportedReplay)?
            .logical(command)?;
        let selected = command
            .statistical_evidence
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)?;
        self.work = self.work.checked_add(selected.work())?;
        self.algorithm_work
            .as_mut()
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)?
            .observe_logical(physical, command);
        super::wave_replay::append_logical_identity(&mut self.family, physical, command);
        self.family.update(selected.family_signature());
        match (
            self.independent_family.as_mut(),
            selected.independent_attention_family_v2(),
        ) {
            (Some(hash), Some(family)) => {
                super::wave_replay::append_logical_identity(hash, physical, command);
                hash.update(family);
            }
            _ => self.independent_family = None,
        }
        Ok(())
    }
    fn replay_work(
        &self,
        shape: &CanonicalWaveCostShape,
    ) -> Result<Option<StructuredReplayWorkV1>, StatisticalEvidenceUnknown> {
        match &self.replay {
            Some(replay) => replay.finish(shape),
            None if matches!(
                shape.graph,
                ActualWaveGraphState::Disabled | ActualWaveGraphState::ConfiguredEager
            ) =>
            {
                Ok(None)
            }
            None => Err(StatisticalEvidenceUnknown::UnsupportedReplay),
        }
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
        if shape.path != ActualWavePath::PlanRuntime {
            return Err(StatisticalEvidenceUnknown::UnsupportedWave);
        }
        self.replay_work(shape)?;
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
        let family_signature = self.family.finalize().into();
        let exact_binding = exact_binding(shape)?;
        let eligible = shape.kind == ActualWaveKind::Decode
            && shape
                .rows
                .iter()
                .all(|row| matches!(row, ActualRowWork::Decode { .. }))
            && shape.numeric_features.as_ref().is_some_and(|numeric| {
                numeric.rows.iter().all(|row| {
                    row.generated_tokens_before
                        .checked_add(1)
                        .is_some_and(|next| next < row.maximum_output_tokens)
                })
            });
        let independent_attention_v2 = self.independent_family.map(|mut hash| {
            let signature = if eligible {
                append_wave_identity(&mut hash, shape, host);
                hash.finalize().into()
            } else {
                // Prefill/mixed/terminal retain the COMPLETE old ordered
                // family. A decode-like count of one cannot broaden scope.
                let mut hash = Sha256::new();
                bytes(&mut hash, b"ferrum.statistical-wave.ordered-promotion.v2");
                hash.update(family_signature);
                hash.finalize().into()
            };
            IndependentAttentionWaveEvidenceV2 {
                schema_version: 2,
                exact_binding,
                family_signature: signature,
                physical_commands: self.count as u32,
                work: self.work,
            }
        });
        Ok(StatisticalWaveEvidenceV1 {
            schema_version: STATISTICAL_ROUTE_WORK_SCHEMA_V1,
            exact_binding,
            family_signature,
            independent_attention_v2,
            structured_capture: None,
            physical_commands: self.count as u32,
            work: self.work,
        })
    }
}

impl StatisticalWaveAccumulator {
    pub(in crate::execution_cost) fn structured_device(
        &mut self,
        shape: &CanonicalWaveCostShape,
        product: CostProductOutput,
        readback: Option<CoreReadbackRoute>,
        retries: u32,
    ) -> Result<DeviceRouteTemplateV1, StatisticalEvidenceUnknown> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let replay = self.replay_work(shape)?;
        // These states contain only the checked selected-command stream. The
        // legacy finish has not appended host policy/categories or row flags.
        DeviceRouteTemplateV1::from_selected_stream(
            self.family.clone().finalize().into(),
            self.independent_family
                .as_ref()
                .map(|hash| hash.clone().finalize().into()),
            self.count,
            self.work,
            shape,
            product,
            readback,
            retries,
            self.algorithm_work
                .take()
                .ok_or(StatisticalEvidenceUnknown::MissingProducer)
                .and_then(|capture| capture.finish(shape, self.count, self.work)),
            replay,
        )
    }
}

pub(super) fn exact_binding(
    shape: &CanonicalWaveCostShape,
) -> Result<[u8; 32], StatisticalEvidenceUnknown> {
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
pub(super) fn exact_binding_parts(
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

pub(super) fn append_command_identity(hash: &mut Sha256, command: CostPhysicalCommand<'_>) {
    number(hash, u64::from(command.command_index));
    number(hash, command.node_index.map_or(u64::MAX, u64::from));
    number(hash, command.command_phase as u64);
    bytes(hash, command.native_op_id.as_bytes());
    bytes(hash, command.batching_form.as_bytes());
    match command.provider {
        Some(provider) => {
            number(hash, 1);
            bytes(hash, provider.provider_id.as_bytes());
            bytes(hash, provider.implementation_fingerprint.as_bytes());
            bytes(hash, provider.operation_fingerprint.as_bytes());
        }
        None => number(hash, 0),
    }
    number(hash, u64::from(command.participant_start));
    number(hash, u64::from(command.participant_count));
    number(hash, command.compute_dispatch_count);
    number(hash, command.transfer_command_count);
}

fn append_wave_identity(
    hash: &mut Sha256,
    shape: &CanonicalWaveCostShape,
    host: &HostRowMultisetCostFeaturesV2,
) {
    number(hash, shape.kind as u64);
    number(hash, shape.path as u64);
    number(hash, shape.graph as u64);
    number(hash, shape.row_order as u64);
    number(hash, shape.recurrent_state_bytes);
    hash.update(host.wave_policy_signature);
    // Initial revision preserves the full physical static role pattern.
    // No assumption of owner/role permutation or cross-width invariance.
    for row in &host.rows {
        number(hash, row.role as u64);
        hash.update(row.categorical_signature);
    }
}
