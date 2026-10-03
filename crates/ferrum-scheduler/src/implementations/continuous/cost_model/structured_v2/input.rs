use super::super::statistical::StatisticalModelInputV1;
use super::*;
use ferrum_interfaces::execution_cost::{
    AlgorithmWorkKindV1, CostWorkloadDomainV1, DeviceNumericWorkV1, HostCostPolicyV2,
    HostRowRoleV2, HostTerminalExpectationV1,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredInputV2 {
    pub(super) owner: StructuredOwnerKeyV2,
    pub(super) domain: [u8; 32],
    /// Membership in a declared finite physical workload, checked while the
    /// original canonical shape and bound recipe are still available.
    pub(super) physical_domain: Option<[u8; 32]>,
    // Both identities share the same owned numeric vectors. Only checked D
    // projection may create the new empirical family.
    pub(super) alternate_template_identity: Option<(StructuredOwnerKeyV2, [u8; 32])>,
    /// Original route transcript: kind, path, graph, row order, total recurrent
    /// bytes, retries. Hashing is deferred until its checked domain binding.
    pub(super) template_route_facts: [u64; 6],
    /// Cached during the original row validation; None means heterogeneous.
    pub(super) homogeneous_host_policy: Option<HostCostPolicyV2>,
    pub(super) settled_terminal_causes: Option<Vec<(u32, ferrum_types::FinishReason)>>,
    pub(super) algorithm_axes: Vec<super::algorithm_universe::AlgorithmAxisV1>,
    pub(super) algorithm_universe: Option<DeclaredAlgorithmUniverseV1>,
    pub(super) basis: Vec<f64>,
    pub(super) support: Vec<u64>,
    pub(super) physical_host_rows: Vec<StructuredHostRowV1>,
    pub(super) pending_positions: Vec<u32>,
    pub(super) length_positions: Vec<u32>,
    pub(super) pending_basis_offset: usize,
    pub(super) pending_support_offset: usize,
    pub(super) completion: Option<super::completion::InputCompletion>,
    pub(super) repetition_offsets: Option<(usize, usize)>,
}
#[derive(Debug, Clone)]
pub struct StructuredQueryV2 {
    pub(super) input: StructuredInputV2,
    /// None preserves Exact versus conditional-empty Unresolved.
    pub(super) pending: Option<PendingQuery>,
    pub(super) repetition_upper_sum: Option<u64>,
}
#[derive(Debug, Clone)]
pub(super) struct PendingQuery {
    pub eligible: Vec<u32>,
    pub constraint: HostPendingConstraintV2,
}
impl StructuredInputV2 {
    /// Borrow the checked input's original `(signature, kind)` roster for
    /// passive diagnostics. Universe alignment does not add its zero-work
    /// algorithms here. This view neither projects work nor grants authority.
    pub fn observation_algorithm_axes(&self) -> impl serde::Serialize + '_ {
        &self.algorithm_axes
    }

    /// Inline value and every owned allocation at its actual capacity. Fixed
    /// owner/domain fields are included in Self; allocator/RSS costs are not.
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        self.retained_numeric_bytes()
    }
    /// Private numerical reconstruction, not a deserializer for a live receipt
    /// or forecast. Original Prepared and settled wire bindings are checked by
    /// the profile10 importer before invoking this shared projection.
    pub(in crate::implementations::continuous) fn from_replay_parts(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        ordered: [u8; 32],
        grouped: Option<[u8; 32]>,
        product: StructuredProductV2,
        readback: CoreReadbackRoute,
        rows: &[StructuredHostRowV1],
        algorithms: &[([u8; 32], AlgorithmWorkKindV1, u64, DeviceNumericWorkV1)],
        replay_counts: Option<[u64; 3]>,
        retries: u32,
    ) -> Result<(Self, StructuredOwnerFactsV2)> {
        match (exact.graph, replay_counts) {
            (
                ferrum_interfaces::execution_cost::ActualWaveGraphState::Disabled
                | ferrum_interfaces::execution_cost::ActualWaveGraphState::ConfiguredEager,
                None,
            )
            | (ferrum_interfaces::execution_cost::ActualWaveGraphState::Warm, Some(_)) => {}
            _ => return Err(StructuredUnknown::MissingEvidence),
        }
        let facts = StructuredOwnerFactsV2::from_replay_parts(
            rows, product, readback, ordered, grouped, algorithms,
        )?;
        let owner = facts.owner_key()?;
        let old = StatisticalModelInputV1::from_future_structured_v2(exact, selected)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let input = Self::project_numeric(
            owner,
            &old,
            rows,
            algorithms.iter().map(|(a, k, n, w)| (*a, *k, *n, *w)),
            replay_counts,
            exact,
            retries,
        )?;
        Ok((input, facts))
    }
    pub fn owner_for(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
    ) -> Result<StructuredOwnerKeyV2> {
        StructuredOwnerFactsV2::validated_prepared(exact, selected, recipe).map(|value| value.owner)
    }
    pub fn from_actual(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
    ) -> Result<Self> {
        // Reuse the full validation's owner and numeric projection. Rebuilding
        // either from the same immutable tuple would repeat hashes and row work.
        let population::ValidatedPreparedV2 { owner, numeric, .. } =
            StructuredOwnerFactsV2::validated_prepared(exact, selected, recipe)?;
        let algorithms = recipe
            .algorithm_work()
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        Self::project_numeric(
            owner,
            &numeric,
            recipe.physical_host_rows(),
            algorithms
                .entries()
                .iter()
                .map(|a| (*a.algorithm().signature(), a.kind(), a.commands(), a.work())),
            recipe.device().replay_work().map(|r| {
                [
                    u64::from(r.replayed_segments()),
                    u64::from(r.logical_commands()),
                    r.native_graph_nodes(),
                ]
            }),
            exact,
            recipe.device().retries(),
        )
    }
    /// The descriptor is numerical scope, never submission authority. Validate
    /// the original receipt/recipe before inspecting its physical limits.
    pub fn from_actual_with_domain(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
        domain: &CostWorkloadDomainV1,
    ) -> Result<Self> {
        Self::from_actual(exact, selected, recipe)?.bind_validated_physical_domain(exact, domain)
    }
    /// Share one original validation between legacy consumers and the new
    /// numerical scope. A scope failure cannot invalidate the original input.
    pub fn from_actual_with_domain_projection(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
        domain: &CostWorkloadDomainV1,
    ) -> Result<(Self, Result<Self>)> {
        let base = Self::from_actual(exact, selected, recipe)?;
        let scoped = base.clone().bind_validated_physical_domain(exact, domain);
        Ok((base, scoped))
    }
    /// Only the already checked profile adapter may attach scope during replay.
    /// Public callers cannot attach a descriptor to arbitrary numerical axes.
    pub(in crate::implementations::continuous) fn bind_validated_physical_domain(
        mut self,
        exact: &CanonicalWaveCostShape,
        domain: &CostWorkloadDomainV1,
    ) -> Result<Self> {
        domain
            .validate_projected_workload(exact, &self.physical_host_rows)
            .map_err(|_| StructuredUnknown::WrongDomain)?;
        if self.cost_template_policy() != StructuredCostTemplatePolicyV1::OrderedV1 {
            return Err(StructuredUnknown::WrongProtocol);
        }
        self.physical_domain = Some(*domain.sha256());
        let mut family = Sha256::new();
        family.update(b"ferrum.installed-algorithm-set-family.v1\0");
        family.update(domain.sha256());
        let mut route = Sha256::new();
        route.update(b"ferrum.numerical-family-route.v1\0");
        for value in self.template_route_facts {
            route.update(value.to_le_bytes());
        }
        family.update(route.finalize());
        family.update(self.owner.algorithm_domain);
        let mut owner = self.owner.clone();
        owner.provider_template =
            StructuredTemplateV2::InstalledAlgorithmSetV1(family.finalize().into());
        let signature = numeric_domain_signature(
            &owner,
            exact.graph == ferrum_interfaces::execution_cost::ActualWaveGraphState::Warm,
            self.repetition_offsets.is_some(),
        )?;
        self.alternate_template_identity = Some((owner, signature));
        Ok(self)
    }
    /// Actual backing capacities for bounded slow-path evidence retention.
    /// This is numerical storage accounting, never an execution authority.
    pub fn retained_numeric_bytes(&self) -> Option<usize> {
        let mut bytes = std::mem::size_of::<Self>()
            .checked_add(
                self.algorithm_axes
                    .capacity()
                    .checked_mul(std::mem::size_of::<
                        super::algorithm_universe::AlgorithmAxisV1,
                    >())?,
            )?
            .checked_add(
                self.algorithm_universe
                    .as_ref()
                    .map_or(Some(0), DeclaredAlgorithmUniverseV1::retained_payload_bytes)?,
            )?;
        for (capacity, element) in [
            (self.basis.capacity(), std::mem::size_of::<f64>()),
            (self.support.capacity(), std::mem::size_of::<u64>()),
            (
                self.physical_host_rows.capacity(),
                std::mem::size_of::<StructuredHostRowV1>(),
            ),
            (
                self.pending_positions.capacity(),
                std::mem::size_of::<u32>(),
            ),
            (self.length_positions.capacity(), std::mem::size_of::<u32>()),
            (
                self.settled_terminal_causes
                    .as_ref()
                    .map_or(0, Vec::capacity),
                std::mem::size_of::<(u32, ferrum_types::FinishReason)>(),
            ),
            (
                self.completion
                    .as_ref()
                    .map_or(0, |c| c.positions.capacity()),
                std::mem::size_of::<u32>(),
            ),
        ] {
            bytes = bytes.checked_add(capacity.checked_mul(element)?)?;
        }
        Some(bytes)
    }
    /// Collector-local accounting. Equality of serialized identities is not
    /// enough: only the exact allocation already owned by the ledger is free.
    pub(in crate::implementations::continuous) fn retained_bytes_with_shared_universe(
        &self,
        shared: Option<&DeclaredAlgorithmUniverseV1>,
    ) -> Option<usize> {
        let bytes = self.retained_numeric_bytes()?;
        match (&self.algorithm_universe, shared) {
            (Some(value), Some(shared)) if value.shares_allocation(shared) => {
                bytes.checked_sub(value.retained_payload_bytes()?)
            }
            _ => Some(bytes),
        }
    }
    pub fn owner(&self) -> &StructuredOwnerKeyV2 {
        &self.owner
    }
    pub fn domain_signature(&self) -> &[u8; 32] {
        &self.domain
    }
    pub fn cost_template_policy(&self) -> StructuredCostTemplatePolicyV1 {
        self.owner.cost_template_policy()
    }
    /// Select a declared empirical interpretation of this already validated
    /// projection. No numerical vector or original execution recipe is changed.
    pub fn with_cost_template_policy(
        mut self,
        policy: StructuredCostTemplatePolicyV1,
    ) -> Result<Self> {
        if self.cost_template_policy() != policy {
            let (mut owner, mut domain) = self
                .alternate_template_identity
                .take()
                .filter(|(owner, _)| owner.cost_template_policy() == policy)
                .ok_or(StructuredUnknown::WrongDomain)?;
            std::mem::swap(&mut self.owner, &mut owner);
            std::mem::swap(&mut self.domain, &mut domain);
            self.alternate_template_identity = Some((owner, domain));
        }
        Ok(self)
    }
    /// Fixed-size identity lookup; no vector clone or alternative prediction.
    pub fn cost_template_identity(
        &self,
        policy: StructuredCostTemplatePolicyV1,
    ) -> Option<(&StructuredOwnerKeyV2, &[u8; 32])> {
        if self.cost_template_policy() == policy {
            Some((&self.owner, &self.domain))
        } else {
            self.alternate_template_identity
                .as_ref()
                .filter(|(owner, _)| owner.cost_template_policy() == policy)
                .map(|(owner, domain)| (owner, domain))
        }
    }
    pub fn physical_domain_signature(&self) -> Option<&[u8; 32]> {
        self.physical_domain.as_ref()
    }
    /// Copy a numerical identity from the original checked projection. This
    /// reads cached facts only: no recipe replay, vector clone or identity hash.
    /// The physical descriptor must already have checked actual rows/state.
    pub fn numerical_family_key(&self) -> Result<NumericalFamilyKeyV1> {
        NumericalFamilyKeyV1::from_validated_input(self)
    }
    pub fn settled_terminal_causes(&self) -> Option<&[(u32, ferrum_types::FinishReason)]> {
        self.settled_terminal_causes.as_deref()
    }
    /// Numerical projection after original live settlement or source receipt
    /// validation. Preserve real causes; capacity termination is not early EOS.
    pub fn with_settled_terminal_causes(
        self,
        causes: &[(u32, ferrum_types::FinishReason)],
    ) -> Result<Self> {
        use ferrum_interfaces::execution_cost::HostContentDomainV1;
        use ferrum_types::FinishReason;
        let positions = causes
            .iter()
            .map(|(position, _)| *position)
            .collect::<Vec<_>>();
        position_moments(&positions)?;
        for &(position, cause) in causes {
            let row = self
                .physical_host_rows
                .get(position as usize)
                .ok_or(StructuredUnknown::InvalidInput)?;
            let emits = row.terminal_expectation != HostTerminalExpectationV1::NoTokenProduced;
            let allowed = emits
                && match cause {
                    FinishReason::Length => {
                        row.terminal_expectation == HostTerminalExpectationV1::LengthBoundary
                    }
                    FinishReason::EOS => matches!(
                        row.installed_policy.empirical_content_domain,
                        Some(HostContentDomainV1::PlainTextInstalledV2(p)) if p.model_eos
                    ),
                    FinishReason::Stop => matches!(
                        row.installed_policy.empirical_content_domain,
                        Some(HostContentDomainV1::PlainTextInstalledV2(p)) if p.user_stop
                    ),
                    _ => false,
                };
            if !allowed {
                return Err(StructuredUnknown::InvalidInput);
            }
        }
        let mut settled = self.with_settled_completion(&positions)?;
        settled.settled_terminal_causes = Some(causes.to_vec());
        Ok(settled)
    }
    pub fn regression_axes(&self) -> &[f64] {
        &self.basis
    }
    pub fn joint_support_coordinates(&self) -> &[u64] {
        &self.support
    }
    pub fn physical_host_rows(&self) -> &[StructuredHostRowV1] {
        &self.physical_host_rows
    }

    fn project_numeric(
        owner: StructuredOwnerKeyV2,
        old: &StatisticalModelInputV1,
        rows: &[StructuredHostRowV1],
        algorithms: impl Iterator<Item = ([u8; 32], AlgorithmWorkKindV1, u64, DeviceNumericWorkV1)>,
        replay_counts: Option<[u64; 3]>,
        exact: &CanonicalWaveCostShape,
        retries: u32,
    ) -> Result<Self> {
        if rows.len() != owner.rows as usize || rows.is_empty() || rows.len() > 128 {
            return Err(StructuredUnknown::InvalidInput);
        }
        let mut pending_positions = Vec::new();
        let mut length_positions = Vec::new();
        // Counts describe actual work classes. Physical position remains audit
        // evidence; none of these bits is interpreted as observed host order.
        let mut host_counts = [0u64; 7];
        let mut homogeneous_host_policy = Some(rows[0].installed_policy);
        for (position, row) in rows.iter().enumerate() {
            if row.physical_position as usize != position {
                return Err(StructuredUnknown::InvalidInput);
            }
            if homogeneous_host_policy != Some(row.installed_policy) {
                homogeneous_host_policy = None;
            }
            match row.role {
                HostRowRoleV2::Decode => {
                    if row.decode_requires_full_logits.is_none()
                        || row.repetition_penalty_bits.is_none()
                        || row.initial_prefill
                        || row.final_prefill
                    {
                        return Err(StructuredUnknown::InvalidInput);
                    }
                    host_counts[0] += 1;
                }
                HostRowRoleV2::Prefill => {
                    if row.decode_requires_full_logits.is_some()
                        || row.repetition_penalty_bits.is_some()
                    {
                        return Err(StructuredUnknown::InvalidInput);
                    }
                    host_counts[1] += 1;
                }
            }
            host_counts[2] += u64::from(row.no_generated_history);
            host_counts[3] += u64::from(row.initial_prefill);
            host_counts[4] += u64::from(row.final_prefill);
            host_counts[5] += u64::from(row.mask_upload_required);
            match row.terminal_expectation {
                HostTerminalExpectationV1::NoTokenProduced => {
                    if row.role != HostRowRoleV2::Prefill || row.final_prefill {
                        return Err(StructuredUnknown::InvalidInput);
                    }
                }
                HostTerminalExpectationV1::TokenMayTerminate => host_counts[6] += 1,
                HostTerminalExpectationV1::LengthBoundary => {
                    host_counts[6] += 1;
                    length_positions.push(position as u32);
                }
            }
            if row.pending_decoded_utf8 {
                pending_positions.push(position as u32);
            }
        }
        let domain = numeric_domain_signature(
            &owner,
            replay_counts.is_some(),
            rows.iter().any(super::completion::installed),
        )?;
        let template_route_facts = [
            exact.kind as u64,
            exact.path as u64,
            exact.graph as u64,
            exact.row_order as u64,
            exact.recurrent_state_bytes,
            u64::from(retries),
        ];
        let mut basis = vec![1.0];
        let mut support = Vec::new();
        let mut algorithm_axes = Vec::new();
        for (signature, kind, commands, work) in algorithms {
            algorithm_axes.push(super::algorithm_universe::AlgorithmAxisV1::new(
                signature, kind,
            ));
            kind.validate_work(work)
                .map_err(|_| StructuredUnknown::InvalidInput)?;
            support.push(commands);
            support.extend(device_coordinates(work));
            basis.push(commands as f64);
            if kind == AlgorithmWorkKindV1::Kernel {
                basis.extend([
                    work.inner_work_units as f64,
                    work.padded_units as f64,
                    work.grid_blocks as f64,
                ]);
            } else if kind == AlgorithmWorkKindV1::LibraryCall {
                // API output elements and reduction work, never a native grid.
                basis.extend([work.logical_units as f64, work.inner_work_units as f64]);
            } else {
                let bytes = work
                    .host_to_device_bytes
                    .checked_add(work.device_to_host_bytes)
                    .and_then(|n| n.checked_add(work.device_to_device_bytes))
                    .and_then(|n| n.checked_add(work.fill_bytes))
                    .ok_or(StructuredUnknown::InvalidInput)?;
                basis.push(bytes as f64);
            }
        }
        if let Some(counts) = replay_counts {
            if counts.iter().any(|n| *n == 0)
                || counts[0] > ferrum_interfaces::execution_cost::MAX_COST_COMMANDS as u64
                || counts[1] > ferrum_interfaces::execution_cost::MAX_COST_COMMANDS as u64
                || counts[1] < counts[0]
                || counts[2] < counts[1]
            {
                return Err(StructuredUnknown::MissingEvidence);
            }
            // Shared topology/domain excludes resident binding. Actual graph
            // work, including launch and logical counts, remains numeric and
            // must be inside the same joint support as all other coordinates.
            support.extend(counts);
            basis.extend(counts.map(|n| n as f64));
        }
        let h = old.host_and_sequence();
        let repetition_support_offset = support.len() + 10;

        support.extend([
            h.rows,
            h.prefill_tokens,
            h.attention_pairs,
            h.kv_tokens_sum,
            h.kv_tokens_max,
            h.prompt_tokens_sum,
            h.prompt_tokens_max,
            h.generated_tokens_sum,
            h.sampling_history_sum,
            h.sampling_history_max,
            h.repetition_tokens_sum,
            h.decoded_prefix_sum,
            h.decoded_text_bytes_sum,
            h.decode_scratch_bytes_sum,
            h.recurrent_bytes,
        ]);
        basis.extend([
            h.prefill_tokens as f64,
            h.attention_pairs as f64,
            h.sampling_history_sum as f64,
            h.decoded_text_bytes_sum as f64,
            h.decode_scratch_bytes_sum as f64,
        ]);
        support.extend(host_counts);
        basis.extend(host_counts.map(|x| x as f64));
        let repetition_offsets = if rows.iter().any(super::completion::installed) {
            let offset = basis.len();
            basis.push(h.repetition_tokens_sum as f64);
            Some((offset, repetition_support_offset))
        } else {
            None
        };
        let completion = if rows.iter().any(super::completion::installed) {
            let positions = rows
                .iter()
                .filter(|row| {
                    super::completion::installed(row)
                        && row.terminal_expectation == HostTerminalExpectationV1::LengthBoundary
                })
                .map(|row| row.physical_position)
                .collect::<Vec<_>>();
            let moments = position_moments(&positions)?;
            let completion = super::completion::InputCompletion {
                positions,
                basis_offset: basis.len(),
                support_offset: support.len(),
                settled: false,
            };
            support.extend(moments);
            basis.extend(moments.map(|x| x as f64));
            Some(completion)
        } else {
            None
        };
        let length = position_moments(&length_positions)?;
        support.extend(length);
        basis.extend(length.map(|x| x as f64));
        let pending_basis_offset = basis.len();
        let pending_support_offset = support.len();
        let pending = position_moments(&pending_positions)?;
        support.extend(pending);
        basis.extend(pending.map(|x| x as f64));
        let value = Self {
            owner,
            domain,
            physical_domain: None,
            alternate_template_identity: None,
            template_route_facts,
            homogeneous_host_policy,
            settled_terminal_causes: None,
            algorithm_axes,
            algorithm_universe: None,
            basis,
            support,
            physical_host_rows: rows.to_vec(),
            pending_positions,
            length_positions,
            pending_basis_offset,
            pending_support_offset,
            completion,
            repetition_offsets,
        };
        let construction_limits = StructuredSettingsV2 {
            max_axes: 4096,
            ..Default::default()
        };
        value.validate(&construction_limits)?;
        Ok(value)
    }
    pub(super) fn validate(&self, settings: &StructuredSettingsV2) -> Result<()> {
        if self.owner.rows as usize != self.physical_host_rows.len()
            || self.owner.rows == 0
            || self.domain == [0; 32]
            || self.basis.first() != Some(&1.0)
            || self
                .basis
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > (1u64 << 53) as f64)
            || self.support.iter().any(|v| *v > (1u64 << 53))
            || self.pending_basis_offset < 3
            || self.pending_support_offset < 3
            || self.pending_basis_offset.checked_add(3) != Some(self.basis.len())
            || self.pending_support_offset.checked_add(3) != Some(self.support.len())
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        if self.basis.len() > settings.max_axes || self.support.len() > settings.max_axes {
            return Err(StructuredUnknown::Capacity);
        }
        let pending = position_moments(&self.pending_positions)?;
        let lengths = position_moments(&self.length_positions)?;
        if self.support[self.pending_support_offset..] != pending
            || self.basis[self.pending_basis_offset..] != pending.map(|v| v as f64)
            || self.support[self.pending_support_offset - 3..self.pending_support_offset] != lengths
            || self.basis[self.pending_basis_offset - 3..self.pending_basis_offset]
                != lengths.map(|v| v as f64)
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        self.validate_completion()
    }
    pub(super) fn same_query_domain(&self, other: &Self) -> Result<()> {
        let (owner, domain) = other
            .cost_template_identity(self.cost_template_policy())
            .ok_or(StructuredUnknown::WrongDomain)?;
        if self.algorithm_universe_signature() != other.algorithm_universe_signature()
            || &self.owner != owner
            || &self.domain != domain
            || self.basis.len() != other.basis.len()
            || self.support.len() != other.support.len()
        {
            return Err(StructuredUnknown::WrongDomain);
        }
        Ok(())
    }
    pub(super) fn same_domain(&self, other: &Self) -> Result<()> {
        if self.algorithm_universe_signature() != other.algorithm_universe_signature()
            || self.owner != other.owner
            || self.domain != other.domain
            || self.basis.len() != other.basis.len()
            || self.support.len() != other.support.len()
        {
            return Err(StructuredUnknown::WrongDomain);
        }
        Ok(())
    }
}
impl StructuredQueryV2 {
    /// Complete owned query payload, including unused pending-position slots.
    /// The input value is embedded in Self and is therefore counted only once.
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(
                self.input
                    .retained_payload_bytes()?
                    .checked_sub(std::mem::size_of::<StructuredInputV2>())?,
            )?
            .checked_add(
                self.pending
                    .as_ref()
                    .map_or(0, |pending| pending.eligible.capacity())
                    .checked_mul(std::mem::size_of::<u32>())?,
            )
    }
    /// Borrow the immutable numerical input shared by evidence consumers.
    pub fn input(&self) -> &StructuredInputV2 {
        &self.input
    }
    pub fn exact(input: StructuredInputV2) -> Self {
        Self {
            input,
            pending: None,
            repetition_upper_sum: None,
        }
    }
    pub fn from_future(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
        forecast: &HostContentForecastV2,
    ) -> Result<Self> {
        forecast
            .validate(exact, recipe)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let input = StructuredInputV2::from_actual(exact, selected, recipe)?;
        let pending = match forecast {
            HostContentForecastV2::Exact => None,
            HostContentForecastV2::Unresolved(p) => Some(PendingQuery {
                eligible: p.eligible_positions().to_vec(),
                constraint: p.constraint(),
            }),
        };
        let repetition_upper_sum = match forecast {
            HostContentForecastV2::Exact => None,
            HostContentForecastV2::Unresolved(p) => p.repetition_upper_sum(),
        };
        Ok(Self {
            input,
            pending,
            repetition_upper_sum,
        })
    }
    pub fn from_future_with_domain(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
        forecast: &HostContentForecastV2,
        domain: &CostWorkloadDomainV1,
    ) -> Result<Self> {
        let mut query = Self::from_future(exact, selected, recipe, forecast)?;
        query.input = query.input.bind_validated_physical_domain(exact, domain)?;
        Ok(query)
    }
    pub fn owner(&self) -> &StructuredOwnerKeyV2 {
        self.input.owner()
    }
    pub fn domain_signature(&self) -> &[u8; 32] {
        self.input.domain_signature()
    }
}
pub(super) fn position_moments(positions: &[u32]) -> Result<[u64; 3]> {
    if positions.len() > 128
        || positions.windows(2).any(|v| v[0] >= v[1])
        || positions.iter().any(|p| *p >= 128)
    {
        return Err(StructuredUnknown::InvalidInput);
    }
    let mut out = [0; 3];
    for p in positions {
        let p = u64::from(*p) + 1;
        out[0] += 1;
        out[1] += p;
        out[2] += p * p;
    }
    Ok(out)
}
fn device_coordinates(d: DeviceNumericWorkV1) -> [u64; 10] {
    [
        d.logical_units,
        d.padded_units,
        d.inner_work_units,
        d.grid_blocks,
        d.peak_scratch_bytes,
        d.staged_weight_bytes,
        d.host_to_device_bytes,
        d.device_to_host_bytes,
        d.device_to_device_bytes,
        d.fill_bytes,
    ]
}

// Preserve the original Ordered hash bytes, including marker order. The new
// policy substitutes only an explicitly tagged owner before these same axes.
fn numeric_domain_signature(
    owner: &StructuredOwnerKeyV2,
    replay: bool,
    installed: bool,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(MODEL_REVISION_V2.as_bytes());
    hash.update(serde_json::to_vec(owner).map_err(|_| StructuredUnknown::InvalidInput)?);
    if replay {
        hash.update(b"ferrum.structured-replay-coordinates.v1\0");
    }
    if installed {
        hash.update(b"ferrum.installed-plain-text-repetition-work.v1\0");
        hash.update(b"ferrum.installed-plain-text-completion.v1\0");
    }
    Ok(hash.finalize().into())
}
