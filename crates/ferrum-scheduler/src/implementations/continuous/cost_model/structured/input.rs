use super::super::statistical::StatisticalModelInputV1;
use super::*;
use ferrum_interfaces::execution_cost::{
    AlgorithmWorkKindV1, CanonicalWaveCostShape, DeviceNumericWorkV1, HostContentDomainV1,
    HostRowRoleV2, HostTerminalExpectationV1, StatisticalWaveEvidenceV1, StructuredHostRowV1,
    UnsettledStructuredWaveEvidenceV1,
};
use sha2::{Digest, Sha256};

/// First declared sharing scope. Other scopes require new basis/qualification
/// semantics, rather than falling through to this one. This deliberately does
/// not claim support for first decode, prefill/mixed, mask, pending UTF8, multiple
/// terminals, EOS/stop/cancel, or extra terminal device/cleanup work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredScopeV1 {
    OrdinaryDecodeSingleLength { rows: usize },
}
impl StructuredScopeV1 {
    pub(super) fn rows(self) -> usize {
        match self {
            Self::OrdinaryDecodeSingleLength { rows } => rows,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredInputV1 {
    pub(super) domain: [u8; 32],
    pub(super) scope: StructuredScopeV1,
    pub(super) basis: Vec<f64>,
    pub(super) support: Vec<u64>,
    /// Raw physical positions survive projection for audit and qualification.
    pub(super) physical_host_rows: Vec<StructuredHostRowV1>,
    pub(super) terminal_position: Option<usize>,
}
impl StructuredInputV1 {
    pub fn domain_signature(&self) -> &[u8; 32] {
        &self.domain
    }
    pub fn scope(&self) -> StructuredScopeV1 {
        self.scope
    }
    pub fn physical_host_rows(&self) -> &[StructuredHostRowV1] {
        &self.physical_host_rows
    }
    pub fn regression_axes(&self) -> &[f64] {
        &self.basis
    }
    pub fn joint_support_coordinates(&self) -> &[u64] {
        &self.support
    }
    pub fn from_future(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        structured: &UnsettledStructuredWaveEvidenceV1,
    ) -> Result<Self> {
        // Legacy exact/statistical equality intentionally omits algorithm-work
        // assignments. Bind this argument to the actual complete sidecar on
        // selected; matching old aggregate work is not sufficient provenance.
        let attached = selected
            .structured_capture()
            .ok_or(StructuredUnknown::MissingEvidence)?
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        if attached.as_ref() != structured {
            return Err(StructuredUnknown::MissingEvidence);
        }
        structured
            .validate_exact(exact)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let old = StatisticalModelInputV1::from_future(exact, selected)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let algorithms = structured
            .algorithm_work()
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        algorithms
            .validate_structure(structured)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        Self::project_numeric(
            &old,
            *structured.device().ordered_template(),
            structured.device().provider_grouped_template().copied(),
            structured.physical_host_rows(),
            algorithms.entries().iter().map(|entry| {
                (
                    *entry.algorithm().signature(),
                    entry.kind(),
                    entry.commands(),
                    entry.work(),
                )
            }),
        )
    }

    /// Numerical replay only. Its caller validates the versioned wire/source;
    /// this cannot construct any live settlement or execution authority.
    pub(in crate::implementations::continuous) fn from_replay_parts(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        ordered: [u8; 32],
        grouped: Option<[u8; 32]>,
        rows: &[StructuredHostRowV1],
        algorithms: impl Iterator<Item = ([u8; 32], AlgorithmWorkKindV1, u64, DeviceNumericWorkV1)>,
    ) -> Result<Self> {
        let old = StatisticalModelInputV1::from_future(exact, selected)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        Self::project_numeric(&old, ordered, grouped, rows, algorithms)
    }

    fn project_numeric(
        old: &StatisticalModelInputV1,
        ordered: [u8; 32],
        grouped: Option<[u8; 32]>,
        rows: &[StructuredHostRowV1],
        algorithms: impl Iterator<Item = ([u8; 32], AlgorithmWorkKindV1, u64, DeviceNumericWorkV1)>,
    ) -> Result<Self> {
        let terminal_position = ordinary_decode_position(rows)?;
        let scope = StructuredScopeV1::OrdinaryDecodeSingleLength { rows: rows.len() };
        let mut domain = Sha256::new();
        domain.update(MODEL_REVISION.as_bytes());
        // This normalization is provider-proved for the device subgroups only;
        // it neither reorders host rows nor grants exact execution permission.
        if let Some(grouped) = grouped {
            domain.update([1]);
            domain.update(grouped);
        } else {
            domain.update([0]);
            domain.update(ordered);
        }
        number(&mut domain, rows.len() as u64);
        let policy = rows[0].installed_policy;
        domain.update(policy.categorical_signature);
        for value in [
            policy.decoder_text_bytes_per_token,
            policy.decoder_scratch_bytes_per_token,
            policy.raw_token_bytes_bound,
            u64::from(rows[0].decode_requires_full_logits.unwrap()),
            u64::from(rows[0].repetition_penalty_bits.unwrap()),
        ] {
            number(&mut domain, value);
        }
        let mut basis = vec![1.0];
        let mut support = Vec::new();
        for (algorithm, kind, commands, work) in algorithms {
            domain.update(algorithm);
            domain.update([match kind {
                AlgorithmWorkKindV1::Kernel => 0,
                AlgorithmWorkKindV1::HostToDevice => 1,
                AlgorithmWorkKindV1::DeviceToHost => 2,
                AlgorithmWorkKindV1::DeviceToDevice => 3,
                AlgorithmWorkKindV1::Fill => 4,
            }]);
            support.push(commands);
            support.extend(device_coordinates(work));
            // A small per-algorithm basis, not one coefficient per raw row or
            // per-position bit. All omitted raw work remains in joint support.
            basis.push(commands as f64);
            if kind == AlgorithmWorkKindV1::Kernel {
                basis.extend([
                    work.inner_work_units as f64,
                    work.padded_units as f64,
                    work.grid_blocks as f64,
                ]);
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
        let h = old.host_and_sequence();
        // Output budget is still exact-bound metadata. It is not consumed work;
        // its typed LengthBoundary consequence is modeled explicitly below.
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
            h.attention_pairs as f64,
            h.sampling_history_sum as f64,
            h.decoded_text_bytes_sum as f64,
            h.decode_scratch_bytes_sum as f64,
        ]);
        let position_work = terminal_basis(terminal_position);
        support.extend(position_work);
        basis.extend(position_work.map(|n| n as f64));
        Ok(Self {
            domain: domain.finalize().into(),
            scope,
            basis,
            support,
            physical_host_rows: rows.to_vec(),
            terminal_position,
        })
    }
    pub(super) fn same_domain(&self, other: &Self) -> Result<()> {
        if self.domain != other.domain
            || self.scope != other.scope
            || self.basis.len() != other.basis.len()
            || self.support.len() != other.support.len()
        {
            return Err(StructuredUnknown::WrongDomain);
        }
        Ok(())
    }
    pub(super) fn validate(&self, settings: &StructuredSettingsV1) -> Result<()> {
        if self.scope.rows() != self.physical_host_rows.len()
            || ordinary_decode_position(&self.physical_host_rows)? != self.terminal_position
            || self.domain == [0; 32]
            || self.basis.len() < 4
            || self.support.len() < 3
            || self.basis[0] != 1.0
            || self.basis.iter().any(|v| !v.is_finite() || *v < 0.)
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        if self.basis.len() > settings.max_axes
            || self.support.len() > settings.max_axes
            || self.support.iter().any(|n| *n > (1 << 53))
        {
            return Err(StructuredUnknown::Capacity);
        }
        let tail = terminal_basis(self.terminal_position);
        if self.support[self.support.len() - 3..] != tail
            || self.basis[self.basis.len() - 3..] != tail.map(|n| n as f64)
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        Ok(())
    }
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
fn number(hash: &mut Sha256, value: u64) {
    hash.update(value.to_le_bytes());
}
pub(super) fn terminal_basis(position: Option<usize>) -> [u64; 3] {
    position.map_or([0; 3], |p| {
        let p = p as u64 + 1;
        [1, p, p * p]
    })
}
pub(super) fn ordinary_decode_position(rows: &[StructuredHostRowV1]) -> Result<Option<usize>> {
    if rows.is_empty() || rows.len() > 128 {
        return Err(StructuredUnknown::UnsupportedScope);
    }
    let first = rows[0];
    let mut terminal = None;
    for (position, row) in rows.iter().enumerate() {
        if row.physical_position as usize != position
            || row.role != HostRowRoleV2::Decode
            || row.installed_policy != first.installed_policy
            || row.installed_policy.empirical_content_domain
                != Some(HostContentDomainV1::PlainTextGreedyV1)
            || row.no_generated_history
            || row.pending_decoded_utf8
            || row.initial_prefill
            || row.final_prefill
            || row.mask_upload_required
            || row.decode_requires_full_logits.is_none()
            || row.decode_requires_full_logits != first.decode_requires_full_logits
            || row.repetition_penalty_bits.is_none()
            || row.repetition_penalty_bits != first.repetition_penalty_bits
        {
            return Err(StructuredUnknown::UnsupportedScope);
        }
        match row.terminal_expectation {
            HostTerminalExpectationV1::TokenMayTerminate => {}
            HostTerminalExpectationV1::LengthBoundary if terminal.is_none() => {
                terminal = Some(position)
            }
            _ => return Err(StructuredUnknown::UnsupportedScope),
        }
    }
    Ok(terminal)
}
