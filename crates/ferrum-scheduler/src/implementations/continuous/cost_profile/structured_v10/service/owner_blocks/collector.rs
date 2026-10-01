use super::*;
use crate::implementations::continuous::cost_model::structured_v2::{
    DeclaredAlgorithmUniverseV1, NonNegativeEnvelopeContractV1, OwnerAlgorithmUniversePolicyV1,
    OwnerInputReadinessDecisionV1, OwnerInputTargetV1,
};
mod blocks;
mod diagnostic;
mod ingest;
mod memory;
mod prepared_tail;
mod replay;
pub use replay::replay_structured_source_v7;

pub(super) enum State {
    Empty,
    Fitted(FittedStructuredModelV2),
    Calibrated(CalibratedStructuredModelV2),
    Qualified(Arc<QualifiedStructuredModelV2>),
    Failed(String),
}
impl State {
    fn phase(&self) -> Option<StructuredPhaseV2> {
        match self {
            Self::Empty => Some(StructuredPhaseV2::Fit),
            Self::Fitted(_) => Some(StructuredPhaseV2::Residual),
            Self::Calibrated(_) => Some(StructuredPhaseV2::Qualification),
            _ => None,
        }
    }
    fn parameters(&self) -> Option<[u8; 32]> {
        match self {
            Self::Fitted(v) => Some(v.parameters_signature()),
            Self::Calibrated(v) => Some(v.parameters_signature()),
            Self::Qualified(v) => Some(v.parameters_signature()),
            _ => None,
        }
    }
    fn retained(&self) -> Option<usize> {
        match self {
            Self::Empty => Some(0),
            Self::Fitted(v) => v.retained_payload_bytes(),
            Self::Calibrated(v) => v.retained_payload_bytes(),
            Self::Qualified(v) => v.retained_payload_bytes(),
            Self::Failed(s) => Some(s.capacity()),
        }
    }
}
pub(super) struct Owner {
    pub(super) scope: StructuredScopeV2,
    pub(super) contract: StructuredOwnerPhaseContractV1,
    pub(super) state: State,
    pub(super) boundary: Option<StructuredOwnerPhaseBoundaryV1>,
    pub(super) samples: Vec<StructuredNumericObservationV2>,
    workspace_bytes: usize,
    sample_heap_bytes: usize,
    maximum_sample_bytes: usize,
    sample_axes: usize,
    pub(super) prior_members: usize,
    pub(super) domain: StructuredServiceDomainFreezeV1,
    pub(super) phases: Vec<StructuredPhaseProvenanceV10>,
    pub(super) oldest: u64,
    pub(super) newest: u64,
    input_target: Option<OwnerInputTargetV1>,
    geometry_visits: u64,
}
/// Population context has no serialized source schema. Both protocols append
/// their own real header bytes before any original offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PopulationSource {
    OwnerBlocksV7,
    PreparedOwnerBlocksV8,
}
#[derive(Clone)]
pub(super) struct PopulationHeader {
    pub(super) capture_identity: [u8; 32],
    pub(super) protocol: [u8; 32],
    pub(super) fingerprint: ProfileFingerprint,
    pub(super) producer: serde_json::Value,
    pub(super) opening: StructuredServiceClockV7,
    pub(super) monotonic_domain: Option<ferrum_interfaces::execution_cost::CostMonotonicDomainV1>,
    pub(super) declaration: StructuredServiceDeclarationV7,
    pub(super) declaration_sha256: [u8; 32],
    pub(super) maximum_file_bytes: u64,
    pub(super) source_kind: PopulationSource,
}
impl From<StructuredServiceHeaderV7> for PopulationHeader {
    fn from(h: StructuredServiceHeaderV7) -> Self {
        Self {
            capture_identity: h.capture_identity,
            protocol: h.protocol,
            fingerprint: h.fingerprint,
            producer: h.producer,
            opening: h.opening,
            monotonic_domain: h.monotonic_domain,
            declaration: h.declaration,
            declaration_sha256: h.declaration_sha256,
            maximum_file_bytes: h.maximum_file_bytes,
            source_kind: PopulationSource::OwnerBlocksV7,
        }
    }
}
/// Receives only the original canonical bytes after the collector accepted
/// them. Storage failure must be contained by the optional sink itself.
pub trait StructuredSourceRecordSinkV1: Send + Sync {
    fn append_original(&mut self, bytes: &[u8], receipt: (u64, [u8; 32]));
}

pub struct StructuredServiceCollectorV7 {
    pub(super) header: PopulationHeader,
    limits: CostProfileLoadLimits,
    prefix: Sha256,
    prefix_bytes: u64,
    maximum_encoded_source_bytes: u64,
    readiness_scalar_visits: u64,
    phase_transition_attempts: u64,
    record_sink: Option<Box<dyn StructuredSourceRecordSinkV1>>,
    pub(super) offered: u64,
    last_fifo: u64,
    block: u64,
    block_count: usize,
    opened: Option<u64>,
    last_observed: u64,
    pub(super) last_close: Option<StructuredServiceClockV7>,
    calls: Vec<u64>,
    frontiers: physical::Frontiers,
    discovery: Option<discovery::DiscoveryWindow>,
    generation_universe: Option<DeclaredAlgorithmUniverseV1>,
    // A seed declares allowed algorithms, not the raw call's numeric projection.
    // Freeze the union only at the first complete ordinary discovery boundary.
    seeded_input_contract: Option<NonNegativeEnvelopeContractV1>,
    block_routes: StructuredServiceRouteCountsV1,
    pub(super) owners: Vec<Owner>,
    pub(super) total_rows: u64,
    numeric_bytes: usize,
    persistent_bytes: usize,
    transition_bytes: usize,
    capacity_reported: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    reserved_peak: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    accepted_reservation_peak: std::sync::atomic::AtomicUsize,
    external_retained_bytes: usize,
    poisoned: bool,
    closed: bool,
    // Source8 terminal audit seal only; never used by the source7 protocol.
    prepared_tail: Option<StructuredPreparedPartialTailV8>,
}
impl StructuredServiceCollectorV7 {
    pub fn new(
        header: StructuredServiceHeaderV7,
        limits: CostProfileLoadLimits,
    ) -> Result<Self, CostProfileError> {
        header.validate()?;
        let bytes = record_bytes_v7(&header)?;
        Self::from_original_header(header.into(), limits, &bytes)
    }
    /// Hash original records incrementally without retaining their encoded
    /// journal. The explicit work budget is separate from offline file reads.
    pub fn new_streaming(
        header: StructuredServiceHeaderV7,
        limits: CostProfileLoadLimits,
        maximum_encoded_source_bytes: std::num::NonZeroU64,
    ) -> Result<Self, CostProfileError> {
        header.validate()?;
        let bytes = record_bytes_v7(&header)?;
        Self::from_original_header_streaming(
            header.into(),
            limits,
            &bytes,
            maximum_encoded_source_bytes,
        )
    }
    /// An optional observer of the exact accepted source7 bytes. It cannot
    /// change protocol acceptance, numerical qualification or source identity.
    pub fn new_streaming_with_record_sink(
        header: StructuredServiceHeaderV7,
        limits: CostProfileLoadLimits,
        maximum_encoded_source_bytes: std::num::NonZeroU64,
        sink: Box<dyn StructuredSourceRecordSinkV1>,
    ) -> Result<Self, CostProfileError> {
        header.validate()?;
        let bytes = record_bytes_v7(&header)?;
        Self::from_original_header_with_record_sink(
            header.into(),
            limits,
            &bytes,
            Some(maximum_encoded_source_bytes),
            sink,
        )
    }
    pub(super) fn from_original_header(
        header: PopulationHeader,
        limits: CostProfileLoadLimits,
        original_header_bytes: &[u8],
    ) -> Result<Self, CostProfileError> {
        let maximum = header
            .maximum_file_bytes
            .min(limits.max_file_bytes.get() as u64);
        Self::from_original_header_with_limit(header, limits, original_header_bytes, maximum, None)
    }
    pub(super) fn from_original_header_streaming(
        header: PopulationHeader,
        limits: CostProfileLoadLimits,
        original_header_bytes: &[u8],
        maximum_encoded_source_bytes: std::num::NonZeroU64,
    ) -> Result<Self, CostProfileError> {
        if header.maximum_file_bytes != maximum_encoded_source_bytes.get() {
            return Err(invalid(
                "streaming source budget differs from original declaration",
            ));
        }
        Self::from_original_header_with_limit(
            header,
            limits,
            original_header_bytes,
            maximum_encoded_source_bytes.get(),
            None,
        )
    }
    pub(super) fn from_original_header_with_record_sink(
        header: PopulationHeader,
        limits: CostProfileLoadLimits,
        original_header_bytes: &[u8],
        maximum_encoded_source_bytes: Option<std::num::NonZeroU64>,
        sink: Box<dyn StructuredSourceRecordSinkV1>,
    ) -> Result<Self, CostProfileError> {
        let maximum = match maximum_encoded_source_bytes {
            Some(budget) if header.maximum_file_bytes == budget.get() => budget.get(),
            Some(_) => {
                return Err(invalid(
                    "streaming source budget differs from original declaration",
                ))
            }
            None => header
                .maximum_file_bytes
                .min(limits.max_file_bytes.get() as u64),
        };
        Self::from_original_header_with_limit(
            header,
            limits,
            original_header_bytes,
            maximum,
            Some(sink),
        )
    }
    fn from_original_header_with_limit(
        header: PopulationHeader,
        limits: CostProfileLoadLimits,
        original_header_bytes: &[u8],
        maximum_encoded_source_bytes: u64,
        record_sink: Option<Box<dyn StructuredSourceRecordSinkV1>>,
    ) -> Result<Self, CostProfileError> {
        limits.validate()?;
        let seeded_input_contract = matches!(
            header.declaration.schedule.algorithm_universe,
            Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
        )
        .then(|| {
            let mut contract = header
                .declaration
                .nonnegative_envelope
                .clone()
                .expect("validated seed envelope");
            contract.algorithm_universe = None;
            contract
        });
        let mut out = Self {
            header,
            limits,
            prefix: Sha256::new(),
            prefix_bytes: 0,
            maximum_encoded_source_bytes,
            readiness_scalar_visits: 0,
            phase_transition_attempts: 0,
            record_sink,
            offered: 0,
            last_fifo: 0,
            block: 0,
            block_count: 0,
            opened: None,
            last_observed: 0,
            last_close: None,
            calls: Vec::new(),
            frontiers: Default::default(),
            discovery: None,
            generation_universe: None,
            seeded_input_contract,
            block_routes: Default::default(),
            owners: Vec::new(),
            total_rows: 0,
            numeric_bytes: 0,
            persistent_bytes: 0,
            transition_bytes: 0,
            capacity_reported: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            reserved_peak: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            accepted_reservation_peak: std::sync::atomic::AtomicUsize::new(0),
            external_retained_bytes: 0,
            poisoned: false,
            closed: false,
            prepared_tail: None,
        };
        out.append_bytes(original_header_bytes)?;
        Ok(out)
    }
    pub(super) fn append(&mut self, value: &impl Serialize) -> Result<(), CostProfileError> {
        let bytes = record_bytes_v7(value)?;
        self.append_bytes(&bytes)
    }
    fn append_bytes(&mut self, bytes: &[u8]) -> Result<(), CostProfileError> {
        let size = self
            .prefix_bytes
            .checked_add(bytes.len() as u64)
            .ok_or(CostProfileError::Limit("source7 byte overflow"))?;
        if size > self.maximum_encoded_source_bytes {
            self.poisoned = true;
            return Err(CostProfileError::Limit("source7 byte capacity"));
        }
        self.prefix.update(&bytes);
        self.prefix_bytes = size;
        if let Some(sink) = &mut self.record_sink {
            sink.append_original(bytes, (size, self.prefix.clone().finalize().into()));
        }
        Ok(())
    }
    pub fn source_receipt(&self) -> (u64, [u8; 32]) {
        (self.prefix_bytes, self.prefix.clone().finalize().into())
    }
    pub fn offered(&self) -> u64 {
        self.offered
    }
    pub fn last_fifo(&self) -> u64 {
        self.last_fifo
    }
    pub fn qualified_children(&self) -> usize {
        self.owners
            .iter()
            .filter(|o| matches!(o.state, State::Qualified(_)))
            .count()
    }
    pub fn audit(&self) -> StructuredServiceAuditV7 {
        StructuredServiceAuditV7 {
            block: self.block,
            offered: self.offered,
            block_offered: self.block_count,
            last_fifo: self.last_fifo,
            source_bytes: self.prefix_bytes,
            readiness_scalar_visits: self.readiness_scalar_visits,
            phase_transition_attempts: self.phase_transition_attempts,
            retained_numeric_bytes: self.numeric_bytes,
            poisoned: self.poisoned,
            closed: self.closed,
            owners: self
                .owners
                .iter()
                .map(|o| StructuredOwnerAuditV7 {
                    owner_attempt_id: o.contract.owner_attempt_id,
                    owner: o.scope.owner.clone(),
                    phase: o.state.phase(),
                    global_offers: o.boundary.map_or(0, |b| {
                        self.offered
                            .saturating_sub(b.first_offered)
                            .saturating_add(1)
                    }),
                    owner_offered: o.domain.owner_offered,
                    eligible: o.samples.len(),
                    qualified: matches!(o.state, State::Qualified(_)),
                    failure: match &o.state {
                        State::Failed(s) => Some(s.clone()),
                        _ => None,
                    },
                })
                .collect(),
        }
    }
    /// Immutable universe fixed by the original first complete Discovery.
    /// Exposing this identity does not choose members or extend its source age.
    pub fn frozen_algorithm_universe(&self) -> Option<&DeclaredAlgorithmUniverseV1> {
        self.generation_universe.as_ref()
    }
    /// A malformed/lost original attempt closes this source. It cannot be
    /// replaced with another offer or silently removed from a denominator.
    pub fn fail(
        &mut self,
        ticket: u64,
        fifo: u64,
        at_ns: u64,
        reason: impl Into<String>,
    ) -> Result<StructuredServiceRecordV7, CostProfileError> {
        if self.closed {
            return Err(invalid("source7 is closed"));
        }
        let reason = reason.into();
        if reason.len() > 4096 {
            return Err(CostProfileError::Limit("source7 failure reason"));
        }
        let (source_prefix_bytes, source_prefix_sha256) = self.source_receipt();
        let r = StructuredServiceRecordV7::Failed {
            ticket,
            fifo,
            at_ns,
            reason,
            source_prefix_bytes,
            source_prefix_sha256,
        };
        self.poisoned = true;
        self.append(&r)?;
        Ok(r)
    }
    pub fn stop(
        &mut self,
        closing: StructuredServiceClockV7,
    ) -> Result<StructuredServiceRecordV7, CostProfileError> {
        if self.closed
            || closing.monotonic_ns
                < self
                    .last_close
                    .map_or(self.header.opening.monotonic_ns, |c| c.monotonic_ns)
        {
            return Err(invalid("source7 stop clock/state differs"));
        }
        let (source_prefix_bytes, source_prefix_sha256) = self.source_receipt();
        let r = StructuredServiceRecordV7::Footer {
            closing,
            offered: self.offered,
            accepted_fifo_cutoff: self.last_fifo,
            incomplete_block: self.opened.is_some(),
            source_prefix_bytes,
            source_prefix_sha256,
        };
        self.append(&r)?;
        self.closed = true;
        Ok(r)
    }
    fn epoch_deadline(&self) -> Result<u64, CostProfileError> {
        self.header
            .opening
            .monotonic_ns
            .checked_add(self.header.declaration.maximum_window_ns)
            .ok_or(CostProfileError::Limit("source7 epoch deadline overflow"))
    }
    pub(super) fn check_retained(&mut self) -> Result<(), CostProfileError> {
        let serial = self.serial_physical_workspace();
        let mut workspace = 0;
        let mut transitions = 0usize;
        let mut n = std::mem::size_of::<Self>()
            .checked_add(self.external_retained_bytes)
            .ok_or(CostProfileError::Limit("source retained overflow"))?;
        let mut add = |v: Option<usize>| -> Result<(), CostProfileError> {
            n = n
                .checked_add(v.ok_or(CostProfileError::Limit("source7 retained overflow"))?)
                .ok_or(CostProfileError::Limit("source7 retained overflow"))?;
            Ok(())
        };
        add(self
            .header
            .declaration
            .nonnegative_envelope
            .as_ref()
            .and_then(|c| c.algorithm_universe.as_ref())
            .map_or(Some(0), |u| u.retained_payload_bytes()))?;
        add(self
            .generation_universe
            .as_ref()
            .map_or(Some(0), |u| u.retained_payload_bytes()))?;
        add(self
            .calls
            .capacity()
            .checked_mul(std::mem::size_of::<u64>()))?;
        add(self.frontiers.retained_heap_bytes())?;
        add(self
            .owners
            .capacity()
            .checked_mul(std::mem::size_of::<Owner>()))?;
        if let Some(d) = &self.discovery {
            add(Some(d.retained_bytes_upper_bound()))?;
        }
        for o in &self.owners {
            add(o
                .contract
                .nonnegative_envelope
                .as_ref()
                .and_then(|c| c.algorithm_universe.as_ref())
                .map_or(Some(0), |u| u.retained_payload_bytes()))?;
            add(o.scope.retained_heap_bytes())?;
            add(Some(
                o.contract
                    .input_target
                    .as_ref()
                    .map_or(0, OwnerInputTargetV1::retained_heap_bytes),
            ))?;
            add(Some(
                o.input_target
                    .as_ref()
                    .map_or(0, OwnerInputTargetV1::retained_heap_bytes),
            ))?;
            add(o.state.retained())?;
            add(o
                .phases
                .capacity()
                .checked_mul(std::mem::size_of::<StructuredPhaseProvenanceV10>()))?;
            if serial {
                add(o
                    .samples
                    .capacity()
                    .checked_mul(std::mem::size_of::<StructuredNumericObservationV2>()))?;
                add(Some(o.sample_heap_bytes))?;
                let transition = self.physical_transition(o, None)?;
                transitions =
                    transitions
                        .checked_add(transition)
                        .ok_or(CostProfileError::Limit(
                            "source7 transition retained overflow",
                        ))?;
                add(Some(transition))?;
                workspace = workspace.max(self.physical_workspace(o, None)?);
            } else {
                add(Some(o.workspace_bytes))?;
            }
        }
        self.persistent_bytes = n;
        self.transition_bytes = transitions;
        n = n
            .checked_add(workspace)
            .ok_or(CostProfileError::Limit("source7 retained overflow"))?;
        self.observe_reservation(n);
        if n > self.header.declaration.maximum_retained_numeric_bytes {
            return Err(self.capacity_error(
                "source7 shared retained capacity",
                self.persistent_bytes,
                workspace,
            ));
        }
        self.observe_accepted_reservation(n);
        self.numeric_bytes = n;
        Ok(())
    }
    pub(super) fn retain_external(&mut self, bytes: usize) -> Result<(), CostProfileError> {
        self.external_retained_bytes = bytes;
        self.check_retained()
    }
    pub(super) fn ensure_active(&self) -> Result<(), CostProfileError> {
        if self.closed || self.poisoned {
            return Err(invalid("source population is closed/poisoned"));
        }
        Ok(())
    }
    pub(super) fn poison(&mut self) {
        self.poisoned = true;
    }
    pub fn push(&mut self, r: &StructuredServiceRecordV7) -> Result<(), CostProfileError> {
        let result = self.push_inner(r);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}
