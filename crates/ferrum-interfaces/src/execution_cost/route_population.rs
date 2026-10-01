//! Pre-submission route facts. These are not execution, sample, or retirement
//! authority. No public constructor or Deserialize can manufacture a receipt.
use super::*;
use crate::vnext::{
    DeviceCostGraphStreamState, DeviceReusableExecutionProgram, DeviceReusableExecutionProgramId,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparedCostRouteClassV1 {
    Warm,
    GraphDisabled,
    OutsideProgramAbsent,
    OutsideProgramNonResident,
    OutsideProgramLayoutAbsent,
    Unknown,
}
impl PreparedCostRouteClassV1 {
    /// The graph states admitted by the original warm-or-disabled population.
    /// This is only a numerical classification: it cannot produce the private
    /// selector, submission or settlement receipt required for an actual member.
    /// Configured eager work can be a valid future recipe while its original
    /// non-reusable dispatch remains outside that declared population.
    pub const fn from_eligible_graph(graph: ActualWaveGraphState) -> Option<Self> {
        match graph {
            ActualWaveGraphState::Warm => Some(Self::Warm),
            ActualWaveGraphState::Disabled => Some(Self::GraphDisabled),
            _ => None,
        }
    }

    pub const fn is_outside(self) -> bool {
        matches!(
            self,
            Self::OutsideProgramAbsent
                | Self::OutsideProgramNonResident
                | Self::OutsideProgramLayoutAbsent
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparedCostRouteReasonV1 {
    ResidentProgram,
    DeclaredGraphUnsupported,
    UnconfiguredStream,
    CatalogEmpty,
    ProgramAbsent,
    ProgramNonResident,
    ProgramPartial,
    ProgramIdentityUnavailable,
    ProgramLayoutAbsent,
    CatalogUnavailable,
    CatalogEpochMismatch,
    SelectionNotUsed,
    RuntimeUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PreparedCostRouteV1 {
    pub(crate) class: PreparedCostRouteClassV1,
    pub(crate) reason: PreparedCostRouteReasonV1,
    pub(crate) program_id: Option<DeviceReusableExecutionProgramId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) non_reusable_wave: Option<NonReusableWaveIdentityV1>,
    pub(crate) lane_id: u64,
    pub(crate) lane_epoch: u64,
    pub(crate) catalog_epoch: Option<u64>,
    pub(crate) graph_state: Option<DeviceCostGraphStreamState>,
    pub(crate) batch_step: Option<u64>,
    pub(crate) batch_invocation: Option<u64>,
}

/// Private proof that the actual claimed wave has no reusable program layout
/// or lane slot. Absence of an observation is not enough to construct this.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct NonReusableWaveIdentityV1 {
    pub(crate) plan_hash: String,
    pub(crate) runtime_implementation_fingerprint: String,
    pub(crate) immediate_sequences: u32,
    pub(crate) immediate_tokens: u64,
    pub(crate) immediate_pages: u64,
}
impl PreparedCostRouteV1 {
    pub const fn graph_state(&self) -> Option<DeviceCostGraphStreamState> {
        self.graph_state
    }
    pub const fn class(&self) -> PreparedCostRouteClassV1 {
        self.class
    }
    pub const fn reason(&self) -> PreparedCostRouteReasonV1 {
        self.reason
    }
    pub fn program_id(&self) -> Option<&DeviceReusableExecutionProgramId> {
        self.program_id.as_ref()
    }
    pub const fn lane_epoch(&self) -> u64 {
        self.lane_epoch
    }
    pub const fn catalog_epoch(&self) -> Option<u64> {
        self.catalog_epoch
    }
}

/// Program and observation come from the same lookup. The caller must submit
/// this selected program; classification cannot request another route.
pub struct PreparedReusableCostSelection<'a> {
    pub(crate) program: Option<&'a DeviceReusableExecutionProgram>,
    pub(crate) route: PreparedCostRouteV1,
}
impl<'a> PreparedReusableCostSelection<'a> {
    pub fn route(&self) -> &PreparedCostRouteV1 {
        &self.route
    }
    pub fn into_parts(
        self,
    ) -> (
        Option<&'a DeviceReusableExecutionProgram>,
        PreparedCostRouteV1,
    ) {
        (self.program, self.route)
    }
}

/// Original preparation facts retained separately from a completed cost shape.
/// In particular these rows never fabricate a graph route or wall sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedCallRouteV1 {
    pub(crate) route: PreparedCostRouteV1,
    pub(crate) selected_at_ns: u64,
    pub(crate) rows: Vec<ActualWaveRow>,
    pub(crate) submitted: Option<SubmittedRouteEvidenceV1>,
}
impl PreparedCallRouteV1 {
    pub fn route(&self) -> &PreparedCostRouteV1 {
        &self.route
    }
    pub const fn selected_at_ns(&self) -> u64 {
        self.selected_at_ns
    }
    pub fn submitted(&self) -> Option<&SubmittedRouteEvidenceV1> {
        self.submitted.as_ref()
    }
    pub fn submitted_graph(&self) -> Option<crate::vnext::DeviceSubmissionGraphEvidence> {
        self.submitted.as_ref().and_then(|s| s.graph)
    }
    pub fn rows(&self) -> &[ActualWaveRow] {
        &self.rows
    }
}

/// Original typed submission identity, copied only after core bound the native
/// attribution to that identity. No deserialization or public constructor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmittedRouteEvidenceV1 {
    pub(crate) batch_step: u64,
    pub(crate) batch_invocation: u64,
    pub(crate) plan_hash: String,
    pub(crate) runtime_implementation_fingerprint: String,
    pub(crate) lane_id: u64,
    pub(crate) submission_started_at_ns: u64,
    pub(crate) graph: Option<crate::vnext::DeviceSubmissionGraphEvidence>,
}
impl SubmittedRouteEvidenceV1 {
    pub const fn submission_started_at_ns(&self) -> u64 {
        self.submission_started_at_ns
    }
}

impl PreparedCallRouteV1 {
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        let mut bytes = std::mem::size_of::<Self>().checked_add(
            self.rows
                .capacity()
                .checked_mul(std::mem::size_of::<ActualWaveRow>())?,
        )?;
        if let Some(program) = &self.route.program_id {
            bytes = bytes.checked_add(program.retained_payload_bytes()?)?;
        }
        if let Some(wave) = &self.route.non_reusable_wave {
            bytes = bytes
                .checked_add(wave.plan_hash.capacity())?
                .checked_add(wave.runtime_implementation_fingerprint.capacity())?;
        }
        if let Some(submitted) = &self.submitted {
            bytes = bytes
                .checked_add(submitted.plan_hash.capacity())?
                .checked_add(submitted.runtime_implementation_fingerprint.capacity())?;
        }
        Some(bytes)
    }
}
