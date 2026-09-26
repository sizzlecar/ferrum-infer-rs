//! Original Prepared evidence, represented by the same four shape fields as
//! host-stage wire. Deserializing this record grants no live receipt authority.
use super::*;
use serde::ser::SerializeStruct;

pub(super) struct CompletedWire<'a> {
    pub reserved: &'a population::ReservedWaveV2,
    pub stages: &'a HostStageEvidenceV1,
    pub queue: Option<HostStageQueueReceipt>,
    pub reconciled: bool,
    pub numeric: Option<&'a StructuredNumericObservationV2>,
    pub stage_binding: [u8; 32],
}
impl Serialize for CompletedWire<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let r = self.reserved;
        let stages = self.stages;
        let mut out = serializer.serialize_struct("Completed", 13)?;
        out.serialize_field("kind", "completed")?;
        out.serialize_field("offered", &r.attempt.offered)?;
        out.serialize_field("member", &r.member)?;
        out.serialize_field("phase", &r.attempt.phase)?;
        out.serialize_field("cohort", &r.attempt.cohort)?;
        out.serialize_field("queue", &self.queue)?;
        out.serialize_field("reconciled", &self.reconciled)?;
        out.serialize_field(
            "host_stages",
            &r.member.map(|_| stages.structured_diagnostic_view()),
        )?;
        out.serialize_field(
            "outside_settlement",
            &r.member.is_none().then_some(Outside {
                stages,
                binding: self.stage_binding,
            }),
        )?;
        out.serialize_field(
            "selected_independent_attention_v2",
            &r.member
                .and_then(|_| stages.statistical_evidence.as_ref())
                .and_then(|v| v.independent_attention_v2()),
        )?;
        out.serialize_field(
            "selected_structured_capture",
            &r.member
                .and_then(|_| stages.statistical_evidence.as_ref())
                .and_then(|v| v.structured_capture())
                .map(|v| v.map(AsRef::as_ref)),
        )?;
        out.serialize_field("numeric", &self.numeric.map(Numeric))?;
        out.serialize_field("conversion_error", &Option::<String>::None)?;
        out.end()
    }
}
struct Numeric<'a>(&'a StructuredNumericObservationV2);
impl Serialize for Numeric<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let s = self.0;
        let mut out = serializer.serialize_struct("Numeric", 7)?;
        out.serialize_field("fifo", &s.ordinal)?;
        out.serialize_field("call_id", &s.call_id)?;
        out.serialize_field("observed_at_ns", &s.observed_at_ns)?;
        out.serialize_field("wall_ns", &s.wall_ns)?;
        out.serialize_field("domain", s.input.domain_signature())?;
        out.serialize_field("basis", s.input.regression_axes())?;
        out.serialize_field("support", s.input.joint_support_coordinates())?;
        out.end()
    }
}
struct Outside<'a> {
    stages: &'a HostStageEvidenceV1,
    binding: [u8; 32],
}
impl Serialize for Outside<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let s = self.stages;
        let mut out = serializer.serialize_struct("OutsideSettlement", 9)?;
        out.serialize_field("call_id", &s.call_id)?;
        out.serialize_field(
            "fingerprint",
            &s.fingerprint
                .as_ref()
                .map(profile::ProfileFingerprint::from),
        )?;
        out.serialize_field("prepare_started_at_ns", &s.prepare_started_at_ns)?;
        out.serialize_field("executor_returned_at_ns", &s.executor_returned_at_ns)?;
        out.serialize_field("finalized_at_ns", &s.finalized_at_ns)?;
        out.serialize_field("full_wall_ns", &s.full_wall_ns)?;
        out.serialize_field("completeness", &s.completeness)?;
        out.serialize_field("rows", &s.rows)?;
        out.serialize_field("stage_binding", &self.binding)?;
        out.end()
    }
}

#[derive(Serialize)]
struct Shape<'a> {
    exact: profile::ProfileWaveShape,
    numeric_features: &'a Option<CanonicalWaveCostFeatures>,
    host_content_features: &'a Option<HostContentCostFeaturesV1>,
    row_multiset_features: &'a Option<HostRowMultisetCostFeaturesV2>,
}
#[derive(Serialize)]
pub(super) struct PreparedWire<'a> {
    exact: Shape<'a>,
    selected: &'a StatisticalWaveEvidenceV1,
    selected_independent_attention_v2: Option<&'a IndependentAttentionWaveEvidenceV2>,
    recipe: &'a UnsettledStructuredWaveEvidenceV1,
    owner_facts: &'a StructuredOwnerFactsV2,
    rows: &'a [PreparedRowBindingV2],
}
impl<'a> PreparedWire<'a> {
    pub fn new(value: &'a PreparedStructuredFactsV2) -> Result<Self, ExportError> {
        value.validate().map_err(numeric_error)?;
        let s = &value.exact;
        let kind = match s.kind {
            ActualWaveKind::Decode => profile::ProfileWaveKind::Decode,
            ActualWaveKind::Prefill => profile::ProfileWaveKind::Prefill,
            ActualWaveKind::Mixed => profile::ProfileWaveKind::Mixed,
            _ => return Err(ExportError::Source("unsupported Prepared wave kind")),
        };
        if s.path != ActualWavePath::PlanRuntime
            || !matches!(
                s.graph,
                ActualWaveGraphState::Disabled
                    | ActualWaveGraphState::Warm
                    | ActualWaveGraphState::ConfiguredEager
            )
            || s.row_order != ActualWaveRowOrder::Ordered
        {
            return Err(ExportError::Source("unsupported Prepared route"));
        }
        let mut decode_kv_tokens = Vec::new();
        let mut prefill_chunks = Vec::new();
        for work in &s.rows {
            match *work {
                ActualRowWork::Decode { kv_tokens } => decode_kv_tokens.push(kv_tokens),
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => prefill_chunks.push(profile::ProfilePrefillShape {
                    offset,
                    count: std::num::NonZeroU32::new(count)
                        .ok_or(ExportError::Source("Prepared prefill count is zero"))?,
                    total_prompt_tokens: std::num::NonZeroU32::new(total_prompt_tokens)
                        .ok_or(ExportError::Source("Prepared prompt total is zero"))?,
                }),
                _ => return Err(ExportError::Source("unsupported Prepared row work")),
            }
        }
        Ok(Self {
            exact: Shape {
                exact: profile::ProfileWaveShape {
                    kind,
                    path: profile::ProfileExecutionPath::PlanRuntime,
                    provider_signature: s.provider_signature,
                    output_policy_signature: s.output_policy_signature,
                    graph_state: match s.graph {
                        ActualWaveGraphState::Warm => profile::ProfileGraphState::Warm,
                        ActualWaveGraphState::ConfiguredEager => {
                            profile::ProfileGraphState::ConfiguredEager
                        }
                        ActualWaveGraphState::Disabled => profile::ProfileGraphState::Disabled,
                        ActualWaveGraphState::Cold => {
                            return Err(ExportError::Source("unsupported Prepared route"))
                        }
                    },
                    order: profile::ProfileBatchOrder::Ordered,
                    decode_kv_tokens,
                    prefill_chunks,
                    recurrent_state_bytes: s.recurrent_state_bytes,
                    restore_bytes: 0,
                    maintenance_bytes: 0,
                    maintenance_units: 0,
                },
                numeric_features: &s.numeric_features,
                host_content_features: &s.host_content_features,
                row_multiset_features: &s.row_multiset_features,
            },
            selected: &value.selected,
            selected_independent_attention_v2: value.selected.independent_attention_v2(),
            recipe: value.recipe.as_ref(),
            owner_facts: &value.owner,
            rows: &value.rows,
        })
    }
}
