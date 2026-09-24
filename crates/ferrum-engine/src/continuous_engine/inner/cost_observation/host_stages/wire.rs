//! Borrowed profile-v2 wire representation: sizing/writing never clones the
//! retained shape vectors or allocates a second complete shape.
use super::*;
use ferrum_scheduler::implementations::continuous::{cost_model::PrefillShape, cost_profile::*};
use serde::ser::{SerializeSeq, SerializeStruct};

struct Prefills<'a>(&'a [PrefillShape]);
impl Serialize for Prefills<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for row in self.0 {
            sequence.serialize_element(&ProfilePrefillShape {
                offset: row.offset,
                count: row.count,
                total_prompt_tokens: row.total_prompt_tokens,
            })?;
        }
        sequence.end()
    }
}
struct Exact<'a>(&'a WaveExecutionShape);
impl Serialize for Exact<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let shape = self.0;
        let mut output = serializer.serialize_struct("ProfileWaveShape", 12)?;
        output.serialize_field("kind", &ProfileWaveKind::from(shape.kind))?;
        output.serialize_field("path", &ProfileExecutionPath::from(shape.path))?;
        output.serialize_field("provider_signature", &shape.provider_signature)?;
        output.serialize_field("output_policy_signature", &shape.output_policy_signature)?;
        output.serialize_field("graph_state", &ProfileGraphState::from(shape.graph_state))?;
        output.serialize_field("order", &ProfileBatchOrder::from(shape.order))?;
        output.serialize_field("decode_kv_tokens", &shape.decode_kv_tokens)?;
        output.serialize_field("prefill_chunks", &Prefills(&shape.prefill_chunks))?;
        output.serialize_field("recurrent_state_bytes", &shape.recurrent_state_bytes)?;
        output.serialize_field("restore_bytes", &shape.restore_bytes)?;
        output.serialize_field("maintenance_bytes", &shape.maintenance_bytes)?;
        output.serialize_field("maintenance_units", &shape.maintenance_units)?;
        output.end()
    }
}
#[derive(Serialize)]
struct Shape<'a> {
    exact: Exact<'a>,
    numeric_features: &'a Option<CanonicalWaveCostFeatures>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host_content_features: &'a Option<HostContentCostFeaturesV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    row_multiset_features: Option<&'a HostRowMultisetCostFeaturesV2>,
}
pub(super) fn serialize_shape<S: serde::Serializer>(
    value: &Option<WaveExecutionShape>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    value
        .as_ref()
        .map(|shape| Shape {
            exact: Exact(shape),
            numeric_features: &shape.numeric_features,
            host_content_features: &shape.host_content_features,
            row_multiset_features: shape.row_multiset_features.as_ref(),
        })
        .serialize(serializer)
}

/// Raw 4/5 and cut 2/3 retain their original shape fields even when the live
/// producer has additional evidence. Raw 6/cut 4 opt into the new field.
/// This borrowed view never clones a row vector or edits retained evidence.
#[derive(Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct ExportEvidence<'a> {
    schema_version: u32,
    call_id: u64,
    fingerprint: Option<ProfileFingerprint>,
    actual_shape: Option<Shape<'a>>,
    prepare_started_at_ns: Option<u64>,
    executor_returned_at_ns: Option<u64>,
    rows: &'a [HostRowStageV1],
    finalized_at_ns: Option<u64>,
    full_wall_ns: Option<u64>,
    completeness: HostStageCompleteness,
}
impl<'a> ExportEvidence<'a> {
    pub(in crate::continuous_engine::inner::cost_observation) fn new(
        evidence: &'a HostStageEvidenceV1,
        row_multiset: bool,
    ) -> Self {
        Self {
            schema_version: evidence.schema_version,
            call_id: evidence.call_id,
            fingerprint: evidence.fingerprint.as_ref().map(ProfileFingerprint::from),
            actual_shape: evidence.actual_shape.as_ref().map(|shape| Shape {
                exact: Exact(shape),
                numeric_features: &shape.numeric_features,
                host_content_features: &shape.host_content_features,
                row_multiset_features: shape
                    .row_multiset_features
                    .as_ref()
                    .filter(|_| row_multiset),
            }),
            prepare_started_at_ns: evidence.prepare_started_at_ns,
            executor_returned_at_ns: evidence.executor_returned_at_ns,
            rows: &evidence.rows,
            finalized_at_ns: evidence.finalized_at_ns,
            full_wall_ns: evidence.full_wall_ns,
            completeness: evidence.completeness,
        }
    }
}
