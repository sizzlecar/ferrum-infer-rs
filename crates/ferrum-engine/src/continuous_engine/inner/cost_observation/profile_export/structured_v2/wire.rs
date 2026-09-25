//! Original Prepared evidence, represented by the same four shape fields as
//! host-stage wire. Deserializing this record grants no live receipt authority.
use super::*;

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
            || s.graph != ActualWaveGraphState::Disabled
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
                    graph_state: profile::ProfileGraphState::Disabled,
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
