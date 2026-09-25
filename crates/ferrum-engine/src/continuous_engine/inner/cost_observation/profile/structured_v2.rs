//! Source3/schema10 startup import. Each query has one immutable owner and no
//! alternative model lookup after missing evidence, unsupported work or expiry.
use super::*;
use file::{ImportedStructuredModelV2, StructuredProfilePhaseV10};
use model::structured_v2::{StructuredQueryV2, StructuredUnknownV2 as Unknown, MODEL_REVISION_V2};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
mod catalog;
mod receipt;

pub(super) struct StructuredSnapshot {
    children: BTreeMap<[u8; 32], ImportedStructuredModelV2>,
}
impl StructuredSnapshot {
    fn single(child: ImportedStructuredModelV2) -> Self {
        Self {
            children: BTreeMap::from([(*child.domain_signature(), child)]),
        }
    }
    pub fn len(&self) -> usize {
        self.children.len()
    }
    fn select(&self, query: &StructuredQueryV2) -> Result<&ImportedStructuredModelV2, Unknown> {
        let child = self
            .children
            .get(query.domain_signature())
            .ok_or(Unknown::WrongDomain)?;
        if child.owner() != query.owner() {
            return Err(Unknown::WrongDomain);
        }
        Ok(child)
    }
}
pub(super) fn load_seed(
    fingerprint: model::ExecutionFingerprint,
    config: &SloCostObservationConfig,
    path: Option<&Path>,
    clock: Option<file::ProfileLoadClock>,
) -> Result<TrainingSeed, FerrumError> {
    let Some(path) = path else {
        return Ok(TrainingSeed {
            trainer: None,
            snapshot: None,
            receipt: None,
        });
    };
    let clock =
        clock.ok_or_else(|| FerrumError::config("structured V2 profile load clock missing"))?;
    let declared = config
        .profile_import
        .declared_local_clock_max_error_ns
        .ok_or_else(|| {
            FerrumError::config("structured V2 profile requires declared local wall-clock accuracy")
        })?;
    if clock.wall_max_error_ns != Some(declared) {
        return Err(FerrumError::config(
            "structured V2 profile clock differs from declared policy",
        ));
    }
    let (snapshot, receipt) = catalog::load(
        path,
        &fingerprint,
        &load_limits(&config.profile_import),
        clock,
        declared,
    )?;
    Ok(TrainingSeed {
        trainer: None,
        snapshot: Some(Arc::new(EngineCostSnapshot {
            inner: Snapshot::StructuredV2(snapshot),
            fingerprint,
        })),
        receipt: Some(receipt),
    })
}
pub(super) fn predict(
    snapshot: &StructuredSnapshot,
    fingerprint: &model::ExecutionFingerprint,
    shape: &model::WaveExecutionShape,
    evidence: Option<&PlanningCostEvidence>,
    local_now: u64,
    version: u64,
) -> Option<PlanningCost> {
    let result = evidence
        .ok_or(Unknown::MissingEvidence)
        .and_then(|e| e.structured_query_v2_for(shape))
        .and_then(|query| predict_query(snapshot, fingerprint, query, local_now, version));
    super::super::query_metrics::record_structured_v2(&result);
    match result {
        Ok(value) => Some(value),
        Err(reason) => {
            tracing::trace!(?reason,kind=?shape.kind,"SLO candidate has no qualified structured V2 cost");
            None
        }
    }
}

// Pure shared lookup: diagnostics neither emit serving-query metrics nor train.
pub(super) fn predict_query(
    snapshot: &StructuredSnapshot,
    fingerprint: &model::ExecutionFingerprint,
    query: &StructuredQueryV2,
    local_now: u64,
    version: u64,
) -> Result<PlanningCost, Unknown> {
    let child = snapshot.select(query)?;
    let (value, model_now) = child.predict_query_local_with_clock(fingerprint, query, local_now)?;
    Ok(PlanningCost {
        typical_ns: value.fitted_upper_ns,
        planning_ns: value.planning_ns,
        model_version: version,
        valid_for_ns: value
            .valid_until_ns
            .checked_sub(model_now)
            .ok_or(Unknown::Clock)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_v2_snapshot_demands_provider_bound_query_and_never_uses_legacy_shape() {
        let fp = model::ExecutionFingerprint {
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        };
        let snapshot = EngineCostSnapshot {
            inner: Snapshot::StructuredV2(StructuredSnapshot {
                children: BTreeMap::new(),
            }),
            fingerprint: fp.clone(),
        };
        assert_eq!(
            snapshot.evidence_requirement(),
            PlanningCostEvidenceRequirement::StructuredV2
        );
        assert_eq!(
            snapshot.planning_boundary(),
            model::CostBoundary::PreparationToHostSettledV1
        );
        let shape = model::WaveExecutionShape {
            kind: model::WaveKind::Decode,
            path: model::WaveExecutionPath::PlanRuntime,
            provider_signature: [5; 32],
            output_policy_signature: [6; 32],
            graph_state: model::WaveGraphState::Disabled,
            order: model::BatchOrderSemantics::Ordered,
            decode_kv_tokens: vec![64],
            prefill_chunks: vec![],
            numeric_features: None,
            host_content_features: None,
            row_multiset_features: None,
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        };
        assert!(snapshot
            .predict_with_evidence(&fp, &shape, None, 0)
            .is_none());
        assert!(matches!(
            snapshot.predict(
                &fp,
                &shape,
                model::CostBoundary::PreparationToHostSettledV1,
                0
            ),
            model::CostPrediction::Unknown(_)
        ));
        assert!(!snapshot.feedback_enabled());
    }
}
