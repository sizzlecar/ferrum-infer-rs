//! Source3/schema10 startup import. Each query has one immutable owner and no
//! alternative model lookup after missing evidence, unsupported work or expiry.
use super::*;
use file::{ImportedStructuredModelV2, StructuredProfilePhaseV10};
use model::structured_v2::{StructuredQueryV2, StructuredUnknownV2 as Unknown, MODEL_REVISION_V2};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
mod catalog;
mod receipt;

#[derive(Clone)]
pub(super) struct StructuredSnapshot {
    children: Arc<BTreeMap<[u8; 32], ImportedStructuredModelV2>>,
    pub feedback: Option<Arc<super::super::selected_feedback::View>>,
}
impl StructuredSnapshot {
    fn single(child: ImportedStructuredModelV2) -> Self {
        Self {
            children: Arc::new(BTreeMap::from([(*child.domain_signature(), child)])),
            feedback: None,
        }
    }
    pub fn prospective_source(
        &self,
        query: &StructuredQueryV2,
    ) -> Option<super::super::prospective_capture::SourceIdentity> {
        let child = self.select(query).ok()?;
        let p = child.provenance();
        Some(super::super::prospective_capture::SourceIdentity {
            domain: *query.domain_signature(),
            profile_sha256: p.file_sha256,
            source_sha256: p.source_sha256,
            parameters_sha256: p.parameters_sha256,
            protocol_sha256: p.protocol,
            capture_identity: p.capture_identity,
        })
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
    pub fn open_feedback(
        &self,
        policy: &ferrum_types::SloStructuredFeedbackPolicy,
        fingerprint: &model::ExecutionFingerprint,
    ) -> Result<Option<super::super::selected_feedback::Monitor>, FerrumError> {
        use super::super::selected_feedback::{Binding, FeedbackKind, Monitor};
        let ferrum_types::SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 {
            policy,
            storage,
        } = policy
        else {
            return Ok(None);
        };
        if self.children.is_empty() {
            return Err(FerrumError::config(
                "structured feedback requires a qualified imported catalog",
            ));
        }
        let mut inventory = Sha256::new();
        inventory.update(b"ferrum.structured-owner-feedback-base.v1\0");
        inventory.update(
            serde_json::to_vec(&file::ProfileFingerprint::from(fingerprint))
                .map_err(profile_error)?,
        );
        let mut sources = Sha256::new();
        let mut parameters = Sha256::new();
        let mut protocols = Sha256::new();
        for (domain, child) in self.children.iter() {
            let (max_wave, max_age) = child.runtime_limits();
            if policy.maximum_family_margin_ns.get() > max_wave
                || policy.maximum_consumption_lag_ns.get() > max_age
            {
                return Err(FerrumError::config(
                    "structured feedback exceeds original child wave/age limits",
                ));
            }
            let p = child.provenance();
            inventory.update(domain);
            inventory.update(p.file_sha256);
            inventory.update(serde_json::to_vec(child.owner()).map_err(profile_error)?);
            inventory.update(serde_json::to_vec(child.scope()).map_err(profile_error)?);
            sources.update(domain);
            sources.update(p.source_sha256);
            parameters.update(domain);
            parameters.update(p.parameters_sha256);
            protocols.update(domain);
            protocols.update(p.protocol);
            protocols.update(p.capture_identity);
            // The provenance clock stores this process's load mapping. It is
            // not a durable source identity and legitimately changes on resume.
            // Original sample age still comes from each immutable child query.
        }
        let mut policy_hash = Sha256::new();
        policy_hash.update(b"ferrum.retrospective-owner-margin.v1\0");
        policy_hash.update(serde_json::to_vec(policy).map_err(profile_error)?);
        let binding = Binding {
            profile_sha256: inventory.finalize().into(),
            source_sha256: sources.finalize().into(),
            fit_sha256: parameters.finalize().into(),
            protocol_sha256: protocols.finalize().into(),
            policy_sha256: policy_hash.finalize().into(),
        };
        Monitor::open_bound(
            policy,
            storage,
            binding,
            self.len(),
            FeedbackKind::StructuredV2,
            Some(self.children.keys().copied().collect::<Vec<_>>().into()),
        )
        .map(Some)
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
    if snapshot
        .feedback
        .as_ref()
        .is_some_and(|view| !view.current())
    {
        return Err(Unknown::RuntimeValidity);
    }
    let child = snapshot.select(query)?;
    let (value, model_now) = child.predict_query_local_with_clock(fingerprint, query, local_now)?;
    let planning_ns = value
        .planning_ns
        .checked_add(
            snapshot
                .feedback
                .as_ref()
                .map_or(0, |view| view.margin(query.domain_signature())),
        )
        .filter(|v| *v <= child.runtime_limits().0)
        .ok_or(Unknown::Numerical)?;
    Ok(PlanningCost {
        typical_ns: value.fitted_upper_ns,
        planning_ns,
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
                children: Arc::new(BTreeMap::new()),
                feedback: None,
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
