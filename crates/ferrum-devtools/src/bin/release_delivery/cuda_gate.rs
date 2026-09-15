//! Bind local Metal/CUDA and optional cloud execution to committed coverage.
use ferrum_bench_core::release_regression::{
    model_schedule::model_task_schedule, Backend, CloudCudaMode, CudaModelLane, MetalModelLane,
    ModelProfile, Plan, PlanInput, ReleaseCudaPolicy, ReleaseMetalPolicy,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn verify_policy(plan: &Plan, catalog: &Value) -> Result<(), String> {
    plan.validate_release_policies()?;
    let policy = plan
        .release_cuda
        .as_ref()
        .ok_or("release plan omits the explicit local/cloud CUDA policy")?;
    let mut committed: ReleaseCudaPolicy = serde_json::from_value(catalog["release_cuda"].clone())
        .map_err(|_| "candidate catalog omits its typed CUDA policy")?;
    let profiles: Vec<ModelProfile> = serde_json::from_value(catalog["profiles"].clone())
        .map_err(|_| "candidate catalog omits its model profiles")?;
    committed.validate(&profiles)?;
    if committed.cloud != CloudCudaMode::Disabled {
        return Err("committed release CUDA policy must default to no cloud rental".into());
    }
    // Only the explicit per-run cloud opt-in can differ from committed policy.
    committed.cloud = policy.cloud;
    if &committed != policy {
        return Err("frozen CUDA lane membership differs from the committed catalog".into());
    }
    let metal_policy = plan
        .release_metal
        .as_ref()
        .ok_or("release plan omits the explicit local/extended Metal policy")?;
    let committed_metal: ReleaseMetalPolicy =
        serde_json::from_value(catalog["release_metal"].clone())
            .map_err(|_| "candidate catalog omits its typed Metal policy")?;
    committed_metal.validate(&profiles)?;
    if &committed_metal != metal_policy {
        return Err("frozen Metal lane membership differs from the committed catalog".into());
    }
    // A profile ID and target do not bind the immutable weights or independent
    // GGUF semantic/tokenizer sources. Prepared tasks must not drift together
    // with a forged selected profile while retaining the same lane membership.
    for selected in &plan.selected {
        if !profiles.iter().any(|profile| profile == &selected.profile) {
            return Err(format!(
                "selected profile {} differs from the committed catalog",
                selected.profile.id
            ));
        }
    }
    // Recompute the partition from committed model capabilities and the frozen
    // impact. A caller cannot move a local/safety obligation into `not_run`, or
    // silently remove an extended obligation from the disclosure.
    let mut input = catalog.clone();
    // README review hashes refine the frozen impact in the planner; they are
    // not a PlanInput field. Keep every other unknown catalog field fail-closed.
    input
        .as_object_mut()
        .ok_or("candidate catalog must be an object")?
        .remove("readme_reviews");
    input["stage"] = serde_json::to_value(plan.stage).map_err(|e| e.to_string())?;
    input["impact"] = serde_json::to_value(&plan.impact).map_err(|e| e.to_string())?;
    input["release_cuda"] = serde_json::to_value(policy).map_err(|e| e.to_string())?;
    input["checks"] = serde_json::json!([]);
    let input: PlanInput = serde_json::from_value(input)
        .map_err(|e| format!("candidate release policies cannot be replanned: {e}"))?;
    let recomputed = ferrum_bench_core::release_regression::plan(&input)?;
    if plan.extended_not_run != recomputed.extended_not_run {
        return Err("extended not-run disclosure differs from committed coverage".into());
    }
    let without_assignments =
        |obligations: &[ferrum_bench_core::release_regression::Obligation]| {
            obligations
                .iter()
                .cloned()
                .map(|mut obligation| {
                    obligation.checkers.clear();
                    obligation
                })
                .collect::<Vec<_>>()
        };
    if without_assignments(&plan.obligations) != without_assignments(&recomputed.obligations) {
        return Err("required coverage differs from the committed release policy partition".into());
    }
    let schedule = model_task_schedule(plan);
    for lane in [CudaModelLane::Local, CudaModelLane::Cloud] {
        let actual: BTreeSet<_> = schedule
            .runs
            .iter()
            .filter(|run| {
                run.profile.target.backend == Backend::Cuda && run.cuda_lane == Some(lane)
            })
            .map(|run| run.profile.id.as_str())
            .collect();
        let expected: BTreeSet<_> = match lane {
            CudaModelLane::Local => policy
                .mandatory_local_profile_ids
                .iter()
                .map(String::as_str)
                .collect(),
            CudaModelLane::Cloud if policy.cloud == CloudCudaMode::Required => policy
                .extended_cloud_profile_ids
                .iter()
                .map(String::as_str)
                .collect(),
            CudaModelLane::Cloud => BTreeSet::new(),
        };
        if actual != expected {
            return Err(format!(
                "CUDA {lane:?} tasks do not exactly cover the enabled lane"
            ));
        }
    }
    let actual_metal: BTreeSet<_> = schedule
        .runs
        .iter()
        .filter(|run| run.profile.target.backend == Backend::Metal)
        .map(|run| {
            if run.metal_lane != Some(MetalModelLane::Local) {
                return Err("Metal tasks include a profile outside the required local lane");
            }
            Ok(run.profile.id.as_str())
        })
        .collect::<Result<_, _>>()?;
    let expected_metal: BTreeSet<_> = metal_policy
        .mandatory_local_profile_ids
        .iter()
        .map(String::as_str)
        .collect();
    if actual_metal != expected_metal {
        return Err("Metal tasks do not exactly cover the required local lane".into());
    }
    Ok(())
}

pub(super) fn verify_jobs(
    policy: &ReleaseCudaPolicy,
    event: &Value,
    local: &BTreeMap<u64, Value>,
    cloud: &BTreeMap<u64, Value>,
) -> Result<(), String> {
    if policy.cloud == CloudCudaMode::Required && event != "workflow_dispatch" {
        return Err("cloud CUDA rental requires an explicit manual release".into());
    }
    let (_, local) = local
        .last_key_value()
        .ok_or("missing required local CUDA job")?;
    if local["status"] != "completed" || local["conclusion"] != "success" {
        return Err("latest required local CUDA job did not succeed".into());
    }
    let (_, cloud) = cloud
        .last_key_value()
        .ok_or("missing optional cloud CUDA job state")?;
    let expected = match policy.cloud {
        CloudCudaMode::Disabled => "skipped",
        CloudCudaMode::Required => "success",
    };
    if cloud["status"] != "completed" || cloud["conclusion"] != expected {
        return Err(format!(
            "cloud CUDA job must be {expected} under the frozen policy"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn policy(cloud: CloudCudaMode) -> ReleaseCudaPolicy {
        ReleaseCudaPolicy {
            cloud,
            mandatory_local_profile_ids: vec!["local-model".into()],
            extended_cloud_profile_ids: vec!["large-model".into()],
            reason: "Local representatives are required; large-model execution is optional".into(),
        }
    }

    fn jobs(conclusion: &str) -> BTreeMap<u64, Value> {
        BTreeMap::from([(1, json!({"status":"completed", "conclusion":conclusion}))])
    }

    #[test]
    fn actual_catalog_replans_both_modes_without_dropping_any_required_coverage() {
        let catalog: Value = serde_json::from_str(include_str!(
            "../../../../../docs/release-regression-catalog.json"
        ))
        .unwrap();
        let mut input = catalog.clone();
        input.as_object_mut().unwrap().remove("readme_reviews");
        input["stage"] = json!("release");
        input["impact"] =
            serde_json::to_value(ferrum_bench_core::release_regression::analyze_paths([
                "crates/ferrum-kernels/src/backend/cuda/mod.rs",
            ]))
            .unwrap();
        let mut input: PlanInput = serde_json::from_value(input).unwrap();
        for cloud in [CloudCudaMode::Disabled, CloudCudaMode::Required] {
            input.release_cuda.as_mut().unwrap().cloud = cloud;
            let plan = ferrum_bench_core::release_regression::plan(&input).unwrap();
            verify_policy(&plan, &catalog).unwrap();
            let pinned_metal = plan
                .selected
                .iter()
                .find(|selected| {
                    selected.profile.target.backend == Backend::Metal
                        && selected.profile.gguf.is_some()
                })
                .unwrap();
            for change_semantic_source in [false, true] {
                let mut drifted = plan.clone();
                let profile = &mut drifted
                    .selected
                    .iter_mut()
                    .find(|selected| selected.profile.id == pinned_metal.profile.id)
                    .unwrap()
                    .profile;
                let other_source = format!("fixture/other-source@{}", "f".repeat(40));
                if change_semantic_source {
                    profile.gguf.as_mut().unwrap().semantic_source = other_source;
                } else {
                    profile.model = other_source;
                }
                assert!(verify_policy(&drifted, &catalog)
                    .unwrap_err()
                    .contains("selected profile"));
            }
            let mut missing_metal = plan.clone();
            missing_metal.release_metal = None;
            assert!(verify_policy(&missing_metal, &catalog).is_err());
            let mut hidden_metal = plan.clone();
            hidden_metal.extended_not_run.retain(|obligation| {
                !matches!(
                    &obligation.scope,
                    ferrum_bench_core::release_regression::ObligationScope::Profile { target, .. }
                        | ferrum_bench_core::release_regression::ObligationScope::Target { target }
                        if target.backend == Backend::Metal
                )
            });
            assert_ne!(hidden_metal.extended_not_run, plan.extended_not_run);
            assert!(verify_policy(&hidden_metal, &catalog).is_err());
            let mut stripped = plan.clone();
            stripped.obligations.retain(|obligation| {
                obligation.layer
                    != ferrum_bench_core::release_regression::EvidenceLayer::BackendNumerics
            });
            assert!(verify_policy(&stripped, &catalog).is_err());
        }
    }

    #[test]
    fn cloud_skip_is_only_accepted_when_disabled_and_local_succeeded() {
        let manual = json!("workflow_dispatch");
        for mode in [CloudCudaMode::Disabled, CloudCudaMode::Required] {
            for local in ["success", "failure", "cancelled", "skipped", "timed_out"] {
                for cloud in ["success", "failure", "cancelled", "skipped", "timed_out"] {
                    let allowed = local == "success"
                        && cloud
                            == if mode == CloudCudaMode::Disabled {
                                "skipped"
                            } else {
                                "success"
                            };
                    assert_eq!(
                        verify_jobs(&policy(mode), &manual, &jobs(local), &jobs(cloud)).is_ok(),
                        allowed
                    );
                }
            }
        }
        assert!(verify_jobs(
            &policy(CloudCudaMode::Required),
            &json!("push"),
            &jobs("success"),
            &jobs("success")
        )
        .is_err());
        assert!(verify_jobs(
            &policy(CloudCudaMode::Disabled),
            &json!("push"),
            &jobs("success"),
            &jobs("skipped")
        )
        .is_ok());
    }

    #[test]
    fn absent_pending_and_newer_failed_cuda_producers_are_not_old_success() {
        let policy = policy(CloudCudaMode::Disabled);
        let manual = json!("workflow_dispatch");
        assert!(verify_jobs(&policy, &manual, &BTreeMap::new(), &jobs("skipped")).is_err());
        assert!(verify_jobs(&policy, &manual, &jobs("success"), &BTreeMap::new()).is_err());
        for latest in [
            json!({"status":"queued", "conclusion":null}),
            json!({"status":"completed", "conclusion":"failure"}),
        ] {
            let mut local = jobs("success");
            local.insert(2, latest);
            assert!(verify_jobs(&policy, &manual, &local, &jobs("skipped")).is_err());
        }
    }
}
