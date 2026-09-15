//! Bind optional cloud execution to the committed policy and actual CI producers.
use ferrum_bench_core::release_regression::{
    model_schedule::model_task_schedule, Backend, CloudCudaMode, CudaModelLane, ModelProfile, Plan,
    PlanInput, ReleaseCudaPolicy,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn verify_policy(plan: &Plan, catalog: &Value) -> Result<(), String> {
    plan.validate_cuda_policy()?;
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
        .map_err(|e| format!("candidate CUDA policy cannot be replanned: {e}"))?;
    let recomputed = ferrum_bench_core::release_regression::plan(&input)?;
    if plan.extended_not_run != recomputed.extended_not_run {
        return Err("extended CUDA not-run disclosure differs from committed coverage".into());
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
        return Err("required coverage differs from the committed CUDA policy partition".into());
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
