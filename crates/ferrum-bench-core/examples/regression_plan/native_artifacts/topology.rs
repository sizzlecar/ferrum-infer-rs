//! Resolve the finite, local release graph before narrowing native-source reach.
//! Unknown Actions expressions, remote callees and graph migrations retain the
//! normal kernel scope; this is not a general workflow interpreter.
use super::*;

fn require(value: &Value, expected: Value, context: &str) -> Result<(), String> {
    if value != &expected {
        return Err(format!("unreviewed artifact binding: {context}"));
    }
    Ok(())
}

fn condition(job: &mut Value, expected: &str) -> Result<(), String> {
    require(&job["if"], json!(expected), "producer condition")?;
    job.as_object_mut()
        .ok_or("producer is not a job")?
        .remove("if");
    Ok(())
}

fn caller(job: &Value, workflow: &str, selector: (&str, &str)) -> Result<(), String> {
    require(&job["uses"], json!(format!("./{workflow}")), "local callee")?;
    require(
        &job["with"][selector.0],
        json!(selector.1),
        "literal backend/platform",
    )?;
    require(
        &job["with"]["release_candidate_sha"],
        json!("${{ github.sha }}"),
        "candidate",
    )?;
    require(
        &job["with"]["publish_release"],
        json!(false),
        "staging safety lock",
    )?;
    require(
        &job["if"],
        json!("needs.prepare.outputs.active == 'true'"),
        "active release",
    )?;
    require(&job["needs"], json!("prepare"), "release preparation")?;
    if job.get("steps").is_some() || job.get("continue-on-error").is_some_and(|v| v != false) {
        return Err("invalid or failure-tolerant reusable caller".into());
    }
    Ok(())
}

fn windows_host(value: &Value, predicate: &str) -> bool {
    if value.as_str().is_some_and(|v| v.starts_with("windows-")) {
        return true;
    }
    let Some(expression) = value.as_str() else {
        return false;
    };
    let Some(choices) = expression
        .strip_prefix(&format!("${{{{ {predicate} && fromJSON('"))
        .and_then(|v| v.strip_suffix("') }}"))
    else {
        return false;
    };
    let Some((yes, no)) = choices.split_once("') || fromJSON('") else {
        return false;
    };
    [yes, no].into_iter().all(|choice| {
        serde_json::from_str::<Vec<String>>(choice).is_ok_and(|labels| {
            labels
                .iter()
                .any(|v| v == "Windows" || v.starts_with("windows-"))
        })
    })
}

pub(super) fn windows_consumer(path: &str, name: &str, job: &Value) -> bool {
    let predicate = if path == WINDOWS_WORKFLOW && name == "build" {
        "inputs.backend == 'cuda'"
    } else if path == CI && name == "windows" {
        "(github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)"
    } else {
        return false;
    };
    windows_host(&job["runs-on"], predicate)
}

fn consumes(job: &Value, dependency: &str, artifact: &str) -> Result<(), String> {
    if !job["needs"]
        .as_array()
        .is_some_and(|needs| needs.contains(&json!(dependency)))
    {
        return Err(format!("artifact consumer does not require {dependency}"));
    }
    let mut found = false;
    for step in mandatory_steps(job)? {
        if step["uses"] == "actions/download-artifact@v4"
            && step["with"]["artifact-ids"]
                .as_str()
                .is_some_and(|ids| ids.split(',').any(|id| id == artifact))
        {
            if step.get("if").is_some() || step.get("continue-on-error").is_some_and(|v| v != false)
            {
                return Err("required artifact download is optional".into());
            }
            found = true;
        }
    }
    if !found {
        return Err(format!("missing required artifact input {artifact}"));
    }
    Ok(())
}

pub(super) fn producer_jobs(
    cuda: &Value,
    windows: &Value,
    delivery: &Value,
) -> Result<(Value, Value), String> {
    caller(
        &delivery["jobs"]["stage-cuda-linux"],
        WORKFLOW,
        ("platform", "linux"),
    )?;
    caller(
        &delivery["jobs"]["stage-cuda"],
        WINDOWS_WORKFLOW,
        ("backend", "cuda"),
    )?;
    caller(
        &delivery["jobs"]["stage-cpu-windows"],
        WINDOWS_WORKFLOW,
        ("backend", "cpu"),
    )?;
    require(
        &cuda["on"]["workflow_call"]["outputs"]["cuda_asset_id"]["value"],
        json!("${{ jobs.linux-x86_64-cuda-sm89.outputs.asset_id }}"),
        "Linux artifact output",
    )?;
    require(
        &windows["on"]["workflow_call"]["outputs"]["asset_id"]["value"],
        json!("${{ jobs.build.outputs.asset_id }}"),
        "Windows artifact output",
    )?;
    let bridge = &cuda["jobs"][WINDOWS];
    require(
        &bridge["uses"],
        json!(format!("./{WINDOWS_WORKFLOW}")),
        "Windows bridge",
    )?;
    require(
        &bridge["with"]["backend"],
        json!("cuda"),
        "Windows bridge backend",
    )?;
    require(
        &bridge["with"]["release_candidate_sha"],
        json!("${{ inputs.release_candidate_sha }}"),
        "bridge candidate",
    )?;
    require(
        &bridge["if"],
        json!("inputs.platform != 'linux'"),
        "bridge condition",
    )?;
    if bridge.get("continue-on-error").is_some_and(|v| v != false) || bridge.get("steps").is_some()
    {
        return Err("optional or malformed Windows bridge".into());
    }

    consumes(
        &delivery["jobs"]["cuda-models"],
        "stage-cuda-linux",
        "${{ needs.stage-cuda-linux.outputs.cuda_asset_id }}",
    )?;
    for (producer, artifact) in [
        (
            "stage-cuda-linux",
            "${{ needs.stage-cuda-linux.outputs.cuda_asset_id }}",
        ),
        ("stage-cuda", "${{ needs.stage-cuda.outputs.asset_id }}"),
        (
            "stage-cpu-windows",
            "${{ needs.stage-cpu-windows.outputs.asset_id }}",
        ),
    ] {
        consumes(&delivery["jobs"]["publish"], producer, artifact)?;
    }

    let mut linux = cuda["jobs"][LINUX].clone();
    condition(&mut linux, "inputs.platform != 'windows'")?;
    mandatory_steps(&linux)?;
    let mut native = windows["jobs"]["build"].clone();
    if !windows_consumer(WINDOWS_WORKFLOW, "build", &native)
        || native["defaults"]["run"]["shell"] != "pwsh"
    {
        return Err("Windows producer does not target a known Windows host".into());
    }
    require(
        &windows["env"]["BACKEND"],
        json!("${{ inputs.backend }}"),
        "callee backend environment",
    )?;
    mandatory_steps(&native)?;
    // Only this literal predicate is known true from the CUDA caller. Other
    // conditions remain intact and cannot pass mandatory staging-step checks.
    for step in native["steps"].as_array_mut().unwrap() {
        if step["if"] == "inputs.backend == 'cuda'" {
            step.as_object_mut().unwrap().remove("if");
        }
    }
    Ok((linux, native))
}
