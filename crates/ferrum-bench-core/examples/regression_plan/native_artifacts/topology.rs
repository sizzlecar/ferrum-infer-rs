//! Resolve the finite, local release graph before narrowing native-source reach.
//! Unknown Actions expressions, remote callees and graph migrations retain the
//! normal kernel scope; this is not a general workflow interpreter.
use super::*;

#[path = "publication.rs"]
mod publication;

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

fn caller(
    job: &Value,
    workflow: &str,
    selector: (&str, &str),
    dependencies: Value,
) -> Result<(), String> {
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
    require(&job["needs"], dependencies, "release preparation and gates")?;
    if job.get("steps").is_some() || job.get("continue-on-error").is_some_and(|v| v != false) {
        return Err("invalid or failure-tolerant reusable caller".into());
    }
    Ok(())
}

fn prerequisites<'a>(job: &'a Value, name: &str) -> Result<Vec<&'a str>, String> {
    match job.get("needs") {
        None => Ok(Vec::new()),
        Some(Value::String(dependency)) => Ok(vec![dependency]),
        Some(Value::Array(dependencies)) => dependencies
            .iter()
            .map(|dependency| {
                dependency
                    .as_str()
                    .ok_or_else(|| format!("nonliteral dependency for {name}"))
            })
            .collect(),
        _ => Err(format!("unresolved dependencies for {name}")),
    }
}

fn model_dependencies(
    jobs: &Value,
    name: &str,
    visiting: &mut BTreeSet<String>,
    checked: &mut BTreeSet<String>,
) -> Result<(), String> {
    if matches!(name, "stage-cuda" | "stage-cpu-windows") {
        return Err("model gate depends on Windows staging".into());
    }
    if checked.contains(name) {
        return Ok(());
    }
    if !visiting.insert(name.to_string()) {
        return Err(format!("cycle in model prerequisites at {name}"));
    }
    let job = jobs
        .get(name)
        .filter(|job| job.is_object())
        .ok_or_else(|| format!("unresolved model prerequisite {name}"))?;
    if job
        .get("continue-on-error")
        .is_some_and(|value| value != false)
    {
        return Err(format!("failure-tolerant model prerequisite {name}"));
    }
    for dependency in prerequisites(job, name)? {
        model_dependencies(jobs, dependency, visiting, checked)?;
    }
    visiting.remove(name);
    checked.insert(name.to_string());
    Ok(())
}

fn windows_cuda_dependencies(delivery: &Value) -> Result<Value, String> {
    // Retain the previously reviewed topology as well as the model-first graph.
    // Comparing immutable workflow snapshots still rejects a graph migration.
    if delivery["jobs"]["stage-cuda"]["needs"] == "prepare" {
        return Ok(json!("prepare"));
    }
    model_first_windows_dependencies(delivery)?;
    Ok(delivery["jobs"]["stage-cuda"]["needs"].clone())
}

pub(super) fn model_first_windows_dependencies(delivery: &Value) -> Result<(), String> {
    let jobs = &delivery["jobs"];
    let expected = json!(["prepare", "metal-models", "cuda-models", "cpu-models"]);
    require(
        &jobs["stage-cuda"]["needs"],
        expected,
        "model-first Windows staging dependencies",
    )?;
    let mut checked = BTreeSet::new();
    for (name, execution) in [
        (
            "metal-models",
            "Execute selected Metal models from the staged archive",
        ),
        (
            "cuda-models",
            "Execute required local CUDA models from the staged archive",
        ),
        (
            "cpu-models",
            "Check CPU load and basic run/serve compatibility",
        ),
    ] {
        // Artifact upload steps may run always(), but the model execution and
        // job itself must remain mandatory and propagate failures to staging.
        step(&jobs[name], execution)?;
        model_dependencies(jobs, name, &mut BTreeSet::new(), &mut checked)?;
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
        json!("prepare"),
    )?;
    caller(
        &delivery["jobs"]["stage-cuda"],
        WINDOWS_WORKFLOW,
        ("backend", "cuda"),
        windows_cuda_dependencies(delivery)?,
    )?;
    caller(
        &delivery["jobs"]["stage-cpu-windows"],
        WINDOWS_WORKFLOW,
        ("backend", "cpu"),
        json!("prepare"),
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
    let mut publish = delivery["jobs"]["publish"].clone();
    if publish.get("if").is_some() {
        publication::verify(&publish)?;
        publish
            .as_object_mut()
            .ok_or("publisher is not a job")?
            .remove("if");
    }
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
        consumes(&publish, producer, artifact)?;
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
