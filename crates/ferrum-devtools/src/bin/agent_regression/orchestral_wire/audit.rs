//! Read-only replay auditing of an already stopped native run. This does not
//! rerun task validation or change the original process/result/acceptance report.

use super::*;
use std::fs::{self, OpenOptions};

pub(crate) fn audit_saved(
    report: &Path,
    task: &str,
    output: &Path,
    format_override: Option<OrchestralToolResultFormat>,
) -> Result<i32> {
    ensure!(
        !task.is_empty() && !task.contains(['/', '\\']) && !matches!(task, "." | ".."),
        "task identity must be one path component"
    );
    let manifest_bytes = fs::read(report.join("manifest.json"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest["schema_version"] == 2 && manifest["agent"]["kind"] == "orchestral",
        "saved report does not declare an Orchestral run"
    );
    let declared: OrchestralToolResultFormat = manifest["agent"]
        .get("tool_result_format")
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()?
        .unwrap_or_default();
    let format = format_override.unwrap_or(declared);
    ensure!(
        format == declared || (format.is_text_parts() && declared.is_text_parts()),
        "saved audit override must remain within the declared tool result format family"
    );
    let task_dir = report.join(task);
    let result_path = task_dir.join("result.json");
    let result_bytes = fs::read(&result_path)?;
    let result: Value = serde_json::from_slice(&result_bytes)?;
    ensure!(
        result["id"] == task,
        "saved result has another task identity"
    );
    let session = result
        .pointer("/orchestral/session_id")
        .and_then(Value::as_str)
        .context("saved result omitted the native session identity")?;
    let public = orchestral_evidence::read(&task_dir.join("journals"), session);
    let requests = report.join("requests");
    let mut records = Vec::new();
    let mut request_hashes = BTreeMap::new();
    for entry in fs::read_dir(&requests)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(index) = name
            .strip_prefix(&format!("{task}-"))
            .and_then(|name| name.strip_suffix(".json"))
            .and_then(|number| number.parse::<u32>().ok())
        else {
            continue;
        };
        let mut record: RequestRecord = serde_json::from_slice(&fs::read(entry.path())?)?;
        ensure!(
            record.task_id == task && record.request_index == index,
            "saved request identity differs from its sidecar"
        );
        let body_bytes = fs::read(requests.join(format!("{task}-{index}.request.json")))?;
        let body: Value = serde_json::from_slice(&body_bytes)?;
        request_hashes.insert(index, sha(&body_bytes));
        record.messages = body["messages"]
            .as_array()
            .context("saved request omitted messages")?
            .clone();
        records.push(record);
    }
    let wire = bind(
        &public,
        &records.iter().collect::<Vec<_>>(),
        &requests,
        format,
    );
    let complete = public.complete() && wire.complete();
    let evidence = json!({
        "schema_version": 1,
        "scope": "Recomputed public journal and uncompacted exact HTTP replay; task validation and timing are not rerun or reclassified",
        "source_report": report, "source_task_result": result_path, "task_id": task,
        "manifest_tool_result_format": declared, "selected_tool_result_format": format,
        "tool_result_format_override": format_override,
        "source_manifest_sha256": sha(&manifest_bytes), "source_task_result_sha256": sha(&result_bytes),
        "request_body_sha256": request_hashes,
        "closed_loop_evidence": complete, "public": public, "wire": wire,
    });
    let destination = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .with_context(|| format!("create new audit {}", output.display()))?;
    serde_json::to_writer_pretty(destination, &evidence)?;
    Ok(if complete { 0 } else { 1 })
}
