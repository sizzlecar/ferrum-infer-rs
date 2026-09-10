//! Correlate actual product output with the engine's persisted request journal.
//! This checks observation acquisition and request closure, not device-clock
//! availability, model state arithmetic or performance.
use super::{model_tasks, Backend, Entrypoint};
use ferrum_types::{FerrumProfileEvent, ProfileEntrypoint, ProfileStatus, ResourceAction};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedRequest {
    pub entrypoint: Entrypoint,
    pub request_id: String,
    pub content: String,
    pub finish_reason: String,
    pub usage: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservabilityEvidence {
    pub requests: Vec<ObservedRequest>,
    pub events: Vec<FerrumProfileEvent>,
}

pub fn verify(
    evidence: &ObservabilityEvidence,
    backend: Backend,
    entrypoints: &[Entrypoint],
    max_tokens: u32,
) -> Result<(), String> {
    let actual: BTreeSet<_> = evidence.requests.iter().map(|r| r.entrypoint).collect();
    if actual != entrypoints.iter().copied().collect()
        || actual.len() != evidence.requests.len()
        || actual.is_empty()
    {
        return Err("observability evidence has missing or duplicate entrypoints".into());
    }
    let mut event_ids = BTreeSet::new();
    for event in &evidence.events {
        event.validate()?;
        if !event_ids.insert(&event.event_id) {
            return Err("observation journal contains a duplicate event identity".into());
        }
    }
    let mut request_ids = BTreeSet::new();
    for request in &evidence.requests {
        if request.request_id.trim().is_empty() || !request_ids.insert(&request.request_id) {
            return Err("observed output has an empty or reused request identity".into());
        }
        if !model_tasks::probe_answer_matches(&request.content, "42")
            || !matches!(request.finish_reason.as_str(), "stop" | "eos")
        {
            return Err(
                "observability probe did not return the completed arithmetic answer".into(),
            );
        }
        let prompt = request.usage["prompt_tokens"].as_u64().filter(|n| *n > 0);
        let completion = request.usage["completion_tokens"]
            .as_u64()
            .filter(|n| *n > 0 && *n <= u64::from(max_tokens));
        let (Some(prompt), Some(completion)) = (prompt, completion) else {
            return Err("observability output lacks bounded, positive token usage".into());
        };
        if prompt.checked_add(completion) != request.usage["total_tokens"].as_u64() {
            return Err("observability output token usage is inconsistent".into());
        }
        let mut phases = Vec::new();
        for phase in [
            "engine_request_open",
            "engine_sequence_terminal_evidence",
            "engine_request_close",
        ] {
            let matching: Vec<_> = evidence
                .events
                .iter()
                .enumerate()
                .filter(|(_, e)| e.request_id == request.request_id && e.phase == phase)
                .collect();
            let [(position, event)] = matching.as_slice() else {
                return Err(format!(
                    "request {} lacks one {phase} event",
                    request.request_id
                ));
            };
            let expected_entrypoint = match request.entrypoint {
                Entrypoint::Run => ProfileEntrypoint::Run,
                Entrypoint::ServeSync | Entrypoint::ServeStream => ProfileEntrypoint::Serve,
            };
            if event.entrypoint != expected_entrypoint
                || event.correlation_id.as_deref() != Some(request.request_id.as_str())
                || event.backend != "actual"
                || model_tasks::run_backend(
                    event
                        .backend_detail
                        .as_ref()
                        .and_then(|detail| detail.get("backend_device"))
                        .unwrap_or(&Value::Null),
                ) != Some(backend)
                || event
                    .attributes
                    .get("backend_device")
                    .is_some_and(|device| model_tasks::run_backend(device) != Some(backend))
                || event.attributes.get("actual_model_smoke") != Some(&Value::Bool(true))
                || event.status == ProfileStatus::Failure
                || event.error.is_some()
                || event.model.as_deref().is_none_or(|s| s.trim().is_empty())
            {
                return Err(format!(
                    "{phase} has mismatched runtime identity or a reported failure"
                ));
            }
            phases.push((*position, *event));
        }
        if !(phases[0].0 < phases[1].0 && phases[1].0 < phases[2].0)
            || phases
                .iter()
                .any(|(_, event)| event.model != phases[0].1.model)
        {
            return Err(
                "observed lifecycle order or model identity changed inside a request".into(),
            );
        }
        let terminal = phases[1].1;
        for (event, action) in [
            (phases[0].1, ResourceAction::RequestOpen),
            (phases[2].1, ResourceAction::RequestClose),
        ] {
            if event.resource.as_ref().is_none_or(|resource| {
                resource.action != action
                    || resource.owner_kind != "request"
                    || resource.owner_id != request.request_id
                    || resource.resource_kind != "request_slot"
            }) {
                return Err(
                    "observation does not describe the actual request resource owner".into(),
                );
            }
        }
        if terminal
            .shape
            .get("prompt_token_count")
            .and_then(Value::as_u64)
            != Some(prompt)
            || terminal
                .shape
                .get("generated_token_count")
                .and_then(Value::as_u64)
                != Some(completion)
            || terminal
                .attributes
                .get("termination")
                .and_then(Value::as_str)
                != Some("completed")
            || !matches!(
                terminal
                    .attributes
                    .get("finish_reason")
                    .and_then(Value::as_str),
                Some("stop" | "eos")
            )
        {
            return Err(
                "engine terminal observation disagrees with actual output usage/completion".into(),
            );
        }
        let closed = phases[2].1;
        if closed
            .attributes
            .get("resource_owner_outstanding_count")
            .and_then(Value::as_u64)
            != Some(0)
            || closed.attributes.contains_key("resource_close_error")
        {
            return Err("completed request reports unclosed resource ownership".into());
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "model_observability_tests.rs"]
pub(super) mod tests;
