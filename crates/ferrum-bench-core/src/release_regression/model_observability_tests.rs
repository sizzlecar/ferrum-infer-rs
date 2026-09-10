use super::*;
use ferrum_types::{ProfileEventKind, ResourceTraceEvent, OBSERVABILITY_PROFILE_SCHEMA_VERSION};
use serde_json::json;
use std::collections::BTreeMap;

fn fixture(entrypoint: Entrypoint) -> ObservabilityEvidence {
    let request_id = "observed-request";
    let events = [
        "engine_request_open",
        "engine_sequence_terminal_evidence",
        "engine_request_close",
    ]
    .into_iter()
    .map(|phase| {
        let terminal = phase == "engine_sequence_terminal_evidence";
        let mut attributes = BTreeMap::from([
            ("backend_device".into(), json!("CPU")),
            ("actual_model_smoke".into(), json!(true)),
        ]);
        let shape = if terminal {
            attributes.insert("termination".into(), json!("completed"));
            attributes.insert("finish_reason".into(), json!("eos"));
            BTreeMap::from([
                ("prompt_token_count".into(), json!(9)),
                ("generated_token_count".into(), json!(3)),
            ])
        } else {
            attributes.insert("resource_owner_outstanding_count".into(), json!(0));
            BTreeMap::from([("resource_amount".into(), Value::Null)])
        };
        FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: 1,
            event_id: phase.into(),
            request_id: request_id.into(),
            correlation_id: Some(request_id.into()),
            entrypoint: if entrypoint == Entrypoint::Run {
                ProfileEntrypoint::Run
            } else {
                ProfileEntrypoint::Serve
            },
            backend: "actual".into(),
            runtime_preset_hash: "runtime-preset".into(),
            phase: phase.into(),
            event_kind: if terminal {
                ProfileEventKind::Instant
            } else {
                ProfileEventKind::Resource
            },
            timestamp: "2026-01-01T00:00:00Z".parse().unwrap(),
            status: if terminal {
                ProfileStatus::DiagnosticOnly
            } else {
                ProfileStatus::Ok
            },
            model: Some("fixture/model".into()),
            duration_us: None,
            memory: None,
            resource: (!terminal).then(|| ResourceTraceEvent {
                owner_kind: "request".into(),
                owner_id: request_id.into(),
                resource_kind: "request_slot".into(),
                action: if phase == "engine_request_open" {
                    ResourceAction::RequestOpen
                } else {
                    ResourceAction::RequestClose
                },
                amount: None,
                before: None,
                after: None,
                capacity: None,
                underflow_amount: None,
                reason: None,
                error_kind: None,
                message: None,
                resource_error_kind: None,
            }),
            error: None,
            replay: None,
            shape,
            backend_detail: Some(BTreeMap::from([("backend_device".into(), json!("CPU"))])),
            attributes,
        }
    })
    .collect();
    ObservabilityEvidence {
        requests: vec![ObservedRequest {
            entrypoint,
            request_id: request_id.into(),
            content: "42".into(),
            finish_reason: "stop".into(),
            usage: json!({"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}),
        }],
        events,
    }
}

pub(crate) fn report_fixture(
    backend: Backend,
    entrypoints: &[Entrypoint],
) -> ObservabilityEvidence {
    let mut combined = ObservabilityEvidence {
        requests: Vec::new(),
        events: Vec::new(),
    };
    for entrypoint in entrypoints {
        let mut next = fixture(*entrypoint);
        let id = format!("request-{entrypoint:?}");
        next.requests[0].request_id = id.clone();
        for event in &mut next.events {
            event.event_id = format!("{id}-{}", event.event_id);
            event.request_id = id.clone();
            event.correlation_id = Some(id.clone());
            if let Some(resource) = &mut event.resource {
                resource.owner_id = id.clone();
            }
            event.attributes.insert(
                "backend_device".into(),
                json!(match backend {
                    Backend::Cpu => "CPU",
                    Backend::Metal => "Metal",
                    Backend::Cuda => "CUDA(0)",
                }),
            );
            event.backend_detail.as_mut().unwrap().insert(
                "backend_device".into(),
                event.attributes["backend_device"].clone(),
            );
        }
        combined.requests.extend(next.requests);
        combined.events.extend(next.events);
    }
    combined
}

#[test]
fn runtime_observations_require_correlated_output_usage_and_closed_ownership() {
    let good = fixture(Entrypoint::Run);
    verify(&good, Backend::Cpu, &[Entrypoint::Run], 32).unwrap();
    let mutations: &[fn(&mut ObservabilityEvidence)] = &[
        |e| {
            e.events.pop();
        },
        |e| {
            e.events[1].request_id = "another-request".into();
        },
        |e| {
            e.events[1].correlation_id = Some("another-request".into());
        },
        |e| {
            e.events[1].backend_detail = None;
        },
        |e| {
            e.events[1]
                .attributes
                .insert("backend_device".into(), json!("Metal"));
        },
        |e| {
            e.events[1]
                .shape
                .insert("generated_token_count".into(), json!(4));
        },
        |e| {
            e.events[2]
                .attributes
                .insert("resource_owner_outstanding_count".into(), json!(1));
        },
        |e| {
            e.events[2].resource.as_mut().unwrap().owner_id = "another-request".into();
        },
        |e| {
            e.events[1].status = ProfileStatus::Failure;
        },
        |e| {
            e.events.swap(0, 2);
        },
        |e| {
            e.requests[0].usage["total_tokens"] = json!(13);
        },
        |e| {
            e.requests[0].content = "wrong".into();
        },
        |e| {
            e.events.push(e.events[0].clone());
        },
    ];
    for mutate in mutations {
        let mut bad = good.clone();
        mutate(&mut bad);
        assert!(verify(&bad, Backend::Cpu, &[Entrypoint::Run], 32).is_err());
    }
}

#[test]
fn sync_output_cannot_cover_streaming_observation_or_reuse_its_request_identity() {
    let mut evidence = fixture(Entrypoint::ServeSync);
    let required = [Entrypoint::ServeSync, Entrypoint::ServeStream];
    assert!(verify(&evidence, Backend::Cpu, &required, 32).is_err());
    let mut stream = fixture(Entrypoint::ServeStream);
    stream.requests[0].request_id = "stream-request".into();
    for event in &mut stream.events {
        event.request_id = "stream-request".into();
        event.correlation_id = Some("stream-request".into());
        event.event_id = format!("stream-{}", event.event_id);
        if let Some(resource) = &mut event.resource {
            resource.owner_id = "stream-request".into();
        }
    }
    evidence.requests.extend(stream.requests);
    evidence.events.extend(stream.events);
    verify(&evidence, Backend::Cpu, &required, 32).unwrap();
    evidence.requests[1].request_id = evidence.requests[0].request_id.clone();
    assert!(verify(&evidence, Backend::Cpu, &required, 32).is_err());
}
