//! Exact opaque identity joins. Client timing is deliberately not an input.
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageTraceBenchAttempt {
    pub endpoint_index: usize,
    pub sample_index: usize,
    pub repetition: usize,
    pub phase: String,
    #[serde(default)]
    pub client_attempt_id: Option<String>,
    #[serde(default)]
    pub server_request_id: Option<String>,
    #[serde(default)]
    pub trace_session_uuid: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTraceJoinStatus {
    Matched,
    MissingCorrelation,
    SessionMismatch,
    NotFound,
    Ambiguous,
    ConflictingCorrelation,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageTraceBenchJoin {
    pub attempt: ImageTraceBenchAttempt,
    pub status: ImageTraceJoinStatus,
    pub matched_request_id: Option<String>,
    pub matched_model_request: Option<String>,
    pub timing_status: ImageStepEvidenceStatus,
    pub queue_total_ns: Option<u64>,
    pub accepted_to_initialization_ns: Option<u64>,
}

/// Missing response headers are expected on cancellation or transport failure:
/// an attempt UUID can still join, but only if unique on both sides. Repeated
/// client identifiers never overwrite another sample or server request.
pub fn join_image_trace_attempts(
    analysis: &ImageContinuousTraceAnalysis,
    attempts: &[ImageTraceBenchAttempt],
    maximum_attempts: usize,
) -> Result<Vec<ImageTraceBenchJoin>, ImageContinuousTraceError> {
    if maximum_attempts == 0 || attempts.len() > maximum_attempts {
        return Err(error(
            0,
            ImageContinuousTraceErrorKind::Capacity,
            "benchmark join exceeds attempt capacity",
        ));
    }
    let mut server = BTreeMap::new();
    let mut clients: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    let mut repeated_attempts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut repeated_servers: BTreeMap<&str, usize> = BTreeMap::new();
    for (index, request) in analysis.timing.requests.iter().enumerate() {
        if server.insert(request.request_id.as_str(), index).is_some() {
            return Err(error(
                0,
                ImageContinuousTraceErrorKind::Invalid,
                "trace repeats server request identity",
            ));
        }
        if let Some(id) = &request.client_attempt_id {
            clients.entry(id).or_default().push(index);
        }
    }
    for attempt in attempts {
        for id in [
            &attempt.client_attempt_id,
            &attempt.server_request_id,
            &attempt.trace_session_uuid,
        ]
        .into_iter()
        .flatten()
        {
            if ferrum_types::parse_image_attempt_id(Some(id)).is_err() {
                return Err(error(
                    0,
                    ImageContinuousTraceErrorKind::Invalid,
                    "benchmark correlation is not a bounded UUID",
                ));
            }
        }
        if !["warmup", "measured"].contains(&attempt.phase.as_str()) {
            return Err(error(
                0,
                ImageContinuousTraceErrorKind::Invalid,
                "benchmark phase is outside the declared schema",
            ));
        }
        if let Some(id) = &attempt.client_attempt_id {
            *repeated_attempts.entry(id).or_default() += 1;
        }
        if let Some(id) = &attempt.server_request_id {
            *repeated_servers.entry(id).or_default() += 1;
        }
    }
    let mut result = Vec::with_capacity(attempts.len());
    for attempt in attempts {
        let (status, matched) = if attempt
            .trace_session_uuid
            .as_ref()
            .is_some_and(|id| Some(id) != analysis.session_uuid.as_ref())
        {
            (ImageTraceJoinStatus::SessionMismatch, None)
        } else if let Some(id) = &attempt.server_request_id {
            match server.get(id.as_str()).copied() {
                _ if repeated_servers
                    .get(id.as_str())
                    .is_some_and(|count| *count > 1) =>
                {
                    (ImageTraceJoinStatus::Ambiguous, None)
                }
                Some(index)
                    if attempt.client_attempt_id.as_ref().is_some_and(|id| {
                        Some(id) != analysis.timing.requests[index].client_attempt_id.as_ref()
                    }) =>
                {
                    (ImageTraceJoinStatus::ConflictingCorrelation, None)
                }
                Some(index) => (ImageTraceJoinStatus::Matched, Some(index)),
                None => (ImageTraceJoinStatus::NotFound, None),
            }
        } else if let Some(id) = &attempt.client_attempt_id {
            match clients.get(id.as_str()) {
                Some(indices)
                    if indices.len() == 1 && repeated_attempts.get(id.as_str()) == Some(&1) =>
                {
                    (ImageTraceJoinStatus::Matched, Some(indices[0]))
                }
                Some(_) => (ImageTraceJoinStatus::Ambiguous, None),
                None => (ImageTraceJoinStatus::NotFound, None),
            }
        } else {
            (ImageTraceJoinStatus::MissingCorrelation, None)
        };
        let request = matched.map(|index| &analysis.timing.requests[index]);
        result.push(ImageTraceBenchJoin {
            attempt: attempt.clone(),
            status,
            matched_request_id: request.map(|request| request.request_id.clone()),
            matched_model_request: request
                .filter(|request| request.model_request_initialized)
                .and_then(|request| request.request.clone()),
            timing_status: if request.is_some() {
                analysis.timing.status
            } else {
                ImageStepEvidenceStatus::NotEvaluated
            },
            queue_total_ns: request.and_then(|request| request.queue_total_ns),
            accepted_to_initialization_ns: request
                .and_then(|request| request.accepted_to_initialization_ns),
        });
    }
    Ok(result)
}
