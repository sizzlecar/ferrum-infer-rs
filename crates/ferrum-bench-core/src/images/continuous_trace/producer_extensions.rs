//! Typed producer metadata. A token observation never proves an image step,
//! a complete native wave, retirement, or a queue/step timing interval.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceSharedOwner {
    pub id: uuid::Uuid,
    pub participants: usize,
}
impl ImageTraceSharedOwner {
    fn validate(
        self,
        limits: &ImageContinuousTraceReadLimits,
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        if self.id.is_nil() || self.participants == 0 || self.participants > limits.maximum_members
        {
            return Err(invalid(
                line,
                "invalid expansion shared owner extent/identity",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceReferenceLatentDigest {
    pub shape: Vec<u64>,
    pub sha256: String,
}
impl ImageTraceReferenceLatentDigest {
    pub(super) fn validate(
        &self,
        editing: bool,
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        let single_image = matches!(self.shape.as_slice(), [1, _, _, _] | [1, _, 1, _, _]);
        let extent = self
            .shape
            .iter()
            .try_fold(1u64, |n, &d| if d == 0 { None } else { n.checked_mul(d) });
        if !editing
            || !single_image
            || extent.and_then(|n| n.checked_mul(4)).is_none()
            || self.sha256.len() != 64
            || !self.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(invalid(
                line,
                "invalid reference latent shape/digest metadata",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ImageTraceExpansionTokenWaveSummary {
    pub observed_records: u64,
    /// Groups having at least one observed row; this does not prove all rows
    /// were recorded or that the original native outcome completed.
    pub observed_wave_groups: u64,
    pub maximum_observed_participants: usize,
    pub completed_rows: u64,
    pub failed_rows: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RowOutcome {
    Completed,
    Failed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NativeOutcome {
    PreparationFailed,
    ExecutionError,
    Completed,
    Interrupted,
    NotSubmitted,
    ContractViolation,
}

fn required_option<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(d)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TokenWave {
    #[serde(deserialize_with = "required_option")]
    shared_owner: Option<ImageTraceSharedOwner>,
    wave: u64,
    participants: usize,
    input_index: usize,
    pub(super) job: u64,
    #[serde(deserialize_with = "required_option")]
    request_id: Option<uuid::Uuid>,
    before_consumed_tokens: u64,
    #[serde(deserialize_with = "required_option")]
    after_consumed_tokens: Option<u64>,
    outcome: RowOutcome,
    native_outcome: NativeOutcome,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OwnerKey {
    Shared(uuid::Uuid),
    Single(u64),
}
struct Owner {
    sealed_participants: usize,
    wave: u64,
    participants: usize,
    native: NativeOutcome,
    inputs: BTreeSet<usize>,
}
struct Request {
    owner: OwnerKey,
    id: Option<uuid::Uuid>,
    after: Option<u64>,
    wave: u64,
    failed: bool,
}
#[derive(Default)]
pub(super) struct TokenWaves {
    owners: BTreeMap<OwnerKey, Owner>,
    requests: BTreeMap<u64, Request>,
    summary: ImageTraceExpansionTokenWaveSummary,
}
impl TokenWaves {
    pub(super) fn event(
        &mut self,
        row: TokenWave,
        limits: &ImageContinuousTraceReadLimits,
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        if row.participants == 0
            || row.participants > limits.maximum_members
            || row.input_index >= row.participants
            || row.request_id.is_some_and(|id| id.is_nil())
            || row
                .after_consumed_tokens
                .is_some_and(|after| after < row.before_consumed_tokens)
            || (row.outcome == RowOutcome::Completed
                && (row
                    .after_consumed_tokens
                    .is_none_or(|after| after <= row.before_consumed_tokens)
                    || !matches!(
                        row.native_outcome,
                        NativeOutcome::Completed | NativeOutcome::Interrupted
                    )))
        {
            return Err(invalid(
                line,
                "invalid expansion token wave extent/frontier",
            ));
        }
        let (key, sealed) = if let Some(owner) = row.shared_owner {
            owner.validate(limits, line)?;
            (OwnerKey::Shared(owner.id), owner.participants)
        } else {
            if row.participants != 1 {
                return Err(invalid(
                    line,
                    "multi-request expansion wave lacks its shared owner",
                ));
            }
            (OwnerKey::Single(row.job), 1)
        };
        if row.participants > sealed {
            return Err(invalid(line, "expansion wave exceeds its sealed owner"));
        }
        if let Some(request) = self.requests.get(&row.job) {
            if request.owner != key
                || request.id != row.request_id
                || request.wave == row.wave
                || request.failed
                || request
                    .after
                    .is_some_and(|after| row.before_consumed_tokens < after)
            {
                return Err(invalid(
                    line,
                    "expansion wave changes request identity or regresses its frontier",
                ));
            }
        } else if self.requests.len() >= limits.maximum_requests {
            return Err(capacity(line));
        }
        let new_wave = if let Some(owner) = self.owners.get_mut(&key) {
            if owner.sealed_participants != sealed || row.wave < owner.wave {
                return Err(invalid(
                    line,
                    "expansion wave changes owner extent or regresses wave identity",
                ));
            }
            let new_wave = row.wave > owner.wave;
            if new_wave {
                owner.wave = row.wave;
                owner.participants = row.participants;
                owner.native = row.native_outcome;
                owner.inputs.clear();
            }
            if owner.participants != row.participants
                || owner.native != row.native_outcome
                || !owner.inputs.insert(row.input_index)
            {
                return Err(invalid(
                    line,
                    "expansion wave repeats an input or disagrees on its native call",
                ));
            }
            new_wave
        } else {
            if self.owners.len() >= limits.maximum_requests {
                return Err(capacity(line));
            }
            self.owners.insert(
                key,
                Owner {
                    sealed_participants: sealed,
                    wave: row.wave,
                    participants: row.participants,
                    native: row.native_outcome,
                    inputs: BTreeSet::from([row.input_index]),
                },
            );
            true
        };
        self.requests.insert(
            row.job,
            Request {
                owner: key,
                id: row.request_id,
                after: row.after_consumed_tokens,
                wave: row.wave,
                failed: row.outcome == RowOutcome::Failed,
            },
        );
        self.summary.observed_records = increment(self.summary.observed_records, line)?;
        if new_wave {
            self.summary.observed_wave_groups = increment(self.summary.observed_wave_groups, line)?;
        }
        self.summary.maximum_observed_participants = self
            .summary
            .maximum_observed_participants
            .max(row.participants);
        match row.outcome {
            RowOutcome::Completed => {
                self.summary.completed_rows = increment(self.summary.completed_rows, line)?
            }
            RowOutcome::Failed => {
                self.summary.failed_rows = increment(self.summary.failed_rows, line)?
            }
        }
        Ok(())
    }
    pub(super) fn observation(
        &self,
        observation: &ImageTracePromptExpansionObservation,
        limits: &ImageContinuousTraceReadLimits,
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        if let Some(owner) = observation.shared_owner {
            owner.validate(limits, line)?;
            if self
                .owners
                .get(&OwnerKey::Shared(owner.id))
                .is_some_and(|state| state.sealed_participants != owner.participants)
            {
                return Err(invalid(
                    line,
                    "expansion summary changes the sealed owner extent",
                ));
            }
        }
        if let Some(request) = self.requests.get(&observation.job) {
            let key = observation
                .shared_owner
                .map_or(OwnerKey::Single(observation.job), |owner| {
                    OwnerKey::Shared(owner.id)
                });
            if request.id != observation.request_id || request.owner != key {
                return Err(invalid(
                    line,
                    "expansion summary changes the observed token request/owner",
                ));
            }
            if let Some(owner) = observation.shared_owner {
                if self.owners[&key].sealed_participants != owner.participants {
                    return Err(invalid(
                        line,
                        "expansion summary changes the sealed owner extent",
                    ));
                }
            }
        }
        Ok(())
    }
    pub(super) fn finish(self) -> Option<ImageTraceExpansionTokenWaveSummary> {
        (self.summary.observed_records > 0).then_some(self.summary)
    }
}
fn increment(n: u64, line: u64) -> Result<u64, ImageContinuousTraceError> {
    n.checked_add(1).ok_or_else(|| capacity(line))
}
fn invalid(line: u64, reason: &str) -> ImageContinuousTraceError {
    error(line, ImageContinuousTraceErrorKind::Invalid, reason)
}
fn capacity(line: u64) -> ImageContinuousTraceError {
    error(
        line,
        ImageContinuousTraceErrorKind::Capacity,
        "expansion diagnostic metadata exceeds reader capacity",
    )
}
