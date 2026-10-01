//! Conditional geometry for the original declared token trajectory. This is
//! neither prefix installation nor evidence that any forced token can execute.
use super::*;
use ferrum_interfaces::Tokenizer;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixSlotV5;
use std::num::NonZeroUsize;

pub(in crate::continuous_engine::inner::calibration) struct GeometryPrefixConstraint {
    /// Binds the original slot to the actual fresh request, before authority
    /// sorting changes physical row order.
    pub request_id: RequestId,
    pub slot: StructuredPrefixSlotV5,
    pub release_generated: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine::inner::calibration) struct GeometryPrefixCondition {
    /// Every hypothesis requires the entire declared cohort's real prefix
    /// preparation and release to succeed. This field is not a release receipt.
    pub release_generated: u32,
    /// Only the first ordinary wave can be a conditional member opportunity;
    /// later waves may terminate or encounter unresolved future host content.
    pub first_ordinary_wave: bool,
}

struct ValidatedPrefix {
    release: u32,
    pending: Vec<bool>,
    unique: Vec<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PrefixHostPoint {
    pub pending: bool,
    pub unique: u64,
}

impl ValidatedPrefix {
    fn validate(
        tokenizer: &dyn Tokenizer,
        maximum_output: u32,
        constraint: &GeometryPrefixConstraint,
        limits: &GeometryProjectionLimits,
        available_bytes: usize,
    ) -> GeometryResult<Self> {
        let slot = &constraint.slot;
        let release = constraint.release_generated;
        if release == 0
            || release >= maximum_output
            || slot.token_ids.len() != release as usize
            || slot.token_bytes.len() != release as usize
        {
            return Err(GeometryProjectionUnknown::InvalidPrefix);
        }
        // The existing validator allocates its raw scratch before returning.
        // Check that bound first, together with both declaration copies, our
        // sorted unique-token scratch and the retained trajectory. UTF-8
        // pending fragments contain at most three bytes each; four may
        // coexist with the raw and combined buffers during validation.
        let bound = tokenizer
            .bounded_token_bytes_bound()
            .ok_or(GeometryProjectionUnknown::InvalidPrefix)?
            .get();
        slot.token_ids
            .len()
            .checked_mul(3 * std::mem::size_of::<ferrum_types::TokenId>())
            .and_then(|n| {
                n.checked_add(
                    (slot.token_ids.len().checked_add(1)?)
                        .checked_mul(std::mem::size_of::<bool>() + std::mem::size_of::<u64>())?,
                )
            })
            .and_then(|n| n.checked_add(bound.checked_mul(2)?))
            .and_then(|n| n.checked_add(4 * 3))
            .filter(|n| *n <= available_bytes)
            .ok_or(GeometryProjectionUnknown::Capacity)?;
        poll_deadline(limits)?;
        let declaration = CalibrationPrefixTokensV1 {
            tokenizer_policy_sha256: slot.tokenizer_policy_sha256,
            token_ids: slot.token_ids.clone(),
            release_generated: release as usize,
        };
        let checked = declaration
            .validate_for_tokenizer(
                tokenizer,
                NonZeroUsize::new(maximum_output as usize)
                    .ok_or(GeometryProjectionUnknown::InvalidPrefix)?,
            )
            .map_err(|_| GeometryProjectionUnknown::InvalidPrefix)?;
        poll_deadline(limits)?;
        let mut scratch = vec![0u8; bound];
        let mut pending_bytes = Vec::new();
        let mut pending = Vec::with_capacity(slot.token_ids.len() + 1);
        let mut unique = Vec::with_capacity(slot.token_ids.len() + 1);
        let mut tokens = Vec::with_capacity(slot.token_ids.len());
        pending.push(false);
        unique.push(0);
        for (&token, expected) in slot.token_ids.iter().zip(&slot.token_bytes) {
            poll_deadline(limits)?;
            let written = tokenizer
                .token_bytes_bounded_into(token, &mut scratch)
                .map_err(|_| GeometryProjectionUnknown::InvalidPrefix)?
                .ok_or(GeometryProjectionUnknown::InvalidPrefix)?;
            let actual = scratch
                .get(..written)
                .ok_or(GeometryProjectionUnknown::InvalidPrefix)?;
            if actual != expected {
                return Err(GeometryProjectionUnknown::InvalidPrefix);
            }
            pending_bytes =
                crate::continuous_engine::advance_pending_utf8_fragment(&pending_bytes, actual)
                    .map_err(|_| GeometryProjectionUnknown::InvalidPrefix)?;
            if let Err(position) = tokens.binary_search(&token.get()) {
                tokens.insert(position, token.get());
            }
            pending.push(!pending_bytes.is_empty());
            unique.push(tokens.len() as u64);
        }
        if pending_bytes != checked.expected_pending_bytes() {
            return Err(GeometryProjectionUnknown::InvalidPrefix);
        }
        poll_deadline(limits)?;
        Ok(Self {
            release,
            pending,
            unique,
        })
    }
}

pub(super) struct PrefixRoots {
    /// Original route-view participant index; never physical row position.
    rows: Vec<Option<ValidatedPrefix>>,
    release: Option<u32>,
    retained_bytes: usize,
}

impl PrefixRoots {
    pub(super) fn bind(
        tokenizer: &dyn Tokenizer,
        ids: &[RequestId],
        roots: &[HostRoot],
        constraints: &[GeometryPrefixConstraint],
        limits: &GeometryProjectionLimits,
        available_bytes: usize,
    ) -> GeometryResult<Self> {
        if constraints.is_empty() {
            return Ok(Self {
                rows: Vec::new(),
                release: None,
                retained_bytes: 0,
            });
        }
        if constraints.len() != ids.len() {
            return Err(GeometryProjectionUnknown::InvalidPrefix);
        }
        let release = constraints[0].release_generated;
        let mut retained = ids
            .len()
            .checked_mul(std::mem::size_of::<Option<ValidatedPrefix>>())
            .filter(|n| *n <= available_bytes)
            .ok_or(GeometryProjectionUnknown::Capacity)?;
        let mut rows: Vec<Option<ValidatedPrefix>> = Vec::with_capacity(ids.len());
        rows.resize_with(ids.len(), || None);
        for constraint in constraints {
            poll_deadline(limits)?;
            let participant = ids
                .iter()
                .position(|id| *id == constraint.request_id)
                .ok_or(GeometryProjectionUnknown::InvalidPrefix)?;
            if rows[participant].is_some() || constraint.release_generated != release {
                return Err(GeometryProjectionUnknown::InvalidPrefix);
            }
            let root = roots
                .iter()
                .find(|r| r.participant == participant)
                .ok_or(GeometryProjectionUnknown::InvalidPrefix)?;
            if root.host.state.generated_tokens_before != 0 || root.host.state.pending_decoded_utf8
            {
                return Err(GeometryProjectionUnknown::InvalidPrefix);
            }
            let checked = ValidatedPrefix::validate(
                tokenizer,
                root.maximum_output,
                constraint,
                limits,
                available_bytes - retained,
            )?;
            retained = retained
                .checked_add(checked.pending.capacity())
                .and_then(|n| {
                    n.checked_add(
                        checked
                            .unique
                            .capacity()
                            .checked_mul(std::mem::size_of::<u64>())?,
                    )
                })
                .filter(|n| *n <= available_bytes)
                .ok_or(GeometryProjectionUnknown::Capacity)?;
            rows[participant] = Some(checked);
        }
        Ok(Self {
            rows,
            release: Some(release),
            retained_bytes: retained,
        })
    }

    pub(super) fn retained_payload_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(super) fn at(&self, participant: usize, generated: u32) -> Option<PrefixHostPoint> {
        let row = self.rows.get(participant)?.as_ref()?;
        if generated > row.release {
            return None;
        }
        Some(PrefixHostPoint {
            pending: row.pending[generated as usize],
            unique: row.unique[generated as usize],
        })
    }

    pub(super) fn repetition_anchor(&self, root: &HostRoot) -> (u64, u64) {
        self.rows
            .get(root.participant)
            .and_then(Option::as_ref)
            .map_or(
                (
                    root.future.repetition.map_or(0, |r| r.0),
                    root.host.state.generated_tokens_before,
                ),
                |row| (row.unique[row.release as usize], u64::from(row.release)),
            )
    }

    pub(super) fn condition(
        &self,
        roots: &[HostRoot],
        point: GeometryProjectionPoint,
    ) -> GeometryResult<Option<GeometryPrefixCondition>> {
        let Some(release_generated) = self.release else {
            return Ok(None);
        };
        let mut first = true;
        if point.rows == 0 || point.rows > roots.len() {
            return Err(GeometryProjectionUnknown::Unreachable);
        }
        for root in roots.iter().filter(|r| r.participant < point.rows) {
            let generated = point
                .sequence_tokens
                .checked_sub(root.prompt)
                .ok_or(GeometryProjectionUnknown::Unreachable)?;
            if generated < release_generated {
                return Err(GeometryProjectionUnknown::Unreachable);
            }
            first &= generated == release_generated;
        }
        Ok(Some(GeometryPrefixCondition {
            release_generated,
            first_ordinary_wave: first,
        }))
    }
}

#[cfg(test)]
mod tests;
