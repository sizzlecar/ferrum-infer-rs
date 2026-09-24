//! Bounded, deterministic feedback policy. No fit/support/clock mutation.
use ferrum_types::SloSelectedFeedbackSettingsV1 as Settings;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum Revocation {
    CorrectionLimit,
    ObservationLag,
    UncomparableObservation,
    FailedOrPartial,
    QueueLoss,
    IdentityOrClock,
    Capacity,
    Arithmetic,
    Persistence,
    WorkerStopped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Binding {
    pub profile_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub fit_sha256: [u8; 32],
    pub protocol_sha256: [u8; 32],
    pub policy_sha256: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WindowItem {
    base_excess_ns: u64,
    qualifying: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Family {
    pub signature: [u8; 32],
    pub margin_ns: u64,
    consecutive: usize,
    window: VecDeque<WindowItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct State {
    pub schema_version: u32,
    pub binding: Binding,
    pub epoch: u64,
    pub session: u64,
    pub revoked: Option<Revocation>,
    pub families: Vec<Family>,
    pub compared: u64,
    pub uncomparable_observations: u64,
    pub failed_or_partial: u64,
    pub queue_drops: u64,
    pub corrections: u64,
}

pub(super) struct Comparison {
    pub family: [u8; 32],
    pub base_planning_ns: u64,
    pub actual_ns: u64,
    pub observed_at_ns: u64,
    pub consumed_at_ns: u64,
}

impl State {
    pub fn new(binding: Binding) -> Self {
        Self {
            schema_version: 1,
            binding,
            epoch: 1,
            session: 0,
            revoked: None,
            families: Vec::new(),
            compared: 0,
            uncomparable_observations: 0,
            failed_or_partial: 0,
            queue_drops: 0,
            corrections: 0,
        }
    }

    pub fn validate(&self, binding: &Binding, policy: &Settings, capacity: usize) -> bool {
        self.schema_version == 1
            && &self.binding == binding
            && self.epoch > 0
            && self.families.len() <= capacity
            && self.families.iter().enumerate().all(|(i, family)| {
                family.margin_ns <= policy.maximum_family_margin_ns.get()
                    && family.window.len() <= policy.window_samples.get()
                    && family.consecutive <= policy.window_samples.get()
                    && self.families[..i]
                        .iter()
                        .all(|prior| prior.signature != family.signature)
            })
    }

    pub fn revoke(&mut self, reason: Revocation) {
        if self.revoked.is_none() {
            self.revoked = Some(reason);
        }
    }

    pub fn compare(&mut self, p: &Settings, capacity: usize, value: Comparison) {
        if self.revoked.is_some() {
            return;
        }
        let Some(lag) = value.consumed_at_ns.checked_sub(value.observed_at_ns) else {
            self.revoke(Revocation::IdentityOrClock);
            return;
        };
        if lag > p.maximum_consumption_lag_ns.get() {
            self.revoke(Revocation::ObservationLag);
            return;
        }
        if value.base_planning_ns == 0 || value.actual_ns == 0 {
            self.revoke(Revocation::IdentityOrClock);
            return;
        }
        let Some(compared) = self.compared.checked_add(1) else {
            self.revoke(Revocation::Arithmetic);
            return;
        };
        self.compared = compared;
        let index = match self
            .families
            .iter()
            .position(|f| f.signature == value.family)
        {
            Some(index) => index,
            None if self.families.len() < capacity => {
                self.families.push(Family {
                    signature: value.family,
                    margin_ns: 0,
                    consecutive: 0,
                    window: VecDeque::new(),
                });
                self.families.len() - 1
            }
            None => {
                self.revoke(Revocation::Capacity);
                return;
            }
        };
        let family = &mut self.families[index];
        let Some(bound) = value.base_planning_ns.checked_add(family.margin_ns) else {
            self.revoke(Revocation::Arithmetic);
            return;
        };
        let item = WindowItem {
            base_excess_ns: value.actual_ns.saturating_sub(value.base_planning_ns),
            qualifying: value.actual_ns.saturating_sub(bound) >= p.trigger_excess_ns.get(),
        };
        family.consecutive = if item.qualifying {
            family
                .consecutive
                .saturating_add(1)
                .min(p.window_samples.get())
        } else {
            0
        };
        if family.window.len() == p.window_samples.get() {
            family.window.pop_front();
        }
        family.window.push_back(item);
        let trigger = family.window.len() == p.window_samples.get()
            && family.consecutive >= p.minimum_consecutive_underestimates.get()
            && family.window.iter().filter(|item| item.qualifying).count()
                >= p.minimum_underestimates.get();
        if !trigger {
            return;
        }
        // Relative to the ORIGINAL frozen bound, never current-bound + error.
        let maximum = family
            .window
            .iter()
            .map(|item| item.base_excess_ns)
            .max()
            .unwrap_or(0);
        let Some(required) = maximum.checked_add(p.correction_padding_ns) else {
            self.revoke(Revocation::Arithmetic);
            return;
        };
        if required > p.maximum_family_margin_ns.get() {
            self.revoke(Revocation::CorrectionLimit);
            return;
        }
        family.margin_ns = family.margin_ns.max(required);
        family.window.clear();
        family.consecutive = 0;
        match self.corrections.checked_add(1) {
            Some(value) => self.corrections = value,
            None => self.revoke(Revocation::Arithmetic),
        }
    }

    pub fn uncomparable(&mut self, p: &Settings) {
        match self.uncomparable_observations.checked_add(1) {
            Some(value) => {
                self.uncomparable_observations = value;
                if value > p.maximum_uncomparable_observations {
                    self.revoke(Revocation::UncomparableObservation);
                }
            }
            None => self.revoke(Revocation::Arithmetic),
        }
    }
    pub fn failed(&mut self, p: &Settings) {
        match self.failed_or_partial.checked_add(1) {
            Some(value) => {
                self.failed_or_partial = value;
                if value > p.maximum_failed_or_partial {
                    self.revoke(Revocation::FailedOrPartial);
                }
            }
            None => self.revoke(Revocation::Arithmetic),
        }
    }
    pub fn dropped(&mut self, p: &Settings, delta: u64) {
        match self.queue_drops.checked_add(delta) {
            Some(value) => {
                self.queue_drops = value;
                if value > p.maximum_queue_drops {
                    self.revoke(Revocation::QueueLoss);
                }
            }
            None => self.revoke(Revocation::Arithmetic),
        }
    }
    #[cfg(test)]
    pub fn same_prediction(&self, other: &Self) -> bool {
        self.revoked == other.revoked
            && self
                .families
                .iter()
                .all(|family| other.margin(&family.signature) == family.margin_ns)
            && other
                .families
                .iter()
                .all(|family| self.margin(&family.signature) == family.margin_ns)
    }
    #[cfg(test)]
    pub fn margin(&self, family: &[u8; 32]) -> u64 {
        self.families
            .iter()
            .find(|f| &f.signature == family)
            .map_or(0, |f| f.margin_ns)
    }
}
