//! Optional geometry memo for one concrete, immutable comparison. It owns only
//! endpoint/member indexes and generation-checked cache handles, never rank
//! clocks, risks, cached errors or independently retained prediction payloads.

use super::*;
#[cfg(test)]
use cache::accounting::TraceClearReason;
use cache::CacheHandle;

#[cfg(test)]
mod tests;

pub(super) type ReplayMembers<'a> =
    dyn Fn(&[GroupMember], &mut PlanBuilder) -> Result<(), Unsupported> + 'a;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::joint_policy) enum TracePhase {
    First,
    Strict,
    Deferred,
    Continuation,
}

impl TracePhase {
    pub(in crate::joint_policy) fn of(round: usize, phase: Option<&ContinuationPhase>) -> Self {
        if round == 0 {
            Self::First
        } else {
            match phase {
                Some(ContinuationPhase::PrefillTargets(_)) => Self::Strict,
                Some(ContinuationPhase::DeferredPrefillTargets(_)) => Self::Deferred,
                None => Self::Continuation,
            }
        }
    }
}

struct WaveTrace {
    phase: TracePhase,
    handles: [Option<CacheHandle>; 2],
    lengths: [usize; 2],
    requests: Box<[usize]>,
}

#[cfg(test)]
pub(super) fn wave_trace_size() -> usize {
    std::mem::size_of::<WaveTrace>()
}

#[derive(Default)]
struct ActionTrace {
    waves: Vec<WaveTrace>,
}

#[derive(Default)]
pub(super) struct TraceStore {
    endpoints: Vec<usize>,
    actions: Vec<ActionTrace>,
    charged_bytes: usize,
}

impl TraceStore {
    fn initial_bytes(&self) -> Option<usize> {
        self.endpoints
            .capacity()
            .checked_mul(std::mem::size_of::<usize>())?
            .checked_add(
                self.actions
                    .capacity()
                    .checked_mul(std::mem::size_of::<ActionTrace>())?,
            )
    }

    fn clear(&mut self, cache: &mut ScenarioCache) {
        // Credit only after releasing capacities and every boxed index list.
        self.actions = Vec::new();
        self.endpoints = Vec::new();
        self.charged_bytes = 0;
        let _ = cache.set_trace_bytes(0);
    }

    pub(super) fn prepare(
        &mut self,
        requests: &[SimulatedRequest],
        actions: usize,
        cache: &parking_lot::Mutex<ScenarioCache>,
        diagnostics: &diagnostics::Recorder,
    ) -> bool {
        if self.actions.len() == actions
            && self.endpoints.len() == requests.len()
            && self
                .endpoints
                .iter()
                .zip(requests)
                .all(|(end, request)| *end == request.scenario_end_tokens)
        {
            return true;
        }
        let mut cache = cache.lock();
        #[cfg(test)]
        cache.note_trace_clear(TraceClearReason::EndpointOrActionChange);
        self.clear(&mut cache);
        let bytes = requests
            .len()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|bytes| {
                actions
                    .checked_mul(std::mem::size_of::<ActionTrace>())
                    .and_then(|actions| bytes.checked_add(actions))
            });
        if !bytes.is_some_and(|bytes| cache.set_trace_bytes(bytes)) {
            return false;
        }
        if self.endpoints.try_reserve_exact(requests.len()).is_err()
            || self.actions.try_reserve_exact(actions).is_err()
            || !self
                .initial_bytes()
                .is_some_and(|bytes| cache.set_trace_bytes(bytes))
        {
            if let Some(bytes) = self.initial_bytes() {
                if cache.set_trace_bytes(bytes) {
                    diagnostics.cache_bytes(cache.retained_bytes());
                }
            }
            #[cfg(test)]
            cache.note_trace_clear(TraceClearReason::PreparationFailure);
            self.clear(&mut cache);
            return false;
        }
        self.endpoints
            .extend(requests.iter().map(|request| request.scenario_end_tokens));
        self.actions.resize_with(actions, ActionTrace::default);
        self.charged_bytes = self.initial_bytes().expect("validated initial capacity");
        diagnostics.cache_bytes(cache.retained_bytes());
        true
    }

    fn wave(&self, candidate: usize, round: usize, phase: TracePhase) -> Option<&WaveTrace> {
        self.actions
            .get(candidate)?
            .waves
            .get(round)
            .filter(|wave| wave.phase == phase)
    }

    fn publish(
        &mut self,
        candidate: usize,
        round: usize,
        phase: TracePhase,
        groups: &[Vec<GroupMember>],
        handles: [Option<CacheHandle>; 2],
        snapshot: &ConditionalSnapshot<'_>,
        now: Instant,
        replayed: bool,
    ) {
        if groups.is_empty() || groups.len() > 2 || candidate >= self.actions.len() {
            return;
        }
        if replayed
            && self
                .wave(candidate, round, phase)
                .is_some_and(|wave| wave.handles == handles)
        {
            // The same canonical geometry was copied before the wave and every
            // group returned its same owner/generation handle. Apply and phase
            // progress already succeeded. No work equality scan or lock here.
            return;
        }
        let mut cache = snapshot.cache.lock();
        // Publication never gives authority to an invalid/evicted handle, even
        // if this wave was valid before a later group replaced its cache entry.
        for (group, handle) in groups.iter().zip(handles) {
            let Some(work) =
                handle.and_then(|handle| cache.handle_work(handle, now, snapshot.rank))
            else {
                return;
            };
            if work.len() != group.len()
                || work
                    .iter()
                    .zip(group)
                    .any(|(work, member)| *work != member.work)
            {
                return;
            }
        }
        let lengths = [groups[0].len(), groups.get(1).map_or(0, Vec::len)];
        if let Some(old) = self.actions[candidate].waves.get_mut(round) {
            if old.phase == phase
                && old.lengths == lengths
                && old
                    .requests
                    .iter()
                    .copied()
                    .eq(groups.iter().flatten().map(|member| member.request))
            {
                old.handles = handles;
                return;
            }
            // This is an optimization miss, not permission to preserve a
            // conflicting tail. Only already executed current geometry wins.
            let removed = self.actions[candidate].waves[round..]
                .iter()
                .map(|wave| std::mem::size_of_val(wave.requests.as_ref()))
                .sum::<usize>();
            self.actions[candidate].waves.truncate(round);
            self.charged_bytes -= removed;
            let _ = cache.set_trace_bytes(self.charged_bytes);
        }
        // Keep only a bounded prefix; never allocate gaps/lookahead work.
        if round != self.actions[candidate].waves.len() {
            return;
        }
        let Some(indexes) = lengths[0].checked_add(lengths[1]) else {
            return;
        };
        let Some(index_bytes) = indexes.checked_mul(std::mem::size_of::<usize>()) else {
            return;
        };
        let current = self.charged_bytes;
        let waves = &mut self.actions[candidate].waves;
        let old_capacity = waves.capacity();
        let Some(minimum_capacity) = waves.len().checked_add(1) else {
            return;
        };
        let preferred_capacity = old_capacity
            .checked_mul(2)
            .unwrap_or(minimum_capacity)
            .max(minimum_capacity);
        let mut capacity = old_capacity.max(minimum_capacity);
        let allocation_bytes = |capacity: usize| {
            capacity
                .checked_sub(old_capacity)?
                .checked_mul(std::mem::size_of::<WaveTrace>())?
                .checked_add(current)?
                .checked_add(index_bytes)
        };
        // Grow descriptor capacity geometrically only when the same existing
        // budget can pay for it. Otherwise try the smallest one-wave growth.
        if old_capacity < minimum_capacity
            && allocation_bytes(preferred_capacity)
                .is_some_and(|bytes| cache.set_trace_bytes(bytes))
        {
            capacity = preferred_capacity;
        } else if !allocation_bytes(capacity).is_some_and(|bytes| cache.set_trace_bytes(bytes)) {
            return;
        }
        if waves.try_reserve_exact(capacity - waves.len()).is_err() {
            let _ = cache.set_trace_bytes(current);
            return;
        }
        // Charge actual retained descriptor capacity, including allocator
        // rounding beyond the requested one-slot growth. If it cannot fit,
        // release all trace storage rather than leave unaccounted capacity. This
        // is a guarded optimization fallback, not an evaluation failure.
        let Some(actual_descriptors) = waves
            .capacity()
            .checked_sub(old_capacity)
            .and_then(|growth| growth.checked_mul(std::mem::size_of::<WaveTrace>()))
            .and_then(|growth| current.checked_add(growth))
        else {
            #[cfg(test)]
            cache.note_trace_clear(TraceClearReason::DescriptorArithmetic);
            self.clear(&mut cache);
            return;
        };
        self.charged_bytes = actual_descriptors;
        let Some(actual) = actual_descriptors.checked_add(index_bytes) else {
            #[cfg(test)]
            cache.note_trace_clear(TraceClearReason::IndexArithmetic);
            self.clear(&mut cache);
            return;
        };
        if !cache.set_trace_bytes(actual) {
            #[cfg(test)]
            cache.note_trace_clear(TraceClearReason::ActualDescriptorCapacityRefused);
            self.clear(&mut cache);
            return;
        }
        // The index reservation is not allocated yet. Retain the descriptor
        // high-water mark even if the subsequent index allocation fails.
        snapshot
            .diagnostics
            .cache_bytes(cache.retained_bytes().saturating_sub(index_bytes));
        let mut requests = Vec::new();
        if requests.try_reserve_exact(indexes).is_err() {
            let _ = cache.set_trace_bytes(actual_descriptors);
            return;
        }
        let allocated = requests
            .capacity()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|bytes| actual_descriptors.checked_add(bytes));
        if !allocated.is_some_and(|bytes| cache.set_trace_bytes(bytes)) {
            drop(requests);
            let _ = cache.set_trace_bytes(actual_descriptors);
            return;
        }
        // The temporary Vec is converted to a boxed exact-length payload.
        // This byte limit bounds retained memo storage, not allocator RSS.
        requests.extend(groups.iter().flatten().map(|member| member.request));
        snapshot.diagnostics.cache_bytes(cache.retained_bytes());
        self.actions[candidate].waves.push(WaveTrace {
            phase,
            handles,
            lengths,
            requests: requests.into_boxed_slice(),
        });
        self.charged_bytes = actual;
        let _ = cache.set_trace_bytes(self.charged_bytes);
    }
}

pub(in crate::joint_policy) struct TraceEvaluation<'a, 'snapshot> {
    store: &'a mut TraceStore,
    snapshot: &'a ConditionalSnapshot<'snapshot>,
    replay: &'a ReplayMembers<'a>,
    now: Instant,
}

impl<'a, 'snapshot> TraceEvaluation<'a, 'snapshot> {
    pub(super) fn new(
        store: &'a mut TraceStore,
        snapshot: &'a ConditionalSnapshot<'snapshot>,
        replay: &'a ReplayMembers<'a>,
        now: Instant,
    ) -> Self {
        Self {
            store,
            snapshot,
            replay,
            now,
        }
    }

    pub(in crate::joint_policy) fn try_plan(
        &mut self,
        candidate: usize,
        round: usize,
        phase: TracePhase,
        plan: &mut PlanBuilder,
    ) -> Result<bool, Unsupported> {
        let Some(wave) = self.store.wave(candidate, round, phase) else {
            return Ok(false);
        };
        let mut replayed = false;
        plan.with_members(|members, plan| {
            let cache = self.snapshot.cache.lock();
            for (index, length) in wave.lengths.iter().copied().enumerate() {
                if length == 0 {
                    continue;
                }
                if !wave.handles[index]
                    .and_then(|handle| cache.handle_work(handle, self.now, self.snapshot.rank))
                    .is_some_and(|work| work.len() == length)
                {
                    return Ok(());
                }
            }
            if members.try_reserve_exact(wave.requests.len()).is_err() {
                return Ok(());
            }
            let mut offset = 0;
            for (index, length) in wave.lengths.iter().copied().enumerate() {
                if length == 0 {
                    continue;
                }
                let work = cache
                    .handle_work(
                        wave.handles[index].expect("validated trace handle"),
                        self.now,
                        self.snapshot.rank,
                    )
                    .expect("cache lock preserves validated entry");
                members.extend(
                    wave.requests[offset..offset + length]
                        .iter()
                        .copied()
                        .zip(work.iter().copied())
                        .map(|(request, work)| GroupMember { request, work }),
                );
                offset += length;
            }
            drop(cache);
            (self.replay)(members, plan)?;
            replayed = true;
            Ok(())
        })?;
        Ok(replayed)
    }

    pub(in crate::joint_policy) fn predict(
        &mut self,
        candidate: usize,
        round: usize,
        group: usize,
        phase: TracePhase,
        replayed: bool,
        query: &CostQuery<'_>,
        now: Instant,
    ) -> (CostPrediction, Option<CacheHandle>) {
        let handle = replayed
            .then(|| self.store.wave(candidate, round, phase))
            .flatten()
            .and_then(|wave| wave.handles.get(group).copied().flatten());
        self.snapshot.predict_with_handle(query, now, handle)
    }

    pub(in crate::joint_policy) fn publish(
        &mut self,
        candidate: usize,
        round: usize,
        phase: TracePhase,
        groups: &[Vec<GroupMember>],
        handles: [Option<CacheHandle>; 2],
        replayed: bool,
    ) {
        self.store.publish(
            candidate,
            round,
            phase,
            groups,
            handles,
            self.snapshot,
            self.now,
            replayed,
        );
    }
}
