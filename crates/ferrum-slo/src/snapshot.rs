use arc_swap::ArcSwap;
use ferrum_execution::execution_attempt::{AttemptPrefixRestore, AttemptWorkKind};
use ferrum_scheduler::cost_prediction::*;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::watch;

/// Private empirical domain: exact ordered product ranges, not kernel shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Member {
    kind: u8,
    start: usize,
    end: usize,
    context: usize,
    restored: Option<usize>,
    internally_reused: Option<usize>,
}

impl Member {
    pub fn from_work(work: CostWork) -> Option<Self> {
        let start = work.requested.start();
        let end = work.requested.end();
        let kind = match work.kind {
            AttemptWorkKind::Prefill => {
                if end > work.context_tokens {
                    return None;
                }
                0
            }
            AttemptWorkKind::Decode => {
                if start.checked_add(1) != Some(end) {
                    return None;
                }
                1
            }
        };
        let restored = match work.prefix_restore {
            AttemptPrefixRestore::NoneObserved => None,
            AttemptPrefixRestore::Restored { tokens } if tokens <= start => Some(tokens),
            AttemptPrefixRestore::Restored { .. } => return None,
        };
        Some(Self {
            kind,
            start,
            end,
            context: work.context_tokens,
            restored,
            internally_reused: work.internally_reused_tokens,
        })
    }
}

pub(super) fn fingerprint(members: impl ExactSizeIterator<Item = Member>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    members.len().hash(&mut hasher);
    for member in members {
        member.hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Clone)]
pub(super) struct Entry {
    pub members: Vec<Member>,
    pub upper: Duration,
    pub samples: usize,
    pub observed: Instant,
    pub expires: Instant,
}

pub(super) struct Snapshot {
    pub revision: u64,
    pub maximum_participants: usize,
    pub entries: HashMap<u64, Vec<Entry>>,
}

impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        self.revision
    }

    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        if query.work.len() > self.maximum_participants {
            return CostPrediction::Unknown(CostUnknownReason::QueryLimit);
        }
        if !query.work_is_exact {
            return CostPrediction::Unknown(CostUnknownReason::InexactWork);
        }
        if query.work.is_empty()
            || query
                .work
                .iter()
                .any(|work| Member::from_work(*work).is_none())
        {
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
        }
        // Hash and collision verification borrow the caller's slice. No query
        // allocates, fits, serializes or locks an execution/scheduler owner.
        let fingerprint = fingerprint(
            query
                .work
                .iter()
                .map(|work| Member::from_work(*work).expect("validated product domain")),
        );
        let Some(entry) = self.entries.get(&fingerprint).and_then(|bucket| {
            bucket.iter().find(|entry| {
                entry.members.len() == query.work.len()
                    && entry
                        .members
                        .iter()
                        .zip(query.work)
                        .all(|(member, work)| Member::from_work(*work).as_ref() == Some(member))
            })
        }) else {
            return CostPrediction::Unknown(CostUnknownReason::NoCoverage);
        };
        if now < entry.observed {
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedDomain);
        }
        if now >= entry.expires {
            return CostPrediction::Unknown(CostUnknownReason::Stale);
        }
        CostPrediction::Known {
            upper_service_time: entry.upper,
            samples: entry.samples,
            observed_at: entry.observed,
            expires_at: entry.expires,
        }
    }
}

pub(super) struct Source {
    snapshots: ArcSwap<Snapshot>,
    revisions: watch::Receiver<u64>,
}

impl Source {
    pub fn new(maximum_participants: usize) -> (Self, watch::Sender<u64>) {
        let (sender, revisions) = watch::channel(0);
        (
            Self {
                snapshots: ArcSwap::from_pointee(Snapshot {
                    revision: 0,
                    maximum_participants,
                    entries: HashMap::new(),
                }),
                revisions,
            },
            sender,
        )
    }
    pub fn revision(&self) -> u64 {
        self.snapshots.load().revision
    }
    pub fn install(&self, snapshot: Snapshot) {
        self.snapshots.store(Arc::new(snapshot));
    }
}

impl CostSnapshotSource for Source {
    fn load(&self) -> Arc<dyn CostSnapshot> {
        self.snapshots.load_full()
    }
    fn subscribe(&self) -> watch::Receiver<u64> {
        self.revisions.clone()
    }
}
