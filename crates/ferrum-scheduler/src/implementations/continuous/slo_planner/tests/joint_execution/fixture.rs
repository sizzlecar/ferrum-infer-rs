//! SharedPool + TestResolver adaptation, not a physical allocator/provider.
use super::*;
use ferrum_interfaces::execution_cost::ActualRowWork;
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Begin,
    Project,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum BudgetFault {
    NoPoll,
    SwallowPoll,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Returned {
    Honest,
    Reverse,
    Duplicate,
    ReplaceAction,
    ReplaceIncarnation,
    BadRow,
    BadRecurrent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Frontier {
    pub key: RequestWorkKey,
    pub context: u32,
    pub commits: u32,
}
fn frontiers(requests: &[RequestSchedulingView]) -> Vec<Frontier> {
    requests
        .iter()
        .map(|r| Frontier {
            key: r.key.clone(),
            context: r.context_tokens,
            commits: r.timing.committed_tokens,
        })
        .collect()
}

#[derive(Debug, Clone)]
pub(super) struct Seen {
    pub remaining: u64,
    pub requests: Vec<RequestSchedulingView>,
}

pub(super) struct JointPool {
    pub free: u64,
    pub revoked: Cell<bool>,
    pub begin_revocations: RefCell<Vec<bool>>,
    pub seen: RefCell<Vec<Seen>>,
    pub now: Cell<u64>,
    pub fault: Option<(Phase, BudgetFault)>,
    pub returned: Returned,
}
impl JointPool {
    pub fn new(free: u64) -> Self {
        Self {
            free,
            revoked: Cell::new(false),
            begin_revocations: RefCell::default(),
            seen: RefCell::default(),
            now: Cell::new(100),
            fault: None,
            returned: Returned::Honest,
        }
    }
    fn budget_fault(
        &self,
        phase: Phase,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) {
        let Some((where_, fault)) = self.fault else {
            return;
        };
        if where_ != phase {
            return;
        }
        // Semantic event, not a particular callback ordinal or real wall time.
        let before = self.now.replace(10_000_000);
        if matches!(fault, BudgetFault::SwallowPoll) {
            let _ = poll();
            // A later successful read cannot erase the callback's failure.
            self.now.set(before);
        }
    }
}

struct JointState<'epoch> {
    pool: &'epoch JointPool,
    snapshot: &'epoch SchedulerSnapshot,
    remaining: u64,
    frontiers: Vec<Frontier>,
}

impl PlanningExecutionContext for JointPool {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        self.budget_fault(Phase::Begin, poll);
        let revoked = self.revoked.get();
        self.begin_revocations.borrow_mut().push(revoked);
        if revoked {
            return Err(PlanningUnknownReason::UnknownResourceEvidence);
        }
        Ok(Arc::new(JointState {
            pool: self,
            snapshot,
            remaining: self.free,
            frontiers: frontiers(&snapshot.requests),
        }))
    }
}

impl<'epoch> PlanningExecutionState<'epoch> for JointState<'epoch> {
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        self.pool.budget_fault(Phase::Project, poll);
        assert_eq!(
            frontiers(input.requests),
            self.frontiers,
            "the callback must receive this branch's full logical frontier"
        );
        self.pool.seen.borrow_mut().push(Seen {
            remaining: self.remaining,
            requests: input.requests.to_vec(),
        });
        let growth = input.work.len() as u64;
        if growth + 1 > self.remaining {
            return Err(PlanningUnknownReason::OutputOrResourceBlocked);
        }
        let mut ordered_work = input.work.to_vec();
        if matches!(self.pool.returned, Returned::Reverse) {
            ordered_work.sort_by_key(|work| {
                std::cmp::Reverse(
                    self.snapshot
                        .requests
                        .iter()
                        .position(|r| r.key == work.key)
                        .unwrap(),
                )
            });
        }
        let ordered_rows: Vec<_> = ordered_work
            .iter()
            .map(|work| {
                *input
                    .rows
                    .iter()
                    .find(|r| r.request.key == work.key)
                    .unwrap()
            })
            .collect();
        // Reuse existing synthetic canonical builder; no new fake provider.
        // Fault cases deliberately do not cooperate with the budget poll.
        let mut canonical = TestResolver
            .resolve(
                &PlanningShapeQuery {
                    snapshot: self.snapshot,
                    prior_waves: &[],
                    kind: input.kind,
                    rows: &ordered_rows,
                    recurrent_state_bytes: input.recurrent_state_bytes,
                },
                &mut || Ok(()),
            )?
            .unwrap();
        let mut next = self.frontiers.clone();
        for row in input.rows {
            let target = next.iter_mut().find(|r| r.key == row.request.key).unwrap();
            match row.work {
                ActualRowWork::Decode { .. } => {
                    target.context += 1;
                    target.commits += 1;
                }
                // This fixture intentionally covers decode wiring only.
                _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
            }
        }
        match self.pool.returned {
            Returned::Honest | Returned::Reverse => {}
            Returned::Duplicate => ordered_work[0] = ordered_work[1].clone(),
            Returned::ReplaceAction => {
                ordered_work[0].action = WaveAction::Prefill {
                    offset: 0,
                    count: n32(1),
                }
            }
            Returned::ReplaceIncarnation => ordered_work[0].key.incarnation += 1,
            Returned::BadRow => canonical.rows[0] = ActualRowWork::Decode { kv_tokens: 999 },
            Returned::BadRecurrent => canonical.recurrent_state_bytes += 1,
        }
        Ok(Some(ProjectedExecution {
            host_content_forecasts: None,
            statistical_evidence: None,
            ordered_work,
            canonical_domain: PlanningShapeDomain::Exact(canonical),
            successor: Arc::new(Self {
                pool: self.pool,
                snapshot: self.snapshot,
                remaining: self.remaining - growth,
                frontiers: next,
            }),
        }))
    }
}

pub(super) struct JointClock<'a>(pub &'a Cell<u64>);
impl PlanningClock for JointClock<'_> {
    fn now_ns(&mut self) -> u64 {
        self.0.get()
    }
}

/// The production helper constructs/validates input and output; this only
/// selects fixture owners. It does not duplicate logical row preparation.
pub(super) fn project_decode<'epoch>(
    snapshot: &SchedulerSnapshot,
    state: &dyn PlanningExecutionState<'epoch>,
    requests: &[RequestSchedulingView],
    indexes: &[usize],
) -> Result<Option<execution::VerifiedExecution<'epoch>>, PlanningUnknownReason> {
    let work: Vec<_> = indexes
        .iter()
        .map(|&i| CandidateWork {
            key: requests[i].key.clone(),
            action: WaveAction::Decode,
        })
        .collect();
    execution::project(
        snapshot,
        requests,
        &work,
        state,
        false,
        PlanningCostEvidenceRequirement::None,
        &mut || Ok(()),
    )
}
