use super::{
    invalid_resource, AdmittedRequestResources, AllocationLifetime, Arc, BTreeMap, Mutex, PlanNode,
    RequestAuthorityId, ResourceId, Serialize, TensorAccess, VNextError,
};
use std::mem;
use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStateHazardAccess {
    Read,
    Write,
}

impl RequestStateHazardAccess {
    fn merge(self, other: Self) -> Self {
        if self == Self::Write || other == Self::Write {
            Self::Write
        } else {
            Self::Read
        }
    }

    fn from_tensor_access(access: TensorAccess) -> Self {
        match access {
            TensorAccess::Read => Self::Read,
            TensorAccess::Write | TensorAccess::ReadWrite => Self::Write,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RequestStateHazardClaimSpec {
    resource_id: ResourceId,
    access: RequestStateHazardAccess,
}

/// Immutable plan-compiled projection from node state effects to physical
/// Request-state resource closures. Runtime arbitration never infers hazards
/// from a model family or from aggregate byte counts.
pub(super) struct RequestStateHazardLayout {
    node_claims: Vec<Arc<[RequestStateHazardClaimSpec]>>,
    full_plan_claims: Arc<[RequestStateHazardClaimSpec]>,
    resource_ids: Arc<[ResourceId]>,
}

enum SelectedRequestStateHazardClaims<'a> {
    Borrowed(&'a [RequestStateHazardClaimSpec]),
    Owned(Vec<RequestStateHazardClaimSpec>),
}

impl SelectedRequestStateHazardClaims<'_> {
    fn as_slice(&self) -> &[RequestStateHazardClaimSpec] {
        match self {
            Self::Borrowed(claims) => claims,
            Self::Owned(claims) => claims,
        }
    }
}

impl RequestStateHazardLayout {
    fn compile(nodes: &[PlanNode]) -> Result<Self, VNextError> {
        let mut full_plan = BTreeMap::<ResourceId, RequestStateHazardAccess>::new();
        let mut node_claims = Vec::with_capacity(nodes.len());
        for node in nodes {
            let mut claims = BTreeMap::<ResourceId, RequestStateHazardAccess>::new();
            for effect in node
                .state_effects()
                .iter()
                .filter(|effect| effect.lifetime() == AllocationLifetime::Request)
            {
                let access = RequestStateHazardAccess::from_tensor_access(effect.access());
                if effect.resource_ids().is_empty() {
                    return Err(invalid_resource(format!(
                        "request state `{}` has no physical hazard closure",
                        effect.state_id()
                    )));
                }
                for resource_id in effect.resource_ids() {
                    claims
                        .entry(resource_id.clone())
                        .and_modify(|current| *current = current.merge(access))
                        .or_insert(access);
                    full_plan
                        .entry(resource_id.clone())
                        .and_modify(|current| *current = current.merge(access))
                        .or_insert(access);
                }
            }
            node_claims.push(Arc::from(
                claims
                    .into_iter()
                    .map(|(resource_id, access)| RequestStateHazardClaimSpec {
                        resource_id,
                        access,
                    })
                    .collect::<Vec<_>>(),
            ));
        }
        let full_plan_claims = full_plan
            .iter()
            .map(|(resource_id, access)| RequestStateHazardClaimSpec {
                resource_id: resource_id.clone(),
                access: *access,
            })
            .collect::<Vec<_>>();
        let resource_ids = full_plan.keys().cloned().collect::<Vec<_>>();
        Ok(Self {
            node_claims,
            full_plan_claims: Arc::from(full_plan_claims),
            resource_ids: Arc::from(resource_ids),
        })
    }

    fn is_empty(&self) -> bool {
        self.resource_ids.is_empty()
    }

    fn selected(
        &self,
        node_indices: &[usize],
    ) -> Result<SelectedRequestStateHazardClaims<'_>, VNextError> {
        if node_indices.is_empty()
            || node_indices.windows(2).any(|pair| pair[0] >= pair[1])
            || node_indices
                .last()
                .is_some_and(|index| *index >= self.node_claims.len())
        {
            return Err(invalid_resource(
                "request-state hazard scope must be non-empty, canonical, and plan-bound",
            ));
        }
        if node_indices.len() == self.node_claims.len()
            && node_indices.iter().copied().eq(0..self.node_claims.len())
        {
            return Ok(SelectedRequestStateHazardClaims::Borrowed(
                &self.full_plan_claims,
            ));
        }
        if node_indices.len() == 1 {
            return Ok(SelectedRequestStateHazardClaims::Borrowed(
                &self.node_claims[node_indices[0]],
            ));
        }
        let mut merged = BTreeMap::<ResourceId, RequestStateHazardAccess>::new();
        for &node_index in node_indices {
            for claim in self.node_claims[node_index].iter() {
                merged
                    .entry(claim.resource_id.clone())
                    .and_modify(|current| *current = current.merge(claim.access))
                    .or_insert(claim.access);
            }
        }
        Ok(SelectedRequestStateHazardClaims::Owned(
            merged
                .into_iter()
                .map(|(resource_id, access)| RequestStateHazardClaimSpec {
                    resource_id,
                    access,
                })
                .collect(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RequestStateHazardKey {
    request: RequestAuthorityId,
    resource_id: ResourceId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStateHazardPoisonCause {
    FailedButQuiescent,
    IndeterminateAfterDrain,
    InFlightOwnerDropped,
    CoordinatorInvariant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RequestStateHazardPoison {
    request: RequestAuthorityId,
    resource_id: ResourceId,
    cause: RequestStateHazardPoisonCause,
    value_generation: u64,
    availability_generation: u64,
}

impl RequestStateHazardPoison {
    pub const fn request(&self) -> RequestAuthorityId {
        self.request
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn cause(&self) -> RequestStateHazardPoisonCause {
        self.cause
    }

    pub const fn value_generation(&self) -> u64 {
        self.value_generation
    }

    pub const fn availability_generation(&self) -> u64 {
        self.availability_generation
    }
}

struct RequestStateHazardCell {
    readers: u32,
    writer: bool,
    waiting_writers: u32,
    value_generation: u64,
    availability_generation: u64,
    poison: Option<RequestStateHazardPoisonCause>,
}

struct RequestStateHazardCoordinatorState {
    cells: BTreeMap<RequestStateHazardKey, RequestStateHazardCell>,
    change_epoch: u64,
    globally_poisoned: bool,
}

/// One plan-local arbiter for every Request-state physical closure. A single
/// mutex makes a mixed-request wave acquisition all-or-nothing; no Nth claim
/// failure can leave an earlier request partially held.
pub(super) struct RequestStateHazardCoordinator {
    layout: RequestStateHazardLayout,
    state: Mutex<RequestStateHazardCoordinatorState>,
    changed: watch::Sender<u64>,
}

impl RequestStateHazardCoordinator {
    pub(super) fn compile(nodes: &[PlanNode]) -> Result<Arc<Self>, VNextError> {
        let layout = RequestStateHazardLayout::compile(nodes)?;
        let (changed, _) = watch::channel(1);
        Ok(Arc::new(Self {
            layout,
            state: Mutex::new(RequestStateHazardCoordinatorState {
                cells: BTreeMap::new(),
                change_epoch: 1,
                globally_poisoned: false,
            }),
            changed,
        }))
    }

    pub(super) fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    pub(super) fn register_request(
        self: &Arc<Self>,
        request: RequestAuthorityId,
    ) -> Result<Option<RequestStateHazardRegistration>, VNextError> {
        if self.is_empty() {
            return Ok(None);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("request-state hazard coordinator is poisoned"))?;
        if state.globally_poisoned {
            return Err(invalid_resource(
                "request-state hazard coordinator is fail-closed",
            ));
        }
        let mut inserted = Vec::with_capacity(self.layout.resource_ids.len());
        for resource_id in self.layout.resource_ids.iter() {
            let key = RequestStateHazardKey {
                request,
                resource_id: resource_id.clone(),
            };
            if state.cells.contains_key(&key) {
                for key in inserted {
                    state.cells.remove(&key);
                }
                state.globally_poisoned = true;
                return Err(invalid_resource(
                    "request-state hazard registration reused a live request authority",
                ));
            }
            state.cells.insert(
                key.clone(),
                RequestStateHazardCell {
                    readers: 0,
                    writer: false,
                    waiting_writers: 0,
                    value_generation: 0,
                    availability_generation: 1,
                    poison: None,
                },
            );
            inserted.push(key);
        }
        Ok(Some(RequestStateHazardRegistration {
            coordinator: Arc::clone(self),
            request,
            registered: true,
        }))
    }

    fn unregister_request(&self, request: RequestAuthorityId) -> Result<(), VNextError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("request-state hazard coordinator is poisoned"))?;
        let keys = self
            .layout
            .resource_ids
            .iter()
            .map(|resource_id| RequestStateHazardKey {
                request,
                resource_id: resource_id.clone(),
            })
            .collect::<Vec<_>>();
        let valid = keys.iter().all(|key| {
            state
                .cells
                .get(key)
                .is_some_and(|cell| cell.readers == 0 && !cell.writer && cell.waiting_writers == 0)
        });
        if !valid {
            state.globally_poisoned = true;
            return Err(invalid_resource(
                "request-state hazard registration dropped with live claims or waiters",
            ));
        }
        for key in keys {
            state.cells.remove(&key);
        }
        Ok(())
    }

    pub(super) fn try_acquire<R>(
        self: &Arc<Self>,
        participants: &[Arc<AdmittedRequestResources<R>>],
        node_indices: &[usize],
    ) -> Result<RequestStateHazardAcquireDecision<R>, VNextError>
    where
        R: super::DeviceRuntime,
    {
        let selected = self.layout.selected(node_indices)?;
        let claims = selected.as_slice();
        if claims.is_empty() {
            return Ok(RequestStateHazardAcquireDecision::Acquired(None));
        }
        let mut requests =
            BTreeMap::<RequestAuthorityId, (Arc<AdmittedRequestResources<R>>, u32)>::new();
        for request in participants {
            if !Arc::ptr_eq(&request.plan.dynamic_pools().request_state_hazards, self) {
                return Err(invalid_resource(
                    "request-state hazard participant belongs to another plan coordinator",
                ));
            }
            let request_authority = request.request_authority();
            if let Some((_, sibling_count)) = requests.get_mut(&request_authority) {
                *sibling_count = sibling_count
                    .checked_add(1)
                    .ok_or_else(|| invalid_resource("request-state sibling count exceeds u32"))?;
            } else {
                requests.insert(request_authority, (Arc::clone(request), 1));
            }
        }
        if requests.is_empty() {
            return Err(invalid_resource(
                "request-state hazard acquisition requires participants",
            ));
        }
        if claims
            .iter()
            .any(|claim| claim.access == RequestStateHazardAccess::Write)
        {
            if let Some((request, (_, sibling_count))) =
                requests.iter().find(|(_, (_, count))| *count > 1)
            {
                return Ok(RequestStateHazardAcquireDecision::SplitRequired(
                    RequestStateHazardSplitRequired {
                        request: *request,
                        sibling_count: *sibling_count,
                        resource_ids: claims
                            .iter()
                            .filter(|claim| claim.access == RequestStateHazardAccess::Write)
                            .map(|claim| claim.resource_id.clone())
                            .collect(),
                    },
                ));
            }
        }

        let requested = requests
            .keys()
            .flat_map(|request| {
                claims
                    .iter()
                    .map(move |claim| ActiveRequestStateHazardClaim {
                        key: RequestStateHazardKey {
                            request: *request,
                            resource_id: claim.resource_id.clone(),
                        },
                        access: claim.access,
                    })
            })
            .collect::<Vec<_>>();
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("request-state hazard coordinator is poisoned"))?;
        if state.globally_poisoned {
            return Err(invalid_resource(
                "request-state hazard coordinator is fail-closed",
            ));
        }
        let mut blockers = Vec::new();
        for claim in &requested {
            let cell = state.cells.get(&claim.key).ok_or_else(|| {
                invalid_resource("request-state hazard claim has no live request registration")
            })?;
            if let Some(cause) = cell.poison {
                return Ok(RequestStateHazardAcquireDecision::Poisoned(
                    RequestStateHazardPoison {
                        request: claim.key.request,
                        resource_id: claim.key.resource_id.clone(),
                        cause,
                        value_generation: cell.value_generation,
                        availability_generation: cell.availability_generation,
                    },
                ));
            }
            let conflict = match claim.access {
                RequestStateHazardAccess::Read => cell.writer || cell.waiting_writers != 0,
                RequestStateHazardAccess::Write => cell.writer || cell.readers != 0,
            };
            if conflict {
                blockers.push(RequestStateHazardBlocker {
                    request: claim.key.request,
                    resource_id: claim.key.resource_id.clone(),
                    requested_access: claim.access,
                    active_readers: cell.readers,
                    active_writer: cell.writer,
                    waiting_writers: cell.waiting_writers,
                    availability_generation: cell.availability_generation,
                });
            }
        }
        if !blockers.is_empty() {
            return Ok(RequestStateHazardAcquireDecision::Deferred(
                RequestStateHazardDeferral {
                    coordinator: Arc::clone(self),
                    observed_change_epoch: state.change_epoch,
                    blockers,
                },
            ));
        }
        if requested.iter().any(|claim| {
            claim.access == RequestStateHazardAccess::Read
                && state
                    .cells
                    .get(&claim.key)
                    .is_some_and(|cell| cell.readers == u32::MAX)
        }) {
            state.globally_poisoned = true;
            return Err(invalid_resource("request-state reader count is exhausted"));
        }
        for claim in &requested {
            let cell = state
                .cells
                .get_mut(&claim.key)
                .expect("validated request-state hazard cell remains registered");
            match claim.access {
                RequestStateHazardAccess::Read => {
                    cell.readers += 1;
                }
                RequestStateHazardAccess::Write => cell.writer = true,
            }
        }
        drop(state);
        Ok(RequestStateHazardAcquireDecision::Acquired(Some(
            RequestStateHazardPermit {
                coordinator: Arc::clone(self),
                requests: Some(requests.into_values().map(|(request, _)| request).collect()),
                claims: requested,
                phase: RequestStateHazardPermitPhase::Prepared,
                finished: false,
            },
        )))
    }

    fn release(
        &self,
        claims: &[ActiveRequestStateHazardClaim],
        disposition: RequestStateHazardReleaseDisposition,
    ) -> Result<(), VNextError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("request-state hazard coordinator is poisoned"))?;
        let valid = claims.iter().all(|claim| {
            state
                .cells
                .get(&claim.key)
                .is_some_and(|cell| match claim.access {
                    RequestStateHazardAccess::Read => cell.readers > 0 && !cell.writer,
                    RequestStateHazardAccess::Write => cell.writer && cell.readers == 0,
                })
        });
        if !valid {
            state.globally_poisoned = true;
            return Err(invalid_resource(
                "request-state hazard release does not own its exact active claims",
            ));
        }
        let next_change_epoch = state
            .change_epoch
            .checked_add(1)
            .ok_or_else(|| invalid_resource("request-state hazard change epoch is exhausted"))?;
        if claims.iter().any(|claim| {
            state.cells.get(&claim.key).is_some_and(|cell| {
                cell.availability_generation == u64::MAX
                    || (claim.access == RequestStateHazardAccess::Write
                        && disposition == RequestStateHazardReleaseDisposition::Succeeded
                        && cell.value_generation == u64::MAX)
            })
        }) {
            state.globally_poisoned = true;
            return Err(invalid_resource(
                "request-state hazard generation is exhausted",
            ));
        }
        for claim in claims {
            let cell = state
                .cells
                .get_mut(&claim.key)
                .expect("validated request-state hazard cell remains registered");
            match claim.access {
                RequestStateHazardAccess::Read => cell.readers -= 1,
                RequestStateHazardAccess::Write => {
                    cell.writer = false;
                    match disposition {
                        RequestStateHazardReleaseDisposition::Succeeded => {
                            cell.value_generation += 1;
                        }
                        RequestStateHazardReleaseDisposition::PreparedAbandoned => {}
                        RequestStateHazardReleaseDisposition::FailedButQuiescent => {
                            cell.poison = Some(RequestStateHazardPoisonCause::FailedButQuiescent);
                        }
                        RequestStateHazardReleaseDisposition::IndeterminateAfterDrain => {
                            cell.poison =
                                Some(RequestStateHazardPoisonCause::IndeterminateAfterDrain);
                        }
                        RequestStateHazardReleaseDisposition::InFlightOwnerDropped => {
                            cell.poison = Some(RequestStateHazardPoisonCause::InFlightOwnerDropped);
                        }
                    }
                }
            }
            cell.availability_generation += 1;
        }
        state.change_epoch = next_change_epoch;
        self.changed.send_replace(next_change_epoch);
        Ok(())
    }

    fn recheck(&self, blockers: &[RequestStateHazardBlocker]) -> Result<bool, VNextError> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("request-state hazard coordinator is poisoned"))?;
        if state.globally_poisoned {
            return Err(invalid_resource(
                "request-state hazard coordinator is fail-closed",
            ));
        }
        for blocker in blockers {
            let key = RequestStateHazardKey {
                request: blocker.request,
                resource_id: blocker.resource_id.clone(),
            };
            let cell = state.cells.get(&key).ok_or_else(|| {
                invalid_resource("request-state hazard waiter lost its request registration")
            })?;
            if let Some(cause) = cell.poison {
                return Err(invalid_resource(format!(
                    "request-state resource `{}` is poisoned by {cause:?}",
                    blocker.resource_id
                )));
            }
            let conflict = match blocker.requested_access {
                RequestStateHazardAccess::Read => cell.writer || cell.waiting_writers != 0,
                RequestStateHazardAccess::Write => cell.writer || cell.readers != 0,
            };
            if conflict && cell.availability_generation == blocker.availability_generation {
                return Ok(false);
            }
            if conflict {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn register_writer_waiters(
        &self,
        blockers: &[RequestStateHazardBlocker],
    ) -> Result<(), VNextError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("request-state hazard coordinator is poisoned"))?;
        let writer_keys = blockers
            .iter()
            .filter(|blocker| blocker.requested_access == RequestStateHazardAccess::Write)
            .map(|blocker| RequestStateHazardKey {
                request: blocker.request,
                resource_id: blocker.resource_id.clone(),
            })
            .collect::<Vec<_>>();
        if writer_keys.iter().any(|key| {
            state
                .cells
                .get(key)
                .is_none_or(|cell| cell.waiting_writers == u32::MAX)
        }) {
            state.globally_poisoned = true;
            return Err(invalid_resource(
                "request-state writer waiter count is exhausted",
            ));
        }
        for key in writer_keys {
            state
                .cells
                .get_mut(&key)
                .expect("validated request-state writer waiter remains registered")
                .waiting_writers += 1;
        }
        Ok(())
    }

    fn unregister_writer_waiters(&self, blockers: &[RequestStateHazardBlocker]) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let mut changed = false;
        for blocker in blockers
            .iter()
            .filter(|blocker| blocker.requested_access == RequestStateHazardAccess::Write)
        {
            let key = RequestStateHazardKey {
                request: blocker.request,
                resource_id: blocker.resource_id.clone(),
            };
            let Some(cell) = state.cells.get_mut(&key) else {
                state.globally_poisoned = true;
                continue;
            };
            if cell.waiting_writers == 0 {
                state.globally_poisoned = true;
            } else {
                cell.waiting_writers -= 1;
                changed = true;
            }
        }
        if changed {
            let Some(next_change_epoch) = state.change_epoch.checked_add(1) else {
                state.globally_poisoned = true;
                return;
            };
            state.change_epoch = next_change_epoch;
            self.changed.send_replace(next_change_epoch);
        }
    }
}

pub(super) struct RequestStateHazardRegistration {
    coordinator: Arc<RequestStateHazardCoordinator>,
    request: RequestAuthorityId,
    registered: bool,
}

impl Drop for RequestStateHazardRegistration {
    fn drop(&mut self) {
        if self.registered {
            let _ = self.coordinator.unregister_request(self.request);
            self.registered = false;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RequestStateHazardBlocker {
    request: RequestAuthorityId,
    resource_id: ResourceId,
    requested_access: RequestStateHazardAccess,
    active_readers: u32,
    active_writer: bool,
    waiting_writers: u32,
    availability_generation: u64,
}

impl RequestStateHazardBlocker {
    pub const fn request(&self) -> RequestAuthorityId {
        self.request
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn requested_access(&self) -> RequestStateHazardAccess {
        self.requested_access
    }

    pub const fn active_readers(&self) -> u32 {
        self.active_readers
    }

    pub const fn active_writer(&self) -> bool {
        self.active_writer
    }

    pub const fn waiting_writers(&self) -> u32 {
        self.waiting_writers
    }

    pub const fn availability_generation(&self) -> u64 {
        self.availability_generation
    }
}

#[derive(Clone, Serialize)]
pub struct RequestStateHazardDeferral {
    #[serde(skip)]
    coordinator: Arc<RequestStateHazardCoordinator>,
    observed_change_epoch: u64,
    blockers: Vec<RequestStateHazardBlocker>,
}

impl std::fmt::Debug for RequestStateHazardDeferral {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestStateHazardDeferral")
            .field("observed_change_epoch", &self.observed_change_epoch)
            .field("blockers", &self.blockers)
            .finish()
    }
}

impl RequestStateHazardDeferral {
    pub const fn observed_change_epoch(&self) -> u64 {
        self.observed_change_epoch
    }

    pub fn blockers(&self) -> &[RequestStateHazardBlocker] {
        &self.blockers
    }

    pub fn register_waiter(&self) -> Result<RequestStateHazardWaitRegistration, VNextError> {
        let receiver = self.coordinator.changed.subscribe();
        self.coordinator.register_writer_waiters(&self.blockers)?;
        let ready = match self.coordinator.recheck(&self.blockers) {
            Ok(ready) => ready,
            Err(error) => {
                self.coordinator.unregister_writer_waiters(&self.blockers);
                return Err(error);
            }
        };
        Ok(RequestStateHazardWaitRegistration {
            coordinator: Arc::clone(&self.coordinator),
            blockers: self.blockers.clone(),
            receiver,
            ready,
            registered: true,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RequestStateHazardSplitRequired {
    request: RequestAuthorityId,
    sibling_count: u32,
    resource_ids: Vec<ResourceId>,
}

impl RequestStateHazardSplitRequired {
    pub const fn request(&self) -> RequestAuthorityId {
        self.request
    }

    pub const fn sibling_count(&self) -> u32 {
        self.sibling_count
    }

    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }
}

#[must_use = "a request-state hazard waiter must be awaited or dropped"]
pub struct RequestStateHazardWaitRegistration {
    coordinator: Arc<RequestStateHazardCoordinator>,
    blockers: Vec<RequestStateHazardBlocker>,
    receiver: watch::Receiver<u64>,
    ready: bool,
    registered: bool,
}

impl RequestStateHazardWaitRegistration {
    pub async fn wait_for_change(mut self) -> Result<u64, VNextError> {
        loop {
            if self.ready || self.coordinator.recheck(&self.blockers)? {
                return Ok(*self.receiver.borrow_and_update());
            }
            self.receiver.changed().await.map_err(|_| {
                invalid_resource("request-state hazard coordinator closed while waiting")
            })?;
        }
    }
}

impl Drop for RequestStateHazardWaitRegistration {
    fn drop(&mut self) {
        if self.registered {
            self.coordinator.unregister_writer_waiters(&self.blockers);
            self.registered = false;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestStateHazardTerminalDisposition {
    Succeeded,
    FailedButQuiescent,
    IndeterminateAfterDrain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestStateHazardReleaseDisposition {
    PreparedAbandoned,
    Succeeded,
    FailedButQuiescent,
    IndeterminateAfterDrain,
    InFlightOwnerDropped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestStateHazardPermitPhase {
    Prepared,
    InFlight,
    Indeterminate,
}

struct ActiveRequestStateHazardClaim {
    key: RequestStateHazardKey,
    access: RequestStateHazardAccess,
}

#[must_use = "request-state hazards must remain owned through the device fence"]
pub struct RequestStateHazardPermit<R>
where
    R: super::DeviceRuntime,
{
    coordinator: Arc<RequestStateHazardCoordinator>,
    requests: Option<Vec<Arc<AdmittedRequestResources<R>>>>,
    claims: Vec<ActiveRequestStateHazardClaim>,
    phase: RequestStateHazardPermitPhase,
    finished: bool,
}

impl<R> RequestStateHazardPermit<R>
where
    R: super::DeviceRuntime,
{
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    pub(crate) fn mark_submission_fence_installed(&mut self) -> Result<(), VNextError> {
        if self.finished || self.phase != RequestStateHazardPermitPhase::Prepared {
            return Err(invalid_resource(
                "request-state hazards cannot install a second submission fence",
            ));
        }
        self.phase = RequestStateHazardPermitPhase::InFlight;
        Ok(())
    }

    pub(crate) fn mark_submission_indeterminate(&mut self) {
        if !self.finished {
            self.phase = RequestStateHazardPermitPhase::Indeterminate;
        }
    }

    pub(crate) fn finish(
        &mut self,
        disposition: RequestStateHazardTerminalDisposition,
    ) -> Result<(), VNextError> {
        if self.finished || self.phase == RequestStateHazardPermitPhase::Prepared {
            return Err(invalid_resource(
                "request-state hazard terminalization requires one installed or indeterminate submission",
            ));
        }
        let disposition = match disposition {
            RequestStateHazardTerminalDisposition::Succeeded => {
                RequestStateHazardReleaseDisposition::Succeeded
            }
            RequestStateHazardTerminalDisposition::FailedButQuiescent => {
                RequestStateHazardReleaseDisposition::FailedButQuiescent
            }
            RequestStateHazardTerminalDisposition::IndeterminateAfterDrain => {
                RequestStateHazardReleaseDisposition::IndeterminateAfterDrain
            }
        };
        self.release(disposition)
    }

    fn release(
        &mut self,
        disposition: RequestStateHazardReleaseDisposition,
    ) -> Result<(), VNextError> {
        self.coordinator.release(&self.claims, disposition)?;
        self.finished = true;
        self.claims.clear();
        drop(self.requests.take());
        Ok(())
    }
}

impl<R> Drop for RequestStateHazardPermit<R>
where
    R: super::DeviceRuntime,
{
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let disposition = match self.phase {
            RequestStateHazardPermitPhase::Prepared => {
                RequestStateHazardReleaseDisposition::PreparedAbandoned
            }
            RequestStateHazardPermitPhase::InFlight
            | RequestStateHazardPermitPhase::Indeterminate => {
                RequestStateHazardReleaseDisposition::InFlightOwnerDropped
            }
        };
        if self.release(disposition).is_err() {
            // Preserve every parent request backing rather than allowing a
            // possibly in-flight device access to observe reused storage.
            mem::forget(self.requests.take());
        }
    }
}

pub(super) enum RequestStateHazardAcquireDecision<R>
where
    R: super::DeviceRuntime,
{
    Acquired(Option<RequestStateHazardPermit<R>>),
    Deferred(RequestStateHazardDeferral),
    SplitRequired(RequestStateHazardSplitRequired),
    Poisoned(RequestStateHazardPoison),
}
