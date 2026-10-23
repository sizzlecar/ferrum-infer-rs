//! Explicit sharing of one successfully initialized, immutable physical weight
//! snapshot. This is not a source/path cache and does not authenticate checkpoint
//! contents. Each attachment owns independent dynamic pools and plan admission.
use super::plan_runtime::{PlanRuntimeStatic, PlanStaticResources, PLAN_RUNTIME_OPEN};
use super::*;
use crate::{ExecutionWeightPlan, ResolvedStorageComponent, RuntimeWorkLimits, WeightId};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct SharedWeightContract {
    weights: ExecutionWeightPlan,
    allocations: Vec<ResourceAllocation>,
    placements: BTreeMap<WeightId, ResolvedStorageComponent>,
}

impl SharedWeightContract {
    pub(crate) fn from_plan(plan: &ExecutionPlan) -> Result<Self, VNextError> {
        let payload = plan.payload();
        let weights = payload.execution_weights().stored()?;
        let allocations = payload.memory().static_allocations();
        if payload.work_limits() != RuntimeWorkLimits::FixedTensor
            || allocations.is_empty()
            || !weights.static_weight_transforms().is_empty()
            || allocations.iter().any(|allocation| {
                allocation.usage() != BufferUsage::Weights
                    || allocation.lifetime() != AllocationLifetime::Plan
                    || !matches!(allocation.kind(), AllocationKind::Value)
            })
        {
            return Err(invalid_resource("shared component weights require fixed-tensor, readonly Stored allocations without static transforms or mutable static state"));
        }
        let static_ids = allocations
            .iter()
            .map(|a| a.resource_id())
            .collect::<BTreeSet<_>>();
        let mut placements = BTreeMap::new();
        for node in payload.nodes() {
            for value in node.values() {
                for component in value.storage().components() {
                    if !static_ids.contains(component.resource_id()) {
                        if value.usage() == BufferUsage::Weights {
                            return Err(invalid_resource("shared weight binding is not part of the immutable static installation"));
                        }
                        continue;
                    }
                    if value.access() != TensorAccess::Read || value.usage() != BufferUsage::Weights
                    {
                        return Err(invalid_resource(
                            "shared static storage is bound by a mutable or non-weight value",
                        ));
                    }
                    let id = component.component_id().ok_or_else(|| {
                        invalid_resource("shared weight placement has no component identity")
                    })?;
                    let schema = weights
                        .schema()
                        .components
                        .iter()
                        .find(|candidate| candidate.id == *id)
                        .ok_or_else(|| {
                            invalid_resource("shared weight component is absent from its schema")
                        })?;
                    if component.length_bytes() != schema.physical_bytes()?
                        || component.element_type() != schema.physical_element_type()
                    {
                        return Err(invalid_resource(
                            "shared weight placement differs from its physical schema",
                        ));
                    }
                    if let Some(previous) = placements.insert(id.clone(), component.clone()) {
                        if previous != *component {
                            return Err(invalid_resource(
                                "shared weight component has inconsistent physical placements",
                            ));
                        }
                    }
                }
            }
        }
        if placements.is_empty()
            || allocations.iter().any(|allocation| {
                !placements
                    .values()
                    .any(|p| p.resource_id() == allocation.resource_id())
            })
        {
            return Err(invalid_resource(
                "shared static allocation lacks a readonly weight placement",
            ));
        }
        Ok(Self {
            weights: weights.clone(),
            allocations: allocations.to_vec(),
            placements,
        })
    }

    /// Consumer plans may omit readonly components, but may neither repack
    /// their arena nor reinterpret the original installation. In particular,
    /// equal names/shapes alone cannot replace the full materializer/schema/
    /// source-map/quality identity or exact component offsets.
    fn supports(&self, consumer: &Self) -> bool {
        self.weights == consumer.weights
            && consumer
                .placements
                .iter()
                .all(|(id, placement)| self.placements.get(id) == Some(placement))
            && consumer.allocations.iter().all(|allocation| {
                self.allocations
                    .iter()
                    .any(|source| allocation_prefix(source, allocation))
            })
    }
}

fn allocation_prefix(source: &ResourceAllocation, consumer: &ResourceAllocation) -> bool {
    source.resource_id() == consumer.resource_id()
        && source.alignment_bytes() == consumer.alignment_bytes()
        && source.usage() == BufferUsage::Weights
        && consumer.usage() == BufferUsage::Weights
        && source.element_type() == consumer.element_type()
        && source.lifetime() == consumer.lifetime()
        && source.kind() == consumer.kind()
        && source.storage() == consumer.storage()
        && source.instance_count() == 1
        && consumer.instance_count() == 1
        && consumer.per_instance_bytes() <= source.per_instance_bytes()
        && consumer.instance_stride_bytes() <= source.instance_stride_bytes()
        && consumer.size_bytes() <= source.size_bytes()
}

/// Original physical installation identity, distinct from an attaching plan.
/// The fingerprint identifies a sealed transaction and its weight contract;
/// it is deliberately not a digest of the uploaded payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SharedComponentWeightsEvidence {
    snapshot_fingerprint: String,
    weight_plan_fingerprint: String,
    source_identity: ResourceTransactionIdentity,
    physical_bytes: u64,
}

impl SharedComponentWeightsEvidence {
    pub fn snapshot_fingerprint(&self) -> &str {
        &self.snapshot_fingerprint
    }
    pub fn weight_plan_fingerprint(&self) -> &str {
        &self.weight_plan_fingerprint
    }
    pub fn source_identity(&self) -> &ResourceTransactionIdentity {
        &self.source_identity
    }
    pub fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }
}

pub(super) struct SharedWeightOwner<R: DeviceRuntime> {
    // Read gate covers compatibility checking and acquisition of a new lease.
    // Close takes the write gate, then releases it before native destruction.
    gate: RwLock<bool>,
    contract: SharedWeightContract,
    pub(super) resources: PlanStaticResources<R>,
    evidence: SharedComponentWeightsEvidence,
}

/// Clone retains exactly the same buffers and one physical capacity claim.
#[must_use = "shared weights retain physical resources until the last owner retires"]
pub struct SharedComponentWeights<R: DeviceRuntime> {
    pub(super) inner: Arc<SharedWeightOwner<R>>,
}

impl<R: DeviceRuntime> Clone for SharedComponentWeights<R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<R: DeviceRuntime> SharedComponentWeights<R> {
    pub fn evidence(&self) -> &SharedComponentWeightsEvidence {
        &self.inner.evidence
    }

    pub fn validate_plan(plan: &ExecutionPlan) -> Result<(), VNextError> {
        SharedWeightContract::from_plan(plan).map(|_| ())
    }

    pub fn validate_for(&self, plan: &ExecutionPlan, runtime: &Arc<R>) -> Result<(), VNextError> {
        let gate = self
            .inner
            .gate
            .read()
            .map_err(|_| invalid_resource("shared weight lifecycle is poisoned"))?;
        if *gate {
            return Err(invalid_resource("shared weights are closing"));
        }
        self.validate_compatible(plan, runtime)
    }

    fn validate_compatible(
        &self,
        plan: &ExecutionPlan,
        runtime: &Arc<R>,
    ) -> Result<(), VNextError> {
        let original = &self.inner.resources;
        let lease = original
            .lease
            .as_ref()
            .expect("live shared weights own their lease");
        let memory = plan.payload().memory();
        if !Arc::ptr_eq(runtime, lease.runtime())
            || original.admission.device_capacity_bytes() != memory.device_capacity_bytes()
            || original.admission.usable_capacity_bytes() != memory.usable_capacity_bytes()
            || original.admission.maximum_active_sequences() != memory.maximum_active_sequences()
            || !self
                .inner
                .contract
                .supports(&SharedWeightContract::from_plan(plan)?)
        {
            return Err(invalid_resource("shared weights differ in runtime instance, physical weight contract/layout, or admission ceiling"));
        }
        Ok(())
    }

    /// Resolve the consumer's already-validated prefix extent to the original
    /// physical slot. The lease still checks the complete source descriptor;
    /// its allocation and capacity claim are never truncated on attachment.
    pub(super) fn plan_static_view(
        &self,
        allocation: &ResourceAllocation,
    ) -> Result<super::LeasedBufferView<'_, R::Buffer>, VNextError> {
        let (slot, source) = self
            .inner
            .contract
            .allocations
            .iter()
            .enumerate()
            .find(|(_, source)| allocation_prefix(source, allocation))
            .ok_or_else(|| {
                invalid_resource("shared consumer allocation differs from its source extent")
            })?;
        self.lease().plan_static_view(slot, source)
    }

    /// Creates only this plan's logical/dynamic authority. No weight allocation,
    /// source read, materialization, upload or additional static claim occurs.
    pub fn bind_plan(
        &self,
        plan: &ExecutionPlan,
        runtime: Arc<R>,
        request_id: RequestIdentity,
    ) -> Result<Arc<PlanRuntimeResources<R>>, VNextError> {
        let gate = self
            .inner
            .gate
            .read()
            .map_err(|_| invalid_resource("shared weight lifecycle is poisoned"))?;
        if *gate {
            return Err(invalid_resource("shared weights are closing"));
        }
        self.validate_compatible(plan, &runtime)?;
        let base = plan.prepare_plan_provisioning(runtime, request_id, true)?;
        let (lifecycle_tx, _) = watch::channel(PLAN_RUNTIME_OPEN);
        let resources = Arc::new(PlanRuntimeResources {
            lifecycle: RwLock::new(()),
            phase: AtomicU8::new(PLAN_RUNTIME_OPEN),
            lifecycle_tx,
            maintenance_controller: base.maintenance_controller,
            dynamic_pools: base.dynamic_pools,
            static_resources: PlanRuntimeStatic::Shared {
                binding: base.binding,
                weights: self.clone(),
            },
            runtime: base.runtime,
            deferred_cleanup_domain: new_deferred_device_cleanup_domain(),
        });
        drop(gate);
        Ok(resources)
    }

    /// Forbids subsequent attachments. Referenced retains this handle for retry;
    /// existing plans and pending completions continue to own their leases.
    pub fn close(
        self,
    ) -> Result<SharedComponentWeightsCloseOutcome<R>, SharedComponentWeightsCloseFailure<R>> {
        {
            let mut gate = self
                .inner
                .gate
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *gate = true;
        }
        match Arc::try_unwrap(self.inner) {
            Err(inner) => Ok(SharedComponentWeightsCloseOutcome::Referenced {
                weights: Self { inner },
            }),
            Ok(owner) => {
                release_shared_owner(owner).map(SharedComponentWeightsCloseOutcome::Closed)
            }
        }
    }

    pub(super) fn lease(&self) -> &StaticProvisioningLease<R> {
        self.inner
            .resources
            .lease
            .as_ref()
            .expect("live shared weights own their lease")
    }

    // A plan is relinquishing its own lease, not closing admission on the cache.
    // into_inner guarantees one caller gets the final owner even if multiple
    // distinct plan roots finish simultaneously.
    pub(super) fn release_attachment(self) -> Result<usize, SharedComponentWeightsCloseFailure<R>> {
        match Arc::into_inner(self.inner) {
            Some(owner) => {
                release_shared_owner(owner).map(|receipt| receipt.released_static_resources)
            }
            None => Ok(0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SharedComponentWeightsCloseReceipt {
    evidence: SharedComponentWeightsEvidence,
    released_static_resources: usize,
}
impl SharedComponentWeightsCloseReceipt {
    pub fn evidence(&self) -> &SharedComponentWeightsEvidence {
        &self.evidence
    }
    pub fn released_static_resources(&self) -> usize {
        self.released_static_resources
    }
}

#[must_use = "referenced weights retain the original physical claim"]
pub enum SharedComponentWeightsCloseOutcome<R: DeviceRuntime> {
    Closed(SharedComponentWeightsCloseReceipt),
    Referenced { weights: SharedComponentWeights<R> },
}

#[must_use = "failed weight retirement retains original buffers and capacity"]
pub struct SharedComponentWeightsCloseFailure<R: DeviceRuntime> {
    pub(super) failure: FailureEnvelope,
    evidence: SharedComponentWeightsEvidence,
    pub(super) resources: PlanStaticResources<R>,
}
impl<R: DeviceRuntime> SharedComponentWeightsCloseFailure<R> {
    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }
    pub fn evidence(&self) -> &SharedComponentWeightsEvidence {
        &self.evidence
    }
    pub fn retry(mut self) -> Result<SharedComponentWeightsCloseReceipt, Self> {
        let count = self.resources.states.len();
        match self.resources.release_all() {
            Ok(_) => Ok(SharedComponentWeightsCloseReceipt {
                evidence: self.evidence,
                released_static_resources: count,
            }),
            Err(error) => {
                self.failure = error.into_failure();
                Err(self)
            }
        }
    }
}

fn release_shared_owner<R: DeviceRuntime>(
    mut owner: SharedWeightOwner<R>,
) -> Result<SharedComponentWeightsCloseReceipt, SharedComponentWeightsCloseFailure<R>> {
    let count = owner.resources.states.len();
    match owner.resources.release_all() {
        Ok(_) => Ok(SharedComponentWeightsCloseReceipt {
            evidence: owner.evidence,
            released_static_resources: count,
        }),
        Err(error) => Err(SharedComponentWeightsCloseFailure {
            failure: error.into_failure(),
            evidence: owner.evidence,
            resources: owner.resources,
        }),
    }
}

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    // Only the initialized transaction typestate calls this with its unique,
    // never-published root, after contract validation and successful handoff.
    pub(crate) fn publish_initialized_weights(
        mut root: Arc<Self>,
        contract: SharedWeightContract,
    ) -> (Arc<Self>, SharedComponentWeights<R>) {
        let unique = Arc::get_mut(&mut root).expect("fresh initialized root is unique");
        let binding = match &unique.static_resources {
            PlanRuntimeStatic::Static(resources) => resources.admission.clone(),
            _ => unreachable!("initialized shared weights require Stored static resources"),
        };
        let resources = match std::mem::replace(
            &mut unique.static_resources,
            PlanRuntimeStatic::NoStatic {
                binding: binding.clone(),
            },
        ) {
            PlanRuntimeStatic::Static(resources) => resources,
            _ => unreachable!("validated static resources"),
        };
        // These private serializable types contain no fallible custom serializers.
        let encoded = serde_json::to_vec(&(&contract, &resources.identity))
            .expect("sealed weight identity serializes");
        let evidence = SharedComponentWeightsEvidence {
            snapshot_fingerprint: format!("{:x}", Sha256::digest(encoded)),
            weight_plan_fingerprint: contract
                .weights
                .fingerprint()
                .expect("validated weight plan fingerprints"),
            source_identity: resources.identity.clone(),
            physical_bytes: resources.admission.plan_static_bytes(),
        };
        let weights = SharedComponentWeights {
            inner: Arc::new(SharedWeightOwner {
                gate: RwLock::new(false),
                contract,
                resources,
                evidence,
            }),
        };
        unique.static_resources = PlanRuntimeStatic::Shared {
            binding,
            weights: weights.clone(),
        };
        (root, weights)
    }
}
