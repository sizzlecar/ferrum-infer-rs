//! Core ownership regression draft; no model/kernel performance assertion.
use super::*;
use crate::vnext::{
    AliasPolicy, CompletionReadbackBatchRequest, CompletionReadbackRequest, DynamicStorageContract,
    ElementType, MemoryPlan, ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueBinding,
    ResolvedValueRole, ResolvedValueStorage, SubmittedWavePredecessor,
};

const ROW_BYTES: u64 = 64;
const BINDING_ROW_BYTES: u64 = 16;
const PARENT_ROWS: u32 = 3;
const INSTANCE_CEILING: u32 = 8;

struct SubmittedParent {
    step: Arc<StepResourceLease<TestRuntime>>,
    handle: CompletionHandle<TestRuntime>,
    claims: ParentClaims,
}

struct ParentClaims {
    step: std::sync::Weak<ClaimedBackingTransaction>,
    wave: std::sync::Weak<ClaimedSubmissionWaveBacking>,
}

struct SuccessorHoldFixture {
    harness: Harness,
    lane: Arc<ExecutionLane<TestRuntime>>,
    reaper: Arc<CompletionReaper<TestRuntime>>,
    sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
    batch: ExecutionBatchParticipants<TestRuntime>,
    bucket: ReusableExecutionBucketSpec,
    small_bucket: ReusableExecutionBucketSpec,
    output_resource: ResourceId,
    binding_resource: ResourceId,
    output_pool: DynamicBackingPoolId,
}

impl SuccessorHoldFixture {
    fn new(with_small_idle_slot: bool) -> Self {
        let profile = linear_profile();
        let node_id = NodeId::new("node/token-output").unwrap();
        let output_resource = ResourceId::new("resource/token-output").unwrap();
        let binding_resource = ResourceId::new("resource/token-binding").unwrap();
        // Reuse the existing descriptor wire-construction boundary. The demand,
        // element type, and output binding are typed; no descriptor is edited
        // after construction and no unknown token is invented.
        let compatibility = json!({
            "version": {"major": 1, "minor": 0},
            "profile": profile,
            "usage": "activations",
            "element_type": "u32",
            "logical_layout_fingerprint": "a".repeat(64),
            "alignment_bytes": 16
        });
        let output_pool: DynamicBackingPoolId = serde_json::from_value(json!(format!(
            "dynamic-pool/sha256/{:x}",
            Sha256::digest(serde_json::to_vec(&canonical_json(compatibility)).unwrap())
        )))
        .unwrap();
        let output: DynamicResourceDescriptor = serde_json::from_value(json!({
            "base_resource_id": output_resource,
            "demand": DynamicResourceDemand::actual_sequences(ROW_BYTES, INSTANCE_CEILING).unwrap(),
            "alignment_bytes": 16,
            "usage": "activations",
            "element_type": "u32",
            "lifetime": "step",
            "kind": "value",
            "storage": {
                "profile": profile,
                "logical_layout_fingerprint": "a".repeat(64)
            },
            "pool_id": output_pool,
            "initialization": StateInitialization::None,
            "theoretical_maximum_instances": INSTANCE_CEILING
        }))
        .unwrap();
        let binding = DynamicResourceDescriptor::resource_test_binding(
            binding_resource.clone(),
            DynamicResourceDemand::actual_sequences(BINDING_ROW_BYTES, INSTANCE_CEILING).unwrap(),
            16,
            node_id.clone(),
            DynamicStorageContract::resource_test_contract(profile, "b".repeat(64)).unwrap(),
            INSTANCE_CEILING,
        )
        .unwrap();
        let output_value = ResolvedValueBinding::new(
            ProgramValueId::new("value/token-output").unwrap(),
            ResolvedValueRole::Output,
            0,
            ResolvedTensorSpec::new(
                vec![ROW_BYTES / 4],
                ElementType::U32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            TensorAccess::Write,
            AliasPolicy::NoAlias,
            BufferUsage::Activations,
            None,
            ResolvedValueStorage::single(output_resource.clone(), 0, ROW_BYTES, ElementType::U32)
                .unwrap(),
        )
        .unwrap();
        let node = PlanNode::resource_test_node_with_binding_and_values(
            node_id,
            binding_resource.clone(),
            vec![output_value],
        );
        let nodes: Arc<[PlanNode]> = Arc::from(vec![node]);
        let mut descriptors = vec![output, binding];
        descriptors.sort_by(|a, b| a.base_resource_id().cmp(b.base_resource_id()));
        let target_limit = 2 * u64::from(PARENT_ROWS) * ROW_BYTES;
        let pools = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, target_limit).unwrap();
        let total_budget = pools
            .iter()
            .map(|pool| pool.provisioning().maximum_resident_bytes())
            .sum();
        let bucket_for = |rows: u32| {
            ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new("test.successor-backing-hold").unwrap(),
                ReusableExecutionCapacity::new(rows, u64::from(rows), u64::from(rows)).unwrap(),
            )
            .unwrap()
        };
        let small_bucket = bucket_for(1);
        let bucket = bucket_for(PARENT_ROWS);
        let resolved = [&small_bucket, &bucket]
            .into_iter()
            .map(|bucket| {
                let rows = u64::from(bucket.capacity().maximum_sequences());
                let mut budgets = descriptors
                    .iter()
                    .map(|descriptor| {
                        let (step, invocation) =
                            if descriptor.lifetime() == AllocationLifetime::Step {
                                (ROW_BYTES * rows, 0)
                            } else {
                                (0, BINDING_ROW_BYTES * rows)
                            };
                        ReusablePoolWorkspaceBudget::new(
                            descriptor.pool_id().clone(),
                            step,
                            invocation,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                budgets.sort_by(|a, b| a.pool_id().cmp(b.pool_id()));
                ResolvedReusableExecutionBucket::new(bucket.clone(), budgets).unwrap()
            })
            .collect();
        let reusable = ReusableExecutionMemoryPlan::new(1, 2, resolved).unwrap();
        let catalog = PoolCatalog {
            pools,
            descriptors,
            pool_id: output_pool.clone(),
            profile,
        };
        let runtime = new_runtime(&catalog, total_budget);
        let harness = harness_with_nodes_and_reusable(
            runtime,
            catalog,
            total_budget,
            false,
            nodes,
            Some(reusable),
        );
        for pool_id in &harness.pool_ids {
            let pool = &harness.root.dynamic_pools.pools[pool_id];
            let bytes = if pool_id == &output_pool {
                // A occupies three units. With no cold slot, the second frame
                // cannot fit the remaining two-unit region plus a separate
                // one-unit growth chunk. With C, reclaiming C joins three units.
                target_limit - if with_small_idle_slot { 0 } else { ROW_BYTES }
            } else {
                pool.domain.pool.provisioning().maximum_resident_bytes()
            };
            harness
                .root
                .maintenance_controller
                .grow_pool(pool_id, bytes)
                .unwrap();
        }
        let lane = harness.root.create_execution_lane().unwrap();
        let sessions = (0..PARENT_ROWS)
            .map(|row| {
                admitted_sequence_with_ceiling(&harness.root, &format!("held-parent-{row}"), 4)
                    .open_session()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
        Self {
            harness,
            lane,
            reaper: CompletionReaper::new(),
            sessions,
            batch,
            bucket,
            small_bucket,
            output_resource,
            binding_resource,
            output_pool,
        }
    }

    fn request(&self, full_tokens: usize) -> StepResourceAdmissionRequest {
        let spans = (0..PARENT_ROWS)
            .map(|row| {
                let tokens = (0..full_tokens)
                    .map(|i| 11 + row * 7 + i as u32)
                    .collect::<Vec<_>>();
                TokenSpanWork::from_token_ids_with_fit(&tokens, full_tokens - 1..full_tokens, 4)
                    .unwrap()
            })
            .collect();
        StepResourceAdmissionRequest::new(
            self.batch.bind_work_shape(spans).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
        .with_reusable_execution_bucket(self.bucket.bucket_id().clone())
    }

    fn submit_parent(&self, full_tokens: usize) -> SubmittedParent {
        let StepResourceAdmissionDecision::Admitted(step) = self
            .batch
            .try_begin_step(self.request(full_tokens), &self.lane)
            .unwrap()
        else {
            panic!("the current parent must fit");
        };
        let wave = prepared_wave(&step);
        let wave_claim = wave.shared_backing_claim();
        assert!(step
            .backing_slices()
            .iter()
            .any(|a| a.resource_id() == &self.output_resource));
        assert!(wave_claim
            .backing_slices()
            .iter()
            .any(|a| a.resource_id() == &self.binding_resource));
        let claims = ParentClaims {
            step: Arc::downgrade(&step.claimed_backing),
            wave: Arc::downgrade(&wave_claim),
        };
        drop(wave_claim);
        self.harness
            .runtime
            .set_fence_behavior(TestFenceBehavior::Pending);
        let handle = submit_fixture_wave_through_reaper(
            &self.harness.root,
            &self.sessions,
            &self.lane,
            wave,
            &self.reaper,
        );
        SubmittedParent {
            step,
            handle,
            claims,
        }
    }

    fn cache_smaller_step(&self) {
        let session = admitted_sequence_with_ceiling(&self.harness.root, "cold-small", 4)
            .open_session()
            .unwrap();
        let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
        let request = StepResourceAdmissionRequest::new(
            batch
                .bind_work_shape(vec![
                    TokenSpanWork::from_token_ids_with_fit(&[97], 0..1, 4).unwrap()
                ])
                .unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
        .with_reusable_execution_bucket(self.small_bucket.bucket_id().clone());
        let StepResourceAdmissionDecision::Admitted(step) =
            batch.try_begin_step(request, &self.lane).unwrap()
        else {
            panic!("one-row cold Step fits beside three-row parent");
        };
        step.try_rollback_unsubmitted().unwrap();
        session.try_abort_if_quiescent().unwrap();
    }

    fn sources(&self) -> CompletionReadbackBatchRequest {
        CompletionReadbackBatchRequest::new(
            (0..PARENT_ROWS)
                .map(|row| {
                    CompletionReadbackRequest::new(
                        NodeId::new("node/token-output").unwrap(),
                        row,
                        self.output_resource.clone(),
                        0,
                        HostTransferLayout::new(ElementType::U32, 1).unwrap(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn successor_request(
        &self,
        predecessor: &SubmittedWavePredecessor<TestRuntime>,
    ) -> StepResourceAdmissionRequest {
        StepResourceAdmissionRequest::new(
            Arc::new(predecessor.bind_next_token_work(self.sources()).unwrap()),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
        .with_reusable_execution_bucket(self.bucket.bucket_id().clone())
    }

    fn deferred_child(
        &self,
        parent: &SubmittedParent,
    ) -> StepAdmissionBackingDeferral<TestRuntime> {
        let predecessor = Arc::new(parent.handle.take_submitted_predecessor().unwrap());
        let request = self.successor_request(&predecessor);
        let decision = self
            .batch
            .try_begin_successor_step(request, &self.lane, predecessor)
            .unwrap();
        let StepResourceAdmissionDecision::BackingDeferred(deferred) = decision else {
            panic!("the second three-row contiguous claim must encounter real pool pressure");
        };
        assert!(deferred
            .evidence()
            .blockers()
            .iter()
            .any(|blocker| blocker.pool_id() == &self.output_pool));
        assert!(matches!(
            parent.handle.poll().unwrap(),
            CompletionObservation::Pending
        ));
        // Live parent forbids trim; growth may consume the remaining one unit,
        // but cannot combine that chunk with the parent's partially free chunk.
        match deferred.maintain().unwrap() {
            DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
            | DynamicDeferredMaintenanceOutcome::Maintained(_) => {}
            other => panic!("live parent maintenance must not evict its slot: {other:?}"),
        }
        assert_eq!(self.harness.runtime.reusable_trim_calls(), 0);
        deferred
    }

    fn retire_parent(&self, parent: SubmittedParent) -> ParentClaims {
        self.harness
            .runtime
            .set_fence_behavior(TestFenceBehavior::Succeeded);
        assert!(matches!(
            parent.handle.wait().unwrap(),
            CompletionObservation::Terminal(_)
        ));
        parent.step.try_retire_normal().unwrap();
        assert_eq!(self.lane.in_flight_count(), 0);
        parent.claims
    }

    fn assert_slots_held(&self, claims: &ParentClaims) {
        let step = claims
            .step
            .upgrade()
            .expect("deferred successor must retain the Step slot lease");
        let wave = claims
            .wave
            .upgrade()
            .expect("deferred successor must retain the Invocation slot lease");
        assert!(step
            .backing_slices()
            .iter()
            .any(|a| a.resource_id() == &self.output_resource));
        assert!(wave
            .backing_slices()
            .iter()
            .any(|a| a.resource_id() == &self.binding_resource));
    }

    fn close(self, aborted: bool) {
        for session in &self.sessions {
            if aborted {
                session.try_abort().unwrap();
            } else {
                session.try_complete().unwrap();
            }
        }
        drop(self.batch);
        drop(self.sessions);
        drop(self.reaper);
        drop(self.lane);
        close_dynamic_test_root(self.harness.root);
    }
}

#[test]
fn successor_deferral_retains_parent_step_and_invocation_until_dropped() {
    let fixture = SuccessorHoldFixture::new(false);
    fixture
        .harness
        .runtime
        .set_reusable_catalog_lifetime(ReusableExecutionCatalogLifetime::OnDemandBounded);
    fixture.harness.runtime.set_reusable_resident_executables(1);
    let epoch = fixture.lane.reusable_execution_epoch();
    let parent = fixture.submit_parent(1);
    let deferred = fixture.deferred_child(&parent);
    let slots = fixture.retire_parent(parent);
    fixture.assert_slots_held(&slots);
    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::WaitForRelease { .. }
    ));
    fixture.assert_slots_held(&slots);
    assert_eq!(fixture.harness.runtime.reusable_trim_calls(), 0);
    assert_eq!(fixture.lane.reusable_execution_epoch(), epoch);
    drop(deferred);
    assert!(slots.step.upgrade().is_none());
    assert!(slots.wave.upgrade().is_none());
    let StepResourceAdmissionDecision::Admitted(next) = fixture
        .batch
        .try_begin_step(fixture.request(2), &fixture.lane)
        .unwrap()
    else {
        panic!("dropping the optional deferral must make the original slot reusable");
    };
    next.try_rollback_unsubmitted().unwrap();
    fixture.close(false);
}

#[test]
fn successor_deferral_reclaims_smaller_idle_slot_and_next_pair_really_fits() {
    let fixture = SuccessorHoldFixture::new(true);
    fixture
        .harness
        .runtime
        .set_reusable_catalog_lifetime(ReusableExecutionCatalogLifetime::OnDemandBounded);
    fixture.harness.runtime.set_reusable_resident_executables(1);
    let parent = fixture.submit_parent(1);
    fixture.cache_smaller_step();
    let deferred = fixture.deferred_child(&parent);
    let slots = fixture.retire_parent(parent);
    fixture.assert_slots_held(&slots);
    assert!(matches!(
        deferred.maintain().unwrap(),
        DynamicDeferredMaintenanceOutcome::RetryAdmission { .. }
    ));
    fixture.assert_slots_held(&slots);
    drop(deferred);

    // A new parent reuses its still-cached first physical frame; another real
    // successor must now acquire B rather than recycling A in a churn loop.
    let parent = fixture.submit_parent(2);
    let predecessor = Arc::new(parent.handle.take_submitted_predecessor().unwrap());
    let request = fixture.successor_request(&predecessor);
    let StepResourceAdmissionDecision::Admitted(child) = fixture
        .batch
        .try_begin_successor_step(request, &fixture.lane, predecessor)
        .unwrap()
    else {
        panic!("useful cold-slot reclamation must make the next simultaneous pair fit");
    };
    let child_wave = prepared_wave(&child);
    for left in parent.step.backing_slices() {
        for right in child.backing_slices() {
            if left.evidence().pool_id() != right.evidence().pool_id() {
                continue;
            }
            for a in left.evidence().segments() {
                for b in right.evidence().segments() {
                    assert!(
                        a.chunk() != b.chunk()
                            || a.offset_bytes() + a.length_bytes() <= b.offset_bytes()
                            || b.offset_bytes() + b.length_bytes() <= a.offset_bytes()
                    );
                }
            }
        }
    }
    drop(child_wave);
    fixture.retire_parent(parent);
    // A prepared Invocation registry is no longer pristine, even without submit.
    child.try_abort().unwrap();
    fixture.close(true);
}
