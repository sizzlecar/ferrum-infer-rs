use super::*;

/// Uses a real capability-closed ExecutionPlan and the production static
/// provisioning adapter. Only the device is the existing CPU test runtime.
pub(in super::super) struct RestoreHarness {
    pub(in super::super) fixture: checkpoint_fixture::Fixture,
    pub(in super::super) root: Arc<PlanRuntimeResources<TestRuntime>>,
    pub(in super::super) runtime: Arc<TestRuntime>,
    pub(in super::super) session: Arc<SequenceSession<TestRuntime>>,
}

impl RestoreHarness {
    pub(in super::super) fn new(spec: checkpoint_fixture::Spec) -> Self {
        let fixture = checkpoint_fixture::Fixture::build(spec).unwrap();
        let descriptor = fixture.catalog.device().clone();
        let mut runtime = TestRuntime::new(
            descriptor.id.clone(),
            descriptor.total_memory_bytes,
            descriptor.dynamic_storage_profiles.clone(),
        );
        runtime.descriptor = descriptor;
        let runtime = Arc::new(runtime);
        let provisioning = fixture
            .plan
            .provision_static(
                Arc::clone(&runtime),
                RequestIdentity::new("restore-fixture").unwrap(),
            )
            .unwrap()
            .into_provisioning();
        let root = match provisioning {
            StaticProvisioning::Required(permit) => {
                let identity = ResourceTransactionIdentity::for_admission(
                    permit.binding(),
                    RunId::new("restore-fixture-run").unwrap(),
                    TransactionId::new("restore-fixture-transaction").unwrap(),
                );
                let driver = RuntimeResourceDriver::new(Arc::clone(&runtime)).unwrap();
                let committed = ResourceTransaction::begin(driver, identity, permit)
                    .unwrap()
                    .reserve()
                    .unwrap()
                    .commit()
                    .unwrap_or_else(|_| panic!("fixture static commit failed"));
                match committed.into_plan_runtime() {
                    Ok(root) => root,
                    Err(failure) => panic!("fixture static handoff failed: {}", failure.error()),
                }
            }
            StaticProvisioning::NoStatic(no_static) => no_static.into_plan_runtime(),
        };
        for pool in fixture.plan.payload().memory().dynamic_pools() {
            root.maintenance_controller
                .grow_pool(pool.pool_id(), pool.provisioning().maximum_resident_bytes())
                .unwrap();
        }
        let session = admitted_sequence_with_ceiling(&root, "restore-target", 16)
            .open_session()
            .unwrap();
        Self {
            fixture,
            root,
            runtime,
            session,
        }
    }

    pub(in super::super) fn new_session(&self, suffix: &str) -> Arc<SequenceSession<TestRuntime>> {
        admitted_sequence_with_ceiling(&self.root, suffix, 16)
            .open_session()
            .unwrap()
    }

    pub(in super::super) fn extend(&self, tokens: usize) {
        match self
            .session
            .try_ensure_backing_covers(
                SequenceResourceExtensionRequest::new(
                    work(tokens),
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
            )
            .unwrap()
        {
            SequenceResourceExtensionDecision::Extended(_)
            | SequenceResourceExtensionDecision::Current(_) => {}
            _ => panic!("resident fixture state must extend while idle"),
        }
    }

    pub(in super::super) fn reserve(&self) -> PreparedSequenceStateTransfer<TestRuntime> {
        reserve_restore(&self.session)
    }

    pub(in super::super) fn close(self) {
        let Self {
            fixture,
            root,
            runtime,
            session,
        } = self;
        session.try_abort_if_quiescent().unwrap();
        drop(session);
        close_dynamic_test_root(root);
        drop(runtime);
        drop(fixture);
    }
}

pub(in super::super) fn reserve_restore(
    session: &Arc<SequenceSession<TestRuntime>>,
) -> PreparedSequenceStateTransfer<TestRuntime> {
    match session
        .try_prepare_state_transfer(
            SequenceStateTransferKind::RestoreWrite,
            session.resources().backing_generation().unwrap(),
        )
        .unwrap()
    {
        SequenceStateTransferPreparation::Prepared(guard) => guard,
        _ => panic!("idle fixture must reserve its fresh target"),
    }
}
