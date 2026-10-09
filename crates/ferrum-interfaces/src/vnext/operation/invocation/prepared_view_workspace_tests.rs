//! Real admitted waves exercise capacity reuse without reusing authority.
use super::*;
use crate::vnext::resource::prepared_view_workspace_test_support::*;
use ferrum_types::InvocationPreparationStrategy;
use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use vnext_device_operation_wave_contract::{
    setup_with_fixture, teardown, test_reusable_program, wave_active_bindings,
};

fn workspace_identity(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    active: &[TrustedActiveSequenceBinding],
) -> BatchOperationIdentity {
    let lane = wave.step_resources().execution_lane();
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, lane).unwrap();
    OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
        &topology,
        active.iter(),
        wave,
        lane,
        InvocationPreparationStrategy::PreparedViewWorkspace,
    )
    .unwrap()
}

fn visit<'wave, 'binding, I, T>(
    workspace: &mut WaveInvocationWorkspace<'wave, TestBuffer>,
    fixture: &'wave Fixture,
    wave: &'wave PreparedStepSubmissionWave<TestRuntime>,
    identity: &'wave BatchOperationIdentity,
    index: usize,
    active: I,
    encode: impl for<'node> FnOnce(BatchedOperationInvocation<'node, TestBuffer>) -> T,
) -> Result<T, VNextError>
where
    I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
{
    let provider = fixture
        .registry
        .bind(&fixture.resolved, wave.nodes()[index].node_id())
        .unwrap();
    workspace.with_reusable_wave_node(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        provider.dispatch(),
        identity,
        identity.materialize_node(index).unwrap(),
        wave,
        index,
        active,
        encode,
    )
}

#[test]
fn prepared_view_workspace_matches_owned_views_across_nodes_and_unequal_windows() {
    with_live_wave_spans(vec![1, 3], |fixture, wave, _, active| {
        let identity = workspace_identity(fixture, wave, active);
        let mut workspace = WaveInvocationWorkspace::new();
        let owners = workspace_test_chunk_owners(&fixture.plan_resources);
        let mut scopes = Vec::new();
        for index in 0..wave.node_count() {
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[index].node_id())
                .unwrap();
            let reference = BatchedOperationInvocation::from_reusable_wave_node(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                identity.materialize_node(index).unwrap(),
                wave,
                index,
                active.iter(),
            )
            .unwrap();
            let expected = snapshot(&reference);
            drop(reference);
            let (actual, scope) = visit(
                &mut workspace,
                fixture,
                wave,
                &identity,
                index,
                active.iter(),
                |invocation| {
                    assert_ne!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
                    (
                        snapshot(&invocation),
                        Arc::clone(&invocation.retained_dependency_scope),
                    )
                },
            )
            .unwrap();
            assert_eq!(actual, expected);
            assert!(scopes.iter().all(|prior| !Arc::ptr_eq(prior, &scope)));
            scopes.push(scope);
            assert!(workspace.is_empty());
            assert!(workspace.view_capacity() > 0);
            assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
        }
        assert_eq!(identity.preparation_snapshot().parts_materialized, 0);
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    });
}

#[test]
fn prepared_view_workspace_rechecks_hot_runtime_participants_and_pool_poison() {
    with_live_wave(2, |fixture, wave, _, active| {
        let identity = workspace_identity(fixture, wave, active);
        let mut workspace = WaveInvocationWorkspace::new();
        visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            0,
            active.iter(),
            |_| (),
        )
        .unwrap();
        let owners = workspace_test_chunk_owners(&fixture.plan_resources);
        let callback = Cell::new(false);
        let mut participant = 0;
        let changing = active.iter().inspect(|_| {
            participant += 1;
            if participant == 2 {
                fixture
                    .runtime_trace
                    .lock()
                    .unwrap()
                    .tamper_buffer_descriptor = true;
            }
        });
        let result = visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            1,
            changing,
            |_| callback.set(true),
        );
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_buffer_descriptor = false;
        assert!(result.is_err());
        assert!(!callback.get());
        assert!(workspace.is_empty());
        assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);

        assert!(visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            1,
            active.iter().rev(),
            |_| ()
        )
        .is_err());
        let poison = RefCell::new(None);
        let mut participant = 0;
        let changing = active.iter().inspect(|_| {
            participant += 1;
            if participant == 2 {
                *poison.borrow_mut() = Some(workspace_test_poison_pools(&fixture.plan_resources));
            }
        });
        assert!(visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            1,
            changing,
            |_| ()
        )
        .is_err());
        drop(poison.into_inner());
        assert!(workspace.is_empty());
        assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
        visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            1,
            active.iter(),
            |_| (),
        )
        .unwrap();
        assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
    });
}

#[test]
fn prepared_view_workspace_clears_views_on_callback_error_and_unwind() {
    with_live_wave(2, |fixture, wave, _, active| {
        let identity = workspace_identity(fixture, wave, active);
        let mut workspace = WaveInvocationWorkspace::new();
        let owners = workspace_test_chunk_owners(&fixture.plan_resources);
        let error = visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            0,
            active.iter(),
            |_| Err::<(), _>(invalid_operation("provider-side test failure")),
        )
        .unwrap();
        assert!(error.is_err());
        assert!(workspace.is_empty());
        assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
        let unwind = catch_unwind(AssertUnwindSafe(|| {
            visit(
                &mut workspace,
                fixture,
                wave,
                &identity,
                0,
                active.iter(),
                |_| panic!("provider-side test unwind"),
            )
        }));
        assert!(unwind.is_err());
        assert!(workspace.is_empty());
        assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
        visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            1,
            active.iter(),
            |_| (),
        )
        .unwrap();
        assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
    });
}

#[test]
fn prepared_view_workspace_returned_retention_owns_real_backing_after_reset() {
    with_live_wave(2, |fixture, wave, _, active| {
        let identity = workspace_identity(fixture, wave, active);
        let mut workspace = WaveInvocationWorkspace::new();
        let owners = workspace_test_chunk_owners(&fixture.plan_resources);
        let retention = visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            0,
            active.iter(),
            |invocation| {
                let view = invocation.participants()[0]
                    .views()
                    .iter()
                    .find(|view| view.shared_backing().is_some())
                    .unwrap();
                view.translate(0, view.descriptor().size_bytes)
                    .unwrap()
                    .iter()
                    .next()
                    .unwrap()
                    .buffer_and_physical_range()
                    .2
            },
        )
        .unwrap();
        assert!(workspace.is_empty());
        assert_ne!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
        // Reusing capacity must neither retain its old views nor release the
        // independently returned owner of the actual admitted allocation.
        visit(
            &mut workspace,
            fixture,
            wave,
            &identity,
            1,
            active.iter(),
            |_| (),
        )
        .unwrap();
        assert_ne!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
        drop(retention);
        assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
    });
}

#[test]
fn prepared_view_workspace_distinguishes_same_node_retention_from_fresh_node_mapping() {
    with_live_wave(2, |fixture, wave, _, active| {
        with_live_wave(2, |donor, _, _, _| {
            let identity = workspace_identity(fixture, wave, active);
            let mut workspace = WaveInvocationWorkspace::new();
            let (pool_id, ordinal) = visit(
                &mut workspace,
                fixture,
                wave,
                &identity,
                0,
                active.iter(),
                |invocation| {
                    let backing = invocation.participants()[0]
                        .views()
                        .iter()
                        .find_map(|view| view.shared_backing())
                        .unwrap();
                    (
                        backing.slice().pool_id().clone(),
                        backing.slice().segments()[0].chunk_ordinal(),
                    )
                },
            )
            .unwrap();
            let owners = workspace_test_chunk_owners(&fixture.plan_resources);
            let replacement = RefCell::new(None);
            let mut participant = 0;
            let changing = active.iter().inspect(|_| {
                participant += 1;
                if participant == 2 {
                    *replacement.borrow_mut() = Some(workspace_test_replace_chunk_from_donor(
                        &fixture.plan_resources,
                        &donor.plan_resources,
                        &pool_id,
                        ordinal,
                    ));
                }
            });
            assert!(visit(
                &mut workspace,
                fixture,
                wave,
                &identity,
                0,
                changing,
                |_| ()
            )
            .is_err());
            drop(replacement.into_inner());
            assert!(workspace.is_empty());
            assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);

            // A later fresh node/construction has no old retained authority.
            // Its acceptance must match the original IP path, not add an
            // invented cross-node Arc-identity requirement.
            let replacement = workspace_test_replace_chunk_from_donor(
                &fixture.plan_resources,
                &donor.plan_resources,
                &pool_id,
                ordinal,
            );
            let current = workspace_test_chunk_owners(&fixture.plan_resources);
            assert_ne!(current, owners);
            let provider = fixture
                .registry
                .bind(&fixture.resolved, wave.nodes()[0].node_id())
                .unwrap();
            let reference = BatchedOperationInvocation::from_reusable_wave_node(
                fixture.runtime.as_ref(),
                &fixture.resolved,
                provider.dispatch(),
                &identity,
                identity.materialize_node(0).unwrap(),
                wave,
                0,
                active.iter(),
            )
            .unwrap();
            let expected = snapshot(&reference);
            drop(reference);
            let actual = visit(
                &mut workspace,
                fixture,
                wave,
                &identity,
                0,
                active.iter(),
                |invocation| snapshot(&invocation),
            )
            .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(
                workspace_test_chunk_owners(&fixture.plan_resources),
                current
            );
            drop(replacement);
            assert_eq!(workspace_test_chunk_owners(&fixture.plan_resources), owners);
        });
    });
}

struct WorkspaceOffTiming;
impl DeviceSubmissionTimingSink for WorkspaceOffTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("Off must not record device timing");
    }
}
impl SubmissionWaveDispatchTimingSink for WorkspaceOffTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        panic!("Off must not record dispatch timing");
    }
}
#[derive(Default)]
struct WorkspaceCapture(Mutex<Vec<InvocationPreparationStats>>);
impl InvocationPreparationSink for WorkspaceCapture {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.lock().unwrap().push(stats);
    }
}

#[test]
fn prepared_view_workspace_owned_replay_commands_complete_after_scoped_views_drop() {
    let (fixture, sequence, session, batch, step) = setup_with_fixture(
        fixture_with_provider_behavior(false, ProviderBehavior::ProgramBinding),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let reaper = CompletionReaper::new();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let identity = workspace_identity(&fixture, &wave, &active);
    let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
        providers.providers(),
        &fixture.resolved,
        &wave,
        &lane,
    )
    .unwrap()
    .unwrap();
    let node_count = u32::try_from(providers.len()).unwrap();
    let program = test_reusable_program(
        program_id,
        node_count,
        vec![],
        vec![DeviceReusableExecutionSegment::new(0, 0, node_count, node_count).unwrap()],
        (0..node_count).collect(),
        vec![],
    );
    let capture = WorkspaceCapture::default();
    let handle = OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        &program,
        SubmissionExecutionPolicy::determinism_replayed(0xa5),
        InvocationPreparationStrategy::PreparedViewWorkspace,
        &WorkspaceOffTiming,
        &capture,
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let (handle, attribution) = handle.into_parts();
    assert!(attribution.is_none());
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    let observations = capture.0.lock().unwrap();
    assert_eq!(observations.len(), 1);
    assert_eq!(observations[0].borrowed_view_nodes, u64::from(node_count));
    assert_eq!(observations[0].parts_materialized, 0);
    assert_eq!(
        fixture.runtime_trace.lock().unwrap().submitted_commands,
        vec![vec![
            TestCommand::CoalescedProgramBinding,
            TestCommand::ReusableExecution
        ]]
    );
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(observations);
    drop(handle);
    drop(providers);
    drop(active);
    drop(reaper);
    drop(lane);
    teardown(fixture, sequence, session, batch, step);
}
