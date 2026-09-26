//! Real core-owned resources and CUDA submission. The tiny fixture provider
//! runs the installed scale kernel and counts actual enqueue calls; it cannot
//! construct a DeviceCommandBatch or mint completion/rollback receipts.
use super::*;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::execution_cost::{
    CoreReadbackRoute, GuardedNotSubmittedReason, HostSubmissionRejection,
};
use ferrum_interfaces::vnext::*;
use half::f16;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

mod cold_graph;
mod configured_eager;
mod family;
mod fixture;
mod library_cost;
mod warm_graph;
use fixture::Fixture;

fn id<T: TryFrom<String>>(value: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.to_owned()).unwrap()
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn span() -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&[1], 0..1).unwrap()
}

#[test]
fn cuda_guard_fixture_family_prepares_through_product_configuration_contract() {
    let family = TypedFamilyRegistration::new(family::Family::default());
    family
        .prepare_with_profile(&serde_json::json!({"width": 4}), &id("fixture.f16"))
        .unwrap();
    assert!(family
        .prepare_with_profile(&serde_json::json!(4), &id("fixture.f16"))
        .is_err());
    assert!(family
        .prepare_with_profile(&serde_json::json!({"width": 5}), &id("fixture.f16"))
        .is_err());
}

#[test]
fn cuda_guard_fixture_binding_has_its_own_declared_operation_contract() {
    let standard = family::scale_contract(false);
    let binding = family::scale_contract(true);
    assert_eq!(standard.descriptor().id, id(CONSTANT_SCALE_OPERATION_ID));
    assert_eq!(
        standard.descriptor().resources.binding,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(
        binding.descriptor().resources.binding,
        ResourcePresenceRequirement::Required
    );
    assert_ne!(standard.descriptor().id, binding.descriptor().id);
    assert_ne!(
        standard.descriptor().fingerprint().unwrap(),
        binding.descriptor().fingerprint().unwrap()
    );
    binding
        .validate_signature(
            &standard.descriptor().inputs,
            &standard.descriptor().outputs,
        )
        .unwrap();
    let family = family::Family::with_program_binding();
    let config = family
        .parse_config(&serde_json::json!({"width": 4}))
        .unwrap();
    let profiles = family.numerical_profiles(&config).unwrap();
    // Typed family preparation independently checks the numerical and semantic
    // operation identities; the actual CUDA test checks provider resource fit.
    let _ = profiles;
    TypedFamilyRegistration::new(family)
        .prepare_with_profile(&serde_json::json!({"width": 4}), &id("fixture.f16"))
        .unwrap();
}

struct Guard {
    reject: bool,
    program_binding: bool,
    calls: AtomicU64,
    enqueues: Arc<AtomicU64>,
}
impl Guard {
    fn new(f: &Fixture, reject: bool) -> Self {
        Self {
            reject,
            program_binding: f.has_program_binding(),
            calls: AtomicU64::new(0),
            enqueues: Arc::clone(&f.enqueues),
        }
    }
}
impl PreparedWaveSubmissionGuard for Guard {
    fn check(
        &self,
        actual: &DeviceSubmissionAttribution,
        readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        assert_eq!(
            self.enqueues.load(Ordering::Relaxed),
            0,
            "actual CUDA compute must not have entered enqueue"
        );
        assert!(actual.graph_evidence().unwrap().proves_unconfigured_eager());
        assert!(actual.replayed_segments().is_empty());
        assert!(actual.commands().iter().all(|command| {
            command.execution_path() == DeviceExecutionPath::Eager
                && command.reusable_graph_node_count().is_none()
        }));
        let bindings = actual
            .commands()
            .iter()
            .filter(|command| {
                command.native_op_id() == core_cost_route::PROGRAM_BINDING_NATIVE_OPERATION
            })
            .collect::<Vec<_>>();
        if self.program_binding {
            let [binding] = bindings.as_slice() else {
                panic!("one real coalesced program-binding upload must reach the final guard")
            };
            assert_eq!(binding.command_phase(), DeviceCommandPhase::DynamicBinding);
            assert_eq!(binding.transfer_command_count(), 1);
            assert_eq!(binding.compute_dispatch_count(), 0);
            assert_eq!(binding.participant_count(), 1);
        } else {
            assert!(bindings.is_empty());
        }
        assert_eq!(readback, CoreReadbackRoute::SubmissionStaged);
        assert_eq!(
            actual
                .commands()
                .iter()
                .map(|c| c.compute_dispatch_count())
                .sum::<u64>(),
            1
        );
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.reject {
            Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::WitnessExpired,
            ))
        } else {
            Ok(())
        }
    }
}
fn accepted(
    value: GuardedWaveSubmissionOutcome<CudaDeviceRuntime>,
) -> CompletionHandle<CudaDeviceRuntime> {
    match value {
        GuardedWaveSubmissionOutcome::Dispatch(Ok(value)) => value.into_parts().0,
        GuardedWaveSubmissionOutcome::Dispatch(Err(error)) => panic!("CUDA dispatch: {error}"),
        GuardedWaveSubmissionOutcome::NotSubmitted(_) => panic!("accepting guard rejected"),
    }
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_core_rejection_rolls_back_and_same_lane_submits() {
    rejected_then_retry(Fixture::new());
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_core_program_binding_stays_eager_and_retries_after_rejection() {
    rejected_then_retry(Fixture::new_program_binding());
}

struct NoHostTiming;
impl DeviceSubmissionTimingSink for NoHostTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        unreachable!("disabled fixture timing cannot record")
    }
}
impl SubmissionWaveDispatchTimingSink for NoHostTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        unreachable!("disabled fixture timing cannot record")
    }
}

#[derive(Default)]
struct HostTiming {
    backend_stages: AtomicU64,
    completion_arms: AtomicU64,
}
impl DeviceSubmissionTimingSink for HostTiming {
    const ENABLED: bool = true;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        self.backend_stages.fetch_add(1, Ordering::Relaxed);
    }
}
impl SubmissionWaveDispatchTimingSink for HostTiming {
    fn record(&self, stage: SubmissionWaveDispatchStage, _: std::time::Duration) {
        if stage == SubmissionWaveDispatchStage::CompletionArm {
            self.completion_arms.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_host_timing_preserves_eager_rejection_retry_and_output() {
    for program_binding in [false, true] {
        let fixture = if program_binding {
            Fixture::new_program_binding()
        } else {
            Fixture::new()
        };
        let timing = HostTiming::default();
        rejected_then_retry_with_timing(fixture, &timing);
        assert!(timing.backend_stages.load(Ordering::Relaxed) > 0);
        // Rejection never creates a completion; only the accepted retry arms one.
        assert_eq!(timing.completion_arms.load(Ordering::Relaxed), 1);
    }
}

fn rejected_then_retry(f: Fixture) {
    rejected_then_retry_with_timing(f, &NoHostTiming);
}

fn rejected_then_retry_with_timing<S: SubmissionWaveDispatchTimingSink>(f: Fixture, timing: &S) {
    let lane_id = f.lane.id();
    let (step, wave) = f.prepare();
    let guard = Guard::new(&f, true);
    let pending = match f.dispatch_with_timing(wave, &guard, timing) {
        GuardedWaveSubmissionOutcome::NotSubmitted(value) => value,
        _ => panic!("expected actual backend rejection"),
    };
    assert_eq!(guard.calls.load(Ordering::Relaxed), 1);
    assert_eq!(f.encoded.load(Ordering::Relaxed), 1);
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 0);
    assert_eq!(f.lane.in_flight_count(), 0);
    assert_eq!(f.lane.cost_readback_available_bytes(), Some(8));
    assert_eq!(f.reaper.retained_count(), 0);
    let receipt = pending
        .reconcile_step(step)
        .unwrap_or_else(|(e, _)| panic!("exact step rollback: {e}"));
    assert_eq!(
        receipt.reason(),
        GuardedNotSubmittedReason::HostRejected(HostSubmissionRejection::WitnessExpired)
    );
    let (step, wave) = f.prepare();
    let guard = Guard::new(&f, false);
    let handle = accepted(f.dispatch_with_timing(wave, &guard, timing));
    assert_eq!(f.lane.id(), lane_id);
    assert_eq!(guard.calls.load(Ordering::Relaxed), 1);
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 1);
    let receipt = match handle.wait_with_readbacks(f.readback()).unwrap() {
        CompletionReadbackBatchObservation::Terminal(value) => value,
        _ => panic!("CUDA readback did not terminate"),
    };
    assert!(matches!(
        receipt.completion().submission_timing(),
        DeviceTimingMeasurement::NotRequested
    ));
    assert!(receipt.readback_timings().is_none());
    let CompletionReadbackDisposition::Succeeded(output) = &receipt.dispositions()[0] else {
        panic!("scale output failed")
    };
    let expected = [2.0_f32, -4.0, 1.0, 8.0]
        .into_iter()
        .flat_map(|v| f16::from_f32(v).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(output.bytes(), expected);
    drop((receipt, handle));
    step.try_retire_normal().unwrap();
    assert_eq!(f.lane.cost_readback_available_bytes(), Some(8));
    f.close(true);
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_core_configured_graph_is_rejected_before_host_gate() {
    let f = Fixture::new();
    f.lane
        .configure_reusable_executables(DeviceReusableExecutionPlan::on_demand(1).unwrap())
        .unwrap();
    let (step, wave) = f.prepare();
    let guard = Guard::new(&f, false);
    let pending = match f.dispatch(wave, &guard) {
        GuardedWaveSubmissionOutcome::NotSubmitted(value) => value,
        _ => panic!("configured graph cannot grant eager guard authority"),
    };
    assert_eq!(guard.calls.load(Ordering::Relaxed), 0);
    assert_eq!(f.encoded.load(Ordering::Relaxed), 1);
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 0);
    assert_eq!(f.lane.in_flight_count(), 0);
    assert_eq!(f.lane.cost_readback_available_bytes(), Some(8));
    let receipt = pending
        .reconcile_step(step)
        .unwrap_or_else(|(e, _)| panic!("graph rejection rollback: {e}"));
    assert_eq!(
        receipt.reason(),
        GuardedNotSubmittedReason::AttributionUnavailable
    );
    f.close(false);
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_core_dropped_handle_keeps_flight_until_terminal_reaped() {
    let f = Fixture::new();
    let (step, wave) = f.prepare();
    let handle = accepted(f.dispatch(wave, &Guard::new(&f, false)));
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 1);
    let step = step
        .try_rollback_unsubmitted()
        .expect_err("submitted work cannot roll back")
        .into_step();
    drop(handle);
    assert_eq!(f.reaper.retained_count(), 1);
    assert_eq!(f.lane.in_flight_count(), 1);
    assert!(f.session.try_abort_if_quiescent().is_err());
    // Device completion alone does not settle core ownership. The reaper must
    // observe the actual native fence before the Step can retire.
    f.runtime.context.synchronize().unwrap();
    f.reaper.poll_bounded(1).unwrap();
    assert_eq!(f.reaper.retained_count(), 0);
    assert_eq!(f.lane.in_flight_count(), 0);
    step.try_retire_normal().unwrap();
    f.close(true);
}
