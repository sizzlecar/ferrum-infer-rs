//! Backend custody fixture: real CUDA allocations, original module permit and
//! production SGEMM+broadcast command. This is not Core admission evidence.
use super::*;
use crate::backend::cuda::vnext_ops::{cuda_tensor_runtime_config, row_bias_retention_fixture};
use ferrum_runtime::{BufferUsage, DeviceCapacityRetention, ResourceId};

fn runtime() -> CudaDeviceRuntime {
    CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
        device: cuda_tensor_runtime_config(0, DeviceId::new("device.cuda.0").unwrap()).unwrap(),
        usable_memory_bytes: 1 << 30,
        bootstrap_memory_bytes: 256 << 20,
        stream_memory_bytes: 16 << 20,
        maximum_streams: 1,
        maximum_in_flight_per_stream: 1,
        maximum_commands_per_submission: 8,
        opaque_memory_profile: "unqualified-row-bias-custody-fixture".into(),
    })
    .unwrap()
}

fn buffers(
    runtime: &CudaDeviceRuntime,
    stream: &CudaDeviceStream,
) -> (Vec<CudaDeviceBuffer>, DeviceCapacityRetention) {
    let quota = runtime.reserve_tensor_opaque(16).unwrap();
    let buffers = [2.0_f32, 3.0, 0.0, 0.25]
        .into_iter()
        .enumerate()
        .map(|(i, value)| {
            let base = stream.stream.clone_htod(&value.to_le_bytes()).unwrap();
            let (pointer, guard) = base.device_ptr(&stream.stream);
            drop(guard);
            CudaDeviceBuffer {
                descriptor: BufferDescriptor {
                    resource_id: ResourceId::new(format!("resource.row_bias.{i}")).unwrap(),
                    size_bytes: 4,
                    alignment_bytes: 4,
                    usage: BufferUsage::Activations,
                    element_type: ElementType::F32,
                },
                runtime_instance: runtime.runtime_instance,
                allocation: Arc::new(CudaAllocation {
                    _base: Some(base),
                    idle_allocation_return: None,
                    aligned_ptr: pointer,
                    requested_bytes: 4,
                    _tensor_owner: runtime.tensor_eager.clone(),
                }),
            }
        })
        .collect();
    (buffers, quota)
}

#[test]
#[ignore = "requires exclusive CUDA; actual module quota, SGEMM/broadcast and completion custody"]
fn shared_row_bias_module_survives_provider_drop_fence_and_uncertain_submission() {
    for uncertain in [false, true] {
        let runtime = runtime();
        let mut stream = runtime.create_stream().unwrap();
        let (buffers, quota) = buffers(&runtime, &stream);
        let regions = buffers.iter().map(|b| b.region(0..4).unwrap()).collect();
        let (provider, mut command) = row_bias_retention_fixture(&runtime, regions).unwrap();
        assert_eq!(command.compute_dispatch_count, 2);
        assert!(runtime.reserve_tensor_provider().is_err());
        drop(provider);
        assert!(
            runtime.reserve_tensor_provider().is_err(),
            "encoded command retains module"
        );
        runtime.ensure_submission_blas(&mut stream, true).unwrap();
        command.tensor_flight = Some(
            stream
                .tensor_lane
                .as_ref()
                .unwrap()
                .reserve_flight()
                .unwrap(),
        );
        stream.state.begin_submission().unwrap();
        if uncertain {
            // The same production command has enqueued real SGEMM and bias.
            // Simulate an unwind before recording the completion fence using
            // the existing real SubmissionGuard/quarantine path.
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut guard = tensor_io::SubmissionGuard::new(&runtime, &stream, vec![command]);
                guard.arm();
                stream.enqueue_command(&guard[0]).unwrap();
                panic!("injected unwind after real row bias enqueue");
            }));
            assert!(result.is_err());
            assert_eq!(runtime.quarantined().len(), 1);
            assert!(
                runtime.reserve_tensor_provider().is_err(),
                "uncertain commands keep original module"
            );
            runtime.synchronize(&mut stream).unwrap();
            assert!(runtime.quarantined().is_empty());
        } else {
            stream.enqueue_command(&command).unwrap();
            let event = stream.stream.record_event(None).unwrap();
            stream.state.submission_recorded().unwrap();
            let fence = CudaDeviceFence {
                event,
                timing: CudaFenceTiming::NotRequested,
                command_timing: CudaFenceCommandTiming::NotRequested,
                attribution: None,
                execution_facts: None,
                stream_state: Arc::clone(&stream.state),
                terminal_accounted: AtomicBool::new(false),
                _stream: Arc::clone(&stream.stream),
                _blas: stream.blas.clone(),
                _commands: vec![command],
            };
            assert!(matches!(
                runtime.wait_fence(&fence).unwrap().into_parts().0,
                DeviceTerminal::Succeeded
            ));
            assert!(
                runtime.reserve_tensor_provider().is_err(),
                "original fence still owns command"
            );
            drop(fence);
        }
        let output = stream
            .stream
            .clone_dtoh(buffers[2].allocation._base.as_ref().unwrap())
            .unwrap();
        assert_eq!(f32::from_le_bytes(output.try_into().unwrap()), 6.25);
        drop(runtime.reserve_tensor_provider().unwrap());
        drop((buffers, quota, stream));
    }
}

#[test]
#[ignore = "requires exclusive CUDA; rejects a different runtime on the same device"]
fn shared_row_bias_rejects_foreign_regions_without_losing_module_capacity() {
    let original = runtime();
    let foreign = runtime();
    let mut stream = foreign.create_stream().unwrap();
    let (buffers, quota) = buffers(&foreign, &stream);
    let regions = buffers.iter().map(|b| b.region(0..4).unwrap()).collect();
    assert!(row_bias_retention_fixture(&original, regions).is_err());
    // Rejection happens before submission and releases the newly loaded owner.
    drop(original.reserve_tensor_provider().unwrap());
    foreign.synchronize(&mut stream).unwrap();
    drop((buffers, quota));
}
