use super::*;
use ferrum_runtime::ResourceId;

pub(crate) fn tensor_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: Mutex<()> = Mutex::new(());
    GUARD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn config() -> MetalTensorRuntimeConfig {
    MetalTensorRuntimeConfig {
        device: MetalDeviceRuntimeConfig {
            device_id: DeviceId::new("device.metal.0").unwrap(),
            runtime_implementation_fingerprint: "d".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                ferrum_runtime::DynamicStorageAllocator::LinearArena,
                ferrum_runtime::DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        },
        usable_memory_bytes: 8 << 20,
        bootstrap_memory_bytes: 1 << 20,
        stream_memory_bytes: 64 << 10,
        maximum_streams: 2,
        maximum_in_flight_per_stream: 1,
        maximum_commands_per_submission: 4,
        opaque_memory_profile: "test.metal.tensor.capacity.v1".into(),
    }
}

#[test]
fn invalid_tensor_runtime_limits_fail_before_device_creation() {
    let original = config();
    original.validate().unwrap();
    let mutations: [fn(&mut MetalTensorRuntimeConfig); 8] = [
        |c| c.maximum_streams = 0,
        |c| c.maximum_in_flight_per_stream = 0,
        |c| c.maximum_commands_per_submission = 0,
        |c| c.maximum_commands_per_submission = usize::MAX,
        |c| c.stream_memory_bytes = u64::MAX,
        |c| c.usable_memory_bytes = c.bootstrap_memory_bytes,
        |c| c.opaque_memory_profile = "invalid\nprofile".into(),
        |c| c.device.device_id = DeviceId::new("device.metal.alias").unwrap(),
    ];
    for mutate in mutations {
        let mut candidate = original.clone();
        mutate(&mut candidate);
        assert!(candidate.validate().is_err());
    }
}

#[test]
fn static_import_releases_converted_host_groups_before_loading_the_next_group() {
    use ferrum_runtime::{
        HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits, WeightComponentRole,
        WeightComponentSpec, WeightEncoding, WeightId,
    };
    let _guard = tensor_test_guard();
    metal::objc::rc::autoreleasepool(|| {
        let runtime = MetalDeviceRuntime::new_tensor_eager(config()).unwrap();
        let page = crate::backend::metal::retained_memory::native_page_size().unwrap();
        // One page-aligned conversion group fits, but two retained groups do not.
        let limit = page * 2;
        let host = runtime
            .tensor_host_account(
                HostMemoryLimits::new(
                    limit,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: 0,
                        output_bytes: 0,
                        transfer_staging_bytes: 0,
                        weight_backing_bytes: limit,
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let weights = runtime
            .allocate_request(
                &BufferRequest::new(
                    ResourceId::new("resource.metal.converted.groups").unwrap(),
                    (page * 2) as u64,
                    64,
                    BufferUsage::Weights,
                    ElementType::U8,
                )
                .unwrap(),
            )
            .unwrap();
        let mut import = runtime.begin_static_weight_import().unwrap().unwrap();
        for (index, value) in [0x35_u8, 0xa6].into_iter().enumerate() {
            let component = WeightComponentSpec {
                id: WeightId::new(format!("weight.metal.converted.{index}")).unwrap(),
                role: WeightComponentRole::Values,
                external_names: vec![format!("projection.{index}.weight")],
                dimensions: vec![page as u64],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::U8,
                },
                required: true,
            };
            let mut backing = host
                .allocate_zeroed(HostMemoryClass::WeightBacking, page)
                .unwrap();
            backing.as_mut().fill(value);
            let payload = WeightComponentPayload::from_managed_host(
                &component,
                component.external_names.clone(),
                vec!["model.safetensors".into()],
                component.dimensions.clone(),
                ElementType::U8,
                backing.freeze(),
            )
            .unwrap();
            import
                .import_component(&payload, &weights, (index * page) as u64)
                .unwrap();
            drop(payload);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
        import.seal().unwrap();
        for (index, value) in [0x35_u8, 0xa6].into_iter().enumerate() {
            let region = weights
                .region((index * page) as u64..((index + 1) * page) as u64)
                .unwrap();
            let pointer = unsafe {
                region
                    .buffer()
                    .contents()
                    .add(region.offset_bytes() as usize)
            };
            // SAFETY: the live checked region owns these initialized shared bytes.
            let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), page) };
            assert!(bytes.iter().all(|&byte| byte == value));
        }
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    });
}

#[test]
fn queues_and_provider_owners_keep_the_same_device_budget_alive() {
    let _guard = tensor_test_guard();
    metal::objc::rc::autoreleasepool(|| {
        let config = config();
        let runtime = MetalDeviceRuntime::new_tensor_eager(config.clone()).unwrap();
        let capacity =
            DeviceCapacityOwner::open(runtime.descriptor(), config.usable_memory_bytes).unwrap();
        let provider = runtime.retain_tensor_provider().unwrap();
        assert!(runtime.retain_tensor_provider().is_err());
        let first = runtime.create_stream().unwrap();
        let second = runtime.create_stream().unwrap();
        assert!(runtime.create_stream().is_err());
        let remaining = config.usable_memory_bytes
            - config.bootstrap_memory_bytes
            - 2 * config.stream_memory_bytes;
        let other_plan = capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, remaining)
            .unwrap();
        assert!(matches!(
            capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1),
            Err(VNextError::DeviceCapacityUnavailable(_))
        ));
        drop(first);
        let replacement = runtime.create_stream().unwrap();
        drop((second, replacement, other_plan, runtime));
        // The provider can outlive both the runtime facade and its queues.
        let rest = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Stream,
                config.usable_memory_bytes - config.bootstrap_memory_bytes,
            )
            .unwrap();
        assert!(capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
        drop(provider);
        let released = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.bootstrap_memory_bytes,
            )
            .unwrap();
        drop((rest, released));
        let all = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.usable_memory_bytes,
            )
            .unwrap();
        assert_eq!(all.bytes(), config.usable_memory_bytes);
    });
}

#[test]
fn failed_queue_creation_returns_its_slot_before_retry() {
    let _guard = tensor_test_guard();
    metal::objc::rc::autoreleasepool(|| {
        let config = config();
        let runtime = MetalDeviceRuntime::new_tensor_eager(config.clone()).unwrap();
        let capacity =
            DeviceCapacityOwner::open(runtime.descriptor(), config.usable_memory_bytes).unwrap();
        let all_available = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.usable_memory_bytes - config.bootstrap_memory_bytes,
            )
            .unwrap();
        for _ in 0..3 {
            assert!(matches!(
                runtime.create_stream(),
                Err(MetalDeviceRuntimeError::Capacity(_))
            ));
        }
        drop(all_available);
        let first = runtime.create_stream().unwrap();
        let second = runtime.create_stream().unwrap();
        assert!(runtime.create_stream().is_err());
        drop((first, second));
    });
}

#[test]
fn exact_tensor_allocations_and_imported_regions_retain_their_runtime_budget() {
    let _guard = tensor_test_guard();
    metal::objc::rc::autoreleasepool(|| {
        let config = config();
        let runtime = MetalDeviceRuntime::new_tensor_eager(config.clone()).unwrap();
        assert!(
            crate::backend::metal::vnext_ops::metal_vnext_operation_registry(&runtime).is_err()
        );
        let capacity =
            DeviceCapacityOwner::open(runtime.descriptor(), config.usable_memory_bytes).unwrap();
        let activation = runtime
            .allocate_request(
                &BufferRequest::new(
                    ResourceId::new("resource.metal.exact.activation").unwrap(),
                    16,
                    16,
                    BufferUsage::Activations,
                    ElementType::Bf16,
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(activation.contiguous_allocation().base.length(), 16);
        let a = activation.region(0..8).unwrap();
        assert!(a
            .overlaps_shared_region(&activation.region(4..12).unwrap())
            .unwrap());
        assert!(!a
            .overlaps_shared_region(&activation.region(8..16).unwrap())
            .unwrap());
        drop((a, activation));

        let weights = runtime
            .allocate_request(
                &BufferRequest::new(
                    ResourceId::new("resource.metal.exact.weight").unwrap(),
                    16,
                    16,
                    BufferUsage::Weights,
                    ElementType::Bf16,
                )
                .unwrap(),
            )
            .unwrap();
        let component = ferrum_runtime::WeightComponentSpec {
            id: ferrum_runtime::WeightId::new("weight.metal.exact").unwrap(),
            role: ferrum_runtime::WeightComponentRole::Values,
            external_names: vec!["projection.weight".into()],
            dimensions: vec![8],
            encoding: ferrum_runtime::WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            },
            required: true,
        };
        let payload = WeightComponentPayload::new(
            &component,
            "projection.weight",
            "model.safetensors",
            vec![8],
            ElementType::Bf16,
            &[0; 16],
        )
        .unwrap();
        let mut import = runtime.begin_static_weight_import().unwrap().unwrap();
        import.import_component(&payload, &weights, 0).unwrap();
        import.seal().unwrap();
        let retained = weights.region(0..16).unwrap();
        assert_eq!(retained.buffer().length(), 16);
        drop((weights, runtime));
        let rest = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.usable_memory_bytes - config.bootstrap_memory_bytes,
            )
            .unwrap();
        assert!(capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
        // A translated region retains the allocation after its arena is gone.
        drop(retained);
        let released = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.bootstrap_memory_bytes,
            )
            .unwrap();
        drop((rest, released));
    });
}

#[test]
fn rejected_submissions_release_flights_but_retained_fences_hold_the_queue() {
    let _guard = tensor_test_guard();
    metal::objc::rc::autoreleasepool(|| {
        let config = config();
        let runtime = MetalDeviceRuntime::new_tensor_eager(config.clone()).unwrap();
        let capacity =
            DeviceCapacityOwner::open(runtime.descriptor(), config.usable_memory_bytes).unwrap();
        let destination = runtime
            .allocate_request(
                &BufferRequest::new(
                    ResourceId::new("resource.metal.tensor.capacity.test").unwrap(),
                    16,
                    16,
                    BufferUsage::Activations,
                    ElementType::Bf16,
                )
                .unwrap(),
            )
            .unwrap();
        let mut stream = runtime.create_stream().unwrap();
        let submit = |command, stream: &mut MetalDeviceStream| {
            runtime.submit_commands(
                stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                DeviceTimingMode::Off,
                &DisabledDeviceSubmissionTimingSink,
            )
        };
        let failed_encode = MetalDeviceCommand::operation(
            "test.reject.before.submit",
            vec![destination.region(0..16).unwrap()],
            |_, _| Err(invalid("injected encode rejection")),
        )
        .unwrap();
        assert!(submit(failed_encode, &mut stream).is_err());
        assert!(runtime
            .encode_upload(
                &[0; 16],
                HostTransferLayout::new(ElementType::Bf16, 8).unwrap(),
                &destination,
                0
            )
            .is_err());
        let zero = || runtime.encode_zero(&destination, 0, 16).unwrap();
        assert!(runtime
            .submit_commands(
                &mut stream,
                (0..5)
                    .map(|_| (DeviceCommandPhase::Compute, Some(0), zero()))
                    .collect(),
                DeviceTimingMode::Off,
                &DisabledDeviceSubmissionTimingSink
            )
            .is_err());
        assert!(runtime
            .submit_commands(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), zero())],
                DeviceTimingMode::Kernel,
                &DisabledDeviceSubmissionTimingSink
            )
            .is_err());
        let fence = submit(zero(), &mut stream).unwrap();
        assert!(runtime
            .wait_fence(&fence)
            .unwrap()
            .terminal()
            .is_succeeded());
        assert!(submit(zero(), &mut stream).is_err());
        drop(fence);
        let final_fence = submit(zero(), &mut stream).unwrap();
        drop((destination, stream, runtime));
        // A fence owns the queue and bootstrap after their public handles drop.
        let remaining = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.usable_memory_bytes
                    - config.bootstrap_memory_bytes
                    - config.stream_memory_bytes,
            )
            .unwrap();
        assert!(capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
        drop(final_fence); // synchronizes before releasing any native owner
        let released = capacity
            .reserve(
                DeviceOpaqueMemoryKind::Bootstrap,
                config.bootstrap_memory_bytes + config.stream_memory_bytes,
            )
            .unwrap();
        drop((remaining, released));
    });
}

#[test]
fn independent_queues_can_submit_before_either_fence_is_waited() {
    let _guard = tensor_test_guard();
    metal::objc::rc::autoreleasepool(|| {
        let runtime = MetalDeviceRuntime::new_tensor_eager(config()).unwrap();
        let mut pending = Vec::new();
        for lane in 0..2 {
            let buffer = runtime
                .allocate_request(
                    &BufferRequest::new(
                        ResourceId::new(format!("resource.metal.concurrent.{lane}")).unwrap(),
                        16,
                        16,
                        BufferUsage::Activations,
                        ElementType::Bf16,
                    )
                    .unwrap(),
                )
                .unwrap();
            let mut stream = runtime.create_stream().unwrap();
            let command = runtime.encode_zero(&buffer, 0, 16).unwrap();
            let fence = runtime
                .submit_commands(
                    &mut stream,
                    vec![(DeviceCommandPhase::Compute, Some(0), command)],
                    DeviceTimingMode::Off,
                    &DisabledDeviceSubmissionTimingSink,
                )
                .unwrap();
            pending.push((stream, fence, buffer));
        }
        // The in-flight bound is one per queue, not one for the whole device.
        assert!(runtime.create_stream().is_err());
        for (_, fence, _) in &pending {
            assert!(runtime.wait_fence(fence).unwrap().terminal().is_succeeded());
        }
        drop(pending);
        drop(runtime.create_stream().unwrap());
    });
}
