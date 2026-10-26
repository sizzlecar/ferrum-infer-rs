use super::*;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits, ResourceId};

fn runtime() -> MetalDeviceRuntime {
    MetalDeviceRuntime::new(MetalDeviceRuntimeConfig {
        device_id: DeviceId::new("device.metal.managed-io").unwrap(),
        runtime_implementation_fingerprint: "b".repeat(64),
        capabilities: BTreeSet::new(),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            ferrum_runtime::DynamicStorageAllocator::LinearArena,
            ferrum_runtime::DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    })
    .unwrap()
}

fn account(aligned: bool) -> HostMemoryAccount {
    let limits = HostMemoryLimits::new(
        1 << 20,
        HostMemoryClassLimits {
            input_preprocess_bytes: 1 << 19,
            output_bytes: 1 << 19,
            transfer_staging_bytes: 0,
            weight_backing_bytes: 0,
        },
    )
    .unwrap();
    if aligned {
        HostMemoryAccount::with_allocation_alignment(
            limits,
            crate::backend::metal::retained_memory::native_page_size().unwrap(),
        )
        .unwrap()
    } else {
        HostMemoryAccount::new(limits)
    }
}

fn buffer(runtime: &MetalDeviceRuntime) -> MetalDeviceBuffer {
    runtime
        .allocate_request(
            &BufferRequest::new(
                ResourceId::new("resource.metal.tensor.io").unwrap(),
                16,
                16,
                BufferUsage::Activations,
                ElementType::Bf16,
            )
            .unwrap(),
        )
        .unwrap()
}

#[test]
fn fixed_tensor_attribution_survives_a_real_metal_submission_without_fake_tokens() {
    let _counter_guard = super::super::counter_capacity_tests::counter_test_guard();
    metal::objc::rc::autoreleasepool(|| {
        let runtime = runtime();
        let destination = buffer(&runtime);
        let work = DeviceCommandLogicalWork::tensor(8, 0).unwrap();
        let command = runtime
            .encode_zero(&destination, 0, 16)
            .unwrap()
            .bind_core_logical_work(work)
            .unwrap();
        let mut stream = runtime.create_stream().unwrap();
        let fence = runtime
            .submit_commands(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                DeviceTimingMode::Kernel,
                &DisabledDeviceSubmissionTimingSink,
            )
            .unwrap();
        let attribution = runtime.submission_attribution(&fence).unwrap();
        let [command] = attribution.commands() else {
            panic!("expected one native zero")
        };
        assert_eq!(command.logical_work(), work.work());
        assert_eq!(command.token_count(), None);
        assert_eq!(command.node_index(), Some(0));
        assert_eq!(command.transfer_command_count(), 1);
        assert!(runtime
            .wait_fence(&fence)
            .unwrap()
            .terminal()
            .is_succeeded());
    });
}

#[test]
fn no_copy_upload_retains_aligned_host_charge_and_readback_writes_only_its_piece() {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = runtime();
        let host = account(true);
        let mut source = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
            .unwrap();
        source
            .as_mut()
            .copy_from_slice(&[0, 63, 128, 63, 0, 64, 64, 64]);
        let source = source.freeze();
        let pointer = source.bytes().as_ptr() as usize;
        let input_charge = host.usage().unwrap().allocated_bytes();
        let destination = buffer(&runtime);
        let layout = HostTransferLayout::new(ElementType::Bf16, 4).unwrap();
        let upload = runtime
            .encode_managed_upload(source, layout, &destination, 4)
            .unwrap()
            .unwrap();
        let page = crate::backend::metal::retained_memory::native_page_size().unwrap();
        assert_eq!(upload.staging[0].contents() as usize, pointer & !(page - 1));
        assert_eq!(host.usage().unwrap().allocated_bytes(), input_charge);
        let work = DeviceCommandLogicalWork::tensor(8, 0).unwrap();
        let zero = runtime
            .encode_zero(&destination, 0, 16)
            .unwrap()
            .bind_core_logical_work(work)
            .unwrap();
        assert_eq!(zero.tensor_work, Some(work));
        assert_eq!(zero.tensor_work.unwrap().token_count(), None);
        let mut stream = runtime.create_stream().unwrap();
        let fence = runtime
            .submit_commands(
                &mut stream,
                vec![
                    (DeviceCommandPhase::DynamicBinding, None, zero),
                    (DeviceCommandPhase::DynamicBinding, None, upload),
                ],
                DeviceTimingMode::Off,
                &DisabledDeviceSubmissionTimingSink,
            )
            .unwrap();
        assert!(runtime
            .wait_fence(&fence)
            .unwrap()
            .terminal()
            .is_succeeded());
        let mut output = host.allocate_zeroed(HostMemoryClass::Output, 16).unwrap();
        output.as_mut().fill(91);
        assert!(runtime
            .readback_managed_into(
                &mut stream,
                &destination,
                CopyRegion::new(4, 2, 8).unwrap(),
                layout,
                &mut output
            )
            .unwrap());
        assert_eq!(&output.as_ref()[..2], &[91; 2]);
        assert_eq!(&output.as_ref()[2..10], &[0, 63, 128, 63, 0, 64, 64, 64]);
        assert_eq!(&output.as_ref()[10..], &[91; 6]);
        drop(fence);
        assert_eq!(
            host.usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::InputPreprocess),
            0
        );
        drop(output);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    });
}

#[test]
fn managed_tensor_io_rejects_foreign_misaligned_or_wrong_class_transfers_before_copy() {
    metal::objc::rc::autoreleasepool(|| {
        let foreign = runtime();
        let runtime = runtime();
        let host = account(true);
        let source = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
            .unwrap()
            .freeze();
        let destination = buffer(&runtime);
        let layout = HostTransferLayout::new(ElementType::Bf16, 4).unwrap();
        for offset in [1, 10, u64::MAX] {
            assert!(runtime
                .encode_managed_upload(source.clone(), layout, &destination, offset)
                .is_err());
        }
        assert!(foreign
            .encode_managed_upload(source.clone(), layout, &destination, 0)
            .is_err());
        let wrong_class = host
            .allocate_zeroed(HostMemoryClass::Output, 8)
            .unwrap()
            .freeze();
        assert!(runtime
            .encode_managed_upload(wrong_class, layout, &destination, 0)
            .is_err());
        let ordinary = account(false)
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
            .unwrap()
            .freeze();
        assert!(runtime
            .encode_managed_upload(ordinary, layout, &destination, 0)
            .unwrap()
            .is_none());
        let mut stream = runtime.create_stream().unwrap();
        let mut output = host.allocate_zeroed(HostMemoryClass::Output, 16).unwrap();
        output.as_mut().fill(37);
        for region in [
            CopyRegion::new(0, 1, 8).unwrap(),
            CopyRegion::new(0, 0, 6).unwrap(),
            CopyRegion::new(10, 0, 8).unwrap(),
        ] {
            assert!(runtime
                .readback_managed_into(&mut stream, &destination, region, layout, &mut output)
                .is_err());
            assert_eq!(output.as_ref(), &[37; 16]);
        }
    });
}
