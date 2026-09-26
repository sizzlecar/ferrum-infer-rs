//! Passive CUDA evidence from checked encoder metadata. No shape or profile
//! can construct a live command, and Disabled creates no per-command builder.
use ferrum_interfaces::execution_cost::{
    SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1,
    StatisticalTransferKindV1, TransferReplayGeometryV1,
};
use ferrum_types::SloStructuredCostCapture;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub(crate) fn builder(
    capture: SloStructuredCostCapture,
    tokens: u64,
) -> Option<SelectedCommandCostBuilderV1> {
    match capture {
        SloStructuredCostCapture::Disabled => None,
        SloStructuredCostCapture::HostSettledV1 => Some(
            SelectedCommandCostBuilderV1::new_with_algorithm_work(tokens),
        ),
    }
}

/// One existing contiguous runtime transfer, after actual span validation.
/// Pinned D2H is the submitted command, not a later synchronous fallback read.
pub(super) fn transfer(
    kind: StatisticalTransferKindV1,
    bytes: u64,
    tokens: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut result = builder(capture, tokens)?;
    let (entry, layout) = match kind {
        StatisticalTransferKindV1::HostToDevice => (
            "cuda.cuMemcpyHtoDAsync",
            b"retained_owned_host_contiguous_upload.v1".as_slice(),
        ),
        StatisticalTransferKindV1::DeviceToDevice => (
            "cuda.cuMemcpyDtoDAsync",
            b"contiguous_same_element_device_copy.v1".as_slice(),
        ),
        StatisticalTransferKindV1::Fill => (
            "cuda.cuMemsetD8Async",
            b"contiguous_device_byte_zero.v1".as_slice(),
        ),
        StatisticalTransferKindV1::DeviceToHost => (
            "cuda.cuMemcpyDtoHAsync",
            b"exclusive_pinned_submission_readback_span.v1".as_slice(),
        ),
    };
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let numerical =
        *NUMERICAL.get_or_init(|| Sha256::digest(b"cuda.byte_exact_transfer.v1").into());
    let class =
        SelectedAlgorithmClassV1::new(entry, 1, numerical, Sha256::digest(layout).into()).ok()?;
    result.transfer(class, kind, bytes).ok()?;
    result.finish().ok()
}

/// Already coalesced physical binding transfers: destination pitch, row bytes,
/// row count. Padding between destination rows is not transferred.
pub(super) fn program_binding(
    transfers: impl IntoIterator<Item = (u64, u64, u64)>,
    tokens: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut result = builder(capture, tokens)?;
    for (stride, row_bytes, rows) in transfers {
        if row_bytes == 0 || rows == 0 || stride < row_bytes {
            return None;
        }
        let bytes = row_bytes.checked_mul(rows)?;
        let entry = if rows == 1 {
            "cuda.cuMemcpyHtoDAsync.binding"
        } else {
            "cuda.cuMemcpy2DAsync.binding"
        };
        let mut layout = Sha256::new();
        // The actual API differs at rows == 1. For 2D, packed and pitched
        // destinations and the number of copied rows remain separate classes.
        // Byte extents/pitch are numeric execution facts, not an algorithm per
        // generated token. Exact geometry is bound independently below.
        layout.update(b"cuda.retained_packed_host_sparse_binding.v2");
        if rows == 1 {
            layout.update(b"contiguous_destination");
        } else {
            layout.update(if stride == row_bytes {
                b"contiguous_destination".as_slice()
            } else {
                b"pitched_destination".as_slice()
            });
            layout.update(rows.to_le_bytes());
        }
        let class = SelectedAlgorithmClassV1::new(
            entry,
            1,
            Sha256::digest(b"cuda.byte_exact_transfer.v1").into(),
            layout.finalize().into(),
        )
        .ok()?;
        result
            .transfer_with_replay_geometry(
                class,
                StatisticalTransferKindV1::HostToDevice,
                bytes,
                TransferReplayGeometryV1 {
                    row_bytes,
                    rows,
                    destination_stride_bytes: stride,
                },
            )
            .ok()?;
    }
    result.finish().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_binding_statistical_domain_uses_route_class_and_numeric_bytes_separately() {
        use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
        let on = SloStructuredCostCapture::HostSettledV1;
        let single = program_binding([(424, 424, 1)], 8, on).unwrap();
        let longer = program_binding([(432, 432, 1)], 8, on).unwrap();
        assert_eq!(single.family_signature(), longer.family_signature());
        assert_eq!(single.work().host_to_device_bytes, 424);
        assert_eq!(longer.work().host_to_device_bytes, 432);
        let pitched = program_binding([(1024, 424, 8)], 8, on).unwrap();
        let longer_pitched = program_binding([(1024, 432, 8)], 8, on).unwrap();
        assert_eq!(
            pitched.family_signature(),
            longer_pitched.family_signature()
        );
        assert_eq!(pitched.work().host_to_device_bytes, 3392);
        assert_eq!(longer_pitched.work().host_to_device_bytes, 3456);
        assert_ne!(single.family_signature(), pitched.family_signature());
        let packed = program_binding([(424, 424, 8)], 8, on).unwrap();
        assert_ne!(pitched.family_signature(), packed.family_signature());
        let fewer_rows = program_binding([(1024, 424, 7)], 8, on).unwrap();
        assert_ne!(pitched.family_signature(), fewer_rows.family_signature());
        let other_pitch = program_binding([(2048, 424, 8)], 8, on).unwrap();
        assert_eq!(pitched.family_signature(), other_pitch.family_signature());
        assert_eq!(pitched.work(), other_pitch.work());
        let template = SelectedReplayAlgorithmTemplateV1::from_selected(&pitched, 8, 0, 1).unwrap();
        assert!(template.validate_binding(&other_pitch).is_err());
        assert!(template.validate_binding(&longer_pitched).is_err());
        template.validate_binding(&pitched).unwrap();
        assert!(program_binding([(1024, 424, 8)], 8, SloStructuredCostCapture::Disabled).is_none());
    }

    #[test]
    fn cuda_selected_core_transfers_keep_direction_bytes_and_logical_binding() {
        use StatisticalTransferKindV1 as K;
        let on = SloStructuredCostCapture::HostSettledV1;
        for kind in [K::HostToDevice, K::DeviceToDevice, K::DeviceToHost, K::Fill] {
            let actual = transfer(kind, 257, 0, on).unwrap();
            actual.validate_command(0, 0, 1).unwrap();
            let rebound = transfer(kind, 257, 3, on).unwrap();
            rebound.validate_command(3, 0, 1).unwrap();
            assert!(rebound.validate_command(0, 0, 1).is_err());
            assert_eq!(actual.work(), rebound.work());
            let work = rebound.work();
            assert_eq!(
                work.host_to_device_bytes,
                if kind == K::HostToDevice { 257 } else { 0 }
            );
            assert_eq!(
                work.device_to_host_bytes,
                if kind == K::DeviceToHost { 257 } else { 0 }
            );
            assert_eq!(
                work.device_to_device_bytes,
                if kind == K::DeviceToDevice { 257 } else { 0 }
            );
            assert_eq!(work.fill_bytes, if kind == K::Fill { 257 } else { 0 });
            rebound
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(&rebound)
                .unwrap();
            assert!(transfer(kind, 0, 3, on).is_none());
            assert!(transfer(kind, 257, 3, SloStructuredCostCapture::Disabled).is_none());
        }
    }
    #[test]
    #[ignore = "requires an actual CUDA device"]
    fn cuda_selected_core_real_offset_transfers_keep_output_and_future_bytes() {
        use super::super::*;
        use ferrum_interfaces::execution_cost::StatisticalTransferKindV1 as K;
        use ferrum_interfaces::vnext::{BufferUsage, ResourceId};
        let runtime = CudaDeviceRuntime::new_with_structured_capture(
            crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
                0,
                DeviceId::new("device.cuda.selected-transfer-test").unwrap(),
                AttentionExecutionPolicy::Portable,
            )
            .unwrap(),
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        let mut stream = runtime.create_stream().unwrap();
        // Private test allocation; every command below uses the production
        // checked encoder. This is transfer evidence, not a live wave receipt.
        let make = |name: &str| {
            let base = stream.stream.alloc_zeros::<u8>(32).unwrap();
            let pointer = base.device_ptr(&stream.stream).0;
            CudaDeviceBuffer {
                descriptor: BufferDescriptor {
                    resource_id: ResourceId::new(name).unwrap(),
                    size_bytes: 32,
                    alignment_bytes: 1,
                    usage: BufferUsage::Transfer,
                    element_type: ElementType::U8,
                },
                runtime_instance: runtime.runtime_instance,
                allocation: Arc::new(CudaAllocation {
                    _base: base,
                    _memory_charge: None,
                    aligned_ptr: pointer,
                    requested_bytes: 32,
                }),
            }
        };
        let source = make("source");
        let destination = make("destination");
        for generation in [1_u8, 19] {
            let input = (0..13)
                .map(|i| generation.wrapping_add(i))
                .collect::<Vec<_>>();
            let commands = [
                (
                    runtime
                        .encode_upload(
                            &input,
                            HostTransferLayout::new(ElementType::U8, 13).unwrap(),
                            &source,
                            3,
                        )
                        .unwrap(),
                    K::HostToDevice,
                    13,
                ),
                (
                    runtime.encode_zero(&destination, 0, 32).unwrap(),
                    K::Fill,
                    32,
                ),
                (
                    runtime
                        .encode_copy(&source, &destination, CopyRegion::new(3, 5, 13).unwrap())
                        .unwrap(),
                    K::DeviceToDevice,
                    13,
                ),
            ];
            for (command, kind, bytes) in &commands {
                let observed = command.statistical_evidence.as_ref().unwrap();
                let future = runtime
                    .cost_core_transfer_evidence(*kind, *bytes, 0)
                    .unwrap();
                assert_eq!(observed.family_signature(), future.family_signature());
                assert_eq!(observed.work(), future.work());
                observed.validate_command(0, 0, 1).unwrap();
                command.enqueue(&stream.stream, &stream.blas).unwrap();
            }
            let actual = runtime
                .readback(
                    &mut stream,
                    &destination,
                    CopyRegion::new(0, 0, 32).unwrap(),
                    HostTransferLayout::new(ElementType::U8, 32).unwrap(),
                )
                .unwrap();
            let mut expected = vec![0; 32];
            expected[5..18].copy_from_slice(&input);
            assert_eq!(actual, expected);
        }
        assert!(runtime
            .encode_copy(&source, &destination, CopyRegion::new(30, 0, 13).unwrap())
            .is_err());
    }
}
