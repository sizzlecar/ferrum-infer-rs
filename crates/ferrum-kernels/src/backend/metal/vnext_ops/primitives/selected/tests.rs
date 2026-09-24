use super::*;
use ferrum_interfaces::vnext::{
    BufferRequest, BufferUsage, DeviceBatchingForm, OperationCostCommand, ResourceId,
};
use half::f16;

#[test]
fn selected_primitive_psos_preserve_type_and_parallel_specialization_identity() {
    let device = Device::system_default().expect("Metal selected primitive PSO test");
    let p = MetalPrimitivePipelines::new(&device).unwrap();
    assert!(std::ptr::eq(
        embedding(&p, EmbeddingPhysicalFormat::Q6K, ElementType::F32)
            .unwrap()
            .pipeline,
        &p.embedding_q6_k_f32
    ));
    assert!(embedding(&p, EmbeddingPhysicalFormat::Q6K, ElementType::U8).is_none());
    let a = rms_evidence(
        &p,
        cost_route::rms_norm_params(1, 256, 1e-6).unwrap(),
        ElementType::F16,
        ElementType::F16,
    )
    .unwrap();
    let b = rms_evidence(
        &p,
        cost_route::rms_norm_params(3, 256, 1e-6).unwrap(),
        ElementType::F16,
        ElementType::F16,
    )
    .unwrap();
    assert_eq!(a.family_signature(), b.family_signature());
    assert_eq!(b.work().inner_work_units, a.work().inner_work_units * 3);
    let f32 = rms_evidence(
        &p,
        cost_route::rms_norm_params(1, 256, 1e-6).unwrap(),
        ElementType::F32,
        ElementType::F16,
    )
    .unwrap();
    assert_ne!(a.family_signature(), f32.family_signature());
    let reduction = |vocab| {
        let mut builder = SelectedCommandCostBuilderV1::new(1);
        push_argmax(
            &mut builder,
            &p,
            cost_route::masked_argmax_params(vocab, 1).unwrap(),
            ElementType::F32,
            masked_argmax_scratch_stride(vocab, ElementType::F32).unwrap(),
        )
        .unwrap();
        builder.finish().unwrap()
    };
    let serial = reduction(u64::from(MASKED_ARGMAX_PARALLEL_MIN_VOCAB) - 1);
    let parallel = reduction(u64::from(MASKED_ARGMAX_PARALLEL_MIN_VOCAB));
    serial.validate_command(1, 1, 0).unwrap();
    parallel.validate_command(1, 2, 0).unwrap();
    assert_ne!(
        serial.family_signature(),
        parallel.family_signature(),
        "same entry string has different function constant and a real finalize dispatch"
    );
    assert!(std::ptr::eq(
        argmax(&p, ElementType::F32, true).unwrap().pipeline,
        &p.parallel_masked_argmax_f32
    ));
}

pub(crate) struct NativeCase {
    pub command: MetalDeviceCommand,
    pub projected: OperationCostCommand,
    /// Whole guarded allocations: read-only values and exact expected outputs.
    pub checks: Vec<(MetalBufferRegion, Vec<u8>)>,
    pub scratch_guards: Vec<(MetalBufferRegion, usize)>,
}
fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
            .to_vec()
    }
}
fn guarded(
    runtime: &MetalDeviceRuntime,
    name: &str,
    data: Vec<u8>,
) -> (MetalBufferRegion, MetalBufferRegion, Vec<u8>) {
    let mut initial = vec![0xa5; 64 + data.len() + 64];
    initial[64..64 + data.len()].copy_from_slice(&data);
    let parent = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new(name).unwrap(),
                initial.len() as u64,
                64,
                BufferUsage::Transfer,
                ElementType::U8,
            )
            .unwrap(),
        )
        .unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            initial.as_ptr(),
            parent
                .buffer()
                .contents()
                .cast::<u8>()
                .add(parent.offset_bytes() as usize),
            initial.len(),
        );
    }
    let child = parent.test_subregion(64..(64 + data.len()) as u64).unwrap();
    (parent, child, initial)
}
fn checked_command(
    route: OperationCostCommand,
    evidence: SelectedCommandCostEvidenceV1,
) -> OperationCostCommand {
    route.with_statistical_evidence(evidence).unwrap()
}
fn changed(mut image: Vec<u8>, output: &[u8]) -> Vec<u8> {
    image[64..64 + output.len()].copy_from_slice(output);
    image
}

/// Uses actual primitive dispatch functions and their shared selected helpers.
/// This tests native receipt/PSO/bytes, not a model-level timing or quality claim.
pub(crate) fn runtime_fixtures(runtime: &MetalDeviceRuntime) -> Vec<NativeCase> {
    let p = Arc::new(MetalPrimitivePipelines::new(runtime.device()).unwrap());
    let mut cases = Vec::new();
    // Dense table lookup, two physical token rows, one participant command.
    {
        let params = cost_route::embedding_params(2, 65, 2).unwrap();
        let (table, table_view, table_bytes) = guarded(
            runtime,
            "primitive.table",
            bytes(&vec![f16::from_f32(2.0); 130]),
        );
        let (ids, ids_view, ids_bytes) = guarded(runtime, "primitive.ids", bytes(&[1u32, 0]));
        let (out, out_view, out_bytes) = guarded(runtime, "primitive.embedding.out", vec![0; 260]);
        let mut b = SelectedCommandCostBuilderV1::new(2);
        push_embedding(
            &mut b,
            &p,
            EmbeddingPhysicalFormat::DenseF16,
            ElementType::F16,
            params,
        )
        .unwrap();
        let evidence = b.finish().unwrap();
        let route = checked_command(
            cost_route::compute_command(
                PrimitiveRoute::TokenEmbedding {
                    transformed_participants: 0,
                },
                1,
                2,
            )
            .unwrap(),
            evidence.clone(),
        );
        let kernels = Arc::clone(&p);
        let command = MetalDeviceCommand::operation(
            route.native_operation(),
            vec![table_view, ids_view, out_view],
            move |encoder, regions| {
                encoder.record_compute_dispatches(1);
                dispatch_embedding(
                    &kernels,
                    encoder.compute_encoder(),
                    EmbeddingPhysicalFormat::DenseF16,
                    &regions[0],
                    &regions[1],
                    &regions[2],
                    params,
                    ElementType::F16,
                );
                Ok(())
            },
        )
        .unwrap()
        .with_work_shape(route.batching(), 1, 2)
        .unwrap()
        .with_statistical_evidence(Some(evidence));
        cases.push(NativeCase {
            command,
            projected: route,
            checks: vec![
                (table, table_bytes),
                (ids, ids_bytes),
                (
                    out,
                    changed(out_bytes, &bytes(&vec![f16::from_f32(2.0); 130])),
                ),
            ],
            scratch_guards: vec![],
        });
    }
    // F32->F16 RMSNorm: constant 2 input and unit coefficients round to 1.
    {
        let params = cost_route::rms_norm_params(2, 64, 1e-6).unwrap();
        let (input, input_view, input_bytes) =
            guarded(runtime, "primitive.rms.in", bytes(&vec![2.0f32; 128]));
        let (weight, weight_view, weight_bytes) =
            guarded(runtime, "primitive.rms.weight", bytes(&vec![f16::ONE; 64]));
        let (out, out_view, out_bytes) = guarded(runtime, "primitive.rms.out", vec![0; 256]);
        let evidence = rms_evidence(&p, params, ElementType::F32, ElementType::F16).unwrap();
        let route = checked_command(
            cost_route::compute_command(PrimitiveRoute::RmsNorm, 2, 2).unwrap(),
            evidence.clone(),
        );
        let kernels = Arc::clone(&p);
        let command = MetalDeviceCommand::operation(
            route.native_operation(),
            vec![input_view, weight_view, out_view],
            move |encoder, regions| {
                encoder.record_compute_dispatches(1);
                dispatch_rms_norm_typed(
                    &kernels,
                    encoder.compute_encoder(),
                    &regions[0],
                    &regions[1],
                    &regions[2],
                    params,
                    ElementType::F32,
                    ElementType::F16,
                );
                Ok(())
            },
        )
        .unwrap()
        .with_work_shape(route.batching(), 2, 2)
        .unwrap()
        .with_statistical_evidence(Some(evidence));
        cases.push(NativeCase {
            command,
            projected: route,
            checks: vec![
                (input, input_bytes),
                (weight, weight_bytes),
                (out, changed(out_bytes, &bytes(&vec![f16::ONE; 128]))),
            ],
            scratch_guards: vec![],
        });
    }
    // Tail of a single pointwise grid, mixed precision master residual.
    {
        let params = cost_route::residual_add_params(1, 127).unwrap();
        let (left, left_view, left_bytes) =
            guarded(runtime, "primitive.add.left", bytes(&vec![1.0f32; 127]));
        let (right, right_view, right_bytes) = guarded(
            runtime,
            "primitive.add.right",
            bytes(&vec![f16::from_f32(2.0); 127]),
        );
        let (out, out_view, out_bytes) = guarded(runtime, "primitive.add.out", vec![0; 508]);
        let evidence = residual_evidence(
            &p,
            params,
            ElementType::F32,
            ElementType::F16,
            ElementType::F32,
            1,
        )
        .unwrap();
        let route = checked_command(
            cost_route::compute_command(PrimitiveRoute::ResidualAdd, 1, 1).unwrap(),
            evidence.clone(),
        );
        let kernels = Arc::clone(&p);
        let command = MetalDeviceCommand::operation(
            route.native_operation(),
            vec![left_view, right_view, out_view],
            move |encoder, regions| {
                encoder.record_compute_dispatches(1);
                dispatch_residual_add_typed(
                    &kernels,
                    encoder.compute_encoder(),
                    &regions[0],
                    &regions[1],
                    &regions[2],
                    params,
                    ElementType::F32,
                    ElementType::F16,
                    ElementType::F32,
                );
                Ok(())
            },
        )
        .unwrap()
        .with_work_shape(route.batching(), 1, 1)
        .unwrap()
        .with_statistical_evidence(Some(evidence));
        cases.push(NativeCase {
            command,
            projected: route,
            checks: vec![
                (left, left_bytes),
                (right, right_bytes),
                (out, changed(out_bytes, &bytes(&vec![3.0f32; 127]))),
            ],
            scratch_guards: vec![],
        });
    }
    for vocab in [257, MASKED_ARGMAX_PARALLEL_MIN_VOCAB] {
        let params = cost_route::masked_argmax_params(u64::from(vocab), 1).unwrap();
        let scratch_bytes =
            masked_argmax_scratch_stride(u64::from(vocab), ElementType::F32).unwrap();
        let mut logits = vec![0.0f32; vocab as usize];
        logits[19] = 1.0;
        let inputs = [
            bytes(&logits),
            vec![1; vocab as usize],
            bytes(&[0u32]),
            bytes(&[0u32, 0]),
            bytes(&[1.0f32]),
        ];
        let mut regions = Vec::new();
        let mut checks = Vec::new();
        for (i, data) in inputs.into_iter().enumerate() {
            let (parent, view, initial) =
                guarded(runtime, &format!("primitive.argmax.{vocab}.{i}"), data);
            regions.push(view);
            checks.push((parent, initial));
        }
        let (out, out_view, out_bytes) = guarded(
            runtime,
            &format!("primitive.argmax.{vocab}.out"),
            bytes(&[u32::MAX]),
        );
        regions.push(out_view);
        checks.push((out, changed(out_bytes, &bytes(&[19u32]))));
        let (scratch, scratch_view, _) = guarded(
            runtime,
            &format!("primitive.argmax.{vocab}.scratch"),
            vec![0; scratch_bytes as usize],
        );
        regions.push(scratch_view);
        let mut b = SelectedCommandCostBuilderV1::new(1);
        push_argmax(&mut b, &p, params, ElementType::F32, scratch_bytes).unwrap();
        let evidence = b.finish().unwrap();
        let route = checked_command(
            cost_route::compute_command(
                PrimitiveRoute::MaskedArgmax {
                    vocabulary_size: vocab,
                },
                1,
                1,
            )
            .unwrap(),
            evidence.clone(),
        );
        let count = route.compute_dispatch_count();
        let kernels = Arc::clone(&p);
        let command =
            MetalDeviceCommand::operation(route.native_operation(), regions, move |encoder, r| {
                encoder.record_compute_dispatches(count);
                dispatch_last_token_masked_argmax(
                    &kernels,
                    encoder.compute_encoder(),
                    &r[0],
                    &r[1],
                    &r[2],
                    &r[3],
                    &r[4],
                    &r[5],
                    &r[6],
                    0,
                    params,
                    ElementType::F32,
                );
                Ok(())
            })
            .unwrap()
            .with_work_shape(DeviceBatchingForm::Scalar, 1, 1)
            .unwrap()
            .with_statistical_evidence(Some(evidence));
        cases.push(NativeCase {
            command,
            projected: route,
            checks,
            scratch_guards: vec![(scratch, scratch_bytes as usize)],
        });
    }
    cases
}
