//! Bind real retained weight slices, then execute the shared production path.
use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};

fn region(
    runtime: &MetalDeviceRuntime,
    name: &str,
    bytes: &[u8],
    dtype: ElementType,
) -> MetalBufferRegion {
    let region = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new(name).unwrap(),
                bytes.len() as u64,
                64,
                BufferUsage::Transfer,
                dtype,
            )
            .unwrap(),
        )
        .unwrap();
    // SAFETY: this fresh retained allocation covers exactly the source bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            bytes.len(),
        );
    }
    region
}

fn read_region(region: &MetalBufferRegion) -> Vec<u8> {
    // SAFETY: callers wait for completion; the retained region bounds the read.
    unsafe {
        std::slice::from_raw_parts(
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            region.length_bytes() as usize,
        )
        .to_vec()
    }
}

fn read_buffer(buffer: &Buffer) -> Vec<u8> {
    // SAFETY: Fixture buffers use shared storage and no command is in flight.
    unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u8>(), buffer.length() as usize)
            .to_vec()
    }
}

#[test]
fn pq2_scale_load_production_dispatch_uses_weight_region_and_fallbacks() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.pq2-scale-load.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let mut pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for output in [ElementType::F16, ElementType::F32] {
        for (rows, width, outputs, offset, expected_aligned) in [
            (1, 384, 16, 18, true),
            (1, 384, 16, 19, false),
            (4, 128, 32, 2, true),
            (4, 128, 32, 3, false),
            (1, 384, 17, 18, false),
        ] {
            let fixture = Fixture::new(runtime.device(), rows, width, outputs, output, true);
            fixture.poison_output();
            let initial_output = read_buffer(&fixture.output);
            fixture.run(&pipelines, &queue, true, 1);
            fixture.assert_cpu();
            let expected = fixture.read();
            let workspace_bytes = read_buffer(&fixture.input);
            let mut weight_bytes = vec![0xcc; offset];
            weight_bytes.extend_from_slice(&fixture.weight_bytes);
            weight_bytes.extend([0xcc; 16]);
            let weight_parent = region(runtime, "scale.weight", &weight_bytes, ElementType::U8);
            let weight = weight_parent
                .test_subregion(offset as u64..(offset + fixture.weight_bytes.len()) as u64)
                .unwrap();
            assert_eq!(weight.offset_bytes() % 2 == 0, offset % 2 == 0);
            let original_input = vec![0; (rows * width) as usize * output.size_bytes() as usize];
            let regions = vec![
                region(runtime, "scale.input", &original_input, output),
                weight,
                region(runtime, "scale.output", &initial_output, output),
                region(
                    runtime,
                    "scale.workspace",
                    &workspace_bytes,
                    ElementType::U8,
                ),
            ];
            let mut launch = LinearLaunch {
                input_region: 0,
                weight_region: 1,
                output_region: 2,
                input_offset_bytes: 0,
                output_offset_bytes: 16,
                activation_type: output,
                format: LinearPhysicalFormat::Native(GgufBlockFormat::Pq2_0),
                params: fixture.params,
                transform: Some(HadamardTransform {
                    block_size: 128,
                    signs_region: None,
                    inverse: false,
                    permutation: None,
                }),
                transform_workspace: None,
                transformed_plan: TransformedLinearPlan::Single,
                plain_plan: PlainLinearPlan::Single,
            };
            launch
                .bind_hadamard_workspace(&pipelines, &regions, 3, 16)
                .unwrap();
            validate_launch_regions(&regions, &[launch]).unwrap();
            let select = |pipelines: &MetalLinearPipelines, params, format, dtype| {
                let (selected, dispatch) = pipelines.hadamard_native_dispatch_for_bindings(
                    format,
                    dtype,
                    params,
                    &regions[3],
                    16,
                    &regions[1],
                );
                (selected as *const ComputePipelineState, dispatch)
            };
            let aligned = match output {
                ElementType::F16 => pipelines
                    .native
                    .pq2_linear_f32_f16_complete_aligned_scale
                    .as_ref(),
                ElementType::F32 => pipelines
                    .native
                    .pq2_linear_f32_complete_aligned_scale
                    .as_ref(),
                _ => unreachable!(),
            }
            .unwrap() as *const ComputePipelineState;
            assert_eq!(
                select(&pipelines, fixture.params, GgufBlockFormat::Pq2_0, output).0 == aligned,
                expected_aligned,
            );
            // Out-of-cohort shapes and formats retain their existing pipelines.
            // These identity-only probes do not execute mismatched payloads.
            for (params, format) in [
                (
                    LinearParams {
                        rows: 32,
                        ..fixture.params
                    },
                    GgufBlockFormat::Pq2_0,
                ),
                (
                    LinearParams {
                        in_features: 129,
                        ..fixture.params
                    },
                    GgufBlockFormat::Pq2_0,
                ),
                (fixture.params, GgufBlockFormat::Iq4Xs),
            ] {
                let expected = pipelines.hadamard_native_dispatch_for_input(
                    format,
                    output,
                    params,
                    &regions[3],
                    16,
                );
                assert_eq!(
                    select(&pipelines, params, format, output),
                    (expected.0 as *const _, expected.1)
                );
            }
            for rows in [20, 31] {
                let params = LinearParams {
                    rows,
                    out_features: 4096,
                    output_stride: 4101,
                    ..fixture.params
                };
                let original = pipelines.hadamard_native_dispatch_for_input(
                    GgufBlockFormat::Pq2_0,
                    ElementType::F16,
                    params,
                    &regions[3],
                    16,
                );
                assert_ne!(original.1, LinearDispatchKind::Pq2CooperativeGemv);
                assert_eq!(
                    select(&pipelines, params, GgufBlockFormat::Pq2_0, ElementType::F16),
                    (original.0 as *const _, original.1),
                );
            }
            let submit = |pipelines: &MetalLinearPipelines| {
                // SAFETY: previous command completed; reset poison and canaries.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        initial_output.as_ptr(),
                        regions[2]
                            .buffer()
                            .contents()
                            .cast::<u8>()
                            .add(regions[2].offset_bytes() as usize),
                        initial_output.len(),
                    );
                }
                let command = queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                // Workspace already contains F32 Hadamard output. Execute the
                // actual shared run/serve projection selector and buffer binding.
                dispatch_transformed_linear(pipelines, encoder, &regions, launch);
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                let actual = read_region(&regions[2]);
                // SAFETY: copy the completed output into the equal-size Fixture
                // buffer so its independent F64 and full-canary oracle is reused.
                assert_eq!(actual.len(), fixture.output.length() as usize);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        actual.as_ptr(),
                        fixture.output.contents().cast::<u8>(),
                        actual.len(),
                    );
                }
                fixture.assert_cpu();
                let output = fixture.read();
                assert_eq!(
                    output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(read_region(&weight_parent), weight_bytes);
                assert_eq!(read_region(&regions[3]), workspace_bytes);
                assert_eq!(read_region(&regions[0]), original_input);
            };
            submit(&pipelines);
            let saved = match output {
                ElementType::F16 => pipelines
                    .native
                    .pq2_linear_f32_f16_complete_aligned_scale
                    .take(),
                ElementType::F32 => pipelines
                    .native
                    .pq2_linear_f32_complete_aligned_scale
                    .take(),
                _ => unreachable!(),
            };
            let original = pipelines.hadamard_native_dispatch_for_input(
                GgufBlockFormat::Pq2_0,
                output,
                fixture.params,
                &regions[3],
                16,
            );
            assert_eq!(
                select(&pipelines, fixture.params, GgufBlockFormat::Pq2_0, output),
                (original.0 as *const _, original.1)
            );
            submit(&pipelines);
            match output {
                ElementType::F16 => {
                    pipelines.native.pq2_linear_f32_f16_complete_aligned_scale = saved
                }
                ElementType::F32 => pipelines.native.pq2_linear_f32_complete_aligned_scale = saved,
                _ => unreachable!(),
            }
            // If the old complete-eight route is unavailable, the aligned
            // PSO alone must not expand its cohort beyond the original route.
            let saved_control = match output {
                ElementType::F16 => pipelines.native.pq2_linear_f32_f16_complete.take(),
                ElementType::F32 => pipelines.native.pq2_linear_f32_complete.take(),
                _ => unreachable!(),
            };
            let original = pipelines.hadamard_native_dispatch_for_input(
                GgufBlockFormat::Pq2_0,
                output,
                fixture.params,
                &regions[3],
                16,
            );
            assert_eq!(
                select(&pipelines, fixture.params, GgufBlockFormat::Pq2_0, output),
                (original.0 as *const _, original.1)
            );
            submit(&pipelines);
            match output {
                ElementType::F16 => pipelines.native.pq2_linear_f32_f16_complete = saved_control,
                ElementType::F32 => pipelines.native.pq2_linear_f32_complete = saved_control,
                _ => unreachable!(),
            }
        }
    }
}
