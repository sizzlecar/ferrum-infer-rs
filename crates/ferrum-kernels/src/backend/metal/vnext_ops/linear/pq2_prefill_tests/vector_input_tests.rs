//! Exercise the production selector with retained regions and real submissions.

use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};

pub(super) fn region<T: Copy>(
    runtime: &MetalDeviceRuntime,
    name: &str,
    values: &[T],
    dtype: ElementType,
) -> MetalBufferRegion {
    let region = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new(name).unwrap(),
                std::mem::size_of_val(values) as u64,
                64,
                BufferUsage::Transfer,
                dtype,
            )
            .unwrap(),
        )
        .unwrap();
    // SAFETY: the allocation has exactly the declared initialized byte length.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            std::mem::size_of_val(values),
        );
    }
    region
}

pub(super) fn read<T: Copy>(region: &MetalBufferRegion) -> Vec<T> {
    // SAFETY: callers read matching element types after command completion.
    unsafe {
        std::slice::from_raw_parts(
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize)
                .cast::<T>(),
            region.length_bytes() as usize / std::mem::size_of::<T>(),
        )
        .to_vec()
    }
}

#[test]
fn pq2_vector_input_production_dispatch_uses_actual_region_and_workspace_binding() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.vector-input.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let mut pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    // Workspace-local offsets obey the production 16-byte allocation rule.
    // A retained parent slice can still move the final physical binding to20B.
    for (rows, outputs, region_offset, expected_vector, wide_scale) in [
        (64, 4096, 0_u64, true, true),
        (64, 4096, 4, false, false),
        (64, 4097, 0, false, false),
        (65, 4096, 0, false, false),
    ] {
        const WIDTH: u32 = 128;
        const WORKSPACE_OFFSET: u64 = 16;
        let fixture = Fixture::new(runtime.device(), rows, WIDTH, outputs, wide_scale);
        let input_count = rows as usize * WIDTH as usize;
        let mut workspace_values = vec![GUARD; (region_offset + WORKSPACE_OFFSET) as usize / 4];
        workspace_values
            .extend_from_slice(&fixture.input_values[INPUT_PREFIX..INPUT_PREFIX + input_count]);
        workspace_values.extend([GUARD; 8]);
        let workspace_allocation = region(
            runtime,
            "vector.workspace",
            &workspace_values,
            ElementType::U8,
        );
        let workspace = workspace_allocation
            .test_subregion(region_offset..workspace_allocation.length_bytes())
            .unwrap();
        let weight_allocation = region(
            runtime,
            "vector.weight",
            &fixture.weight_bytes,
            ElementType::U8,
        );
        let weights = weight_allocation
            .test_subregion(WEIGHT_PREFIX as u64..fixture.weight_bytes.len() as u64 - 16)
            .unwrap();
        let original_input = vec![f16::ZERO; input_count];
        let regions = vec![
            workspace,
            weights,
            region(
                runtime,
                "vector.output",
                &fixture.initial_output,
                ElementType::F16,
            ),
            region(
                runtime,
                "vector.original-input",
                &original_input,
                ElementType::F16,
            ),
        ];
        let mut launch = LinearLaunch {
            input_region: 3,
            weight_region: 1,
            output_region: 2,
            input_offset_bytes: 0,
            output_offset_bytes: (OUTPUT_PREFIX * 2) as u64,
            activation_type: ElementType::F16,
            format: LinearPhysicalFormat::Native(GgufBlockFormat::Pq2_0),
            params: fixture.params,
            transform: Some(HadamardTransform {
                block_size: WIDTH,
                signs_region: None,
                inverse: false,
                permutation: None,
            }),
            transform_workspace: None,
            transformed_plan: TransformedLinearPlan::Single,
            plain_plan: PlainLinearPlan::Single,
        };
        launch
            .bind_hadamard_workspace(&pipelines, &regions, 0, WORKSPACE_OFFSET)
            .unwrap();
        validate_launch_regions(&regions, &[launch]).unwrap();
        let (selected, dispatch) = pipelines.hadamard_native_dispatch_for_input(
            GgufBlockFormat::Pq2_0,
            ElementType::F16,
            fixture.params,
            &regions[0],
            WORKSPACE_OFFSET,
        );
        let (scalar, scalar_dispatch) = pipelines.hadamard_native_dispatch(
            GgufBlockFormat::Pq2_0,
            ElementType::F16,
            fixture.params,
        );
        let vector = pipelines
            .native
            .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
            .as_ref()
            .unwrap();
        assert_eq!(std::ptr::eq(selected, vector), expected_vector);
        assert_eq!(dispatch, scalar_dispatch);
        if !expected_vector {
            assert!(std::ptr::eq(selected, scalar));
        }
        for (format, dtype) in [
            (GgufBlockFormat::Pq2_0, ElementType::F32),
            (GgufBlockFormat::Iq4Xs, ElementType::F16),
        ] {
            let actual = pipelines.hadamard_native_dispatch_for_input(
                format,
                dtype,
                fixture.params,
                &regions[0],
                WORKSPACE_OFFSET,
            );
            let original = pipelines.hadamard_native_dispatch(format, dtype, fixture.params);
            assert!(std::ptr::eq(actual.0, original.0));
            assert_eq!(actual.1, original.1);
        }
        // Populate transformed F32 input directly to isolate the actual
        // post-Hadamard production dispatch. The independent literal oracle
        // describes these exact values, not a second implementation's output.
        let submit = |pipelines: &MetalLinearPipelines| {
            // SAFETY: previous submit is complete; reset all output guards and
            // NaN poison before each production path executes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    fixture.initial_output.as_ptr(),
                    regions[2]
                        .buffer()
                        .contents()
                        .cast::<u8>()
                        .add(regions[2].offset_bytes() as usize)
                        .cast::<f16>(),
                    fixture.initial_output.len(),
                );
            }
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            dispatch_transformed_linear(pipelines, encoder, &regions, launch);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(read::<f32>(&workspace_allocation), workspace_values);
            assert_eq!(read::<u8>(&weight_allocation), fixture.weight_bytes);
            assert_eq!(read::<f16>(&regions[3]), original_input);
            fixture.validate_output_values(&read::<f16>(&regions[2]))
        };
        let actual = submit(&pipelines);
        let vector_pso = pipelines
            .native
            .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input
            .take();
        let missing = pipelines.hadamard_native_dispatch_for_input(
            GgufBlockFormat::Pq2_0,
            ElementType::F16,
            fixture.params,
            &regions[0],
            WORKSPACE_OFFSET,
        );
        let original = pipelines.hadamard_native_dispatch(
            GgufBlockFormat::Pq2_0,
            ElementType::F16,
            fixture.params,
        );
        assert!(std::ptr::eq(missing.0, original.0));
        assert_eq!(missing.1, original.1);
        assert_eq!(submit(&pipelines), actual);
        pipelines
            .native
            .pq2_gemm_input_f32_output_f16_m64_full_tiles_vector_input = vector_pso;
        // The underlying allocation has enough bytes, but the retained view
        // does not. Eligibility and launch validation must honor the view.
        let mut short_regions = regions.clone();
        let short_len = WORKSPACE_OFFSET + input_count as u64 * 4 - 1;
        short_regions[0] = regions[0].test_subregion(0..short_len).unwrap();
        assert!(short_regions[0].buffer().length() > short_len + short_regions[0].offset_bytes());
        let selected = pipelines.hadamard_native_dispatch_for_input(
            GgufBlockFormat::Pq2_0,
            ElementType::F16,
            fixture.params,
            &short_regions[0],
            WORKSPACE_OFFSET,
        );
        let scalar = pipelines.hadamard_native_dispatch(
            GgufBlockFormat::Pq2_0,
            ElementType::F16,
            fixture.params,
        );
        assert!(std::ptr::eq(selected.0, scalar.0));
        let mut invalid = launch;
        invalid.transform_workspace = None;
        assert!(invalid
            .bind_hadamard_workspace(&pipelines, &short_regions, 0, WORKSPACE_OFFSET)
            .is_err());
    }
}
