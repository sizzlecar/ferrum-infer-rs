use super::*;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, OperationCostCommand, ResourceId};
use half::f16;

fn launch(rows: u64, input: u64, output: u32) -> LinearLaunch {
    linear_launch(
        PreparedLinearPart {
            region: 1,
            format: LinearPhysicalFormat::Q4K,
            output_offset: 0,
            out_features: output,
            transform: None,
        },
        0,
        2,
        rows,
        input,
        u64::from(output),
        0,
        0,
    )
    .unwrap()
}

#[test]
fn numeric_grid_preserves_actual_padding_and_checked_boundaries() {
    let p = launch(33, 1024, 1025).params;
    let mma = grid(p, LinearDispatchKind::TiledGemm).unwrap();
    assert_eq!(mma.groups, [2, 17, 1]);
    assert_eq!(mma.padded_outputs, 64 * 1088);
    let scalar = grid(p, LinearDispatchKind::CooperativeGemv).unwrap();
    assert_eq!(scalar.groups, [257, 33, 1]);
    assert_eq!(scalar.padded_outputs, 1028 * 33);
    assert!(grid(p, LinearDispatchKind::NativeTiledGemm).is_none());
    let maximum = LinearParams {
        rows: u32::MAX,
        out_features: u32::MAX,
        ..p
    };
    assert!(grid(maximum, LinearDispatchKind::TiledGemm).is_none());
}

#[test]
fn actual_pipeline_catalog_separates_b4_split_m32_and_staged_work() {
    let device = Device::system_default().expect("selected algorithm evidence requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let m8_split = launch(8, 1024, 1024);
    let m8 = launch(8, 512, 1024);
    let m16 = launch(16, 512, 1024);
    let m32 = launch(32, 512, 1024);
    let split = dense(&pipelines, &[m8_split], 8).unwrap();
    let eight = dense(&pipelines, &[m8], 8).unwrap();
    let sixteen = dense(&pipelines, &[m16], 16).unwrap();
    let thirtytwo = dense(&pipelines, &[m32], 32).unwrap();
    split.validate_command(8, 2, 0).unwrap();
    eight.validate_command(8, 1, 0).unwrap();
    assert_ne!(split.family_signature(), eight.family_signature());
    assert_eq!(
        eight.family_signature(),
        sixteen.family_signature(),
        "HEAD uses the same actual M32 kernel for unsplit B8 and B16"
    );
    assert_eq!(
        sixteen.family_signature(),
        thirtytwo.family_signature(),
        "same actual M32 kernel and layout, different numeric work"
    );
    assert_eq!(
        sixteen.work().logical_units * 2,
        thirtytwo.work().logical_units
    );
    let mut before = SelectedCommandCostBuilderV1::new(255);
    projection(
        &mut before,
        &pipelines,
        launch(255, 1024, 1024),
        Some(staged_prefill::StagingPolicy::SwiGlu),
        2 << 20,
    )
    .unwrap();
    let before = before.finish().unwrap();
    let mut after = SelectedCommandCostBuilderV1::new(256);
    projection(
        &mut after,
        &pipelines,
        launch(256, 1024, 1024),
        Some(staged_prefill::StagingPolicy::SwiGlu),
        2 << 20,
    )
    .unwrap();
    let after = after.finish().unwrap();
    before.validate_command(255, 1, 0).unwrap();
    after.validate_command(256, 2, 0).unwrap();
    assert_ne!(before.family_signature(), after.family_signature());
    assert_eq!(before.work().staged_weight_bytes, 0);
    assert_eq!(after.work().staged_weight_bytes, 2 << 20);
    let mut unmapped = launch(8, 512, 1024);
    unmapped.format = LinearPhysicalFormat::Native(GgufBlockFormat::Iq4Xs);
    assert!(dense(&pipelines, &[unmapped], 8).is_none());
}

/// Test-only data factory, using the actual retained launch and command paths.
/// Zero Q4 coefficients provide an exact output check without creating a new
/// numerical tolerance policy. This is not model quality qualification.
pub(crate) fn runtime_fixture(
    runtime: &MetalDeviceRuntime,
) -> (
    MetalDeviceCommand,
    OperationCostCommand,
    Vec<MetalBufferRegion>,
) {
    let pipelines = Arc::new(MetalLinearPipelines::new(runtime.device()).unwrap());
    let make = |name: &str, bytes: &[u8], ty| {
        let region = runtime
            .allocate_test_region(
                &BufferRequest::new(
                    ResourceId::new(name).unwrap(),
                    bytes.len() as u64,
                    64,
                    BufferUsage::Transfer,
                    ty,
                )
                .unwrap(),
            )
            .unwrap();
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
    };
    let input = vec![f16::from_f32(1.0); 4 + 8 * 1024 + 4];
    let output = vec![f16::from_f32(333.0); 4 + 8 * 1027 + 4];
    let as_bytes = |values: &[f16]| unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2).to_vec()
    };
    let regions = vec![
        make("statistics.input", &as_bytes(&input), ElementType::F16),
        make(
            "statistics.weight",
            &vec![0_u8; 1024 * 4 * 144],
            ElementType::U8,
        ),
        make("statistics.output", &as_bytes(&output), ElementType::F16),
    ];
    let mut selected = launch(8, 1024, 1024);
    selected.params.output_stride = 1027;
    selected.params.output_column_offset = 1;
    selected.input_offset_bytes = 8;
    selected.output_offset_bytes = 8;
    selected.plain_plan = PlainLinearPlan::for_launch(selected);
    validate_launch_regions(&regions, &[selected]).unwrap();
    let projected = cost_route::dense_command_selected(&pipelines, 8, 8, &[selected]).unwrap();
    let statistics = dense(&pipelines, &[selected], 8).unwrap();
    assert_eq!(projected.statistical_evidence(), Some(&statistics));
    let retained = regions.clone();
    let command =
        MetalDeviceCommand::operation("vnext_dense_linear", regions, move |encoder, regions| {
            encoder.record_compute_dispatches(selected.dispatch_count());
            dispatch_linear(&pipelines, encoder.compute_encoder(), regions, selected);
            Ok(())
        })
        .unwrap()
        .with_work_shape(DeviceBatchingForm::Packed, 8, 8)
        .unwrap()
        .with_statistical_evidence(Some(statistics));
    (command, projected, retained)
}
