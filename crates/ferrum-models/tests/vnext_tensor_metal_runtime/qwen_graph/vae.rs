use super::*;

fn config() -> QwenImage21VaeDecoderConfig {
    // Full five-block architecture, including three first-frame temporal
    // shortcut maps and the final spatial shortcut; only widths/image reduced.
    QwenImage21VaeDecoderConfig::new(
        2,
        2,
        vec![1, 2, 4, 8, 8],
        2,
        vec![true, true, true, false],
        2,
        1,
        2,
    )
    .unwrap()
}
fn encoder_config() -> QwenImage21VaeEncoderConfig {
    QwenImage21VaeEncoderConfig::new(
        2,
        2,
        vec![1, 2, 4, 8, 8],
        2,
        vec![false, true, true, true],
        2,
        16,
        32,
    )
    .unwrap()
}
pub(super) fn maximum_command_count() -> usize {
    let prepared = TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config()).unwrap(),
            &id(QwenImage21VaeDecoderFamily::METAL_PROFILE_ID),
        )
        .unwrap();
    let decoder = prepared.program().inputs().len()
        + prepared
            .program()
            .blocks()
            .iter()
            .map(|b| b.nodes.len())
            .sum::<usize>();
    let encoder = TypedFamilyRegistration::new(QwenImage21VaeEncoderFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(encoder_config()).unwrap(),
            &id(QwenImage21VaeEncoderFamily::METAL_PROFILE_ID),
        )
        .unwrap();
    decoder.max(
        encoder.program().inputs().len()
            + encoder
                .program()
                .blocks()
                .iter()
                .map(|b| b.nodes.len())
                .sum::<usize>(),
    )
}
pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
) {
    let config = config();
    let input = fixture::inputs(&config.input_tensor_specs().unwrap());
    let options = config.compile_options().unwrap();
    let inspected = execute(
        runtime,
        registry,
        catalog,
        host,
        QwenImage21VaeDecoderFamily::new().unwrap(),
        &config,
        QwenImage21VaeDecoderFamily::METAL_PROFILE_ID,
        &options,
        &input,
        true,
    );
    let ordinary = execute(
        runtime,
        registry,
        catalog,
        host,
        QwenImage21VaeDecoderFamily::new().unwrap(),
        &config,
        QwenImage21VaeDecoderFamily::METAL_PROFILE_ID,
        &options,
        &input,
        false,
    );
    assert_eq!(ordinary.len(), 1);
    let output = id::<ProgramValueId>(QwenImage21VaeDecoderFamily::OUTPUT_ID);
    assert_eq!(ordinary[&output], inspected[&output]);
    assert_eq!(ordinary[&output].len(), 2 * 4 * 16 * 32);
    eprintln!("Metal full five-block VAE decoder: all numerical boundaries, first-frame shortcuts, and original RGBA-only output agree");
    let config = encoder_config();
    let input = fixture::inputs(&config.input_tensor_specs().unwrap());
    let options = config.compile_options().unwrap();
    let inspected = execute(
        runtime,
        registry,
        catalog,
        host,
        QwenImage21VaeEncoderFamily::new().unwrap(),
        &config,
        QwenImage21VaeEncoderFamily::METAL_PROFILE_ID,
        &options,
        &input,
        true,
    );
    let ordinary = execute(
        runtime,
        registry,
        catalog,
        host,
        QwenImage21VaeEncoderFamily::new().unwrap(),
        &config,
        QwenImage21VaeEncoderFamily::METAL_PROFILE_ID,
        &options,
        &input,
        false,
    );
    assert_eq!(ordinary.len(), 1);
    let output = id::<ProgramValueId>(QwenImage21VaeEncoderFamily::OUTPUT_ID);
    assert_eq!(ordinary[&output], inspected[&output]);
    assert_eq!(ordinary[&output].len(), 2 * 2 * 1 * 2);
    eprintln!("Metal full five-block VAE encoder: downsample/first-frame mean, all boundaries and original posterior-mean output agree");
}
