pub(in crate::p_h3_image_vae) use super::test_fixture::{
    architecture, archive, bytes, limits, originals,
};
use super::*;
use ferrum_quantization::SafetensorsArchive;
use safetensors::Dtype;
type Family = PH3ImageVaeEncoderFamily;
type Values = BTreeMap<ProgramValueId, Vec<f32>>;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn config(batch: u64) -> PH3ImageVaeEncoderConfig {
    PH3ImageVaeEncoderConfig::new(architecture(true), batch, 32, 48).unwrap()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            128 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 48 << 20,
                transfer_staging_bytes: 32 << 20,
                weight_backing_bytes: 32 << 20,
            },
        )
        .unwrap(),
    )
}
fn input(c: &PH3ImageVaeEncoderConfig, perturb: bool) -> Vec<f32> {
    let per = (3 * c.height() * c.width()) as usize;
    (0..c.batch() as usize * per)
        .map(|i| {
            ((i * 7 + i / per * 17) % 67) as f32 / 32. - 0.875
                + 0.000123
                + if perturb && i / per == 1 { 0.625 } else { 0. }
        })
        .collect()
}
#[test]
fn full_released_source_range_is_all_116_encoder_parameters_plus_two_quant_parameters() {
    let c = PH3ImageVaeEncoderConfig::new(architecture(false), 2, 32, 48).unwrap();
    let entries = weights::entries(&c).unwrap();
    assert_eq!(
        entries
            .iter()
            .filter(|e| e.name.starts_with("encoder."))
            .count(),
        116
    );
    assert_eq!(
        entries
            .iter()
            .filter(|e| e.name.starts_with("quant_conv."))
            .count(),
        2
    );
    assert!(entries
        .iter()
        .all(|e| e.name.starts_with("encoder.") || e.name.starts_with("quant_conv.")));
    for (name, shape) in [
        ("encoder.conv_out.weight", vec![48, 1024, 3, 3, 3]),
        (
            "encoder.down.5.block.0.nin_shortcut.weight",
            vec![1024, 512, 1, 1, 1],
        ),
        (
            "encoder.down.5.block.1.conv2.weight",
            vec![1024, 1024, 3, 3, 3],
        ),
        ("quant_conv.weight", vec![48, 48, 1, 1, 1]),
    ] {
        assert_eq!(
            entries.iter().find(|e| e.name == name).unwrap().original,
            shape
        );
    }
    let family = Family::new(limits()).unwrap();
    let ModelWeights::Stored(schema) = family.weights(&c).unwrap() else {
        unreachable!()
    };
    let prepared = TypedFamilyRegistration::new(family)
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &id(Family::REFERENCE_PROFILE_ID),
        )
        .unwrap();
    let options = c.compile_options().unwrap();
    assert_eq!(
        options.tensor_specs()[&id::<ProgramValueId>(Family::OUTPUT_ID)].dimensions,
        [2, 48, 2, 3]
    );
    for node in prepared.program().blocks().iter().flat_map(|b| &b.nodes) {
        for output in &node.outputs {
            assert!(options.tensor_specs().contains_key(output));
        }
    }
    let physical = schema
        .components
        .iter()
        .map(|c| c.physical_bytes().unwrap())
        .sum::<u64>();
    let mut low = limits();
    low.max_stored_bytes = physical - 1;
    assert!(Family::new(low).unwrap().weights(&c).is_err());
    let mut low = limits();
    low.max_plan_nodes = c.plan_nodes().unwrap() - 1;
    assert!(Family::new(low).unwrap().weights(&c).is_err());
}
#[test]
fn full_encoder_original_f32_source_packs_every_last_temporal_tap_and_refunds_failed_grants() {
    let c = config(2);
    let tensors = originals(&c);
    let (_dir, source) = archive(&tensors);
    let host = host();
    let checkpoint =
        PH3ImageVaeEncoderCheckpoint::new(&c, &source, host.clone(), limits()).unwrap();
    let ModelWeights::Stored(schema) = Family::new(limits()).unwrap().weights(&c).unwrap() else {
        unreachable!()
    };
    for component in &schema.components {
        let e = weights::entries(&c)
            .unwrap()
            .into_iter()
            .find(|e| e.name == component.external_names[0])
            .unwrap();
        let original = source.tensor(&e.name).unwrap();
        let payload = checkpoint.component(component).unwrap();
        if e.original.len() == 5 && e.original[2] > 1 {
            let plane = (e.original[3] * e.original[4] * 4) as usize;
            for pair in 0..(e.original[0] * e.original[1]) as usize {
                let start = (pair * 3 + 2) * plane;
                assert_eq!(
                    &payload.bytes()[pair * plane..(pair + 1) * plane],
                    &original.bytes()[start..start + plane]
                );
            }
            assert!(host.usage().unwrap().committed_bytes() > 0);
            assert!(payload.retained_host_memory().is_some());
        } else {
            assert_eq!(payload.bytes().as_ptr(), original.bytes().as_ptr());
        }
        drop(payload);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    let small = HostMemoryAccount::new(
        HostMemoryLimits::new(
            16,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 16,
            },
        )
        .unwrap(),
    );
    let limited = PH3ImageVaeEncoderCheckpoint::new(&c, &source, small.clone(), limits()).unwrap();
    assert!(limited.component(&schema.components[0]).is_err());
    assert_eq!(small.usage().unwrap().committed_bytes(), 0);
    let mut bad = tensors;
    let conv = bad.get_mut("quant_conv.weight").unwrap();
    conv.0 = Dtype::BF16;
    conv.2.truncate(conv.2.len() / 2);
    let (_dir, source) = archive(&bad);
    assert!(PH3ImageVaeEncoderCheckpoint::new(&c, &source, host, limits()).is_err());
}
mod execution;
mod oracle;
