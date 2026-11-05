use super::*;
use ferrum_quantization::SafetensorsArchive;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
type Family = PH3ImageVaeEncoderVerticalFamily;
type Tensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
type Values = BTreeMap<ProgramValueId, Vec<f32>>;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn architecture(tiny: bool) -> PH3ImageVaeEncoderArchitecture {
    PH3ImageVaeEncoderArchitecture::from_hf_json(&serde_json::to_vec(&serde_json::json!({
    "_class_name":"AutoencoderKLPH3Image","in_channels":3,"latent_channels":24,
    "block_out_channels":if tiny{vec![4,8]}else{vec![128,256,256,512,512,1024]},"layers_per_block":2,
    "spatial_downsample_factors":if tiny{vec![2,2]}else{vec![2,2,2,2,1,1]},
    "temporal_downsample_factors":if tiny{vec![1,2]}else{vec![1,2,2,1,1,1]},
    "norm_num_groups":if tiny{2}else{32},"norm_eps":0.000001,"spatial_padding_mode":"reflect",
})).unwrap()).unwrap()
}
fn config(batch: u64) -> PH3ImageVaeEncoderVerticalConfig {
    PH3ImageVaeEncoderVerticalConfig::new(architecture(true), batch, 17, 19, 1).unwrap()
}
fn limits() -> PH3ImageVaeEncoderVerticalLimits {
    PH3ImageVaeEncoderVerticalLimits {
        max_metadata_bytes: 1 << 20,
        max_stored_bytes: 128 << 20,
        max_plan_nodes: 64,
    }
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            16 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4 << 20,
                output_bytes: 4 << 20,
                transfer_staging_bytes: 4 << 20,
                weight_backing_bytes: 4 << 20,
            },
        )
        .unwrap(),
    )
}
fn bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
fn originals(c: &PH3ImageVaeEncoderVerticalConfig) -> Tensors {
    weights::entries(c)
        .into_iter()
        .map(|entry| {
            let count = entry.original.iter().product::<u64>() as usize;
            let shape = entry
                .original
                .iter()
                .map(|&n| n as usize)
                .collect::<Vec<_>>();
            let values = (0..count).map(|i| {
                if shape.len() == 5 {
                    let plane = shape[3] * shape[4];
                    let temporal = i / plane % shape[2];
                    let base = ((i * 7 + i / plane) % 17) as f32 / 2048. - 0.003 + 0.000001;
                    if temporal + 1 == shape[2] {
                        base
                    } else if temporal == 0 {
                        base + 2.
                    } else {
                        base - 3.
                    }
                } else if entry.name.contains("norm") && entry.name.ends_with("weight") {
                    0.75 + (i % 5) as f32 / 16. + 0.000123
                } else {
                    ((i * 3) % 11) as f32 / 256. - 0.015625 + 0.000123
                }
            });
            let payload = bytes(values);
            (entry.name, (Dtype::F32, shape, payload))
        })
        .collect()
}
fn archive(t: &Tensors) -> (tempfile::TempDir, SafetensorsArchive) {
    let d = tempfile::tempdir().unwrap();
    let views = t
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &d.path().join("weights.safetensors")).unwrap();
    let source = SafetensorsArchive::open_single_file(d.path(), "weights.safetensors").unwrap();
    (d, source)
}
fn input(c: &PH3ImageVaeEncoderVerticalConfig, perturb: bool) -> Vec<f32> {
    let per = (c.architecture().in_channels() * c.height() * c.width()) as usize;
    (0..c.batch() as usize * per)
        .map(|i| {
            ((i * 7 + i / per * 13) % 43) as f32 / 32. - 0.625
                + 0.000123
                + if perturb && i / per == 1 { 0.875 } else { 0. }
        })
        .collect()
}
#[test]
fn released_f32_header_inventory_and_literal_single_frame_shapes_are_checked() {
    let c = PH3ImageVaeEncoderVerticalConfig::new(architecture(false), 2, 33, 35, 1).unwrap();
    let entries = weights::entries(&c);
    for (name, shape) in [
        ("encoder.conv_in.weight", vec![128, 3, 3, 3, 3]),
        (
            "encoder.down.1.block.0.conv1.weight",
            vec![256, 128, 3, 3, 3],
        ),
        (
            "encoder.down.1.block.0.conv2.weight",
            vec![256, 256, 3, 3, 3],
        ),
        (
            "encoder.down.1.block.0.nin_shortcut.weight",
            vec![256, 128, 1, 1, 1],
        ),
        (
            "encoder.down.1.downsample.conv.weight",
            vec![256, 256, 3, 3, 3],
        ),
    ] {
        assert_eq!(
            entries.iter().find(|e| e.name == name).unwrap().original,
            shape
        );
    }
    let f = Family::new(limits()).unwrap();
    let prepared = TypedFamilyRegistration::new(f)
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &id(Family::REFERENCE_PROFILE_ID),
        )
        .unwrap();
    let options = c.compile_options().unwrap();
    for node in prepared.program().blocks().iter().flat_map(|b| &b.nodes) {
        for output in &node.outputs {
            assert!(options.tensor_specs().contains_key(output));
        }
    }
    assert_eq!(
        options.tensor_specs()[&id::<ProgramValueId>(Family::OUTPUT_ID)].dimensions,
        [2, 256, 16, 17]
    );
    assert!(PH3ImageVaeEncoderVerticalConfig::new(architecture(false), 1, 17, 19, 2).is_err());
}
#[test]
fn last_temporal_tap_is_packed_per_channel_pair_under_real_weightbacking_and_wrong_source_dtype_rejects(
) {
    let c = config(2);
    let data = originals(&c);
    let (_d, source) = archive(&data);
    let host = host();
    let checkpoint =
        PH3ImageVaeEncoderVerticalCheckpoint::new(&c, &source, host.clone(), limits()).unwrap();
    let ModelWeights::Stored(schema) = Family::new(limits()).unwrap().weights(&c).unwrap() else {
        unreachable!()
    };
    for component in &schema.components {
        let entry = weights::entries(&c)
            .into_iter()
            .find(|e| e.name == component.external_names[0])
            .unwrap();
        let original = source.tensor(&entry.name).unwrap();
        let payload = checkpoint.component(component).unwrap();
        if entry.original.len() == 5 && entry.original[2] > 1 {
            let plane = (entry.original[3] * entry.original[4] * 4) as usize;
            let pairs = (entry.original[0] * entry.original[1]) as usize;
            for pair in 0..pairs {
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
    let mut bad = data;
    let w = bad.get_mut("encoder.conv_in.weight").unwrap();
    w.0 = Dtype::BF16;
    w.2.truncate(w.2.len() / 2);
    let (_d, wrong) = archive(&bad);
    assert!(PH3ImageVaeEncoderVerticalCheckpoint::new(&c, &wrong, host, limits()).is_err());
}
mod execution;
mod oracle;
