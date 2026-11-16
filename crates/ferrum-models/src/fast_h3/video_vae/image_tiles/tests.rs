use super::*;
use crate::p_h3_image_vae::encoder::{
    CausalVaeEncoderSourceLayout, PH3ImageVaeEncoderConfig, PH3ImageVaeEncoderFamily,
    PH3ImageVaeEncoderLimits,
};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::collections::BTreeMap;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 32 << 20,
                output_bytes: 32 << 20,
                transfer_staging_bytes: 32 << 20,
                weight_backing_bytes: 32 << 20,
            },
        )
        .unwrap(),
    )
}
fn architecture() -> FastH3VideoVaeConfig {
    let mut c = serde_json::to_value(super::super::tests::config()).unwrap();
    c["block_out_channels"] = serde_json::json!([2, 2, 2, 2, 2, 2]);
    c["spatial_downsample_factors"] = serde_json::json!([2, 2, 2, 2, 1, 1]);
    c["temporal_downsample_factors"] = serde_json::json!([1, 2, 2, 1, 1, 1]);
    c["norm_num_groups"] = serde_json::json!(1);
    FastH3VideoVaeConfig::parse(&serde_json::to_vec(&c).unwrap(), 4096).unwrap()
}
fn archive(a: &FastH3VideoVaeConfig) -> (tempfile::TempDir, Arc<SafetensorsArchive>) {
    let c = PH3ImageVaeEncoderConfig::new(a.causal_encoder_architecture().unwrap(), 2, 32, 32)
        .unwrap()
        .with_source_layout(CausalVaeEncoderSourceLayout::Diffusers);
    let family = PH3ImageVaeEncoderFamily::new(PH3ImageVaeEncoderLimits {
        max_metadata_bytes: 1 << 20,
        max_plan_nodes: 1024,
        max_stored_bytes: 1 << 20,
    })
    .unwrap();
    let weights = family.weights(&c).unwrap();
    let mut originals = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    for component in &weights.stored_schema().unwrap().components {
        let name = component.external_names[0].clone();
        let mut shape = component
            .dimensions
            .iter()
            .map(|&n| n as usize)
            .collect::<Vec<_>>();
        if shape.len() == 4 {
            shape.insert(2, shape[2]);
        }
        let salt = name.bytes().map(usize::from).sum::<usize>();
        let values = (0..shape.iter().product::<usize>())
            .flat_map(|i| {
                let v = if name.contains("norm") && name.ends_with("weight") {
                    1. + (i % 3) as f32 / 16.
                } else {
                    ((i * 7 + salt) % 17) as f32 / 128. - 0.0625
                };
                v.to_le_bytes()
            })
            .collect();
        originals.insert(name, (shape, values));
    }
    let dir = tempfile::tempdir().unwrap();
    let views = originals
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &dir.path().join("encoder.safetensors")).unwrap();
    let archive =
        Arc::new(SafetensorsArchive::open_single_file(dir.path(), "encoder.safetensors").unwrap());
    (dir, archive)
}
fn read(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}

#[test]
fn original_tile_session_commits_raw_neighbours_once_and_retains_complete_moments() {
    let a = architecture();
    let (_dir, source) = archive(&a);
    let host = host();
    let limits = FastH3ImageEncoderLimits {
        maximum_metadata_bytes: 1 << 20,
        maximum_stored_bytes: 1 << 20,
        maximum_plan_nodes: 1024,
        maximum_input_bytes: 2 << 20,
        maximum_output_bytes: 1 << 20,
    };
    // Public default uses the fixed baseline geometry, never the request's full
    // extent as a replacement for per-tile normalization.
    let baseline = PreparedFastH3TiledImageEncoder::prepare(
        &a,
        source.clone(),
        1,
        320,
        320,
        FastH3ImageSpatialPolicy::default(),
        limits,
        FastH3ImageEncoderProfile::Reference,
        &host,
    )
    .unwrap();
    assert_eq!(baseline.tile_count(), 4);
    assert_eq!(
        baseline.tile_encoder().input_spec().dimensions(),
        [1, 3, 256, 256]
    );
    drop(baseline);
    // Scaled traversal exercises the same complete six-stage CNN and all
    // residual blocks while keeping this hardware-independent test small.
    let encoder = PreparedFastH3ImageEncoder::prepare(
        &a,
        source,
        2,
        32,
        32,
        limits,
        FastH3ImageEncoderProfile::Reference,
        &host,
    )
    .unwrap();
    let p = PreparedFastH3TiledImageEncoder {
        fingerprint: encoder.family().fingerprint().unwrap(),
        encoder,
        plan: PH3ImageVaeTilePlan::new(
            48,
            48,
            16,
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: 32,
                minimum_overlap_pixels: 16,
                max_tiles: 4,
            },
        )
        .unwrap(),
        input: spec(vec![2, 3, 48, 48]).unwrap(),
        output: spec(vec![2, 4, 3, 3]).unwrap(),
    };
    let composition =
        ReferenceVNextComposition::create(id("device.reference.keyframe_tiles")).unwrap();
    let d = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        p.family().clone(),
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        p.compile_options(),
    )
    .unwrap();
    let weights = p.weight_source(host.clone());
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&weights),
            ComponentInitializationConfig {
                run_id: id("run.tile.init"),
                request_id: id("request.tile.init"),
                transaction_id: id("transaction.tile.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 128).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let mut pixels = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 2 * 3 * 48 * 48 * 4)
        .unwrap();
    for (i, b) in pixels.as_mut().chunks_exact_mut(4).enumerate() {
        b.copy_from_slice(
            &(((i * 7 + i / (3 * 48 * 48) * 13) % 67) as f32 / 32. - 0.875).to_le_bytes(),
        );
    }
    let input = ManagedHostTensor::new(
        id("value.normalized_pixels"),
        p.input_spec().clone(),
        pixels.freeze(),
    )
    .unwrap();
    assert!(p
        .begin(&input, id("request.foreign"), &self::host())
        .is_err());
    let mut request = p.begin(&input, id("request.tiles"), &host).unwrap();
    // Independent crop from full NCHW bytes: both requests/channels stay distinct.
    let mut raw = Vec::new();
    for index in 0..4 {
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 2 * 3 * 32 * 32 * 4)
            .unwrap();
        for bc in 0..6 {
            for y in 0..32 {
                for x in 0..32 {
                    let src = ((bc * 48 + y + index / 2 * 16) * 48 + x + index % 2 * 16) * 4;
                    let dst = ((bc * 32 + y) * 32 + x) * 4;
                    bytes.as_mut()[dst..dst + 4]
                        .copy_from_slice(&input.bytes().bytes()[src..src + 4]);
                }
            }
        }
        let tensor = ManagedHostTensor::new(
            id(PreparedFastH3ImageEncoder::INPUT_ID),
            p.tile_encoder().input_spec().clone(),
            bytes.freeze(),
        )
        .unwrap();
        let output = session
            .execute(
                id(format!("run.raw.{index}")),
                id("request.raw"),
                vec![tensor],
            )
            .unwrap();
        raw.push(output[0].bytes().bytes().to_vec());
    }
    for index in 0..4 {
        assert_eq!(
            request
                .advance(&mut session, id(format!("run.tile.{index}")))
                .unwrap(),
            index == 3
        );
        assert_eq!(request.executed_tiles(), index + 1);
    }
    assert!(request.advance(&mut session, id("run.extra")).is_err());
    let moments = request.finish().unwrap();
    assert_eq!(moments.tensor().dimensions(), [2, 4, 3, 3]);
    // With a single latent overlap, the seam takes its raw upper row and then
    // raw left column. It must not recursively reuse a blended predecessor.
    let mut expected = vec![0.; 2 * 4 * 3 * 3];
    for index in 0..4 {
        let mut tile = (0..2 * 4 * 2 * 2)
            .map(|i| read(&raw[index], i))
            .collect::<Vec<_>>();
        let row = index / 2;
        let col = index % 2;
        for bc in 0..8 {
            if row > 0 {
                for x in 0..2 {
                    tile[bc * 4 + x] = read(&raw[index - 2], bc * 4 + 2 + x);
                }
            }
            if col > 0 {
                for y in 0..2 {
                    tile[bc * 4 + y * 2] = read(&raw[index - 1], bc * 4 + y * 2 + 1);
                }
            }
            for y in 0..if row == 0 { 1 } else { 2 } {
                for x in 0..if col == 0 { 1 } else { 2 } {
                    expected[(bc * 3 + row + y) * 3 + col + x] = tile[bc * 4 + y * 2 + x];
                }
            }
        }
    }
    for (i, x) in expected.iter().enumerate() {
        assert_eq!(
            read(moments.bytes().bytes(), i).to_bits(),
            x.to_bits(),
            "moment {i}"
        );
    }
    assert!(session.close().is_ok());
    drop(weights);
    drop(p);
    drop(input);
    drop(compiled);
    drop(composition);
    assert!(host.owns(moments.bytes()));
    drop(moments);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
