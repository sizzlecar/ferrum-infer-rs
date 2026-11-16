use super::*;
use crate::fast_h3::video_vae::{
    sample_fast_h3_video_posterior, FastH3ImageSpatialPolicy, FastH3VideoTemporalPolicy,
    PreparedFastH3TiledVideoEncoder,
};

#[test]
#[ignore = "requires CUDA, explicit original checkpoint case and reference geometry"]
fn released_tiled_video_reference_completes_original_clips_and_normalized_posterior() {
    let (case, a) = read_case();
    let geometry = case
        .reference_geometry
        .expect("explicit reference geometry");
    let archive =
        Arc::new(SafetensorsArchive::open_with_index(&case.vae_dir, &case.index_file).unwrap());
    let prepared = PreparedFastH3TiledVideoEncoder::prepare(
        &a,
        archive,
        geometry,
        FastH3VideoTemporalPolicy::AllFrames,
        FastH3ImageSpatialPolicy::FastVideoTiled {
            maximum_tiles: 1024,
        },
        FastH3VideoEncoderLimits {
            maximum_metadata_bytes: 1 << 20,
            maximum_plan_nodes: 4096,
            maximum_stored_bytes: case.maximum_stored_bytes,
            maximum_activation_bytes: case.maximum_activation_bytes,
        },
        FastH3VideoEncoderProfile::Cuda,
    )
    .unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            case.host_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: case.host_bytes,
                transfer_staging_bytes: case.host_bytes,
                output_bytes: case.host_bytes,
                weight_backing_bytes: case.host_bytes,
            },
        )
        .unwrap(),
    );
    let (runtime, registry, catalog) = composition(case.device_bytes);
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let catalog = materializers.augment_catalog(catalog).unwrap();
    let d = runtime.descriptor();
    let encoder = prepared.tile_encoder();
    let compiled = CompiledTensorComponent::compile(
        encoder.family().clone(),
        &registry,
        &catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &materializers,
        encoder.compile_options(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(encoder.weight_source()),
            ComponentInitializationConfig {
                run_id: id("run.released_tiles.init").unwrap(),
                request_id: id("request.released_tiles.init").unwrap(),
                transaction_id: id("transaction.released_tiles.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(8 << 20, 256).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let mut data = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            prepared.input_spec().minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for (i, value) in data.as_mut().iter_mut().enumerate() {
        *value = ((i * 7 + i / 31) % 251) as u8;
    }
    let input_sha256 = format!("{:x}", Sha256::digest(data.as_ref()));
    let pixels = ManagedHostTensor::new(
        id("value.released_tiles.pixels").unwrap(),
        prepared.input_spec().clone(),
        data.freeze(),
    )
    .unwrap();
    let start = Instant::now();
    let mut request = prepared
        .begin(
            &pixels,
            id("request.released_tiles.execute").unwrap(),
            &host,
        )
        .unwrap();
    while !request
        .advance(&mut session, id("run.released_tiles.execute").unwrap())
        .unwrap()
    {}
    let completed_clips = request.completed_clips();
    let executed_tiles = request.executed_tiles();
    let moments = request.finish().unwrap();
    let posterior = sample_fast_h3_video_posterior(
        &a,
        moments.tensor(),
        moments.bytes(),
        id("value.released_tiles.posterior").unwrap(),
        &host,
    )
    .unwrap();
    let execute_copy_posterior_seconds = start.elapsed().as_secs_f64();
    let dims = posterior.tensor().dimensions();
    let expected_frames = geometry.frames.div_ceil(a.clip_length() as u64)
        * (a.clip_length() as u64).div_ceil(a.temporal_compression_ratio().unwrap() as u64)
        - a.token_drop() as u64;
    assert_eq!(
        dims,
        [
            geometry.batch,
            a.latent_channels() as u64,
            expected_frames,
            geometry.height / a.spatial_compression_ratio().unwrap() as u64,
            geometry.width / a.spatial_compression_ratio().unwrap() as u64
        ]
    );
    assert!(posterior
        .bytes()
        .bytes()
        .chunks_exact(4)
        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
    std::fs::create_dir_all(&case.output_directory).unwrap();
    for (name, bytes) in [
        ("tiled-moments.f32", moments.bytes()),
        ("tiled-posterior.f32", posterior.bytes()),
    ] {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(case.output_directory.join(name))
            .unwrap()
            .write_all(bytes.bytes())
            .unwrap();
    }
    eprintln!(
        "{}",
        serde_json::json!({"input_sha256":input_sha256,"input_shape":prepared.input_spec().dimensions(),"output_shape":dims,
        "moments_sha256":format!("{:x}",Sha256::digest(moments.bytes().bytes())),"posterior_sha256":format!("{:x}",Sha256::digest(posterior.bytes().bytes())),
        "completed_clips":completed_clips,"executed_tiles":executed_tiles,"output_peak_plan_bytes":prepared.output_peak_bytes(&host).unwrap(),
        "execute_copy_posterior_seconds":execute_copy_posterior_seconds})
    );
    assert!(session.close().is_ok());
    drop((moments, posterior, pixels, compiled, prepared));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
