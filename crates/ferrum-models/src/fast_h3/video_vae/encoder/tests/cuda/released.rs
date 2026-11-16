//! Reusable original-checkpoint qualification. A bounded explicit case file
//! supplies hardware/model paths and capacity; no machine/commit is a test gate.
use super::*;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::PathBuf,
    time::Instant,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    vae_dir: PathBuf,
    vae_config: PathBuf,
    index_file: String,
    geometry: FastH3VideoEncoderGeometry,
    #[serde(default)]
    reference_geometry: Option<FastH3VideoEncoderGeometry>,
    host_bytes: usize,
    device_bytes: u64,
    maximum_stored_bytes: u64,
    maximum_activation_bytes: u64,
    output_directory: PathBuf,
}
fn read_case() -> (Case, FastH3VideoVaeConfig) {
    let path = std::env::var_os("FERRUM_TEST_VIDEO_ENCODER_CASE").expect("explicit case file");
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .take(1_048_577)
        .read_to_end(&mut bytes)
        .unwrap();
    assert!(bytes.len() <= 1_048_576);
    let case: Case = serde_json::from_slice(&bytes).unwrap();
    let mut metadata = Vec::new();
    std::fs::File::open(&case.vae_config)
        .unwrap()
        .take(1_048_577)
        .read_to_end(&mut metadata)
        .unwrap();
    let a = FastH3VideoVaeConfig::parse(&metadata, 1 << 20).unwrap();
    (case, a)
}
#[test]
#[ignore = "requires CUDA, explicit cuDNN and FERRUM_TEST_VIDEO_ENCODER_CASE original checkpoint case"]
fn released_causal_encoder_preserves_past_frames_and_retires_original_weights() {
    let (case, a) = read_case();
    assert_eq!(
        case.geometry.frames,
        a.clip_length() as u64,
        "qualify one complete original clip"
    );
    assert!(case.geometry.frames > 1);
    let archive =
        Arc::new(SafetensorsArchive::open_with_index(&case.vae_dir, &case.index_file).unwrap());
    let limits = FastH3VideoEncoderLimits {
        maximum_metadata_bytes: 1 << 20,
        maximum_plan_nodes: 4096,
        maximum_stored_bytes: case.maximum_stored_bytes,
        maximum_activation_bytes: case.maximum_activation_bytes,
    };
    let prepared = PreparedFastH3VideoEncoder::prepare(
        &a,
        archive,
        case.geometry,
        limits,
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
    let compiled = CompiledTensorComponent::compile(
        prepared.family().clone(),
        &registry,
        &catalog,
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        &materializers,
        prepared.compile_options(),
    )
    .unwrap();
    let init = Instant::now();
    let mut session = compiled
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(prepared.weight_source()),
            ComponentInitializationConfig {
                run_id: id("run.released_video_encoder.init").unwrap(),
                request_id: id("request.released_video_encoder.init").unwrap(),
                transaction_id: id("transaction.released_video_encoder.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(8 << 20, 256).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let init_seconds = init.elapsed().as_secs_f64();
    std::fs::create_dir_all(&case.output_directory).unwrap();
    let g = case.geometry;
    let source_len = prepared.input_spec().minimum_storage_bytes().unwrap() as usize;
    let mut previous: Vec<f32> = vec![];
    for trial in 0..2 {
        let mut data = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, source_len)
            .unwrap();
        for (i, word) in data.as_mut().chunks_exact_mut(4).enumerate() {
            let value = ((i * 7 + i / 31) % 97) as f32 / 64. - 0.75;
            let last = (i as u64 / (g.height * g.width)) % g.frames == g.frames - 1;
            let value = value
                + if trial == 1 && last && i % 3 == 0 {
                    0.5
                } else {
                    0.
                };
            word.copy_from_slice(&value.to_le_bytes());
        }
        let input_sha256 = format!("{:x}", Sha256::digest(data.as_ref()));
        let input = ManagedHostTensor::new(
            id(PreparedFastH3VideoEncoder::INPUT_ID).unwrap(),
            prepared.input_spec().clone(),
            data.freeze(),
        )
        .unwrap();
        let start = Instant::now();
        let outputs = session
            .execute(
                id("run.released_video_encoder.execute").unwrap(),
                id("request.released_video_encoder.execute").unwrap(),
                vec![input],
            )
            .unwrap();
        let execution_seconds = start.elapsed().as_secs_f64();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].tensor(), prepared.output_spec());
        let output = &outputs[0];
        let actual: Vec<f32> = output
            .bytes()
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(actual.iter().all(|x| x.is_finite()));
        let dims = output.tensor().dimensions();
        if trial == 1 {
            for (i, v) in actual.iter().enumerate() {
                let time = (i as u64 / (dims[3] * dims[4])) % dims[2];
                if time * (a.temporal_compression_ratio().unwrap() as u64) < g.frames - 1 {
                    assert_eq!(
                        *v, previous[i],
                        "released encoder future affected earlier output"
                    );
                }
            }
            assert_ne!(
                actual, previous,
                "released encoder failed to consume final frame"
            );
        }
        let path = case.output_directory.join(format!("moments-{trial}.f32"));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(output.bytes().bytes())
            .unwrap();
        eprintln!(
            "{}",
            serde_json::json!({"trial":trial,"input_sha256":input_sha256,"output_shape":dims,"output_sha256":format!("{:x}",Sha256::digest(output.bytes().bytes())),
            "output_bytes":output.bytes().len(),"initialization_seconds":init_seconds,"execute_copy_seconds":execution_seconds,"past_frames_equal":trial==1})
        );
        previous = actual;
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    drop((compiled, prepared));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

mod tiled;
