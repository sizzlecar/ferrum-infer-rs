use super::{
    config::{Configuration, Input},
    retained_failure::Failure,
};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_cnn_f32_providers, CudaTensorOperation},
    vnext_runtime::{CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_models::fast_h3::video_vae::{
    sample_fast_h3_keyframe_posterior, FastH3ImageEncoderLimits, FastH3ImageEncoderProfile,
    FastH3VideoVaeConfig, PreparedFastH3ImageEncoder, PreparedFastH3TiledImageEncoder,
    FAST_H3_IMAGE_TILE_PROFILE, FAST_H3_KEYFRAME_POSTERIOR_PROFILE,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    sync::Arc,
    time::Instant,
};
fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).expect("diagnostic ID")
}
fn failure(e: impl std::fmt::Display + 'static) -> Failure {
    Failure::retain("prepare", e, ())
}
pub(super) fn run(c: Configuration) -> Result<Value, Failure> {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            c.host_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: c.host_bytes,
                transfer_staging_bytes: c.host_bytes,
                output_bytes: c.host_bytes,
                weight_backing_bytes: c.host_bytes,
            },
        )
        .map_err(failure)?,
    );
    let mut bytes = Vec::new();
    File::open(&c.vae_config)
        .map_err(failure)?
        .take(c.maximum_metadata_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(failure)?;
    let architecture =
        FastH3VideoVaeConfig::parse(&bytes, c.maximum_metadata_bytes).map_err(failure)?;
    let archive =
        Arc::new(SafetensorsArchive::open_with_index(&c.vae_dir, &c.index_file).map_err(failure)?);
    let [batch, _, height, width] = c.input_shape;
    let prepared = Arc::new(
        PreparedFastH3TiledImageEncoder::prepare(
            &architecture,
            archive,
            batch,
            height,
            width,
            c.spatial,
            FastH3ImageEncoderLimits {
                maximum_metadata_bytes: c.maximum_metadata_bytes,
                maximum_stored_bytes: c.maximum_stored_bytes,
                maximum_plan_nodes: c.maximum_plan_nodes,
                maximum_input_bytes: c.host_bytes as u64,
                maximum_output_bytes: c.host_bytes as u64,
            },
            FastH3ImageEncoderProfile::Cuda,
            &host,
        )
        .map_err(failure)?,
    );
    let mut input = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            c.input_bytes().map_err(failure)?,
        )
        .map_err(failure)?;
    match &c.input {
        Input::File { path } => {
            let mut f = File::open(path).map_err(failure)?;
            if f.metadata().map_err(failure)?.len() != input.len() as u64 {
                return Err(failure("normalized F32 input length differs"));
            }
            f.read_exact(input.as_mut()).map_err(failure)?;
        }
        Input::DiagnosticDyadic => {
            for (i, word) in input.as_mut().chunks_exact_mut(4).enumerate() {
                let x = (1 + i % 31) as f32 / 64.;
                let x = if (i / 31) % 2 == 0 { x } else { -x };
                word.copy_from_slice(&x.to_le_bytes());
            }
        }
    }
    if !input
        .as_ref()
        .chunks_exact(4)
        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
    {
        return Err(failure("nonfinite image input"));
    }
    let input = ManagedHostTensor::new(
        id(PreparedFastH3ImageEncoder::INPUT_ID),
        prepared.input_spec().clone(),
        input.freeze(),
    )
    .map_err(failure)?;
    let input_sha = format!("{:x}", Sha256::digest(input.bytes().bytes()));
    let nodes = prepared
        .family()
        .program()
        .blocks()
        .iter()
        .map(|b| b.nodes.len())
        .sum::<usize>();
    let device = cuda_tensor_runtime_config(c.device, id(format!("device.cuda.{}", c.device)))
        .map_err(failure)?;
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device,
            usable_memory_bytes: c.device_bytes,
            bootstrap_memory_bytes: c.bootstrap_bytes,
            stream_memory_bytes: c.stream_bytes,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: nodes
                .checked_mul(2)
                .and_then(|n| n.checked_add(16))
                .ok_or_else(|| failure("command count overflow"))?,
            opaque_memory_profile: "fast_h3.original_t1_encoder.v1".into(),
        })
        .map_err(failure)?,
    );
    let providers = cuda_vnext_cnn_f32_providers(
        &runtime,
        &CudaCudnnConfig {
            library_path: c.cudnn_library.clone(),
            dependency_libraries: c.cudnn_dependencies.clone(),
            opaque_memory_bytes: c.cudnn_opaque_bytes,
            workspace_limit_bytes: c.cudnn_workspace_bytes,
            opaque_memory_profile: "fast_h3.original_t1_encoder.cudnn9.v1".into(),
        },
        &[
            CudaTensorOperation::GroupNormSiluNchwF32,
            CudaTensorOperation::PlainResidualAddNchwF32,
        ],
    )
    .map_err(|e| Failure::retain("providers", e, runtime.clone()))?;
    let registry = Arc::new(
        OperationRuntimeRegistry::new(
            vec![
                Box::new(conv2d_f32_contract().map_err(failure)?),
                Box::new(group_norm_silu_nchw_f32_contract().map_err(failure)?),
                Box::new(plain_residual_add_nchw_f32_contract().map_err(failure)?),
            ],
            providers,
        )
        .map_err(|e| Failure::retain("registry", e, runtime.clone()))?,
    );
    let materializers = WeightMaterializerRegistry::identity_only().map_err(failure)?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.fast_h3.keyframe_encoder"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .map_err(failure)?;
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .map_err(failure)?,
        )
        .map_err(failure)?;
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            prepared.family().clone(),
            &registry,
            &catalog,
            &TensorComponentPolicy::new(
                c.device_bytes,
                d.dynamic_storage_profiles.iter().copied().collect(),
            )
            .map_err(failure)?,
            &materializers,
            prepared.compile_options(),
        )
        .map_err(|e| Failure::retain("compile", e, (runtime.clone(), registry.clone())))?,
    );
    let plan_hash = compiled
        .executable()
        .execution_plan()
        .plan_hash()
        .to_owned();
    let schema = match prepared.family().weights() {
        ModelWeights::Stored(s) => s,
        ModelWeights::None => return Err(failure("encoder must retain original stored weights")),
    };
    let source_identity = json!({"fingerprint":schema.fingerprint().map_err(failure)?,"components":schema.components.len(),"selected_bytes":schema.components.iter().map(|c|c.physical_bytes()).collect::<Result<Vec<_>,_>>().map_err(failure)?.iter().sum::<u64>()});
    let source = prepared.weight_source(host.clone());
    let phase = Arc::new((
        prepared.clone(),
        runtime.clone(),
        registry.clone(),
        compiled.clone(),
        input.clone(),
    ));
    let mut session = compiled
        .initialize_retained(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.keyframe.init"),
                request_id: id("request.keyframe.init"),
                transaction_id: id("transaction.keyframe.init"),
                static_policy: StaticInitializationPolicy::new(
                    u64::try_from((c.host_bytes / 8).clamp(4096, 8 << 20)).map_err(failure)?,
                    64,
                )
                .map_err(failure)?,
                executor: TensorExecutorConfig::default(),
            },
            phase.clone(),
        )
        .map_err(|e| Failure::retain("initialize", e, phase.clone()))?;
    let began = Instant::now();
    let mut request = match prepared.begin(&input, id("request.keyframe"), &host) {
        Ok(r) => r,
        Err(e) => return Err(Failure::retain("tile_prepare", e, (session, phase))),
    };
    loop {
        let run = id(format!("run.keyframe.tile.{}", request.completed_tiles()));
        match request.advance(session.session_mut(), run) {
            Ok(true) => break,
            Ok(false) => {}
            Err(e) => {
                return Err(Failure::retain(
                    "tile_execute_or_commit",
                    e,
                    (session, phase, request),
                ))
            }
        }
    }
    let executed_tiles = request.executed_tiles();
    let out = match request.finish() {
        Ok(m) => m,
        Err(e) => return Err(Failure::retain("tile_finish", e, (session, phase))),
    };
    let elapsed = began.elapsed().as_secs_f64();
    let (returned, close) = session
        .close()
        .map_err(|e| Failure::retain("close", e, phase.clone()))?;
    let output_shape = out.tensor().dimensions().to_vec();
    let output_sha = format!("{:x}", Sha256::digest(out.bytes().bytes()));
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&c.output_file)
        .map_err(failure)?;
    f.write_all(out.bytes().bytes()).map_err(failure)?;
    f.flush().map_err(failure)?;
    let posterior_report = if let Some(path) = &c.posterior_file {
        let posterior = sample_fast_h3_keyframe_posterior(
            &architecture,
            out.tensor(),
            out.bytes(),
            id("value.fast_h3.normalized_keyframe"),
            &host,
        )
        .map_err(failure)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(failure)?;
        file.write_all(posterior.bytes().bytes()).map_err(failure)?;
        file.sync_all().map_err(failure)?;
        json!({"path":path,"shape":posterior.tensor().dimensions(),"bytes":posterior.bytes().len(),
            "sha256":format!("{:x}",Sha256::digest(posterior.bytes().bytes())),
            "profile":FAST_H3_KEYFRAME_POSTERIOR_PROFILE,"dtype":"F32 little-endian normalized NC1HW",
            "seed_per_reference":42,"torch_avx2_bitwise_parity_claim":false})
    } else {
        Value::Null
    };
    let output_bytes = out.bytes().len();
    let released = close.released_static_resources();
    drop(out);
    drop(returned);
    drop(phase);
    drop(source);
    drop(prepared);
    drop(input);
    drop(compiled);
    let remaining = host.usage().map_err(failure)?.committed_bytes();
    if remaining != 0 {
        return Err(failure(format!("Host retirement incomplete: {remaining}")));
    }
    Ok(
        json!({"schema":"ferrum.fast_h3.keyframe_encoder.v1","configuration":c,"input_sha256":input_sha,"output_shape":output_shape,"output_bytes":output_bytes,"output_sha256":output_sha,"output_dtype":"F32 little-endian posterior moments N(2C)HW","source":source_identity,"plan_hash":plan_hash,"nodes_per_tile":nodes,"executed_tiles":executed_tiles,"tile_blend_profile":FAST_H3_IMAGE_TILE_PROFILE,"execution_seconds":elapsed,"released_static_resources":released,"host_committed_after_retirement":remaining,"posterior_sampling_executed":c.posterior_file.is_some(),"posterior":posterior_report,"scope":"full original T1 causal CNN with explicit spatial policy and original raw-neighbour moment stitching; posterior F16 boundary preserved; upstream binary parity and video encoder unqualified"}),
    )
}
