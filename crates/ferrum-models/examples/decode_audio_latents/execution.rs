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
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_cnn_f32_with_transpose_providers,
        CudaTensorOperation,
    },
    vnext_runtime::{CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_models::fast_h3::audio_vae::{
    FastH3AudioDecoderLimits, FastH3AudioLatentLayout, FastH3AudioVaeConfig,
    PreparedFastH3AudioDecoder,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};
fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).expect("diagnostic ID")
}
fn failure(e: impl std::fmt::Display + 'static) -> Failure {
    Failure::retain("prepare", e, ())
}
struct Count<'a> {
    source: &'a PreparedFastH3AudioDecoder,
    reads: AtomicUsize,
}
impl WeightComponentSource for Count<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.component(c)
    }
}
pub(super) fn run(c: Configuration) -> Result<Value, Failure> {
    let began = Instant::now();
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
    let a = FastH3AudioVaeConfig::parse(&bytes, c.maximum_metadata_bytes).map_err(failure)?;
    let parent = c
        .vae_file
        .parent()
        .ok_or_else(|| failure("audio source has no directory"))?;
    let name = c
        .vae_file
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| failure("audio source basename must be UTF8"))?;
    let archive = Arc::new(SafetensorsArchive::open_single_file(parent, name).map_err(failure)?);
    let prepared = Arc::new(
        PreparedFastH3AudioDecoder::prepare(
            &a,
            archive,
            c.requests,
            c.latent_samples,
            &host,
            FastH3AudioDecoderLimits {
                maximum_metadata_bytes: c.maximum_metadata_bytes,
                maximum_source_bytes: c.maximum_source_bytes,
                maximum_stored_bytes: c.maximum_stored_bytes,
                maximum_plan_nodes: c.maximum_plan_nodes,
                maximum_activation_bytes: c.maximum_activation_bytes,
            },
        )
        .map_err(failure)?,
    );
    let input_bytes = usize::try_from(
        prepared
            .input_spec()
            .minimum_storage_bytes()
            .map_err(failure)?,
    )
    .map_err(failure)?;
    let mut raw = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, input_bytes)
        .map_err(failure)?;
    match &c.input {
        Input::File { path } => {
            let mut f = File::open(path).map_err(failure)?;
            if f.metadata().map_err(failure)?.len() != input_bytes as u64 {
                return Err(failure("audio file extent differs from explicit layout"));
            }
            f.read_exact(raw.as_mut()).map_err(failure)?;
        }
        Input::DiagnosticDyadic => {
            for (i, b) in raw.as_mut().chunks_exact_mut(4).enumerate() {
                let x = (1 + i % 31) as f32 / 64.;
                let x = if (i / 31) % 2 == 0 { x } else { -x };
                b.copy_from_slice(&x.to_le_bytes());
            }
        }
    }
    let input_sha = format!("{:x}", Sha256::digest(raw.as_ref()));
    let batch = prepared.input_spec().dimensions()[0];
    let channels = a.latent_channels();
    let shape = match c.input_layout {
        FastH3AudioLatentLayout::PackedStereoTokens => vec![batch * c.latent_samples, channels],
        FastH3AudioLatentLayout::MonoBatchNcl => vec![batch, channels, c.latent_samples],
    };
    let raw = ManagedHostTensor::new(
        id("value.audio.source"),
        ResolvedTensorSpec::new(shape, ElementType::F32, ResolvedTensorLayout::Contiguous)
            .map_err(failure)?,
        raw.freeze(),
    )
    .map_err(failure)?;
    let input = prepared
        .managed_input(&raw, c.input_layout, c.latent_units, &host)
        .map_err(failure)?;
    drop(raw);
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
                .checked_mul(3)
                .and_then(|n| n.checked_add(16))
                .ok_or_else(|| failure("command count overflow"))?,
            opaque_memory_profile: "fast_h3.original_audio_decoder.v1".into(),
        })
        .map_err(failure)?,
    );
    let providers = cuda_vnext_cnn_f32_with_transpose_providers(
        &runtime,
        &CudaCudnnConfig {
            library_path: c.cudnn_library.clone(),
            dependency_libraries: c.cudnn_dependencies.clone(),
            opaque_memory_bytes: c.cudnn_opaque_bytes,
            workspace_limit_bytes: c.cudnn_workspace_bytes,
            opaque_memory_profile: "fast_h3.audio_decoder.cudnn9.v1".into(),
        },
        &[
            CudaTensorOperation::PlainResidualAddNchwF32,
            CudaTensorOperation::ScaleRank4F32,
            CudaTensorOperation::FirResample1dF32,
            CudaTensorOperation::SnakeBetaLogNchwF32,
            CudaTensorOperation::ClampRank4F32,
        ],
    )
    .map_err(|e| Failure::retain("providers", e, runtime.clone()))?;
    let registry = Arc::new(
        OperationRuntimeRegistry::new(
            vec![
                Box::new(conv2d_f32_contract().map_err(failure)?),
                Box::new(conv_transpose2d_f32_contract().map_err(failure)?),
                Box::new(plain_residual_add_nchw_f32_contract().map_err(failure)?),
                Box::new(scale_rank4_f32_contract().map_err(failure)?),
                Box::new(fir_resample1d_f32_contract().map_err(failure)?),
                Box::new(snake_beta_log_nchw_f32_contract().map_err(failure)?),
                Box::new(clamp_rank4_f32_contract().map_err(failure)?),
            ],
            providers,
        )
        .map_err(|e| Failure::retain("registry", e, runtime.clone()))?,
    );
    let materializers = WeightMaterializerRegistry::identity_only().map_err(failure)?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.fast_h3.audio_decoder"),
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
        ModelWeights::None => return Err(failure("audio decoder requires original source")),
    };
    let source_fingerprint = schema.fingerprint().map_err(failure)?;
    let components = schema.components.len();
    let phase = Arc::new((
        prepared.clone(),
        runtime.clone(),
        registry.clone(),
        compiled.clone(),
        input.clone(),
    ));
    let source = Count {
        source: &prepared,
        reads: AtomicUsize::new(0),
    };
    let mut session = compiled
        .initialize_retained(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.audio.init"),
                request_id: id("request.audio.init"),
                transaction_id: id("transaction.audio.init"),
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
    if source.reads.load(Ordering::Relaxed) != components {
        return Err(Failure::retain(
            "source",
            "selected original/derived components were not uploaded exactly once",
            (session, phase),
        ));
    }
    let prepare_seconds = began.elapsed().as_secs_f64();
    let began = Instant::now();
    let outputs = match session.session_mut().execute(
        id("run.audio.forward"),
        id("request.audio.forward"),
        vec![input.clone()],
    ) {
        Ok(o) => o,
        Err(e) => return Err(Failure::retain("execute", e, (session, phase))),
    };
    let execution_seconds = began.elapsed().as_secs_f64();
    if outputs.len() != 1 {
        return Err(Failure::retain(
            "output",
            "audio must publish one waveform",
            (session, phase),
        ));
    }
    let (returned, close) = session
        .close()
        .map_err(|e| Failure::retain("close", e, phase.clone()))?;
    let waveform = prepared
        .interleave_stereo(&outputs[0], &host)
        .map_err(failure)?;
    std::fs::create_dir_all(&c.output_dir).map_err(failure)?;
    let mut artifacts = Vec::new();
    for request in 0..waveform.requests() {
        let data = waveform.request_bytes(request).map_err(failure)?;
        let path = c.output_dir.join(format!("request-{request}.wav"));
        write_wave(
            &path,
            data,
            waveform.sampling_rate(),
            waveform.samples_per_channel(),
        )
        .map_err(failure)?;
        artifacts.push(json!({"request":request,"path":path,"sample_frames":waveform.samples_per_channel(),"interleaved_f32_sha256":format!("{:x}",Sha256::digest(data))}));
    }
    let report = json!({"schema":"ferrum.fast_h3.audio_decoder.v1","configuration":c,"input_sha256":input_sha,"nodes":nodes,"plan_hash":plan_hash,"original_source_tensors":prepared.original_source_tensors(),"original_source_bytes":prepared.original_source_bytes(),"stored_components":components,"source_component_reads":source.reads.load(Ordering::Relaxed),"source_fingerprint":source_fingerprint,"sampling_rate":waveform.sampling_rate(),"stereo_channels":2,"artifacts":artifacts,"prepare_seconds":prepare_seconds,"execute_and_copy_seconds":execution_seconds,"released_static_resources":close.released_static_resources(),"scope":"complete original F32 decoder; explicit two mono items per request; diagnostic input is not generation quality or a throughput benchmark"});
    drop(waveform);
    drop(outputs);
    drop(returned);
    drop(phase);
    drop(source);
    drop(prepared);
    drop(input);
    drop(compiled);
    let remaining = host.usage().map_err(failure)?.committed_bytes();
    if remaining != 0 {
        return Err(failure(format!(
            "audio Host retirement incomplete: {remaining}"
        )));
    }
    let mut report = report;
    report["host_committed_after_retirement"] = json!(remaining);
    Ok(report)
}
fn write_wave(path: &std::path::Path, data: &[u8], rate: u32, samples: u64) -> anyhow::Result<()> {
    let count = u32::try_from(samples)?;
    let length = u32::try_from(data.len())?;
    anyhow::ensure!(
        data.len() as u64
            == samples
                .checked_mul(8)
                .ok_or_else(|| anyhow::anyhow!("wave extent overflow"))?,
        "stereo wave extent differs"
    );
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(
        &length
            .checked_add(50)
            .ok_or_else(|| anyhow::anyhow!("RIFF size overflow"))?
            .to_le_bytes(),
    )?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&18u32.to_le_bytes())?;
    f.write_all(&3u16.to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(
        &rate
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("wave rate overflow"))?
            .to_le_bytes(),
    )?;
    f.write_all(&8u16.to_le_bytes())?;
    f.write_all(&32u16.to_le_bytes())?;
    f.write_all(&0u16.to_le_bytes())?;
    f.write_all(b"fact")?;
    f.write_all(&4u32.to_le_bytes())?;
    f.write_all(&count.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&length.to_le_bytes())?;
    f.write_all(data)?;
    f.flush()?;
    Ok(())
}
