use super::{config::Configuration, retained_failure::Failure};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentInitializationReceipt,
        ComponentWeightSource, TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use ferrum_models::fast_h3::video_vae::*;
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
#[path = "input.rs"]
mod input;
#[path = "runtime.rs"]
mod runtime;
fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).expect("diagnostic ID")
}
fn failure(e: impl std::fmt::Display + 'static) -> Failure {
    Failure::retain("prepare", e, ())
}
struct Counted {
    source: FastH3VideoClipSource,
    reads: AtomicUsize,
}
impl WeightComponentSource for Counted {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.component(c)
    }
}
struct Phase {
    bundle: runtime::Bundle,
    compiled: Arc<CompiledTensorComponent<CudaDeviceRuntime>>,
    source: Arc<Counted>,
    latent: ManagedHostBytes,
    cosine: ManagedHostTensor,
    sine: ManagedHostTensor,
}
pub(super) fn run(c: Configuration) -> Result<Value, Failure> {
    c.validate().map_err(failure)?;
    let started = Instant::now();
    let mut bytes = Vec::new();
    File::open(&c.vae_config)
        .map_err(failure)?
        .take(c.maximum_metadata_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(failure)?;
    let architecture =
        FastH3VideoVaeConfig::parse(&bytes, c.maximum_metadata_bytes).map_err(failure)?;
    if c.latent_shape[1] != architecture.latent_channels() {
        return Err(failure("latent channels differ from original VAE"));
    }
    let parent = c
        .packed_decoder
        .parent()
        .ok_or_else(|| failure("packed decoder parent missing"))?;
    let name = c
        .packed_decoder
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| failure("decoder filename is not UTF-8"))?;
    let archive = SafetensorsArchive::open_single_file(parent, name).map_err(failure)?;
    let package = Arc::new(
        FastH3VideoDecoderPackage::new(
            &architecture,
            archive,
            FastH3VideoDecoderWeightLimits {
                maximum_tensors: c.maximum_tensors,
                maximum_source_bytes: c.maximum_source_bytes,
                maximum_metadata_bytes: c.maximum_metadata_bytes,
            },
        )
        .map_err(failure)?,
    );
    let source_payload_bytes = package.source_bytes();
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
    let latent = input::load(&c, &host).map_err(failure)?;
    let latent_sha = format!("{:x}", Sha256::digest(latent.bytes()));
    let [batch, _, time, height, width] = c.latent_shape;
    let ratio = architecture.spatial_compression_ratio().map_err(failure)?;
    let pixel_height = height
        .checked_mul(ratio)
        .ok_or_else(|| failure("pixel height overflow"))?;
    let pixel_width = width
        .checked_mul(ratio)
        .ok_or_else(|| failure("pixel width overflow"))?;
    let mut temporal = FastH3VideoTemporalAssembler::new(
        &architecture,
        time,
        batch,
        pixel_height,
        pixel_width,
        FastH3VideoTemporalLimits {
            maximum_output_frames: c.maximum_output_frames,
            maximum_buffer_bytes: c.host_bytes,
        },
    )
    .map_err(failure)?;
    let plan = temporal.plan().clone();
    let spatial =
        FastH3VideoSpatialPlan::new(&architecture, c.latent_shape, c.spatial, c.host_bytes)
            .map_err(failure)?;
    let grid = spatial
        .clip_grid(plan.chunk(0).map_err(failure)?.latent_count)
        .map_err(failure)?;
    let sequence = grid[1..]
        .iter()
        .try_fold(1u64, |n, v| n.checked_mul(*v))
        .and_then(|n| n.checked_add(architecture.register_tokens() as u64 + 1))
        .ok_or_else(|| failure("clip rows overflow"))?;
    let rows = sequence
        .checked_mul(batch as u64)
        .ok_or_else(|| failure("batch rows overflow"))?;
    let nodes = architecture
        .decoder_layers()
        .checked_mul(32)
        .and_then(|n| n.checked_add(10))
        .ok_or_else(|| failure("node count overflow"))?;
    let limits = FastH3VideoClipLimits {
        blocks: FastH3VideoDecoderBlocksLimits {
            maximum_rows: rows,
            maximum_blocks: architecture.decoder_layers(),
            maximum_stored_bytes: c.device_bytes,
            maximum_metadata_bytes: c.maximum_metadata_bytes,
        },
        maximum_plan_nodes: nodes,
        maximum_activation_bytes: c.device_bytes,
    };
    let (config, source) =
        FastH3VideoClipSource::prepare(package, grid, &host, limits).map_err(failure)?;
    let prepared =
        TypedFamilyRegistration::new(FastH3VideoClipFamily::new(limits).map_err(failure)?)
            .prepare_with_profile(
                &serde_json::to_value(&config).map_err(failure)?,
                &NumericalProfileId::new(FastH3VideoClipFamily::PROFILE_ID).map_err(failure)?,
            )
            .map_err(failure)?;
    let ModelWeights::Stored(schema) = prepared.weights() else {
        return Err(failure("full decoder requires Stored source"));
    };
    let source_components = schema.components.len();
    let schema_sha = schema.fingerprint().map_err(failure)?;
    let selected_bytes = schema
        .components
        .iter()
        .try_fold(0u64, |n, v| n.checked_add(v.physical_bytes().ok()?))
        .ok_or_else(|| failure("source extent overflow"))?;
    let bundle = runtime::build(
        &c,
        architecture.decoder_layers(),
        nodes
            .checked_add(16)
            .ok_or_else(|| failure("command count overflow"))?,
    )
    .map_err(failure)?;
    let descriptor = bundle.runtime.descriptor();
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            prepared,
            &bundle.registry,
            &bundle.catalog,
            &TensorComponentPolicy::new(
                c.device_bytes,
                descriptor
                    .dynamic_storage_profiles
                    .iter()
                    .copied()
                    .collect(),
            )
            .map_err(failure)?,
            &bundle.materializers,
            &config.compile_options(limits).map_err(failure)?,
        )
        .map_err(|e| {
            Failure::retain(
                "compile",
                e,
                (bundle.runtime.clone(), bundle.registry.clone()),
            )
        })?,
    );
    let plan_hash = compiled
        .executable()
        .execution_plan()
        .plan_hash()
        .to_owned();
    let memory = compiled.executable().execution_plan().payload().memory();
    let plan_memory = json!({"static_bytes":memory.static_bytes(),"minimum_step_bytes":memory.minimum_step_bytes()});
    let (cosine, sine) = config
        .managed_bf16_frequencies(&host, limits)
        .map_err(failure)?;
    // Refuse to mix output with any prior or partial run.
    std::fs::create_dir(&c.output_dir).map_err(failure)?;
    let mut files = (0..batch)
        .map(|b| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(c.output_dir.join(format!("batch-{b}.rgb")))
        })
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(failure)?;
    let source = Arc::new(Counted {
        source,
        reads: AtomicUsize::new(0),
    });
    let phase = Arc::new(Phase {
        bundle,
        compiled: compiled.clone(),
        source,
        latent,
        cosine,
        sine,
    });
    let mut session = compiled
        .initialize_retained(
            phase.bundle.runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(phase.source.as_ref()),
            ComponentInitializationConfig {
                run_id: id("run.video_decode.init"),
                request_id: id("request.video_decode.init"),
                transaction_id: id("transaction.video_decode.init"),
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
    let initialization = match session.session().initialization_receipt() {
        ComponentInitializationReceipt::Initialized { receipt } => {
            json!({"components":receipt.uploaded_component_count(),"bytes":receipt.uploaded_bytes()})
        }
        _ => {
            return Err(Failure::retain(
                "initialize receipt",
                "expected actual source upload",
                (session, phase),
            ))
        }
    };
    let prepare_seconds = started.elapsed().as_secs_f64();
    let execution_started = Instant::now();
    macro_rules! owned {
        ($stage:literal,$expr:expr) => {
            match $expr {
                Ok(v) => v,
                Err(e) => return Err(Failure::retain($stage, e, (session, phase))),
            }
        };
    }
    let mut written = 0usize;
    for ordinal in 0..plan.chunk_count() {
        let chunk = owned!("temporal descriptor", plan.chunk(ordinal));
        let mut canvas = owned!(
            "spatial canvas",
            spatial.begin_canvas(chunk.raw_frames, &host)
        );
        for tile in 0..spatial.tile_count() {
            let input = owned!(
                "latent tile",
                spatial.managed_tile_input(
                    &plan,
                    ordinal,
                    tile,
                    c.latent_units,
                    &phase.latent,
                    &host
                )
            );
            let outputs = owned!(
                "tile execution",
                session.session_mut().execute(
                    id(format!("run.video_decode.{ordinal}.{tile}")),
                    id(format!("request.video_decode.{ordinal}.{tile}")),
                    vec![input, phase.cosine.clone(), phase.sine.clone()]
                )
            );
            if outputs.len() != 1 {
                return Err(Failure::retain(
                    "tile output",
                    "expected one patch tensor",
                    (session, phase),
                ));
            }
            let raw = owned!(
                "unpack tile",
                config.unpack_execution(&outputs[0], &host, limits)
            );
            owned!("spatial raw-neighbour stitch", canvas.push(tile, &raw));
        }
        let raw = owned!("complete spatial canvas", canvas.finish());
        let pending = owned!(
            "temporal prepare",
            temporal.prepare_chunk(ordinal, &raw, &host)
        );
        if let Some(chunk) = pending.commit() {
            let rgb = owned!("pixel conversion", chunk.managed_rgb8(&host, c.host_bytes));
            owned!("write RGB", write_rgb(&rgb, &mut files, written));
            written += rgb.nthwc_shape()[1];
        }
        if phase.source.reads.load(Ordering::Relaxed) != source_components {
            return Err(Failure::retain(
                "source reuse",
                "source was reread after initialization",
                (session, phase),
            ));
        }
        eprintln!(
            "decoded chunk {}/{}; finalized frames {}/{}",
            ordinal + 1,
            plan.chunk_count(),
            written,
            plan.output_frames()
        );
    }
    let pending = owned!("temporal finish", temporal.prepare_finish(&host));
    if let Some(chunk) = pending.commit() {
        let rgb = owned!("tail pixels", chunk.managed_rgb8(&host, c.host_bytes));
        owned!("write tail RGB", write_rgb(&rgb, &mut files, written));
        written += rgb.nthwc_shape()[1];
    }
    if written != plan.output_frames() || !temporal.is_finished() {
        return Err(Failure::retain(
            "final frame count",
            "incomplete temporal output",
            (session, phase),
        ));
    }
    let execution_seconds = execution_started.elapsed().as_secs_f64();
    let (returned, receipt) = session
        .close()
        .map_err(|e| Failure::retain("close", e, phase.clone()))?;
    for f in &mut files {
        f.flush()
            .map_err(|e| Failure::retain("flush", e, phase.clone()))?;
    }
    let released_static = receipt.released_static_resources();
    drop(returned);
    drop(phase);
    drop(compiled);
    drop(temporal);
    let host_remaining = host.usage().map_err(failure)?.committed_bytes();
    if host_remaining != 0 {
        return Err(failure(format!(
            "Host retirement incomplete: {host_remaining} bytes"
        )));
    }
    let report = json!({"schema":"ferrum.fast_h3.video_decode.v1","scope":"actual full declared-depth video decoder; no DiT/text/audio or MP4 encoding","configuration":c,
        "decoder_layers":architecture.decoder_layers(),"decoder_nodes":nodes,"rotary_profile":FastH3VideoClipConfig::ROTARY_PROFILE_ID,"temporal_profile":FastH3VideoTemporalAssembler::NUMERICAL_PROFILE_ID,
        "input_sha256":latent_sha,"original_package_payload_bytes":source_payload_bytes,"selected_source_components":source_components,"selected_source_bytes":selected_bytes,"source_schema_fingerprint":schema_sha,
        "plan_hash":plan_hash,"plan_memory":plan_memory,"initialization":initialization,"chunks":plan.chunk_count(),"frames":written,"pixel_height":pixel_height,"pixel_width":pixel_width,
        "artifact":"one batch-N.rgb per batch; contiguous uint8 [frames,height,width,3]","spatial_policy":c.spatial,"spatial_tiles_per_chunk":spatial.tile_count(),"spatial_blend_profile":FastH3VideoSpatialPlan::BLEND_PROFILE_ID,"frame_rate":c.frame_rate,"preparation_seconds":prepare_seconds,"execution_and_copy_seconds":execution_seconds,
        "finite_outputs_checked":true,"released_static_resources":released_static,"host_committed_bytes_after_retirement":host_remaining,"quality_parity":"not established by this forward"});
    let mut manifest = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(c.output_dir.join("manifest.json"))
        .map_err(failure)?;
    manifest
        .write_all(&serde_json::to_vec_pretty(&report).map_err(failure)?)
        .map_err(failure)?;
    Ok(report)
}
fn write_rgb(rgb: &FastH3VideoRgbChunk, files: &mut [File], start: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        rgb.output_frame_start() == start && rgb.nthwc_shape()[0] == files.len(),
        "RGB output routing mismatch"
    );
    let stride = rgb.bytes().len() / files.len();
    for (file, bytes) in files
        .iter_mut()
        .zip(rgb.bytes().bytes().chunks_exact(stride))
    {
        file.write_all(bytes)?;
    }
    Ok(())
}
