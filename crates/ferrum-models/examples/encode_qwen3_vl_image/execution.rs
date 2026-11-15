use super::{
    config::{self, Attention, Configuration, Image, Presentation},
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
        cuda_tensor_runtime_config, cuda_vnext_cnn_f32_providers_with_segmented_sdpa,
        cuda_vnext_tensor_providers, CudaTensorOperation as Op,
    },
    vnext_runtime::{
        CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig, CudnnSdpaPlanPolicy,
    },
};
use ferrum_models::{qwen3_vl_vision::*, qwen_image21::image_input::ManagedRgba8Image};
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
    T::try_from(s.into()).expect("constant/diagnostic ID")
}
fn fail(e: impl std::fmt::Display + 'static) -> Failure {
    Failure::retain("prepare", e, ())
}
struct Count<'a> {
    inner: Qwen3VLVisionCheckpointSource<'a>,
    reads: AtomicUsize,
}
impl WeightComponentSource for Count<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.component(c)
    }
}
pub(super) fn run(c: Configuration) -> Result<Value, Failure> {
    let start = Instant::now();
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
        .map_err(fail)?,
    );
    let metadata =
        config::read(&c.encoder_dir.join("config.json"), c.maximum_metadata_bytes).map_err(fail)?;
    let raw = serde_json::from_slice(&metadata).map_err(fail)?;
    let architecture = Qwen3VLVisionArchitecture::from_hf_value(&raw).map_err(fail)?;
    let processor_bytes = config::read(&c.processor, c.maximum_metadata_bytes).map_err(fail)?;
    let image_limits = c.image_limits().map_err(fail)?;
    let processor = Qwen3VLVisionImageProcessor::from_source_bytes(
        &processor_bytes,
        Qwen3VLVisionProcessorLimits {
            maximum_metadata_bytes: c.maximum_metadata_bytes,
            image: image_limits,
            maximum_patch_rows: c.maximum_patch_rows,
            maximum_patch_bytes: c.maximum_patch_bytes,
        },
    )
    .map_err(fail)?;
    let image = match &c.image {
        Image::Png { path } => {
            let mut f = File::open(path).map_err(fail)?;
            let n = usize::try_from(f.metadata().map_err(fail)?.len()).map_err(fail)?;
            if n > c.maximum_image_encoded_bytes {
                return Err(fail("encoded image exceeds capacity"));
            }
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, n)
                .map_err(fail)?;
            f.read_exact(bytes.as_mut()).map_err(fail)?;
            ManagedRgba8Image::decode_png(&host, &bytes.freeze(), image_limits).map_err(fail)?
        }
        Image::DiagnosticRgb { width, height } => {
            let n = u64::from(*width)
                .checked_mul(u64::from(*height))
                .ok_or_else(|| fail("image extent overflow"))?;
            if *width == 0
                || *height == 0
                || *width > c.maximum_image_axis
                || *height > c.maximum_image_axis
                || n > c.maximum_image_pixels as u64
            {
                return Err(fail("diagnostic image exceeds capacity"));
            }
            let bytes = usize::try_from(
                n.checked_mul(4)
                    .ok_or_else(|| fail("image extent overflow"))?,
            )
            .map_err(fail)?;
            let mut data = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
                .map_err(fail)?;
            for (i, pixel) in data.as_mut().chunks_exact_mut(4).enumerate() {
                pixel.copy_from_slice(&[
                    (i % 251) as u8,
                    ((i * 3) % 251) as u8,
                    ((i * 7) % 251) as u8,
                    255,
                ]);
            }
            ManagedRgba8Image::from_rgba8(&host, data.freeze(), *width, *height, image_limits)
                .map_err(fail)?
        }
    };
    let original_image_sha = format!("{:x}", Sha256::digest(image.bytes().bytes()));
    let (image, presentation) = match &c.presentation {
        Presentation::ProcessorInput => (image, Value::Null),
        Presentation::FastH3Reference {
            short_edge,
            vae_pixels_output,
        } => {
            use ferrum_models::fast_h3::{
                FastH3ReferenceImagePlan, FAST_H3_REFERENCE_IMAGE_PROFILE,
            };
            let plan = FastH3ReferenceImagePlan::new(
                image.width(),
                image.height(),
                *short_edge,
                image_limits,
            )
            .map_err(fail)?;
            let prepared = plan.prepare(&image, &host).map_err(fail)?;
            drop(image);
            let pixels = prepared
                .managed_vae_pixels(&host, c.host_bytes)
                .map_err(fail)?;
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(vae_pixels_output)
                .map_err(fail)?;
            file.write_all(pixels.bytes().bytes()).map_err(fail)?;
            file.sync_all().map_err(fail)?;
            let report = json!({"profile":FAST_H3_REFERENCE_IMAGE_PROFILE,"short_edge":short_edge,
                "prepared_width":prepared.image().width(),"prepared_height":prepared.image().height(),
                "vae_pixels":{"path":vae_pixels_output,"dtype":"F32 little-endian normalized NCHW",
                    "shape":pixels.tensor().dimensions(),"sha256":format!("{:x}",Sha256::digest(pixels.bytes().bytes()))}});
            (prepared.into_image(), report)
        }
    };
    let image_sha = format!("{:x}", Sha256::digest(image.bytes().bytes()));
    let image_plan = processor
        .prepare_image(architecture, image.width(), image.height())
        .map_err(fail)?;
    let checkpoint = Arc::new(
        PreparedQwen3VLVisionCheckpoint::prepare(
            image_plan.config().clone(),
            Arc::new(config::archive(&c).map_err(fail)?),
            &NumericalProfileId::new(Qwen3VLVisionFamily::CUDA_PROFILE_ID).map_err(fail)?,
            Qwen3VLVisionCheckpointLimits {
                maximum_metadata_bytes: c.maximum_metadata_bytes,
                maximum_plan_nodes: c.maximum_plan_nodes,
                maximum_source_bytes: c.maximum_source_bytes,
            },
        )
        .map_err(fail)?,
    );
    let mut inputs = checkpoint
        .managed_position_inputs(&host)
        .map_err(fail)?
        .to_vec();
    inputs.push(image_plan.managed_patches(&host, &image).map_err(fail)?);
    drop(image);
    let family = checkpoint.family();
    let nodes = family
        .program()
        .blocks()
        .iter()
        .map(|b| b.nodes.len())
        .sum::<usize>();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(c.device, id(format!("device.cuda.{}", c.device)))
                .map_err(fail)?,
            usable_memory_bytes: c.device_bytes,
            bootstrap_memory_bytes: c.bootstrap_bytes,
            stream_memory_bytes: c.stream_bytes,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: nodes
                .checked_mul(2)
                .and_then(|n| n.checked_add(16))
                .ok_or_else(|| fail("command capacity overflow"))?,
            opaque_memory_profile: "qwen3vl.original_bf16_vision.v1".into(),
        })
        .map_err(fail)?,
    );
    let selected = [
        Op::AffineLinear,
        Op::PatchProjection,
        Op::AffineLayerNorm,
        Op::GeluErf,
        Op::GeluTanh,
        Op::HalfRotary,
        Op::ReshapeCopy,
        Op::SegmentedAttention,
        Op::RowGather,
        Op::PlainResidualAdd,
    ];
    let providers = match &c.attention {
        Attention::SequentialDiagnostic => cuda_vnext_tensor_providers(&runtime, &selected),
        Attention::Cudnn {
            library,
            dependencies,
            opaque_bytes,
            workspace_bytes,
            cache_bytes,
        } => cuda_vnext_cnn_f32_providers_with_segmented_sdpa(
            &runtime,
            &CudaCudnnConfig {
                library_path: library.clone(),
                dependency_libraries: dependencies.clone(),
                opaque_memory_bytes: *opaque_bytes,
                workspace_limit_bytes: *workspace_bytes,
                opaque_memory_profile: "qwen3vl.vision.cudnn9.v1".into(),
            },
            CudnnSdpaPlanPolicy {
                maximum_plans: 1,
                maximum_live_bindings: usize::try_from(checkpoint.config().architecture().depth())
                    .map_err(fail)?,
                opaque_memory_bytes: *cache_bytes,
                workspace_limit_bytes: *workspace_bytes,
            },
            &selected,
        ),
    }
    .map_err(|e| Failure::retain("providers", e, runtime.clone()))?;
    let registry = Arc::new(
        OperationRuntimeRegistry::new(
            vec![
                Box::new(conv2d_f32_contract().map_err(fail)?),
                Box::new(affine_linear_bf16_contract().map_err(fail)?),
                Box::new(patch_projection_bf16_contract().map_err(fail)?),
                Box::new(affine_layer_norm_bf16_contract().map_err(fail)?),
                Box::new(gelu_erf_bf16_contract().map_err(fail)?),
                Box::new(gelu_tanh_bf16_contract().map_err(fail)?),
                Box::new(half_rotary_bf16_contract().map_err(fail)?),
                Box::new(reshape_copy_bf16_contract().map_err(fail)?),
                Box::new(segmented_attention_bf16_contract().map_err(fail)?),
                Box::new(row_gather_bf16_contract().map_err(fail)?),
                Box::new(plain_residual_add_bf16_contract().map_err(fail)?),
            ],
            providers,
        )
        .map_err(|e| Failure::retain("registry", e, runtime.clone()))?,
    );
    let materializers = WeightMaterializerRegistry::identity_only().map_err(fail)?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.qwen3vl.original_vision"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .map_err(fail)?;
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .map_err(fail)?,
        )
        .map_err(fail)?;
    let compiled = Arc::new(
        CompiledTensorComponent::compile(
            family.clone(),
            &registry,
            &catalog,
            &TensorComponentPolicy::new(
                c.device_bytes,
                d.dynamic_storage_profiles.iter().copied().collect(),
            )
            .map_err(fail)?,
            &materializers,
            checkpoint.compile_options(),
        )
        .map_err(|e| Failure::retain("compile", e, (runtime.clone(), registry.clone())))?,
    );
    let schema = family.stored_weight_schema().map_err(fail)?;
    let component_count = schema.components.len();
    let fingerprint = schema.fingerprint().map_err(fail)?;
    let plan_hash = compiled
        .executable()
        .execution_plan()
        .plan_hash()
        .to_owned();
    let source = Count {
        inner: checkpoint.managed_source(&host),
        reads: AtomicUsize::new(0),
    };
    let phase = Arc::new((
        checkpoint.clone(),
        runtime.clone(),
        registry.clone(),
        compiled.clone(),
        inputs.clone(),
    ));
    let mut session = compiled
        .initialize_retained(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.vision.init"),
                request_id: id("request.vision.init"),
                transaction_id: id("transaction.vision.init"),
                static_policy: StaticInitializationPolicy::new(
                    u64::try_from((c.host_bytes / 8).clamp(4096, 8 << 20)).map_err(fail)?,
                    64,
                )
                .map_err(fail)?,
                executor: TensorExecutorConfig::default(),
            },
            phase.clone(),
        )
        .map_err(|e| Failure::retain("initialize", e, phase.clone()))?;
    if source.reads.load(Ordering::Relaxed) != component_count {
        return Err(Failure::retain(
            "source",
            "visual components must upload exactly once",
            (session, phase),
        ));
    }
    let prepare_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let outputs = match session.session_mut().execute(
        id("run.vision.forward"),
        id("request.vision.forward"),
        inputs.clone(),
    ) {
        Ok(o) => o,
        Err(e) => return Err(Failure::retain("execute", e, (session, phase))),
    };
    let execute_seconds = start.elapsed().as_secs_f64();
    let (returned, close) = session
        .close()
        .map_err(|e| Failure::retain("close", e, phase.clone()))?;
    let expected = [
        Qwen3VLVisionFamily::OUTPUT_ID,
        Qwen3VLVisionFamily::DEEPSTACK_IDS[0],
        Qwen3VLVisionFamily::DEEPSTACK_IDS[1],
        Qwen3VLVisionFamily::DEEPSTACK_IDS[2],
    ];
    if outputs.len() != expected.len() {
        return Err(fail(
            "vision must publish main plus all three DeepStack outputs",
        ));
    }
    std::fs::create_dir_all(&c.output_dir).map_err(fail)?;
    let mut artifacts = Vec::new();
    for (stage, value) in expected.into_iter().enumerate() {
        let o = outputs
            .iter()
            .find(|o| o.value_id().as_str() == value)
            .ok_or_else(|| fail("missing visual output"))?;
        if o.tensor().element_type() != ElementType::Bf16
            || o.tensor().dimensions()
                != [
                    checkpoint.config().merged_rows().map_err(fail)?,
                    checkpoint.config().architecture().output_size(),
                ]
            || !o
                .bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| u16::from_le_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80)
        {
            return Err(fail("visual output shape/type/finite check failed"));
        }
        let name = if stage == 0 {
            "merged.bf16".into()
        } else {
            format!("deepstack-{}.bf16", stage - 1)
        };
        let path = c.output_dir.join(name);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(fail)?;
        file.write_all(o.bytes().bytes()).map_err(fail)?;
        file.flush().map_err(fail)?;
        artifacts.push(json!({"value_id":value,"file":path,"element_type":"bf16","dimensions":o.tensor().dimensions(),"sha256":format!("{:x}",Sha256::digest(o.bytes().bytes()))}));
    }
    let grid = &checkpoint.config().grids()[0];
    let mut report = json!({"schema":"ferrum.qwen3_vl.vision_forward.v1","configuration":c,"metadata_sha256":format!("{:x}",Sha256::digest(metadata)),"processor_sha256":format!("{:x}",Sha256::digest(processor_bytes)),"input_rgba_sha256":image_sha,"original_rgba_sha256":original_image_sha,"presentation":presentation,"grid":[grid.temporal(),grid.height(),grid.width()],"merge_size":checkpoint.config().architecture().merge_size(),"layers":checkpoint.config().architecture().depth(),"deepstack_layers":checkpoint.config().architecture().deepstack_layers(),"source_fingerprint":fingerprint,"selected_source_bytes":checkpoint.selected_source_bytes(),"stored_components":component_count,"source_component_reads":source.reads.load(Ordering::Relaxed),"plan_hash":plan_hash,"nodes":nodes,"outputs":artifacts,"preprocessing_profile":image_plan.preprocessing_profile(),"position_profile":QWEN3_VL_VISION_POSITION_PROFILE,"prepare_seconds":prepare_seconds,"execute_and_copy_seconds":execute_seconds,"released_static_resources":close.released_static_resources(),"scope":"complete original BF16 visual graph and all DeepStack outputs; numerical/resize parity with upstream Torch binary and generation quality remain unqualified"});
    drop(outputs);
    drop(returned);
    drop(phase);
    drop(source);
    drop(checkpoint);
    drop(inputs);
    drop(compiled);
    let remaining = host.usage().map_err(fail)?.committed_bytes();
    if remaining != 0 {
        return Err(fail(format!(
            "visual Host retirement incomplete: {remaining}"
        )));
    }
    report["host_committed_after_retirement"] = json!(remaining);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(c.output_dir.join("manifest.json"))
        .map_err(fail)?;
    serde_json::to_writer_pretty(&mut file, &report).map_err(fail)?;
    file.flush().map_err(fail)?;
    Ok(report)
}
