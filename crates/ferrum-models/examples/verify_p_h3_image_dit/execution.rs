use super::{
    artifacts, data, shared::id, source, trajectory, Arguments, Attention, DenseLinearSource,
    InputBoundary, Trajectory,
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutionOutput, TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::*,
    vnext_runtime::{
        CudaCudnnConfig, CudaDeviceRuntime, CudaTensorRuntimeConfig, CudnnSdpaPlanPolicy,
    },
};
use ferrum_models::p_h3_image::*;
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use ferrum_runtime::*;
use half::bf16;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::Path,
    sync::Arc,
};

fn file_sha(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn initial(args: &Arguments, name: &str) -> Result<ComponentInitializationConfig> {
    Ok(ComponentInitializationConfig {
        run_id: id(format!("run.dit.oracle.{name}"))?,
        request_id: id(format!("request.dit.oracle.{name}"))?,
        transaction_id: id(format!("transaction.dit.oracle.{name}"))?,
        static_policy: StaticInitializationPolicy::new(
            args.cuda.transfer_staging_bytes.try_into()?,
            64,
        )?,
        executor: TensorExecutorConfig::default(),
    })
}

pub(super) fn run(args: Arguments) -> Result<()> {
    let config_bytes = artifacts::read_bounded(&args.config, args.max_metadata_bytes)?;
    let metadata_bytes = artifacts::read_bounded(&args.input_metadata, args.max_metadata_bytes)?;
    let capture: data::Capture = serde_json::from_slice(&metadata_bytes)?;
    let transformer = PH3ImageTransformerConfig::from_hf_json(&config_bytes)?;
    let inputs = data::open_archive(&args.input_bundle, args.max_input_bytes)?;
    let basis = inputs.tensor("time_basis")?;
    ensure!(
        basis.shape().len() == 2 && basis.dtype() == safetensors::Dtype::F32,
        "time basis must be original F32 rank2"
    );
    let basis_rows = basis.shape()[0];
    let (tags, times) = capture.routing(&transformer, basis_rows)?;
    let source = Arc::new(ComfyCanonicalWeightSource::new(
        SafetensorsArchive::open_single_file(&args.model_dir, &args.checkpoint_file)?,
    )?);
    let checkpoint_sha = file_sha(&args.model_dir.join(&args.checkpoint_file))?;
    ensure!(
        checkpoint_sha.eq_ignore_ascii_case(&capture.checkpoint_sha256),
        "actual original checkpoint SHA differs from reference capture"
    );
    let mut adapter = PH3ImageDenoiserAdapter::new(
        transformer.clone(),
        source,
        PH3ImageDenoiserLimits {
            max_metadata_bytes: args.max_metadata_bytes,
            max_plan_nodes: args.max_plan_nodes,
            max_stored_bytes: args.cuda.usable_memory_bytes,
        },
    )?;
    adapter.set_attention_policy(match args.attention {
        Attention::Sequential => PH3ImageDenoiserAttentionPolicy::Sequential,
        Attention::CudnnSdpa => PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1,
    })?;
    let prepare = |phase| {
        adapter.prepare(
            phase,
            basis_rows,
            vec![adapter.request(
                id("request.dit.oracle.actual")?,
                &tags,
                capture.target()?,
                capture.reference(&transformer)?,
                &times,
            )?],
        )
    };
    let first = prepare(PH3ImageDenoiserPhase::InstallFullWeights)?;
    let steady = prepare(PH3ImageDenoiserPhase::DenoiseRefinedText)?;
    let execution_family = data::observed_family(
        steady.family(),
        args.capture_first_block,
        args.first_block_taps,
        args.input_boundary,
        steady.compile_options(),
        args.dense_linear_source,
    )?;
    let execution_options = data::execution_options(
        steady.compile_options(),
        &execution_family,
        args.input_boundary,
    )?;
    ensure!(
        serde_json::to_vec(&execution_family)?.len() <= args.max_metadata_bytes,
        "observed family exceeds metadata budget"
    );
    let bindings = data::input_bindings(args.input_boundary);
    let selection = if args.dense_linear_source == DenseLinearSource::Bf16Roundtrip {
        Some(source::select(
            first.family().program(),
            first.family().stored_weight_schema()?,
            first.source(),
            usize::try_from(args.cuda.transfer_staging_bytes)?,
        )?)
    } else {
        None
    };
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        args.host_memory_bytes,
        HostMemoryClassLimits {
            input_preprocess_bytes: args.host_memory_bytes,
            output_bytes: args.host_memory_bytes,
            transfer_staging_bytes: args.host_memory_bytes,
            // This account has alignment 1; the wrapper additionally checks
            // physical backing against this one-chunk class ceiling.
            weight_backing_bytes: selection.as_ref().map_or(0, |s| s.chunk_bytes),
        },
    )?);
    let mut managed = Vec::with_capacity(bindings.len());
    let mut input_receipts = Vec::with_capacity(bindings.len());
    for &(name, value_id) in bindings {
        let value_id: ProgramValueId = id(value_id)?;
        ensure!(
            execution_family.program().inputs().contains(&value_id),
            "capture port is not a declared family input"
        );
        let expected = execution_options
            .tensor_specs()
            .get(&value_id)
            .context("missing actual input spec")?;
        let tensor = inputs.tensor(name)?;
        ensure!(
            tensor.shape() == expected.dimensions
                && tensor.element_type() == Some(expected.element_type)
                && expected.layout == ResolvedTensorLayout::Contiguous,
            "actual {name} shape/dtype differs from prepared family"
        );
        data::finite(&tensor)?;
        let mut bytes =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, tensor.bytes().len())?;
        bytes.as_mut().copy_from_slice(tensor.bytes());
        managed.push(ManagedHostTensor::new(
            value_id,
            ResolvedTensorSpec::new(
                expected.dimensions.clone(),
                expected.element_type,
                expected.layout.clone(),
            )?,
            bytes.freeze(),
        )?);
        input_receipts.push(json!({"name":name,"shape":tensor.shape(),"dtype":format!("{:?}",tensor.dtype()),"bytes":tensor.bytes().len(),"sha256":artifacts::sha256(tensor.bytes())}));
    }
    ensure!(
        execution_family.program().inputs().len() == bindings.len(),
        "unexpected selected input ABI"
    );
    let artifact = artifacts::Artifacts::create(&args.artifact_dir)?;
    // Validate the reference terminal rounding/sign boundary before any GPU work.
    let mut expected_output_bytes = 0usize;
    for (returned, raw, shape, is_image) in [
        (
            "returned_image",
            "final_layer_image",
            capture.target_shape.to_vec(),
            true,
        ),
        (
            "returned_audio",
            "final_layer_audio",
            capture.audio_shape.to_vec(),
            false,
        ),
    ] {
        let returned = inputs.tensor(returned)?;
        let raw = inputs.tensor(raw)?;
        let patch = transformer.patch_size();
        let expected = if is_image {
            vec![
                (shape[3] / patch[1]) * (shape[4] / patch[2]),
                shape[1] * patch[1] * patch[2],
            ]
        } else {
            vec![4, shape[1]]
        };
        ensure!(
            raw.dtype() == safetensors::Dtype::F32 && raw.shape() == expected,
            "final-layer raw positive tensor shape/dtype differs"
        );
        let count = usize::try_from(
            expected[0]
                .checked_mul(expected[1])
                .context("output extent overflow")?,
        )?;
        expected_output_bytes = expected_output_bytes
            .checked_add(count.checked_mul(6).context("output bytes overflow")?)
            .context("output total overflow")?;
        ensure!(
            args.trajectory != Trajectory::Single || expected_output_bytes <= args.max_output_bytes,
            "native BF16 + reference F32 payloads exceed output limit"
        );
        let mut checker = data::TerminalCheck::new(&raw, std::io::sink());
        if is_image {
            data::reference_image(&returned, capture.target_shape, patch, &mut checker)?;
        } else {
            data::reference_audio(&returned, capture.audio_shape, &mut checker)?;
        }
        checker.finish()?;
    }
    if args.capture_first_block {
        for &(_, value) in data::taps(
            args.input_boundary,
            args.capture_first_block,
            args.first_block_taps,
        ) {
            let bytes =
                data::tap_tensor_spec(steady.compile_options(), value)?.minimum_storage_bytes()?;
            expected_output_bytes = expected_output_bytes
                .checked_add(usize::try_from(bytes)?)
                .context("tap output budget overflow")?;
        }
        ensure!(
            expected_output_bytes <= args.max_output_bytes,
            "native taps plus native/reference velocity payloads exceed output limit"
        );
    }
    let mut declaration = json!({"schema":"ferrum.ph3.actual_dit.v1","phase":"refined","input_boundary":args.input_boundary,"trajectory":args.trajectory,"dense_linear_source":args.dense_linear_source,"source_chunk_bytes":selection.as_ref().map(|s|s.chunk_bytes),"inputs":input_receipts,"config_sha256":artifacts::sha256(&config_bytes),"capture_metadata_sha256":artifacts::sha256(&metadata_bytes),"capture_bundle_sha256":file_sha(&args.input_bundle)?,"checkpoint_sha256":checkpoint_sha,"attention":format!("{:?}",args.attention),"cuda":args.cuda,"host_capacity_bytes":args.host_memory_bytes,"output_payload_limit_bytes":args.max_output_bytes,"capture_first_block":args.capture_first_block,"first_block_taps":args.first_block_taps,"source_family_fingerprint":steady.family().fingerprint()?,"execution_family_fingerprint":execution_family.fingerprint()?,"execution_program_fingerprint":execution_family.program().fingerprint()?,"execution_numerical_profile":execution_family.numerical_profile().id,"execution_config":execution_family.canonical_config(),"provenance":capture.provenance,"scope":"independent actual-input DiT diagnostic; trajectory mode additionally uses captured original F32 solver state and actual per-step bases, with exact reference CPU replay required before GPU; no native RNG, TE, preprocessing, conditioning cache, image pipeline or serving validation","host_scope":"managed ports/readbacks/staging/trajectory state charged; immutable checkpoint and capture mmap plus bounded metadata are outside Host ledger; trajectory output budget covers final F32 latent; single-forward output budget includes taps plus native/reference velocities","terminal_reference":"BF16(final positive F32) exactly equals negated returned BF16 in checked packed order, including signed zero"});
    let mut trajectory = if args.trajectory == Trajectory::OriginalEuler {
        Some(trajectory::EulerRun::prepare(
            &inputs,
            &capture,
            transformer.patch_size(),
            &host,
            &artifact,
            args.max_output_bytes,
        )?)
    } else {
        None
    };
    if let Some(selected) = &args.input_override {
        let run = trajectory
            .as_mut()
            .context("override requires a validated official trajectory")?;
        let original_bytes = usize::try_from(args.input_bundle.metadata()?.len())?;
        let remaining = args
            .max_input_bytes
            .checked_sub(original_bytes)
            .context("official bundle exceeds combined input budget")?;
        let replacement = data::open_archive(&selected.bundle, remaining)?;
        let replacement_bytes = usize::try_from(selected.bundle.metadata()?.len())?;
        let file = json!({"path":selected.bundle,"bytes":replacement_bytes,"sha256":file_sha(&selected.bundle)?,
            "maximum_combined_input_bytes":args.max_input_bytes,"official_file_bytes":original_bytes,
            "budget":"both complete input files, including ignored tensors; immutable mmap outside managed Host ledger"});
        run.bind_override(replacement, selected.group, file)?;
        declaration["input_override"] = json!(run.override_receipt());
        declaration["scope"] = json!("independent original-Euler trajectory with one explicitly substituted production input group after exact official replay and initial-port validation; no service/performance claim");
    }
    artifact.json("declaration.json", &declaration)?;
    let result = execute(
        &args,
        &first,
        &steady,
        &execution_family,
        &execution_options,
        selection.as_ref(),
        &host,
        managed,
        trajectory.as_mut(),
    );
    let (outputs, initialization, source_receipt) = match result {
        Ok(v) => v,
        Err(e) => {
            artifact.json(
                "failure.json",
                &json!({"error":format!("{e:#}"),"complete":false}),
            )?;
            return Err(e);
        }
    };
    if let Some(trajectory) = trajectory {
        ensure!(
            outputs.is_empty(),
            "trajectory retained unexpected model outputs"
        );
        let receipt = trajectory.finish(&artifact, args.max_metadata_bytes)?;
        ensure!(
            host.usage()?.committed_bytes() == 0,
            "trajectory Host owners remain after close"
        );
        artifact.json(
            "complete.json",
            &json!({"complete":true,"trajectory":receipt,
            "initialization":initialization,"dense_linear_source":source_receipt,
            "native_file_sha256":file_sha(&artifact.directory().join("final-target.latent"))?,
            "host_retained_bytes":0}),
        )?;
        println!(
            "Actual-input DiT trajectory: {}",
            artifact.directory().display()
        );
        return Ok(());
    }
    ensure!(
        outputs.len()
            == 2 + data::taps(
                args.input_boundary,
                args.capture_first_block,
                args.first_block_taps
            )
            .len(),
        "output count differs from explicit velocity/tap selection"
    );
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    let mut receipts = Vec::new();
    let mut written_ids = vec![
        PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID,
        PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID,
    ];
    for (id, name, returned) in [
        (
            PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID,
            "image_velocity",
            "returned_image",
        ),
        (
            PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID,
            "audio_velocity",
            "returned_audio",
        ),
    ] {
        let output = outputs
            .iter()
            .find(|o| o.value_id().as_str() == id)
            .context("missing declared velocity output")?;
        let reference = inputs.tensor(returned)?;
        let expected = if name == "image_velocity" {
            let p = transformer.patch_size();
            vec![
                capture.target_shape[3] / p[1] * (capture.target_shape[4] / p[2]),
                capture.target_shape[1] * p[1] * p[2],
            ]
        } else {
            vec![4, capture.audio_shape[1]]
        };
        ensure!(
            host.owns(output.bytes())
                && output.tensor().element_type() == ElementType::Bf16
                && output.tensor().dimensions() == expected
                && output.bytes().bytes().len() as u64
                    == output.tensor().minimum_storage_bytes()?,
            "velocity owner/dtype/shape/extent differs"
        );
        ensure!(
            output
                .bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite()),
            "nonfinite native velocity"
        );
        let end = offset
            .checked_add(output.bytes().bytes().len())
            .context("native output offset overflow")?;
        header.insert(
            name.into(),
            json!({"dtype":"BF16","shape":expected,"data_offsets":[offset,end]}),
        );
        offset = end;
        let file = OpenOptions::new().write(true).create_new(true).open(
            artifact
                .directory()
                .join(format!("reference-{name}-positive.f32")),
        )?;
        let mut writer = BufWriter::with_capacity(16384, file);
        if name == "image_velocity" {
            data::reference_image(
                &reference,
                capture.target_shape,
                transformer.patch_size(),
                &mut writer,
            )?;
        } else {
            data::reference_audio(&reference, capture.audio_shape, &mut writer)?;
        }
        writer.flush()?;
        writer.get_ref().sync_all()?;
        receipts.push(json!({"name":name,"shape":expected,"dtype":"BF16","positive_velocity":true,"bytes":output.bytes().bytes().len(),"sha256":artifacts::sha256(output.bytes().bytes())}));
    }
    if args.capture_first_block {
        for &(name, id) in data::taps(
            args.input_boundary,
            args.capture_first_block,
            args.first_block_taps,
        ) {
            let output = outputs
                .iter()
                .find(|o| o.value_id().as_str() == id)
                .context("missing requested hidden output")?;
            let expected = data::tap_tensor_spec(steady.compile_options(), id)?;
            ensure!(
                host.owns(output.bytes())
                    && output.tensor() == &expected
                    && output.bytes().bytes().len() as u64
                        == output.tensor().minimum_storage_bytes()?,
                "hidden output owner/type/shape/extent differs"
            );
            ensure!(
                output
                    .bytes()
                    .bytes()
                    .chunks_exact(2)
                    .all(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite()),
                "nonfinite hidden output"
            );
            let end = offset
                .checked_add(output.bytes().bytes().len())
                .context("hidden output offset overflow")?;
            header.insert(
                name.into(),
                json!({"dtype":"BF16","shape":expected.dimensions(),"data_offsets":[offset,end]}),
            );
            offset = end;
            receipts.push(json!({"name":name,"value_id":id,"shape":expected.dimensions(),"dtype":"BF16","bytes":output.bytes().bytes().len(),"sha256":artifacts::sha256(output.bytes().bytes())}));
            written_ids.push(id);
        }
    }
    let mut header = serde_json::to_vec(&header)?;
    ensure!(
        header.len() <= args.max_metadata_bytes,
        "output header exceeds metadata bound"
    );
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(artifact.directory().join("native-velocity.safetensors"))?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    for id in written_ids {
        file.write_all(
            outputs
                .iter()
                .find(|o| o.value_id().as_str() == id)
                .unwrap()
                .bytes()
                .bytes(),
        )?;
    }
    file.sync_all()?;
    drop(outputs);
    ensure!(
        host.usage()?.committed_bytes() == 0,
        "Host owners remain after explicit close/output drop"
    );
    artifact.json("complete.json",&json!({"complete":true,"outputs":receipts,"initialization":initialization,"dense_linear_source":source_receipt,"native_file_sha256":file_sha(&artifact.directory().join("native-velocity.safetensors"))?,"host_retained_bytes":0,"comparison":"not automatically thresholded; reference-positive raw F32 can be compared with native BF16 using compare_tensor_payloads"}))?;
    println!(
        "Actual-input DiT output: {}",
        artifact.directory().display()
    );
    Ok(())
}

fn execute(
    args: &Arguments,
    first: &PreparedPH3ImageDenoiser,
    steady: &PreparedPH3ImageDenoiser,
    execution_family: &PreparedModelFamily,
    execution_options: &ProgramPlanCompileOptions,
    selection: Option<&source::Selection>,
    host: &HostMemoryAccount,
    inputs: Vec<ManagedHostTensor>,
    trajectory: Option<&mut trajectory::EulerRun<'_>>,
) -> Result<(
    Vec<TensorExecutionOutput>,
    serde_json::Value,
    serde_json::Value,
)> {
    let o = &args.cuda;
    let runtime = Arc::new(CudaDeviceRuntime::new_tensor_eager(
        CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(o.device, id(format!("device.cuda.{}", o.device))?)?,
            usable_memory_bytes: o.usable_memory_bytes,
            bootstrap_memory_bytes: o.bootstrap_memory_bytes,
            stream_memory_bytes: o.stream_memory_bytes,
            maximum_streams: o.maximum_streams,
            maximum_in_flight_per_stream: o.maximum_in_flight_per_stream,
            maximum_commands_per_submission: o.maximum_commands_per_submission,
            opaque_memory_profile: o.opaque_memory_profile.clone(),
        },
    )?);
    let selected = [
        CudaTensorOperation::AffineLinear,
        CudaTensorOperation::AffineLinearF32,
        CudaTensorOperation::CastF32ToBf16,
        CudaTensorOperation::ReshapeCopy,
        CudaTensorOperation::RowGather,
        CudaTensorOperation::BatchedRowConcat,
        CudaTensorOperation::RowRmsNormSingleRound,
        CudaTensorOperation::ModulationScale,
        CudaTensorOperation::RowSelectedModulationBf16,
        CudaTensorOperation::RowSelectedAddcmulBf16,
        CudaTensorOperation::PlainResidualAdd,
        CudaTensorOperation::FusedHeadRmsPartialRope,
        CudaTensorOperation::Addcmul,
        CudaTensorOperation::SegmentedAttention,
        CudaTensorOperation::RowSelectedModulationBf16F32,
        CudaTensorOperation::ComfyConvRotW4A8,
        CudaTensorOperation::ComfyConvRotSwiGluF32W4A8,
        CudaTensorOperation::ComfyConvRotW8A8,
        CudaTensorOperation::ComfyConvRotSwiGluF32W8A8,
    ];
    let providers = match args.attention {
        Attention::Sequential => cuda_vnext_tensor_providers(&runtime, &selected)?,
        Attention::CudnnSdpa => cuda_vnext_cnn_f32_providers_with_segmented_sdpa(
            &runtime,
            &CudaCudnnConfig {
                library_path: args.cudnn_library.clone().unwrap(),
                dependency_libraries: vec![],
                opaque_memory_bytes: args.cudnn_opaque_bytes,
                workspace_limit_bytes: args.cudnn_workspace_bytes,
                opaque_memory_profile: "h3.actual_dit.cudnn9.declared.v1".into(),
            },
            CudnnSdpaPlanPolicy {
                maximum_plans: 64,
                maximum_live_bindings: 2048,
                opaque_memory_bytes: args.sdpa_opaque_bytes,
                workspace_limit_bytes: args.cudnn_workspace_bytes,
            },
            &selected,
        )?,
    };
    let mut contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(affine_linear_bf16_contract()?),
        Box::new(affine_linear_f32_contract()?),
        Box::new(cast_f32_to_bf16_contract()?),
        Box::new(reshape_copy_bf16_contract()?),
        Box::new(row_gather_bf16_contract()?),
        Box::new(batched_row_concat_bf16_contract()?),
        Box::new(row_rms_norm_single_round_bf16_contract()?),
        Box::new(modulation_scale_bf16_contract()?),
        Box::new(row_selected_modulation_bf16_contract()?),
        Box::new(row_selected_addcmul_bf16_contract()?),
        Box::new(plain_residual_add_bf16_contract()?),
        Box::new(fused_head_rms_partial_rope_bf16_contract()?),
        Box::new(addcmul_bf16_contract()?),
        Box::new(segmented_attention_bf16_contract()?),
        Box::new(row_selected_modulation_bf16_f32_contract()?),
        Box::new(comfy_convrot_w4a8_bf16_contract()?),
        Box::new(comfy_convrot_swiglu_f32_w4a8_bf16_contract()?),
        Box::new(comfy_convrot_w8a8_bf16_contract()?),
        Box::new(comfy_convrot_swiglu_f32_w8a8_bf16_contract()?),
    ];
    if args.attention == Attention::CudnnSdpa {
        contracts.push(Box::new(conv2d_f32_contract()?));
    }
    let registry = OperationRuntimeRegistry::new(contracts, providers)?;
    let materializers = WeightMaterializerRegistry::identity_only()?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.h3.actual_dit")?,
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )?;
    let catalog =
        materializers.augment_catalog(registry.capability_catalog(d.clone(), vec![engine])?)?;
    let policy = TensorComponentPolicy::new(
        d.total_memory_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )?;
    let declared_install = data::installation_family(first.family(), args.dense_linear_source)?;
    let first_plan = CompiledTensorComponent::compile(
        declared_install,
        &registry,
        &catalog,
        &policy,
        &materializers,
        first.compile_options(),
    )?;
    let steady_plan = CompiledTensorComponent::compile(
        execution_family.clone(),
        &registry,
        &catalog,
        &policy,
        &materializers,
        execution_options,
    )?;
    ensure!(
        first_plan
            .executable()
            .execution_plan()
            .payload()
            .execution_weights()
            == steady_plan
                .executable()
                .execution_plan()
                .payload()
                .execution_weights(),
        "full/refined physical installation differs"
    );
    let transformed = selection
        .map(|selected| {
            source::RoundtripSource::new(
                first.source(),
                selected,
                host,
                first_plan.executable().execution_plan(),
            )
        })
        .transpose()?;
    let installation_source: &dyn WeightComponentSource = match &transformed {
        Some(source) => source,
        None => first.source(),
    };
    let (first_session, weights) = first_plan
        .initialize_shared_weights(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(installation_source),
            initial(args, "install")?,
        )
        .map_err(|e| anyhow::anyhow!("original shared initialization: {e}"))?;
    let initialization = serde_json::to_value(first_session.initialization_receipt())?;
    // Hash coverage is published only after the original Core installation
    // has completed every upload/fence, never merely after range conversion.
    let source_receipt = match &transformed {
        Some(source) => source.receipt()?,
        None => json!({"policy":"archive","canonical_archive_bytes":true}),
    };
    // InstallFullWeights is never executed. Its session retires before opening
    // the refined session; only the original sealed shared installation survives.
    ensure!(
        matches!(
            first_session
                .close()
                .map_err(|e| anyhow::anyhow!("install close: {:?}", e.failure()))?,
            PlanRuntimeCloseOutcome::Closed(_)
        ),
        "install session did not close"
    );
    let mut session = steady_plan
        .initialize_with_shared_weights(runtime, host.clone(), &weights, initial(args, "refined")?)
        .map_err(|e| anyhow::anyhow!("shared attachment: {e}"))?;
    if args.capture_first_block
        || args.input_boundary != InputBoundary::Prepared
        || args.dense_linear_source != DenseLinearSource::Archive
    {
        let plan = session.executable().execution_plan().payload();
        ensure!(
            plan.family_id() == execution_family.family_id()
                && plan.prepared_family_fingerprint() == execution_family.fingerprint()?
                && plan.program_fingerprint() == execution_family.program().fingerprint()?,
            "observed attachment differs from declared diagnostic family/program"
        );
        let ferrum_execution::component::ComponentInitializationReceipt::SharedWeights { evidence } =
            session.initialization_receipt()
        else {
            anyhow::bail!("observed execution requires original shared installation")
        };
        ensure!(
            evidence == weights.evidence()
                && evidence.weight_plan_fingerprint()
                    == plan.execution_weights().stored()?.fingerprint()?,
            "observed attachment changed original full physical weights"
        );
    } else {
        steady.validate_attached_session(&session, &weights)?;
    }
    let result = if let Some(trajectory) = trajectory {
        trajectory
            .run(inputs, |step, inputs| {
                session
                    .execute(
                        id(format!("run.dit.oracle.step{step}"))?,
                        id(format!("request.dit.oracle.step{step}"))?,
                        inputs,
                    )
                    .map_err(|e| anyhow::anyhow!("actual DiT step {step}: {e}"))
            })
            .map(|()| Vec::new())
    } else {
        session
            .execute(
                id("run.dit.oracle.execute")?,
                id("request.dit.oracle.execute")?,
                inputs,
            )
            .map_err(|e| anyhow::anyhow!("actual DiT execution: {e}"))
    };
    // Preserve the original execution result while closing; an error's owned
    // retirement behavior remains authoritative and is never a successful close.
    let close = session.close();
    let weights_close = weights.close();
    let outputs = result.map_err(|e| anyhow::anyhow!("actual DiT execution: {e}"))?;
    ensure!(
        matches!(
            close.map_err(|e| anyhow::anyhow!("refined close: {:?}", e.failure()))?,
            PlanRuntimeCloseOutcome::Closed(_)
        ),
        "refined session did not close"
    );
    ensure!(
        matches!(
            weights_close.map_err(|e| anyhow::anyhow!("shared weight close: {:?}", e.failure()))?,
            SharedComponentWeightsCloseOutcome::Closed(_)
        ),
        "shared installation still has live attachments"
    );
    Ok((outputs, initialization, source_receipt))
}
