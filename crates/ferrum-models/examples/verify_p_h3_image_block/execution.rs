use super::{artifacts, Arguments};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::*,
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_models::p_h3_image::*;
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use ferrum_runtime::*;
use half::bf16;
use serde_json::json;
use std::{sync::Arc, time::Instant};

fn id<T: TryFrom<String>>(value: impl Into<String>) -> Result<T>
where
    T::Error: std::fmt::Display,
{
    T::try_from(value.into()).map_err(|e| anyhow::anyhow!("{e}"))
}
pub fn run(args: Arguments) -> Result<()> {
    let config_bytes = artifacts::read_bounded(&args.config, args.max_metadata_bytes)?;
    let transformer = PH3ImageTransformerConfig::from_hf_json(&config_bytes)?;
    let source = ComfyCanonicalWeightSource::new(SafetensorsArchive::open_single_file(
        &args.model_dir,
        &args.checkpoint_file,
    )?)?;
    let table = source.archive().tensor("adaln_t_table")?;
    ensure!(
        table.element_type() == Some(ElementType::F32) && table.shape().len() == 2,
        "requires original F32 curve time table"
    );
    let curve = PH3ImageCurveTimeEmbedding::new(
        usize::try_from(table.shape()[0])?,
        usize::try_from(table.shape()[1])?,
    )?;
    let tags = vec![PH3ImageModality::Text; args.text_rows];
    let target = PH3ImageReferenceImage::new(args.latent_height, args.latent_width)?;
    let layout = PH3ImageRequestLayout::new(&transformer, &tags, target, &[])?;
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        args.host_memory_bytes,
        HostMemoryClassLimits {
            input_preprocess_bytes: args.host_memory_bytes,
            transfer_staging_bytes: args.host_memory_bytes,
            output_bytes: args.host_memory_bytes,
            weight_backing_bytes: args.host_memory_bytes,
        },
    )?);
    let mut time_rows = host.allocate_zeroed(
        HostMemoryClass::InputPreprocess,
        layout
            .rows()
            .checked_mul(8)
            .context("time row bytes overflow")?,
    )?;
    let indices: &mut [u64] = bytemuck::try_cast_slice_mut(time_rows.as_mut())
        .map_err(|e| anyhow::anyhow!("time row alignment: {e}"))?;
    let mut requests = Vec::with_capacity(args.batch);
    for request in 0..args.batch {
        indices.fill(u64::try_from(request)?);
        requests.push(PH3ImageCurveBlockRequest::from_layout(
            layout,
            indices,
            args.max_metadata_bytes,
        )?);
    }
    drop(time_rows);
    let config = PH3ImageCurveBlockConfig::from_checkpoint(
        &transformer,
        curve,
        args.block_index,
        u64::try_from(args.batch)?,
        requests,
        &source,
    )?;
    let limits = PH3ImageCurveBlockLimits {
        max_metadata_bytes: args.max_metadata_bytes,
        max_stored_bytes: args.cuda.usable_memory_bytes,
        max_plan_nodes: 4096,
    };
    let checkpoint = PH3ImageCurveBlockCheckpoint::new(&config, &source, limits)?;
    let layouts = vec![layout; args.batch];
    let packed = PH3ImagePackedBatchLayout::new(&layouts)?;
    let position_bytes = packed
        .rows()
        .checked_mul(24)
        .context("position bytes overflow")?;
    let mut positions = host.allocate_zeroed(HostMemoryClass::InputPreprocess, position_bytes)?;
    let position_view: &mut [[f64; 3]] = bytemuck::try_cast_slice_mut(positions.as_mut())
        .map_err(|e| anyhow::anyhow!("position alignment: {e}"))?;
    packed.write_positions(position_view)?;
    let rotary = PH3ImageCheckpointRotary::new(&transformer, &source)?;
    let (cosine, sine) = rotary.managed_bf16_frequencies(
        &host,
        position_view,
        id(PH3ImageCurveBlockFamily::COSINE_ID)?,
        id(PH3ImageCurveBlockFamily::SINE_ID)?,
    )?;
    drop(positions);
    let times: Vec<f32> = (0..args.batch)
        .map(|i| ((i + 1) as f64 / (args.batch + 1) as f64) as f32)
        .collect();
    let basis =
        checkpoint.managed_basis(&host, &times, PH3ImageCurveLerpArithmetic::SeparateF32)?;
    let hidden = usize::try_from(config.hidden_size())?;
    let input_bytes = packed
        .rows()
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(2))
        .context("activation bytes overflow")?;
    let mut input = host.allocate_zeroed(HostMemoryClass::InputPreprocess, input_bytes)?;
    for (index, out) in input.as_mut().chunks_exact_mut(2).enumerate() {
        let row = index / hidden;
        let col = index % hidden;
        let value = (((row % 101) * 37 + (col % 101) * 13) % 101) as f32 / 128.0 - 50.0 / 128.0;
        out.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    let input = ManagedHostTensor::new(
        id(PH3ImageCurveBlockFamily::INPUT_ID)?,
        ResolvedTensorSpec::new(
            vec![config.rows()?, config.hidden_size()],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?,
        input.freeze(),
    )?;
    let inputs = vec![input, basis, cosine, sine];
    let artifacts = artifacts::Artifacts::create(&args.artifact_dir)?;
    artifacts.json("declaration.json",&json!({
        "schema":"ferrum.p_h3_image.block_execution.v1","config":config,
        "config_sha256":artifacts::sha256(&config_bytes),"checkpoint":args.model_dir.join(&args.checkpoint_file),
        "cuda":args.cuda,"host_capacity_bytes":args.host_memory_bytes,"time_rows":times,
        "rotary_profile":PH3ImageCheckpointRotary::NUMERICAL_PROFILE_ID,
        "scope":"one actual original-weight block; synthetic hidden states; no image, independent full-size oracle or performance qualification",
        "host_scope":"tensor buffers, initialization staging and outputs are charged; immutable checkpoint mmap and bounded control metadata are outside Host allocation capacity"
    }))?;
    for tensor in &inputs {
        artifacts.json(
            &format!("{}.json", tensor.value_id()),
            &json!({
            "shape":tensor.tensor().dimensions(),"dtype":tensor.tensor().element_type(),
            "sha256":artifacts::sha256(tensor.bytes().bytes())}),
        )?;
    }
    let result = execute(&args, &config, &source, &host, inputs, &artifacts);
    match result {
        Ok(()) => {
            println!(
                "Original-weight H3 block execution completed: {}",
                artifacts.directory().display()
            );
            Ok(())
        }
        Err(e) => {
            artifacts.json(
                "failure.json",
                &json!({"error":format!("{e:#}"),"scope":"execution diagnostic"}),
            )?;
            Err(e)
        }
    }
}

fn execute(
    args: &Arguments,
    config: &PH3ImageCurveBlockConfig,
    source: &ComfyCanonicalWeightSource,
    host: &HostMemoryAccount,
    inputs: Vec<ManagedHostTensor>,
    artifacts: &artifacts::Artifacts,
) -> Result<()> {
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
    let mut providers = cuda_vnext_tensor_providers(
        &runtime,
        &[
            CudaTensorOperation::CastF32ToBf16,
            CudaTensorOperation::ReshapeCopy,
            CudaTensorOperation::RowGather,
            CudaTensorOperation::BatchedRowConcat,
            CudaTensorOperation::RowRmsNormSingleRound,
            CudaTensorOperation::ModulationScale,
            CudaTensorOperation::PlainResidualAdd,
            CudaTensorOperation::FusedHeadRmsPartialRope,
            CudaTensorOperation::Addcmul,
            CudaTensorOperation::SegmentedAttention,
            CudaTensorOperation::ComfyConvRotW4A8,
            CudaTensorOperation::ComfyConvRotSwiGluF32W4A8,
        ],
    )?;
    providers.push(cuda_vnext_affine_linear_f32_provider(&runtime)?);
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(affine_linear_f32_contract()?),
        Box::new(cast_f32_to_bf16_contract()?),
        Box::new(reshape_copy_bf16_contract()?),
        Box::new(row_gather_bf16_contract()?),
        Box::new(batched_row_concat_bf16_contract()?),
        Box::new(row_rms_norm_single_round_bf16_contract()?),
        Box::new(modulation_scale_bf16_contract()?),
        Box::new(plain_residual_add_bf16_contract()?),
        Box::new(fused_head_rms_partial_rope_bf16_contract()?),
        Box::new(addcmul_bf16_contract()?),
        Box::new(segmented_attention_bf16_contract()?),
        Box::new(comfy_convrot_w4a8_bf16_contract()?),
        Box::new(comfy_convrot_swiglu_f32_w4a8_bf16_contract()?),
    ];
    let registry = OperationRuntimeRegistry::new(contracts, providers)?;
    let materializers = WeightMaterializerRegistry::identity_only()?;
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.h3.block_probe")?,
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )?;
    let catalog =
        materializers.augment_catalog(registry.capability_catalog(d.clone(), vec![engine])?)?;
    let family =
        TypedFamilyRegistration::new(PH3ImageCurveBlockFamily::new(PH3ImageCurveBlockLimits {
            max_metadata_bytes: args.max_metadata_bytes,
            max_stored_bytes: o.usable_memory_bytes,
            max_plan_nodes: 4096,
        })?)
        .prepare_with_profile(
            &serde_json::to_value(config)?,
            &id(PH3ImageCurveBlockFamily::cuda_profile_id(config))?,
        )?;
    let policy = TensorComponentPolicy::new(
        d.total_memory_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )?;
    let component = CompiledTensorComponent::compile(
        family,
        &registry,
        &catalog,
        &policy,
        &materializers,
        &config.compile_options()?,
    )?;
    let mut session = component
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::Stored(source),
            ComponentInitializationConfig {
                run_id: id("run.h3.probe.init")?,
                request_id: id("request.h3.probe.init")?,
                transaction_id: id("transaction.h3.probe.init")?,
                static_policy: StaticInitializationPolicy::new(
                    u64::try_from(o.transfer_staging_bytes)?,
                    64,
                )?,
                executor: TensorExecutorConfig::default(),
            },
        )
        .map_err(|error| anyhow::anyhow!("component initialization: {error}"))?;
    let begin = Instant::now();
    let execution = session.execute(
        id("run.h3.probe.execute")?,
        id("request.h3.probe.execute")?,
        inputs,
    );
    let elapsed_seconds = begin.elapsed().as_secs_f64();
    // Close even if execution failed. Completed readback owners remain valid
    // after device retirement, and validation/output-I/O cannot bypass close.
    let close = session.close();
    let execution = execution.map_err(|error| anyhow::anyhow!("component execution: {error}"));
    let close = match close {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => Ok(()),
        Ok(_) => Err(anyhow::anyhow!("component did not close cleanly")),
        Err(error) => Err(anyhow::anyhow!("component close: {:?}", error.failure())),
    };
    let outputs = match (execution, close) {
        (Ok(outputs), Ok(())) => outputs,
        (Err(execution), Ok(())) => return Err(execution),
        (Ok(_), Err(close)) => return Err(close),
        (Err(execution), Err(close)) => {
            return Err(anyhow::anyhow!("{execution:#}; additionally {close:#}"));
        }
    };
    ensure!(
        outputs.len() == 1 && outputs[0].value_id().as_str() == PH3ImageCurveBlockFamily::OUTPUT_ID,
        "production block must return exactly its final output"
    );
    let output = &outputs[0];
    ensure!(
        host.owns(output.bytes())
            && output.tensor().element_type() == ElementType::Bf16
            && output.tensor().dimensions() == [config.rows()?, config.hidden_size()]
            && output.bytes().bytes().len() as u64 == output.tensor().minimum_storage_bytes()?,
        "output owner, BF16 dtype, exact shape or storage extent differs"
    );
    let mut finite = true;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for bytes in output.bytes().bytes().chunks_exact(2) {
        let v = bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32();
        finite &= v.is_finite();
        minimum = minimum.min(v);
        maximum = maximum.max(v);
    }
    artifacts.bytes("output.bf16", output.bytes().bytes())?;
    artifacts.json("output.json",&json!({"shape":output.tensor().dimensions(),
        "dtype":output.tensor().element_type(),"finite":finite,"minimum":minimum,"maximum":maximum,
        "sha256":artifacts::sha256(output.bytes().bytes()),"elapsed_seconds":elapsed_seconds,
        "timing_scope":"single cold diagnostic execution including transfer/readback; not a benchmark"}))?;
    drop(outputs);
    ensure!(
        host.usage()?.committed_bytes() == 0,
        "retained Host buffers remain after close/drop"
    );
    artifacts.json(
        "result.json",
        &json!({"execution_completed":true,"finite_output":finite,
        "host_retained_bytes":0,"semantic_oracle":"not performed at original model width"}),
    )?;
    ensure!(finite, "original-weight block produced nonfinite output");
    Ok(())
}
