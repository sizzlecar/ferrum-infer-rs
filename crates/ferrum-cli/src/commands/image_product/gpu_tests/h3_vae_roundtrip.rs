//! Explicit official-weight codec diagnostic, without vision/TE/DiT inference.
//! Inputs: FERRUM_TEST_IMAGE_PRODUCT_CONFIG, FERRUM_TEST_IMAGE_REFERENCE_PNG,
//! FERRUM_TEST_IMAGE_ARTIFACT_DIR (new directory, e.g. r56f-vae-roundtrip).
//! Both modes consume the same managed UnitRgb image and unchanged VAE policy.
//! Zero noise selects the posterior mean; native seed42 is not Torch CPU noise.
//! Optional FERRUM_TEST_COMFY_SAVED_LATENT must be a new-format F32 SaveLatent
//! file with the same geometry. Native 4D and explicit Comfy T=1 views export
//! identical payload bytes; no legacy scaling is applied.
use super::super::{
    p_h3::{PH3ImageBackendConfig, PH3ImageProductConfig},
    source,
};
use super::{required_path, write_json_new, HashedWriter};
use anyhow::{ensure, Context, Result};
use ferrum_execution::{component::ComponentWeightSource, TensorSerialActor};
use ferrum_models::{
    image_pixels::OwnedImagePixels,
    p_h3_image::{PH3ImageEditPixelPlan, PH3ImageEditPixelPolicy},
    p_h3_image_vae::{
        PH3ImageVaeAdapter, PH3ImageVaeDecoderPreset, PH3ImageVaeEncoderPreset,
        PH3ImageVaePipelineError, PH3ImageVaePixelDomain, PreparedPH3ImageVaeDecoder,
        PreparedPH3ImageVaeEncoder,
    },
    qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image},
};
use ferrum_output::png::{write_rgba8_png, PngDimensions, PngLimits};
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use ferrum_runtime::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read},
    path::Path,
    sync::Arc,
    time::Instant,
};
mod cuda;
mod latent_io;

fn id<T: TryFrom<String>>(value: impl Into<String>) -> Result<T>
where
    T::Error: std::fmt::Display,
{
    T::try_from(value.into()).map_err(|e| anyhow::anyhow!("{e}"))
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn input_png(
    host: &HostMemoryAccount,
    path: &Path,
    limits: ImageInputLimits,
) -> Result<ManagedHostBytes> {
    let mut file = File::open(path)?;
    let length = usize::try_from(file.metadata()?.len())?;
    ensure!(
        length > 0 && length <= limits.maximum_encoded_bytes(),
        "reference PNG exceeds the ordinary input bound"
    );
    let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, length)?;
    file.read_exact(bytes.as_mut())?;
    ensure!(
        file.read(&mut [0_u8; 1])? == 0,
        "reference PNG changed length while reading"
    );
    Ok(bytes.freeze())
}

// B1 contiguous NCHW: expose per-channel ranges, not an invented quality gate.
fn f32_statistics(bytes: &[u8], channels: usize) -> Result<Value> {
    ensure!(
        channels > 0
            && !bytes.is_empty()
            && bytes.len() % 4 == 0
            && bytes.len() / 4 % channels == 0,
        "statistics require complete nonempty F32 channel planes"
    );
    let plane_bytes = bytes.len() / channels;
    let values = bytes
        .chunks_exact(plane_bytes)
        .map(|plane| {
            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;
            let (mut sum, mut squares, mut finite, mut nonfinite) =
                (0_f64, 0_f64, 0_usize, 0_usize);
            for raw in plane.chunks_exact(4) {
                let value = f64::from(f32::from_le_bytes(raw.try_into().unwrap()));
                if value.is_finite() {
                    finite += 1;
                    minimum = minimum.min(value);
                    maximum = maximum.max(value);
                    sum += value;
                    squares += value * value;
                } else {
                    nonfinite += 1;
                }
            }
            json!({"finite_count":finite,"nonfinite_count":nonfinite,
            "min":(finite > 0).then_some(minimum),"max":(finite > 0).then_some(maximum),
            "mean":(finite > 0).then(||sum / finite as f64),
            "rms":(finite > 0).then(||(squares / finite as f64).sqrt())})
        })
        .collect::<Vec<_>>();
    Ok(json!({"elements":bytes.len()/4,"channels":values,
        "finite":bytes.chunks_exact(4).all(|v|f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
        "sha256":format!("{:x}",Sha256::digest(bytes))}))
}

fn difference(left: &[u8], right: &[u8]) -> Result<Value> {
    ensure!(
        !left.is_empty() && left.len() == right.len() && left.len() % 4 == 0,
        "comparison requires identical nonempty F32 extents"
    );
    let (mut maximum, mut absolute, mut squares) = (0_f64, 0_f64, 0_f64);
    for (a, b) in left.chunks_exact(4).zip(right.chunks_exact(4)) {
        let a = f64::from(f32::from_le_bytes(a.try_into().unwrap()));
        let b = f64::from(f32::from_le_bytes(b.try_into().unwrap()));
        ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite comparison operand"
        );
        let delta = (a - b).abs();
        maximum = maximum.max(delta);
        absolute += delta;
        squares += delta * delta;
    }
    let n = (left.len() / 4) as f64;
    Ok(
        json!({"elements":left.len()/4,"max_abs":maximum,"mae":absolute/n,"rmse":(squares/n).sqrt()}),
    )
}

fn checkpoint_latent_statistics(
    source: &ComfyCanonicalWeightSource,
    channels: usize,
) -> Result<Value> {
    let mut statistics = serde_json::Map::new();
    let mut combined = Sha256::new();
    for name in ["latents_mean", "latents_std"] {
        let tensor = source.archive().tensor(name)?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32 && tensor.shape() == [channels as u64],
            "checkpoint {name} must retain its original F32 channel shape"
        );
        let values = tensor
            .bytes()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        ensure!(
            values
                .iter()
                .all(|v| v.is_finite() && (name != "latents_std" || *v > 0.0)),
            "invalid original checkpoint normalization values"
        );
        let bits = tensor
            .bytes()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        combined.update(tensor.bytes());
        statistics.insert(name.to_owned(), json!({"dtype":"F32","shape":tensor.shape(),
            "source_file":tensor.source_file(),"values":values,"bits_u32":bits,
            "byte_count":tensor.bytes().len(),"sha256":format!("{:x}",Sha256::digest(tensor.bytes()))}));
    }
    Ok(
        json!({"source":"actual selected checkpoint mapping shared with the VAE adapter",
        "channels":channels,"tensors":statistics,"concatenation_order":["latents_mean","latents_std"],
        "combined_bytes":channels*8,"combined_sha256":format!("{:x}",combined.finalize())}),
    )
}

fn write_png(host: &HostMemoryAccount, path: &Path, output: &OwnedImagePixels) -> Result<Value> {
    let pixels = output.view(host)?;
    let [batch, _, height, width] = pixels.rgba_dimensions();
    ensure!(batch == 1, "roundtrip emits exactly one image");
    let mut scratch = host.allocate_zeroed(HostMemoryClass::Output, 16 * 1024)?;
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = HashedWriter {
        writer: file,
        sha256: Sha256::new(),
    };
    let result = write_rgba8_png(
        &mut writer,
        PngDimensions::new(width as u64, height as u64)?,
        PngLimits::default(),
        |sink| {
            pixels
                .write_frame_rgba8(0, scratch.as_mut(), sink)
                .map_err(io::Error::other)
        },
    )?;
    writer.writer.sync_all()?;
    Ok(
        json!({"path":path,"width":width,"height":height,"color_type":"rgba8",
        "encoded_bytes":result.encoded_bytes(),"sha256":format!("{:x}",writer.sha256.finalize())}),
    )
}

fn pipeline_error(error: PH3ImageVaePipelineError<cuda::CudaDeviceRuntime>) -> anyhow::Error {
    match error {
        PH3ImageVaePipelineError::Preparation(e) => e,
        PH3ImageVaePipelineError::Execution(e) => cuda::retained(format!("VAE execution: {e}"), e),
    }
}

fn decode_latent(
    name: &str,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<cuda::CudaDeviceRuntime>,
    decoder: &PreparedPH3ImageVaeDecoder,
    decoder_session: &mut ferrum_execution::component::TensorComponentSession<
        cuda::CudaDeviceRuntime,
    >,
    latent: &ManagedHostTensor,
) -> Result<(OwnedImagePixels, Value)> {
    ensure!(
        latent.tensor() == &decoder.normalized_latent_spec()?,
        "codec normalized latent boundaries differ"
    );
    let requirements = decoder.host_requirements(host)?;
    let scope = host.reserve_scope(&requirements.scope_claims())?;
    decoder_session.set_output_host_account(scope.account())?;
    let start = Instant::now();
    let mut consumer = decoder.bind(
        decoder_session,
        latent,
        id(format!("request.vae_roundtrip.{name}.decode"))?,
        scope.account(),
        scope.account(),
    )?;
    loop {
        let index = consumer.completed_tiles();
        if consumer
            .advance(
                actor,
                id(format!("run.vae_roundtrip.{name}.decode.{index}"))?,
            )
            .map_err(pipeline_error)?
        {
            break;
        }
    }
    let decode_tiles = consumer.executed_tiles();
    let output = consumer.finish()?;
    let decode_s = start.elapsed().as_secs_f64();
    decoder_session.set_output_host_account(host.clone())?;
    scope.close()?;
    let decoder_reclaimed = decoder_session.try_release_idle_dynamic_backing()?;
    Ok((
        output,
        json!({"decode_s":decode_s,"decoder_tiles":decode_tiles,
        "decoder_idle_release":decoder_reclaimed}),
    ))
}

fn roundtrip(
    name: &str,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<cuda::CudaDeviceRuntime>,
    encoder: &PreparedPH3ImageVaeEncoder,
    encoder_session: &mut ferrum_execution::component::TensorComponentSession<
        cuda::CudaDeviceRuntime,
    >,
    decoder: &PreparedPH3ImageVaeDecoder,
    decoder_session: &mut ferrum_execution::component::TensorComponentSession<
        cuda::CudaDeviceRuntime,
    >,
    pixels: &ManagedHostTensor,
    noise: &ManagedHostTensor,
) -> Result<(ManagedHostTensor, OwnedImagePixels, Value)> {
    let requirements = encoder.host_requirements(host)?;
    let scope = host.reserve_scope(&requirements.scope_claims())?;
    encoder_session.set_output_host_account(scope.account())?;
    let start = Instant::now();
    let mut consumer = encoder.bind(
        encoder_session,
        pixels,
        PH3ImageVaePixelDomain::UnitRgb,
        noise,
        id(format!("request.vae_roundtrip.{name}.encode"))?,
        scope.account(),
        scope.account(),
    )?;
    loop {
        let index = consumer.completed_tiles();
        if consumer
            .advance(
                actor,
                id(format!("run.vae_roundtrip.{name}.encode.{index}"))?,
            )
            .map_err(pipeline_error)?
        {
            break;
        }
    }
    let encode_tiles = consumer.executed_tiles();
    let encode_s = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let latent = consumer.finish(id(format!("value.vae_roundtrip.{name}.latent"))?)?;
    let posterior_s = start.elapsed().as_secs_f64();
    drop(consumer);
    encoder_session.set_output_host_account(host.clone())?;
    scope.close()?;
    ensure!(
        latent.tensor() == &decoder.normalized_latent_spec()?,
        "codec normalized latent boundaries differ"
    );
    // Ordinary idle release: no live resource is skipped to lower the peak.
    let encoder_reclaimed = encoder_session.try_release_idle_dynamic_backing()?;
    let (output, decode) = decode_latent(name, host, actor, decoder, decoder_session, &latent)?;
    Ok((
        latent,
        output,
        json!({"mode":name,"encode_s":encode_s,"posterior_s":posterior_s,"decode_s":decode["decode_s"],
        "encoder_tiles":encode_tiles,"decoder_tiles":decode["decoder_tiles"],
        "encoder_idle_release":encoder_reclaimed,"decoder_idle_release":decode["decoder_idle_release"]}),
    ))
}

#[test]
#[ignore = "requires explicit original PNG, ordinary H3 CUDA config/weights, pinned operators/cuDNN and an exclusive GPU window"]
fn cuda_official_h3_vae_roundtrip_mean_and_native_posterior_retire_owners() -> Result<()> {
    let config_path = required_path("FERRUM_TEST_IMAGE_PRODUCT_CONFIG")?;
    let input_path = required_path("FERRUM_TEST_IMAGE_REFERENCE_PNG")?;
    let directory = required_path("FERRUM_TEST_IMAGE_ARTIFACT_DIR")?;
    let comfy_latent =
        std::env::var_os("FERRUM_TEST_COMFY_SAVED_LATENT").map(std::path::PathBuf::from);
    ensure!(
        comfy_latent
            .as_ref()
            .is_none_or(|p| !p.as_os_str().is_empty()),
        "explicit Comfy latent path must not be empty"
    );
    // Refuse replacement of any existing result, before CUDA is constructed.
    fs::create_dir(&directory)?;
    let result = execute(
        &config_path,
        &input_path,
        &directory,
        comfy_latent.as_deref(),
    );
    if let Err(error) = &result {
        if let Err(record_error) = write_json_new(
            &directory.join("failure.json"),
            &json!({"status":"failed","error":format!("{error:#}"),
            "retirement_proved":false,"quality_status":"not_evaluated"}),
        ) {
            // Preserve the original execution/fence owner even when the
            // diagnostic destination itself fails.
            eprintln!("roundtrip failure record could not be written: {record_error:#}");
        }
    }
    result
}

fn execute(
    config_path: &Path,
    input_path: &Path,
    directory: &Path,
    comfy_latent: Option<&Path>,
) -> Result<()> {
    let config = PH3ImageProductConfig::load(config_path)?;
    let PH3ImageBackendConfig::Cuda(backend) = &config.backend;
    ensure!(
        config.vae_decoder_tile_batch.maximum_tiles() == 1,
        "VAE interoperability control requires the single-tile decoder policy"
    );
    let resolved = backend.resolved(config.vae_spatial_policy, config.vae_decoder_tile_batch)?;
    let target = config
        .serving
        .as_ref()
        .context("declare the original edit target in serving.default_width/default_height")?;
    ensure!(
        target.default_width <= resolved.pipeline.maximum_width
            && target.default_height <= resolved.pipeline.maximum_height,
        "edit target exceeds configured product dimensions"
    );
    let host = HostMemoryAccount::new(resolved.host.clone());
    let start = Instant::now();
    let encoded = input_png(&host, input_path, ImageInputLimits::default())?;
    let input_sha = format!("{:x}", Sha256::digest(encoded.bytes()));
    let decoded = ManagedRgba8Image::decode_png(&host, &encoded, ImageInputLimits::default())?;
    let (width, height) = (decoded.width(), decoded.height());
    let plan = PH3ImageEditPixelPlan::new(
        width,
        height,
        target.default_width,
        target.default_height,
        PH3ImageEditPixelPolicy::default(),
        ImageInputLimits::default(),
    )?;
    let (vae_width, vae_height) = plan.vae_dimensions();
    let input = plan.prepare(&host, &decoded)?;
    drop(input.vision);
    let pixels = input.vae;
    let preprocess_s = start.elapsed().as_secs_f64();
    drop((encoded, decoded));
    let metadata_path = config.metadata_directory.join("vae/config.json");
    let metadata = source::read_bounded(
        &metadata_path,
        usize::try_from(config.metadata_limits.configuration_bytes)?,
    )?;
    let archive_path = &config.archives.vae;
    let archive = SafetensorsArchive::open_single_file(
        archive_path.parent().context("VAE archive parent")?,
        archive_path
            .file_name()
            .and_then(|v| v.to_str())
            .context("VAE archive filename")?,
    )?;
    let vae_source = Arc::new(ComfyCanonicalWeightSource::new(archive)?);
    let adapter = PH3ImageVaeAdapter::new(
        Arc::clone(&vae_source),
        &metadata,
        ferrum_engine::image_generation::p_h3::PH3ImageAssetLimits::default().vae,
    )?;
    let checkpoint_statistics =
        checkpoint_latent_statistics(&vae_source, adapter.latent_channels() as usize)?;
    let encoder = adapter.prepare_encoder(
        1,
        vae_width as usize,
        vae_height as usize,
        config.vae_spatial_policy,
        PH3ImageVaeEncoderPreset::CudaF32Fma,
    )?;
    let decoder = adapter.prepare_decoder(
        1,
        vae_width as usize,
        vae_height as usize,
        config.vae_spatial_policy,
        PH3ImageVaeDecoderPreset::F32A8FullAttention,
    )?;
    // Validate/read optional external bytes before CUDA construction. The one
    // native input owner is charged to the same configured account; there is
    // no unaccounted file-sized Vec or additional capacity ledger.
    let imported = comfy_latent
        .map(|path| latent_io::read(&host, path, &decoder.normalized_latent_spec()?))
        .transpose()?;
    let context = json!({"config":config_path,"config_sha256":file_sha256(config_path)?,
        "input":input_path,"input_sha256":input_sha,
        "geometry":{"source_png":[width,height],"edit_target":[target.default_width,target.default_height],
            "target_source":"ordinary product serving defaults; configure to match original edit request",
            "vision_pixels":plan.vision_dimensions(),"vae_pixels":[vae_width,vae_height],
            "vae_input_shape":pixels.tensor().dimensions(),"posterior_noise_shape":encoder.posterior_noise_spec()?.dimensions(),
            "normalized_latent_shape":encoder.normalized_latent_spec()?.dimensions(),
            "reconstruction_shape":decoder.pixel_output_spec()?.dimensions()},
        "vae_archive":archive_path,"vae_archive_sha256":file_sha256(archive_path)?,
        "vae_metadata":metadata_path,"vae_metadata_sha256":format!("{:x}",Sha256::digest(&metadata)),
        "checkpoint_latent_statistics":checkpoint_statistics,
        "spatial_policy":config.vae_spatial_policy,"pixel_profile":ferrum_models::p_h3_image::PH3_IMAGE_EDIT_PIXEL_PROFILE,
        "comfy_saved_latent":imported.as_ref().map(|(_,record)|record),
        "preprocess_s":preprocess_s,"unit_rgb":f32_statistics(pixels.bytes().bytes(),3)?,
        "bypassed_inference":["prompt_expansion","vision_encoder","text_encoder","text_connector","dit"],
        "posterior_note":"zero supplied noise selects mean; native CUDA Philox F32 seed42 does not equal Torch CPU Generator seed42",
        "quality_status":"not_evaluated","latency_scope":"host wall per stage, not GPU busy time or throughput qualification"});
    let original_pixel_sha = format!("{:x}", Sha256::digest(pixels.bytes().bytes()));
    write_json_new(&directory.join("request.json"), &context)?;
    let execution = cuda::Execution::new(&resolved)?;
    let runtime_weak = Arc::downgrade(&execution.runtime);
    let mut actor = TensorSerialActor::new(execution.runtime.clone())?;
    let (mut enc_session, enc_init) = execution.initialize(
        "encoder",
        encoder.family().clone(),
        encoder.compile_options(),
        ComponentWeightSource::Stored(&encoder.weight_source(host.clone())),
        &host,
    )?;
    let (mut dec_session, dec_init) = execution.initialize(
        "decoder",
        decoder.family().clone(),
        decoder.compile_options(),
        ComponentWeightSource::Stored(&decoder.weight_source(host.clone())),
        &host,
    )?;
    let work = (|| -> Result<Value> {
        let spec = encoder.posterior_noise_spec()?;
        let zero = ManagedHostTensor::new(
            id("value.vae_roundtrip.zero_noise")?,
            spec.clone(),
            host.allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                usize::try_from(spec.minimum_storage_bytes()?)?,
            )?
            .freeze(),
        )?;
        let (mean_latent, mean_pixels, mean_timing) = roundtrip(
            "mean",
            &host,
            &mut actor,
            &encoder,
            &mut enc_session,
            &decoder,
            &mut dec_session,
            &pixels,
            &zero,
        )?;
        drop(zero);
        let mean_png = write_png(&host, &directory.join("mean.png"), &mean_pixels)?;
        let mean_artifact = latent_io::write_pair(directory, "mean", &mean_latent)?;
        let (noise, noise_record) = execution.posterior_noise(&spec, &host, &mut actor)?;
        let noise_stats =
            f32_statistics(noise.bytes().bytes(), adapter.latent_channels() as usize)?;
        let (sample_latent, sample_pixels, sample_timing) = roundtrip(
            "native-posterior42",
            &host,
            &mut actor,
            &encoder,
            &mut enc_session,
            &decoder,
            &mut dec_session,
            &pixels,
            &noise,
        )?;
        let sample_png = write_png(
            &host,
            &directory.join("native-posterior42.png"),
            &sample_pixels,
        )?;
        let sample_artifact =
            latent_io::write_pair(directory, "native-posterior42", &sample_latent)?;
        let imported_result = if let Some((latent, record)) = &imported {
            let (output, timing) = decode_latent(
                "comfy-saved-latent",
                &host,
                &mut actor,
                &decoder,
                &mut dec_session,
                latent,
            )?;
            let png = write_png(&host, &directory.join("comfy-saved-latent.png"), &output)?;
            Some(json!({"source":record,"timing":timing,"png":png,
                "latent":f32_statistics(latent.bytes().bytes(),adapter.latent_channels() as usize)?,
                "versus_mean_latent":difference(latent.bytes().bytes(),mean_latent.bytes().bytes())?,
                "versus_native_posterior42_latent":difference(latent.bytes().bytes(),sample_latent.bytes().bytes())?,
                "reconstruction":f32_statistics(output.bytes().bytes(),3)?,
                "versus_mean_reconstruction":difference(output.bytes().bytes(),mean_pixels.bytes().bytes())?,
                "versus_native_posterior42_reconstruction":difference(output.bytes().bytes(),sample_pixels.bytes().bytes())?,
                "versus_unit_rgb":difference(output.bytes().bytes(),pixels.bytes().bytes())?}))
        } else {
            None
        };
        ensure!(
            format!("{:x}", Sha256::digest(pixels.bytes().bytes())) == original_pixel_sha,
            "the original UnitRgb owner changed during roundtrip"
        );
        Ok(
            json!({"mean":{"timing":mean_timing,"png":mean_png,"latent_artifact":mean_artifact,"latent":f32_statistics(mean_latent.bytes().bytes(),adapter.latent_channels() as usize)?,
                "reconstruction":f32_statistics(mean_pixels.bytes().bytes(),3)?,"versus_unit_rgb":difference(mean_pixels.bytes().bytes(),pixels.bytes().bytes())?},
            "native_posterior42":{"timing":sample_timing,"png":sample_png,"latent_artifact":sample_artifact,"latent":f32_statistics(sample_latent.bytes().bytes(),adapter.latent_channels() as usize)?,
                "reconstruction":f32_statistics(sample_pixels.bytes().bytes(),3)?,"versus_unit_rgb":difference(sample_pixels.bytes().bytes(),pixels.bytes().bytes())?},
            "posterior_noise":{"execution":noise_record,"statistics":noise_stats},
            "comfy_saved_latent":imported_result,
            "between_modes":{"normalized_latent":difference(mean_latent.bytes().bytes(),sample_latent.bytes().bytes())?,
                "reconstruction":difference(mean_pixels.bytes().bytes(),sample_pixels.bytes().bytes())?}}),
        )
    })();
    drop(actor);
    // Even a failed inspection attempts the original sessions' normal close;
    // any remaining owner is retained in the error and cannot produce success.
    let encoder_close = cuda::close(enc_session);
    let decoder_close = cuda::close(dec_session);
    // If more than one boundary fails, retain every original error owner.
    let mut failures = Vec::new();
    let mut collect = |result: Result<Value>| match result {
        Ok(value) => value,
        Err(error) => {
            failures.push(error);
            Value::Null
        }
    };
    let observations = collect(work);
    let encoder_close = collect(encoder_close);
    let decoder_close = collect(decoder_close);
    if !failures.is_empty() {
        let message = failures
            .iter()
            .map(|e| format!("{e:#}"))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(cuda::retained(message, failures));
    }
    let memory_after_close = execution.runtime.memory_availability()?;
    drop((pixels, imported, encoder, decoder, adapter, execution));
    let runtime_destroyed = runtime_weak.upgrade().is_none();
    ensure!(
        runtime_destroyed,
        "CUDA runtime still has a retained owner after all components retired"
    );
    let usage = host.usage()?;
    ensure!(
        usage.committed_bytes() == 0 && usage.allocated_bytes() == 0,
        "Host backing or grants survived all roundtrip owners"
    );
    write_json_new(
        &directory.join("result.json"),
        &json!({"status":"completed","context":context,"observations":observations,
        "initialization":{"encoder":enc_init,"decoder":dec_init},"retirement":{"encoder":encoder_close,"decoder":decoder_close,
            "host_final_committed_bytes":usage.committed_bytes(),"host_final_allocated_bytes":usage.allocated_bytes(),
            "runtime_owner_destroyed":runtime_destroyed,"device_memory_after_component_close":memory_after_close},
        "quality_status":"not_evaluated","torch_cpu_noise_equivalent":false,
        "measurement_limits":["No raw mean/logvar export from public encoder consumer; statistics describe normalized posterior successors.",
            "Reconstruction preserves the actual post-preprocessing VAE geometry and is compared with that same UnitRgb input, without resizing outputs.",
            "Final RGB statistics follow decoder normalization/clamping, not every raw intermediate.",
            "Native .latent files preserve actual 4D F32 shape; .comfy.latent companions insert only a T=1 metadata view and retain identical payload bytes. New-format external input must be F32 with exact matching geometry; no implicit scale or dtype conversion is applied.",
            "Closed receipts and runtime destruction prove owner retirement; the sampled device availability is not a global zero-VRAM or peak measurement.",
            "This diagnostic does not validate edit semantics, upstream numerical parity, or performance SLOs."]}),
    )
}

#[test]
fn roundtrip_statistics_preserve_channel_planes_and_report_nonfinite() {
    let bytes = [1_f32, 3., -2., 2.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let stats = f32_statistics(&bytes, 2).unwrap();
    assert_eq!(stats["channels"][0]["mean"], 2.);
    assert_eq!(stats["channels"][1]["min"], -2.);
    assert_eq!(stats["channels"][1]["rms"], 2.);
    let invalid = [f32::NAN, f32::INFINITY]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let stats = f32_statistics(&invalid, 1).unwrap();
    assert_eq!(stats["finite"], false);
    assert_eq!(stats["channels"][0]["nonfinite_count"], 2);
    assert!(stats["channels"][0]["min"].is_null());
    assert!(difference(&bytes, &invalid).is_err());
    assert!(difference(&invalid, &invalid).is_err());
}

#[test]
fn roundtrip_difference_is_f64_and_rejects_invalid_extents() {
    let left = [1_f32, 2.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let right = [4_f32, -2.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let stats = difference(&left, &right).unwrap();
    assert_eq!(stats["max_abs"], 4.);
    assert_eq!(stats["mae"], 3.5);
    assert_eq!(stats["rmse"], (12.5_f64).sqrt());
    assert!(difference(&[], &[]).is_err());
    assert!(f32_statistics(&left, 3).is_err());
}

#[test]
fn roundtrip_latent_statistics_read_the_actual_checkpoint_values_and_bits() {
    let directory = tempfile::tempdir().unwrap();
    let mut mean_bits = [0u32; 24];
    mean_bits[0] = 0x8000_0000;
    mean_bits[1] = 0x3f80_0001;
    let mut std_bits = [1f32.to_bits(); 24];
    std_bits[0] = 0x3f00_0001;
    let mean = mean_bits
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let std = std_bits
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let views = [
        (
            "latents_mean",
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![24], &mean).unwrap(),
        ),
        (
            "latents_std",
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![24], &std).unwrap(),
        ),
    ];
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("stats.safetensors"),
    )
    .unwrap();
    let source = ComfyCanonicalWeightSource::new(
        SafetensorsArchive::open_single_file(directory.path(), "stats.safetensors").unwrap(),
    )
    .unwrap();
    let result = checkpoint_latent_statistics(&source, 24).unwrap();
    assert_eq!(result["combined_bytes"], 192);
    assert_eq!(
        result["tensors"]["latents_mean"]["bits_u32"][0],
        json!(0x8000_0000u32)
    );
    assert_eq!(
        result["tensors"]["latents_mean"]["bits_u32"][1],
        json!(0x3f80_0001u32)
    );
    assert_eq!(
        result["tensors"]["latents_std"]["bits_u32"][0],
        json!(0x3f00_0001u32)
    );
    let mut combined = Sha256::new();
    combined.update(&mean);
    combined.update(&std);
    assert_eq!(
        result["combined_sha256"],
        format!("{:x}", combined.finalize())
    );
    assert!(checkpoint_latent_statistics(&source, 23).is_err());
}
