//! Execute the product audio-reference encoder with original weights and a
//! bounded, explicitly laid out F32 waveform; keep qualification logic in Rust.
#[cfg(feature = "cuda-fast-h3")]
mod native {
    use anyhow::{ensure, Context, Result};
    use ferrum_engine::video_generation::fast_h3::{
        conditioning::{encode_audio_reference, ConditioningCompilation},
        cuda::{FastH3CudaComposition, FastH3CudaConfig},
    };
    use ferrum_execution::{
        component::ComponentInitializationConfig, TensorExecutorConfig, TensorSerialActor,
    };
    use ferrum_models::fast_h3::audio_vae::{FastH3AudioComponentLimits, FastH3AudioVaeConfig};
    use ferrum_quantization::SafetensorsArchive;
    use ferrum_runtime::*;
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use std::{
        fs::{File, OpenOptions},
        io::{Read, Write},
        path::{Path, PathBuf},
        sync::Arc,
        time::Instant,
    };

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Layout {
        ChannelMajor,
        Interleaved,
    }
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Configuration {
        cuda: FastH3CudaConfig,
        vae_config: PathBuf,
        vae_file: PathBuf,
        input_f32_file: PathBuf,
        input_layout: Layout,
        channels: u64,
        samples_per_channel: u64,
        sampling_rate: u32,
        output_f32_file: PathBuf,
        maximum_metadata_bytes: usize,
        maximum_source_bytes: u64,
        maximum_stored_bytes: u64,
        maximum_activation_bytes: u64,
        maximum_plan_nodes: usize,
    }
    fn read_metadata(path: &Path, maximum: usize) -> Result<Vec<u8>> {
        ensure!(
            maximum > 0 && maximum < isize::MAX as usize,
            "invalid metadata bound"
        );
        let mut bytes = Vec::new();
        File::open(path)?
            .take(maximum as u64 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= maximum, "metadata exceeds configured bound");
        Ok(bytes)
    }
    fn file_sha(path: &Path) -> Result<String> {
        let mut f = File::open(path)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = f.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    pub(super) fn run() -> Result<()> {
        let mut args = std::env::args().skip(1);
        ensure!(
            args.next().as_deref() == Some("--configuration"),
            "usage: encode_audio_reference --configuration FILE.json"
        );
        let path = args.next().context("missing configuration")?;
        ensure!(args.next().is_none(), "unexpected argument");
        let c: Configuration = serde_json::from_slice(&read_metadata(Path::new(&path), 1 << 20)?)?;
        c.cuda.validate()?;
        let metadata = read_metadata(&c.vae_config, c.maximum_metadata_bytes)?;
        let a = FastH3AudioVaeConfig::parse(&metadata, c.maximum_metadata_bytes)?;
        ensure!(
            c.sampling_rate == a.sampling_rate() && c.channels > 0 && c.samples_per_channel > 0,
            "requires positive original-rate mono channels"
        );
        let bytes = c
            .channels
            .checked_mul(c.samples_per_channel)
            .and_then(|n| n.checked_mul(4))
            .context("input extent overflow")?;
        ensure!(
            bytes <= c.maximum_activation_bytes && bytes <= c.cuda.host_bytes as u64 / 2,
            "waveform exceeds bounded Host capacity"
        );
        ensure!(!c.output_f32_file.exists(), "output must be a new artifact");
        let mut f = File::open(&c.input_f32_file)?;
        ensure!(
            f.metadata()?.is_file() && f.metadata()?.len() == bytes,
            "raw waveform file differs from declared shape"
        );
        let archive_sha = file_sha(&c.vae_file)?;
        let began = Instant::now();
        let cuda = Arc::new(FastH3CudaComposition::new(&c.cuda)?);
        let mut raw = cuda
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, usize::try_from(bytes)?)?;
        f.read_exact(raw.as_mut())?;
        let mut extra = [0u8];
        ensure!(
            f.read(&mut extra)? == 0,
            "waveform file changed during read"
        );
        let input_sha = format!("{:x}", Sha256::digest(raw.as_ref()));
        let raw = raw.freeze();
        let raw = match c.input_layout {
            Layout::ChannelMajor => raw,
            Layout::Interleaved => {
                let mut channel_major = cuda
                    .host
                    .allocate_zeroed(HostMemoryClass::InputPreprocess, usize::try_from(bytes)?)?;
                for channel in 0..c.channels {
                    for sample in 0..c.samples_per_channel {
                        let src = ((sample * c.channels + channel) * 4) as usize;
                        let dst = ((channel * c.samples_per_channel + sample) * 4) as usize;
                        channel_major.as_mut()[dst..dst + 4]
                            .copy_from_slice(&raw.bytes()[src..src + 4]);
                    }
                }
                drop(raw);
                channel_major.freeze()
            }
        };
        let waveform = ManagedHostTensor::new(
            ProgramValueId::new("value.audio.reference.source")?,
            ResolvedTensorSpec::new(
                vec![c.channels, 1, c.samples_per_channel],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            raw,
        )?;
        let archive = Arc::new(SafetensorsArchive::open_single_file(
            c.vae_file.parent().context("source directory")?,
            c.vae_file
                .file_name()
                .and_then(|s| s.to_str())
                .context("source basename")?,
        )?);
        let registry = cuda.registry(None)?;
        let catalog = cuda.catalog(&registry)?;
        let compilation = ConditioningCompilation {
            registry: &registry,
            catalog: &catalog,
            policy: &cuda.policy,
            materializers: &cuda.materializers,
        };
        let mut actor = TensorSerialActor::new(cuda.runtime.clone())?;
        let initialized_seconds = began.elapsed().as_secs_f64();
        let started = Instant::now();
        let encoded = encode_audio_reference(
            &a,
            archive,
            waveform,
            c.sampling_rate,
            &compilation,
            cuda.runtime.clone(),
            &cuda.host,
            &mut actor,
            FastH3AudioComponentLimits {
                maximum_metadata_bytes: c.maximum_metadata_bytes,
                maximum_source_bytes: c.maximum_source_bytes,
                maximum_stored_bytes: c.maximum_stored_bytes,
                maximum_activation_bytes: c.maximum_activation_bytes,
                maximum_plan_nodes: c.maximum_plan_nodes,
            },
            |stage| {
                Ok(ComponentInitializationConfig {
                    run_id: RunId::new(format!("run.qualification.{stage}"))?,
                    request_id: RequestIdentity::new("request.audio.reference")?,
                    transaction_id: TransactionId::new(format!(
                        "transaction.qualification.{stage}"
                    ))?,
                    static_policy: StaticInitializationPolicy::new(
                        c.cuda.staging_bytes as u64,
                        64,
                    )?,
                    executor: TensorExecutorConfig::default(),
                })
            },
        );
        let output = match encoded {
            Ok(output) => output,
            Err(error) => {
                eprintln!(
                    "{}",
                    serde_json::json!({"stage":error.stage,"error":error.to_string(),"retirement":"not claimed; original failure and actor owners retained until process exit"})
                );
                std::mem::forget((error, actor, registry, catalog, cuda));
                std::process::exit(1);
            }
        };
        let encoder_seconds = started.elapsed().as_secs_f64();
        let output_shape = output.tensor().dimensions().to_vec();
        let output_sha = format!("{:x}", Sha256::digest(output.bytes().bytes()));
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for word in output.bytes().bytes().chunks_exact(4) {
            let value = f32::from_le_bytes(word.try_into().unwrap());
            ensure!(value.is_finite(), "nonfinite normalized posterior mode");
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        let mut artifact = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&c.output_f32_file)?;
        artifact.write_all(output.bytes().bytes())?;
        artifact.flush()?;
        drop(output);
        drop(actor);
        let remaining = cuda.host.usage()?.committed_bytes();
        ensure!(
            remaining == 0,
            "audio reference encoder retained {remaining} Host bytes after completion"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"schema":"ferrum.fast_h3.audio_reference_encoder.v1","configuration":c,"config_sha256":format!("{:x}",Sha256::digest(&metadata)),"original_archive_sha256":archive_sha,"input_sha256":input_sha,"normalized_mode_sha256":output_sha,"normalized_mode_shape":output_shape,"minimum":minimum,"maximum":maximum,"runtime_initialization_seconds":initialized_seconds,"encoder_prepare_execute_and_copy_seconds":encoder_seconds,"host_committed_after_retirement":remaining,"scope":"Complete original F32 encoder trunk, causal projection, posterior mode and original normalization through the product conditioning executor. Not an end-to-end generation performance or quality claim."})
            )?
        );
        Ok(())
    }
}
fn main() {
    #[cfg(feature = "cuda-fast-h3")]
    if let Err(error) = native::run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
    #[cfg(not(feature = "cuda-fast-h3"))]
    {
        eprintln!("encode_audio_reference requires --features cuda-fast-h3");
        std::process::exit(1);
    }
}
