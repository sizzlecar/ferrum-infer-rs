use crate::shared::{id, TensorPolicy};
use anyhow::{bail, ensure, Context, Result};
use ferrum_diffusion::{pack_spatial_into, SingleFrameLatentShape};
use ferrum_execution::{PreparedTensorExecutor, TensorExecutorConfig};
use ferrum_kernels::backend::cuda::{
    vnext_ops::*,
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};
use ferrum_models::normal_noise::{NormalNoiseConfig, NormalNoiseFamily};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
    sync::Arc,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    seed: u64,
    shape: NormalNoiseConfig,
    device: usize,
    usable_memory_bytes: u64,
    bootstrap_memory_bytes: u64,
    stream_memory_bytes: u64,
    opaque_memory_profile: String,
    maximum_host_bytes: usize,
    output_directory: PathBuf,
}
pub fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .context("usage: generate_normal_noise CONFIG.json")?;
    ensure!(
        args.next().is_none(),
        "expected exactly one explicit JSON configuration"
    );
    let file = File::open(&path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= 65536,
        "configuration must be a regular file <=64KiB"
    );
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 65536, "configuration grew beyond 64KiB");
    let args: Arguments = serde_json::from_slice(&bytes)?;
    let report = run(&args)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run(args: &Arguments) -> Result<Value> {
    let output_spec = args.shape.output_tensor()?;
    let geometry =
        PhiloxNormalBf16Geometry::new(&NormalNoiseConfig::state_tensor()?, &output_spec)?;
    let output_bytes = usize::try_from(geometry.output_bytes())?;
    let output_peak = output_bytes
        .checked_mul(2)
        .context("packed/NCHW Host peak overflows")?;
    let peak = output_peak.checked_add(16).context("Host peak overflows")?;
    ensure!(
        peak <= args.maximum_host_bytes,
        "noise requires {peak} managed Host bytes before device construction"
    );
    ensure!(
        !args.output_directory.exists(),
        "output directory already exists"
    );
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        peak,
        HostMemoryClassLimits {
            input_preprocess_bytes: 16,
            output_bytes: output_peak,
            transfer_staging_bytes: 0,
            weight_backing_bytes: 0,
        },
    )?);
    let runtime = Arc::new(CudaDeviceRuntime::new_tensor_eager(
        CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(
                args.device,
                id(format!("device.cuda.{}", args.device))?,
            )?,
            usable_memory_bytes: args.usable_memory_bytes,
            bootstrap_memory_bytes: args.bootstrap_memory_bytes,
            stream_memory_bytes: args.stream_memory_bytes,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: args.opaque_memory_profile.clone(),
        },
    )?);
    let device_geometry = CudaPhiloxDeviceGeometry::query(&runtime)?;
    let launch = device_geometry.launch(geometry.element_count())?;
    let provider = cuda_vnext_philox_normal_bf16_provider(&runtime)?;
    let provider_fingerprint = provider
        .descriptor()
        .provider_implementation_fingerprint()
        .to_owned();
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(philox_normal_bf16_contract()?)],
        vec![provider],
    )?;
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.normal_noise")?,
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )?;
    let catalog = registry.capability_catalog(descriptor.clone(), vec![engine])?;
    let family = TypedFamilyRegistration::new(NormalNoiseFamily::new(
        CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID,
    )?)
    .prepare_with_profile(
        &serde_json::to_value(&args.shape)?,
        &NumericalProfileId::new(CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID)
            .map_err(anyhow::Error::msg)?,
    )?;
    let policy = TensorPolicy {
        capacity_bytes: args.usable_memory_bytes,
        storage_profiles: descriptor
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    };
    let compilation = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy,
        &registry.planning(),
        &args.shape.compile_options()?,
    )?;
    let executable = compilation.executable();
    ensure!(
        matches!(
            executable.execution_plan().payload().execution_weights(),
            ExecutionWeights::None
        ),
        "normal generation unexpectedly needs weights"
    );
    let resources = match executable
        .execution_plan()
        .provision_static(runtime.clone(), id("request.noise.provision")?)?
        .into_provisioning()
    {
        StaticProvisioning::NoStatic(p) => p.into_plan_runtime(),
        StaticProvisioning::Required(_) => {
            bail!("normal generation unexpectedly needs static resources")
        }
    };
    let result = (|| -> Result<Value> {
        let mut executor = PreparedTensorExecutor::new(
            executable.clone(),
            registry.bind_plan(executable)?,
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )?;
        let state = PhiloxSeedOffset::fresh(args.seed);
        let mut input = host.allocate_zeroed(HostMemoryClass::InputPreprocess, 16)?;
        input.as_mut().copy_from_slice(&state.little_endian_bytes());
        let input = ManagedHostTensor::new(
            id(NormalNoiseFamily::STATE_ID)?,
            NormalNoiseConfig::state_tensor()?,
            input.freeze(),
        )?;
        let outputs = executor
            .execute(
                id("run.normal_noise.generate")?,
                id("request.normal_noise.generate")?,
                vec![input],
            )
            .map_err(|e| anyhow::anyhow!("noise execution: {e}"))?;
        let [output] = outputs.as_slice() else {
            bail!("normal generation returned an unexpected output count");
        };
        ensure!(
            output.tensor() == &output_spec
                && output.value_id().as_str() == NormalNoiseFamily::OUTPUT_ID
                && host.owns(output.bytes()),
            "noise output port/account mismatch"
        );
        let nchw = output.bytes().bytes();
        ensure!(
            nchw.chunks_exact(2)
                .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80),
            "nonfinite normal output"
        );
        let dims = geometry.shape();
        let shape = SingleFrameLatentShape::new(
            1,
            usize::try_from(dims[1])?,
            usize::try_from(dims[2])?,
            usize::try_from(dims[3])?,
        )?;
        let mut packed = host.allocate_zeroed(HostMemoryClass::Output, output_bytes)?;
        // [u8;2] is alignment-one opaque BF16. No uncharged tensor conversion.
        pack_spatial_into(
            shape,
            bytemuck::try_cast_slice::<u8, [u8; 2]>(nchw).map_err(anyhow::Error::msg)?,
            bytemuck::try_cast_slice_mut::<u8, [u8; 2]>(packed.as_mut())
                .map_err(anyhow::Error::msg)?,
        )?;
        fs::create_dir(&args.output_directory)?;
        write_new(args.output_directory.join("noise.nchw.bf16"), nchw)?;
        write_new(
            args.output_directory.join("noise.packed.bf16"),
            packed.as_ref(),
        )?;
        Ok(
            json!({"configuration":args,"profile":CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_ID,
            "numerical_identity":CUDA_PHILOX_NORMAL_BF16_NUMERICAL_PROFILE_IDENTITY,"build_identity_sha256":CUDA_PHILOX_BUILD_IDENTITY_SHA256,
            "provider_fingerprint":provider_fingerprint,"runtime_fingerprint":descriptor.runtime_implementation_fingerprint,
            "device_geometry":{"multiprocessors":device_geometry.multiprocessors(),"max_threads_per_multiprocessor":device_geometry.max_threads_per_multiprocessor()},
            "launch":{"blocks":launch.blocks(),"threads_per_block":launch.threads_per_block(),"normal4_rounds_per_thread":launch.rounds(),"offset":0,"offset_advance":launch.offset_advance()},
            "nchw":{"file":"noise.nchw.bf16","shape":dims,"bytes":output_bytes,"sha256":hash(nchw)},
            "packed":{"file":"noise.packed.bf16","shape":shape.sequence_dims(),"bytes":output_bytes,"sha256":hash(packed.as_ref())},
            "dtype":"BF16","byte_order":"little-endian","finite":true,"torch_byte_parity":"not_verified",
            "opaque_allowances":"explicit caller declaration; not measured capacity qualification"}),
        )
    })();
    let close = match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            ensure!(
                receipt.released_static_resources() == 0,
                "unexpected static release"
            );
            Ok(())
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => Err(anyhow::anyhow!(
            "noise root still has {strong_count} owners"
        )),
        Err(e) => Err(anyhow::anyhow!(
            "noise root close failed: {:?}",
            e.failure()
        )),
    };
    let usage = host.usage()?;
    let clean = if usage.committed_bytes() == 0 && usage.allocated_bytes() == 0 {
        close
    } else {
        Err(anyhow::anyhow!(
            "noise Host owners remain: committed={}, allocated={}; root close={close:?}",
            usage.committed_bytes(),
            usage.allocated_bytes()
        ))
    };
    let mut report = match (result, clean) {
        (Ok(v), Ok(())) => v,
        (Err(e), Ok(())) | (Ok(_), Err(e)) => return Err(e),
        (Err(e), Err(c)) => return Err(e.context(format!("noise cleanup also failed: {c:#}"))),
    };
    report["cleanup"] = json!({"managed_host_committed_bytes":0,"managed_host_allocated_bytes":0,"plan_root":"closed"});
    write_new(
        args.output_directory.join("receipt.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}
fn write_new(path: PathBuf, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
