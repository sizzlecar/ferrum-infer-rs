//! Release-only orchestration of the existing locked source builder.
//! The historical production set supplies exact source plans and provider rows;
//! every source byte is recovered by its locked digest before a fresh build.
//! No CUDA runtime, device, publication, or application inference is involved.

use std::{
    collections::BTreeMap,
    error::Error,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

use clap::{Parser, Subcommand};
use ferrum_native_ops::{
    CudaNativeBuildUnit, NativeOperatorArtifactSetLock, ResolvedCudaNativeBuildCoverage,
    ResolvedNativeOperatorArtifactSet,
};
use ferrum_native_ops_builder::{
    assemble_native_operator_set, materialize_native_operator_package_spec,
    package_native_operator, run_native_operator_source_build, NativeOperatorPackageRequest,
    NativeOperatorPackageSpecRequest, NativeOperatorSetRequest, NativeOperatorSourceBuildPlan,
    NativeOperatorSourceBuildRequest,
};
use ferrum_types::NativeOperatorProviderCatalog;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Parser)]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Resolve all historical artifacts before exporting their canonical catalog.
    Reference {
        #[arg(long, value_enum, default_value = "sm120")]
        architecture: Architecture,
        #[arg(long)]
        lock: PathBuf,
        #[arg(long)]
        catalog_output: PathBuf,
    },
    /// Rebuild the complete historical set and add the explicitly locked FP8 unit.
    Build(Box<Build>),
    /// Resolve bytes/ABI/evidence and require both production and FP8 coverage.
    Verify {
        #[arg(long, value_enum, default_value = "sm120")]
        architecture: Architecture,
        #[arg(long)]
        lock: PathBuf,
    },
}

#[derive(clap::Args)]
struct Build {
    #[arg(long, value_enum, default_value = "sm120")]
    architecture: Architecture,
    #[arg(long)]
    reference_lock: PathBuf,
    #[arg(long)]
    reference_workspace: PathBuf,
    #[arg(long)]
    source_bundle: PathBuf,
    /// Additional pinned source tree for historical plans; locked digests still apply.
    #[arg(long)]
    historical_source: Option<PathBuf>,
    #[arg(long)]
    llama_source: PathBuf,
    #[arg(long)]
    workspace: PathBuf,
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    fp8_source: PathBuf,
    #[arg(long)]
    fp8_plan: PathBuf,
    #[arg(long)]
    work: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    object_cache: PathBuf,
    #[arg(long)]
    builder_sha: String,
    #[arg(long)]
    cuda_toolkit_root: PathBuf,
    #[arg(long)]
    nvcc: PathBuf,
    #[arg(long)]
    cc: PathBuf,
    #[arg(long)]
    cxx: PathBuf,
    #[arg(long)]
    ar: PathBuf,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum Architecture {
    Sm89,
    Sm120,
}

impl Architecture {
    fn compute_capability(self) -> &'static str {
        match self {
            Self::Sm89 => "sm_89",
            Self::Sm120 => "sm_120",
        }
    }

    fn package_definition(self) -> &'static str {
        match self {
            Self::Sm89 => "native-operators/cuda/package-definitions/cutlass-fp8-sm89.json",
            Self::Sm120 => "native-operators/cuda/package-definitions/cutlass-fp8.json",
        }
    }

    fn required_units(self) -> &'static [CudaNativeBuildUnit] {
        match self {
            Self::Sm89 => SM89_REQUIRED,
            Self::Sm120 => PRODUCTION,
        }
    }
}

const SM89_REQUIRED: &[CudaNativeBuildUnit] = &[
    CudaNativeBuildUnit::Marlin,
    CudaNativeBuildUnit::VllmMarlin,
    CudaNativeBuildUnit::VllmMoeMarlin,
    CudaNativeBuildUnit::VllmPagedAttentionV2,
];

// The Linux production feature closure, including indexed text attention and
// vllm-moe-marlin's dense vllm-marlin repacking dependency. The addressed and
// indexed attention units share one artifact; units are not artifact counts.
const PRODUCTION: &[CudaNativeBuildUnit] = &[
    CudaNativeBuildUnit::Marlin,
    CudaNativeBuildUnit::VllmMarlin,
    CudaNativeBuildUnit::VllmMoeMarlin,
    CudaNativeBuildUnit::VllmPagedAttentionV2,
    CudaNativeBuildUnit::VllmIndexedPagedAttentionV2,
    CudaNativeBuildUnit::UpstreamLinear,
    CudaNativeBuildUnit::UpstreamExtraLinear,
    CudaNativeBuildUnit::UpstreamQ6F32Linear,
];

fn main() -> Result<()> {
    match Arguments::parse().command {
        Command::Reference {
            architecture,
            lock,
            catalog_output,
        } => {
            let reference = resolve(&lock, architecture, false)?;
            let catalog = reference_catalog(&reference)?;
            write_new(&catalog_output, &catalog.canonical_json_bytes()?)?;
        }
        Command::Build(arguments) => build(&arguments)?,
        Command::Verify { lock, architecture } => {
            let set = resolve(&lock, architecture, true)?;
            println!(
                "verified {} native artifacts; production + CUTLASS FP8",
                set.artifacts.len()
            );
        }
    }
    Ok(())
}

fn resolve(
    path: &Path,
    architecture: Architecture,
    fp8: bool,
) -> Result<ResolvedNativeOperatorArtifactSet> {
    let set = NativeOperatorArtifactSetLock::load_and_resolve_for_target(
        path,
        Some(architecture.compute_capability()),
        "x86_64-unknown-linux-gnu",
    )?;
    let required = architecture
        .required_units()
        .iter()
        .copied()
        .chain(fp8.then_some(CudaNativeBuildUnit::CutlassFp8ScaledMm));
    ResolvedCudaNativeBuildCoverage::resolve(&set, required)?;
    Ok(set)
}

fn reference_catalog(
    set: &ResolvedNativeOperatorArtifactSet,
) -> Result<NativeOperatorProviderCatalog> {
    let first = set.artifacts.first().ok_or("empty reference set")?;
    let root = set
        .lock_path
        .parent()
        .ok_or("reference lock has no parent")?;
    let catalog: NativeOperatorProviderCatalog =
        read_json(&root.join(&first.lock.g03_catalog.path))?;
    catalog.validate()?;
    if catalog.canonical_sha256()? != set.g03_catalog_sha256 {
        return Err("reference catalog is not canonical".into());
    }
    Ok(catalog)
}

fn validate_catalog_extension(
    original: &NativeOperatorProviderCatalog,
    combined: &NativeOperatorProviderCatalog,
) -> Result<()> {
    original.validate()?;
    combined.validate()?;
    if original.schema_version != combined.schema_version || original.backend != combined.backend {
        return Err("catalog schema/backend changed".into());
    }
    let fp8 = "provider.cuda.dense_linear_fp8.bf16.cutlass";
    if original.providers.iter().any(|row| row.provider_id == fp8)
        || combined.providers.len() != original.providers.len() + 1
        || original
            .providers
            .iter()
            .any(|row| !combined.providers.contains(row))
        || combined
            .providers
            .iter()
            .filter(|row| {
                row.provider_id == fp8 && row.operation_id == "operation.dense_linear_fp8.bf16"
            })
            .count()
            != 1
    {
        return Err("combined catalog must preserve every production row and add only FP8".into());
    }
    Ok(())
}

fn build(args: &Build) -> Result<()> {
    let reference = resolve(&args.reference_lock, args.architecture, false)?;
    let base_catalog = reference_catalog(&reference)?;
    let combined: NativeOperatorProviderCatalog = read_json(&args.catalog)?;
    validate_catalog_extension(&base_catalog, &combined)?;
    let catalog_sha = combined.canonical_sha256()?;
    if digest_file(&args.catalog)? != catalog_sha {
        return Err("combined catalog must use canonical bytes".into());
    }
    fs::create_dir(&args.work)?;
    fs::create_dir(&args.output)?;
    let reference_root = args
        .reference_lock
        .parent()
        .ok_or("reference lock has no parent")?;
    let mut prepared = Vec::new();
    for artifact in &reference.artifacts {
        let plan_path = reference_root.join(&artifact.lock.source_build_plan.path);
        let plan: NativeOperatorSourceBuildPlan = read_json(&plan_path)?;
        if plan.operator != artifact.lock.operator {
            return Err("reference source plan operator mismatch".into());
        }
        let unit = CudaNativeBuildUnit::from_artifact_operator(&plan.operator)
            .ok_or_else(|| format!("unsupported release native operator {}", plan.operator))?;
        if unit == CudaNativeBuildUnit::CutlassFp8ScaledMm {
            return Err("reference unexpectedly already contains FP8".into());
        }
        let source = args.work.join(format!("{}-source", unit.as_str()));
        materialize_source(&plan, unit, args, &source)?;
        let spec = reference_root.join(&artifact.lock.package_spec.path);
        prepared.push((unit, source, plan_path, spec));
    }
    let fp8: NativeOperatorSourceBuildPlan = read_json(&args.fp8_plan)?;
    if fp8.operator != CudaNativeBuildUnit::CutlassFp8ScaledMm.artifact_operator() {
        return Err("FP8 source plan has wrong operator".into());
    }
    let spec = args.work.join("cutlass-fp8.package.json");
    materialize_native_operator_package_spec(&NativeOperatorPackageSpecRequest {
        definition_path: args.workspace.join(args.architecture.package_definition()),
        g03_catalog_path: args.catalog.clone(),
        output_path: spec.clone(),
    })?;
    // Reject a missing historical source before spending time on any compiler.
    let mut receipts = Vec::new();
    for (unit, source, plan, spec) in prepared {
        receipts.push(build_package(
            args,
            unit.as_str(),
            &source,
            &plan,
            &spec,
            &args.reference_workspace,
        )?);
    }
    receipts.push(build_package(
        args,
        "cutlass-fp8",
        &args.fp8_source,
        &args.fp8_plan,
        &spec,
        &args.workspace,
    )?);
    let lock = args.output.join("native-operator-set.lock.json");
    assemble_native_operator_set(&NativeOperatorSetRequest {
        expected_receipt_sha256: receipts
            .iter()
            .map(|path| digest_file(path))
            .collect::<Result<_>>()?,
        receipt_paths: receipts,
        expected_g03_catalog_sha256: catalog_sha,
        compute_capability: args.architecture.compute_capability().into(),
        output_lock_path: lock.clone(),
    })?;
    let completed = resolve(&lock, args.architecture, true)?;
    for old in &reference.artifacts {
        let new = completed
            .artifacts
            .iter()
            .find(|new| new.lock.operator == old.lock.operator)
            .ok_or("a production artifact was lost")?;
        if new.lock.source_package_sha256 != old.lock.source_package_sha256
            || new.lock.source_build_plan.sha256 != old.lock.source_build_plan.sha256
            || new.lock.operation_bindings != old.lock.operation_bindings
            || new.lock.required_exports != old.lock.required_exports
        {
            return Err(format!(
                "production source/contract changed for {}",
                old.lock.operator
            )
            .into());
        }
    }
    println!(
        "staged native lock: {} sha256={}",
        lock.display(),
        digest_file(&lock)?
    );
    Ok(())
}

fn build_package(
    args: &Build,
    name: &str,
    source: &Path,
    plan: &Path,
    spec: &Path,
    licenses: &Path,
) -> Result<PathBuf> {
    let build = args.work.join(format!("{name}-build"));
    run_native_operator_source_build(&NativeOperatorSourceBuildRequest {
        plan_path: plan.to_owned(),
        source_root: source.to_owned(),
        output_dir: build.clone(),
        compute_capability: args.architecture.compute_capability().into(),
        builder_sha: args.builder_sha.clone(),
        nvcc_path: args.nvcc.clone(),
        ccbin_path: args.cxx.clone(),
        ar_path: args.ar.clone(),
        cuda_toolkit_root: args.cuda_toolkit_root.clone(),
        nvcc_threads: 1,
        object_cache_dir: args.object_cache.clone(),
        plan_only: false,
    })?;
    let package = args.output.join("packages").join(name);
    package_native_operator(&NativeOperatorPackageRequest {
        spec_path: spec.to_owned(),
        source_root: source.to_owned(),
        license_root: licenses.to_owned(),
        source_build_receipt_path: build.join("source-build.receipt.json"),
        source_build_plan_path: plan.to_owned(),
        g03_catalog_path: args.catalog.clone(),
        abi_contract_path: args
            .workspace
            .join("native-operators/abi/ferrum-native-abi-v2.json"),
        output_dir: package.clone(),
        cc: args.cc.clone(),
        ar: args.ar.clone(),
    })?;
    Ok(package.join("package.receipt.json"))
}

fn materialize_source(
    plan: &NativeOperatorSourceBuildPlan,
    unit: CudaNativeBuildUnit,
    args: &Build,
    output: &Path,
) -> Result<()> {
    fs::create_dir(output)?;
    let directory = match unit {
        CudaNativeBuildUnit::Marlin => "marlin-instance-policy",
        CudaNativeBuildUnit::VllmMarlin => "vllm-marlin",
        CudaNativeBuildUnit::VllmMoeMarlin => "vllm-moe-marlin-profiler-free",
        CudaNativeBuildUnit::VllmPagedAttentionV2
        | CudaNativeBuildUnit::VllmIndexedPagedAttentionV2 => "vllm-paged-attention-indexed",
        CudaNativeBuildUnit::UpstreamLinear => "upstream-linear",
        CudaNativeBuildUnit::UpstreamExtraLinear => "upstream-extra-linear",
        CudaNativeBuildUnit::UpstreamQ6F32Linear => "upstream-q6-f32-linear",
        CudaNativeBuildUnit::CutlassFp8ScaledMm => {
            return Err("FP8 uses its observed source closure".into())
        }
    };
    let mut files = BTreeMap::new();
    for file in plan.translation_units.iter().chain(&plan.headers) {
        if let Some(previous) = files.insert(&file.path, &file.sha256) {
            if previous != &file.sha256 {
                return Err("conflicting locked source digests".into());
            }
        }
    }
    for (relative, sha) in files {
        let relative_path = checked_relative(relative)?;
        let repository_file = Path::new("native-operators/cuda")
            .join(directory)
            .join(relative_path.file_name().ok_or("source file has no name")?);
        let mut candidates = vec![
            (args.source_bundle.as_path(), relative_path.to_owned()),
            (args.reference_workspace.as_path(), repository_file.clone()),
            (args.workspace.as_path(), repository_file),
        ];
        if let Some(historical) = &args.historical_source {
            candidates.push((historical.as_path(), relative_path.to_owned()));
        }
        if let Ok(llama) = relative_path.strip_prefix("llama") {
            candidates.push((args.llama_source.as_path(), llama.to_owned()));
        }
        copy_matching_source(relative, sha, &candidates, &output.join(relative_path))?;
    }
    for directory in &plan.include_dirs {
        fs::create_dir_all(output.join(checked_relative(directory)?))?;
    }
    Ok(())
}

fn checked_relative(value: &str) -> Result<&Path> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains(['\\', ':'])
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!("invalid relative source path: {value}").into());
    }
    Ok(path)
}

fn copy_matching_source(
    relative: &str,
    expected: &str,
    candidates: &[(&Path, PathBuf)],
    output: &Path,
) -> Result<()> {
    for (root, path) in candidates {
        let candidate = root.join(path);
        if !candidate.is_file() {
            continue;
        }
        let canonical = candidate.canonicalize()?;
        if !canonical.starts_with(root.canonicalize()?) {
            return Err("source symlink escapes admitted root".into());
        }
        if digest_file(&canonical)? != expected {
            continue;
        }
        fs::create_dir_all(output.parent().ok_or("source output has no parent")?)?;
        let mut target = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        std::io::copy(&mut fs::File::open(canonical)?, &mut target)?;
        target.sync_all()?;
        if digest_file(output)? != expected {
            return Err("source changed during materialization".into());
        }
        return Ok(());
    }
    Err(format!("no pinned source matches {relative} sha256={expected}; refusing to alter the production plan").into())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn digest_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
#[path = "release_cuda_native/tests.rs"]
mod tests;
