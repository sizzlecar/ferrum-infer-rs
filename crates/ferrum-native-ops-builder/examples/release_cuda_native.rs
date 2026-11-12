//! Release-only orchestration of the existing locked source builder.
//! The historical production set supplies source plans and provider rows; the
//! qualified Q6 prefill extension uses its current checked-in source lock.
//! Every source byte is recovered by its locked digest before a fresh build.
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
    assemble_native_operator_set, lock_native_operator_source_definition,
    materialize_native_operator_package_spec, materialize_native_operator_source_overlay,
    package_native_operator, run_native_operator_source_build, NativeOperatorPackageRequest,
    NativeOperatorPackageSpec, NativeOperatorPackageSpecRequest, NativeOperatorSetRequest,
    NativeOperatorSourceBuildPlan, NativeOperatorSourceBuildRequest,
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
    /// Rebuild the historical set plus locked FP8 and architecture-specific XQA.
    Build(Box<Build>),
    /// Resolve bytes/ABI/evidence and require complete architecture-specific coverage.
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
    /// Architecture of the source-plan reference; every archive is rebuilt for the target.
    #[arg(long, value_enum)]
    reference_architecture: Option<Architecture>,
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
    /// CPU source declaration generated from this workspace, required for SM120.
    #[arg(long)]
    gpt_oss_source_catalog: Option<PathBuf>,
    #[arg(long)]
    fp8_source: PathBuf,
    #[arg(long)]
    fp8_plan: PathBuf,
    #[arg(long)]
    fused_f32_source: PathBuf,
    #[arg(long)]
    fused_f32_plan: PathBuf,
    #[arg(long)]
    fused_int8_source: PathBuf,
    #[arg(long)]
    fused_int8_plan: PathBuf,
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

    fn cuda_toolkit(self) -> &'static str {
        match self {
            Self::Sm89 => "12.4",
            Self::Sm120 => "12.8",
        }
    }
}

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
                "verified {} native artifacts; complete architecture-specific release coverage",
                set.artifacts.len()
            );
        }
    }
    Ok(())
}

fn resolve(
    path: &Path,
    architecture: Architecture,
    final_set: bool,
) -> Result<ResolvedNativeOperatorArtifactSet> {
    let set = NativeOperatorArtifactSetLock::load_and_resolve_for_target(
        path,
        Some(architecture.compute_capability()),
        "x86_64-unknown-linux-gnu",
    )?;
    ResolvedCudaNativeBuildCoverage::resolve(
        &set,
        required_units_for_set(architecture, final_set),
    )?;
    let q6 = set
        .artifacts
        .iter()
        .find(|artifact| {
            artifact.lock.operator == CudaNativeBuildUnit::UpstreamQ6F32Linear.artifact_operator()
        })
        .ok_or("production set is missing the Q6 projection artifact")?;
    require_q6_f16_exports(&q6.lock.required_exports, final_set)?;
    Ok(set)
}

fn require_q6_f16_exports(exports: &[String], prefill: bool) -> Result<()> {
    for required in ferrum_native_ops::upstream_q6_f16_linear::UPSTREAM_Q6_F16_LINEAR_EXPORTS {
        if !exports.iter().any(|export| export == required) {
            return Err(format!("production Q6 artifact is missing F16 export {required}").into());
        }
    }
    let required = ferrum_native_ops::upstream_q6_f16_linear::UPSTREAM_Q6_F16_PREFILL_PLAN_EXPORT;
    if prefill && !exports.iter().any(|export| export == required) {
        return Err(format!("release Q6 artifact is missing prefill export {required}").into());
    }
    Ok(())
}

fn required_units_for_set(
    architecture: Architecture,
    final_set: bool,
) -> impl Iterator<Item = CudaNativeBuildUnit> {
    PRODUCTION
        .iter()
        .copied()
        .chain(final_set.then_some(CudaNativeBuildUnit::CutlassFp8ScaledMm))
        .chain(final_set.then_some(CudaNativeBuildUnit::CutlassFusedF32Attention))
        .chain(final_set.then_some(CudaNativeBuildUnit::CutlassInt8FusedOutput))
        .chain(
            (final_set && matches!(architecture, Architecture::Sm120))
                .then_some(CudaNativeBuildUnit::FlashinferXqa),
        )
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
    gpt_source: Option<&NativeOperatorProviderCatalog>,
) -> Result<()> {
    original.validate()?;
    combined.validate()?;
    if original.schema_version != combined.schema_version || original.backend != combined.backend {
        return Err("catalog schema/backend changed".into());
    }
    let mut expected = original.clone();
    if let Some(source) = gpt_source {
        source.validate()?;
        if source.schema_version != original.schema_version
            || source.backend != original.backend
            || source.providers.len() != original.providers.len()
        {
            return Err(
                "GPT-OSS source declaration must preserve the reference catalog shape".into(),
            );
        }
        let key = (
            "operation.gpt_oss.causal_paged_attention",
            "provider.cuda.gpt_oss.causal_paged_attention.f16",
        );
        let mut replacements = 0;
        for (old, declared) in expected.providers.iter_mut().zip(&source.providers) {
            if (old.operation_id.as_str(), old.provider_id.as_str()) == key {
                if old.operation_id != declared.operation_id
                    || old.provider_id != declared.provider_id
                    || old.operation_contract_version != declared.operation_contract_version
                    || old.operation_fingerprint != declared.operation_fingerprint
                    || old.provider_version != declared.provider_version
                {
                    return Err("GPT-OSS source declaration changed a versioned contract".into());
                }
                *old = declared.clone();
                replacements += 1;
            } else if old != declared {
                return Err("GPT-OSS source declaration changed an unrelated provider".into());
            }
        }
        if replacements != 1 {
            return Err(
                "GPT-OSS source declaration must contain its one exact provider row".into(),
            );
        }
    }
    let additions = [
        (
            "operation.dense_linear_fp8.bf16",
            "provider.cuda.dense_linear_fp8.bf16.cutlass",
        ),
        (
            "operation.dense_linear_fp8_pair.bf16",
            "provider.cuda.dense_linear_fp8_pair.bf16.cutlass",
        ),
        (
            "operation.dense_linear_fp8_triple.bf16",
            "provider.cuda.dense_linear_fp8_triple.bf16.cutlass",
        ),
        (
            "operation.structured_gqa_attention_fused.f32",
            "provider.cuda.structured_gqa_attention_fused.f32.cutlass",
        ),
        (
            "operation.comfy_convrot_codebook_w4a8.bf16",
            "provider.cuda.comfy_convrot_codebook_w4a8.bf16.cutlass_int8_fused_output",
        ),
        (
            "operation.comfy_convrot_codebook_w4a8.swiglu_f32.bf16",
            "provider.cuda.comfy_convrot_w4a8.swiglu_f32.bf16.cutlass_int8_fused_output",
        ),
        (
            "operation.comfy_convrot_w8a8.bf16",
            "provider.cuda.comfy_convrot_w8a8.bf16.cutlass_int8_fused_output",
        ),
        (
            "operation.comfy_convrot_w8a8.swiglu_f32.bf16",
            "provider.cuda.comfy_convrot_w8a8.swiglu_f32.bf16.cutlass_int8_fused_output",
        ),
    ];
    if original.providers.iter().any(|row| {
        additions
            .iter()
            .any(|(_, provider)| row.provider_id == *provider)
    }) || combined.providers.len() != original.providers.len() + additions.len()
        || expected
            .providers
            .iter()
            .any(|row| !combined.providers.contains(row))
        || additions.iter().any(|(operation, provider)| {
            combined
                .providers
                .iter()
                .filter(|row| row.provider_id == *provider && row.operation_id == *operation)
                .count()
                != 1
        })
    {
        return Err("combined catalog must preserve production rows (or the exact GPT-OSS source declaration) and add FP8, fused F32 attention and fused INT8 output providers".into());
    }
    Ok(())
}

fn build(args: &Build) -> Result<()> {
    let reference = resolve(
        &args.reference_lock,
        args.reference_architecture.unwrap_or(args.architecture),
        false,
    )?;
    let base_catalog = reference_catalog(&reference)?;
    let combined: NativeOperatorProviderCatalog = read_json(&args.catalog)?;
    let gpt_source = match args.architecture {
        Architecture::Sm120 => Some(read_json::<NativeOperatorProviderCatalog>(
            args.gpt_oss_source_catalog
                .as_deref()
                .ok_or("SM120 release requires the CPU-generated GPT-OSS source catalog")?,
        )?),
        Architecture::Sm89 => {
            if args.gpt_oss_source_catalog.is_some() {
                return Err("SM89 release does not consume the SM120 XQA declaration".into());
            }
            None
        }
    };
    validate_catalog_extension(&base_catalog, &combined, gpt_source.as_ref())?;
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
        let unit = CudaNativeBuildUnit::from_artifact_operator(&artifact.lock.operator)
            .ok_or_else(|| {
                format!(
                    "unsupported release native operator {}",
                    artifact.lock.operator
                )
            })?;
        // This operator has a qualified prefill extension. Recover its current
        // locked bytes, rather than silently rebuilding the decode-only archive
        // carried by the historical set. Other operators keep their own plans.
        let plan_path = if unit == CudaNativeBuildUnit::UpstreamQ6F32Linear {
            args.workspace
                .join("native-operators/cuda/source-locks/upstream-q6-f32-linear.plan.json")
        } else {
            reference_root.join(&artifact.lock.source_build_plan.path)
        };
        let plan: NativeOperatorSourceBuildPlan = read_json(&plan_path)?;
        if plan.operator != artifact.lock.operator {
            return Err("reference source plan operator mismatch".into());
        }
        if unit == CudaNativeBuildUnit::CutlassFp8ScaledMm {
            return Err("reference unexpectedly already contains FP8".into());
        }
        let source = args.work.join(format!("{}-source", unit.as_str()));
        materialize_source(&plan, unit, args, &source)?;
        // Only source plans and provider contracts cross architectures. The
        // archive is always freshly compiled below, and packaging verifies the
        // target/toolkit declaration against that actual source-build receipt.
        let reference_spec: NativeOperatorPackageSpec =
            if unit == CudaNativeBuildUnit::UpstreamQ6F32Linear {
                let declared = args.work.join("upstream-q6.source.package.json");
                materialize_native_operator_package_spec(&NativeOperatorPackageSpecRequest {
                    definition_path: args.workspace.join(
                        "native-operators/cuda/package-definitions/upstream-q6-f32-linear.json",
                    ),
                    g03_catalog_path: args.catalog.clone(),
                    output_path: declared.clone(),
                })?;
                read_json(&declared)?
            } else {
                read_json(&reference_root.join(&artifact.lock.package_spec.path))?
            };
        let spec = args
            .work
            .join(format!("{}-target.package.json", unit.as_str()));
        write_new(
            &spec,
            &serde_json::to_vec_pretty(&package_spec_for_target(
                reference_spec,
                args.architecture,
            ))?,
        )?;
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
    let xqa = if matches!(args.architecture, Architecture::Sm120) {
        Some(prepare_xqa(args)?)
    } else {
        None
    };
    for (unit, name, source, plan) in [
        (
            CudaNativeBuildUnit::CutlassFusedF32Attention,
            "cutlass-fused-f32-attention",
            &args.fused_f32_source,
            &args.fused_f32_plan,
        ),
        (
            CudaNativeBuildUnit::CutlassInt8FusedOutput,
            "cutlass-int8-fused-output",
            &args.fused_int8_source,
            &args.fused_int8_plan,
        ),
    ] {
        let declared: NativeOperatorSourceBuildPlan = read_json(plan)?;
        if declared.operator != unit.artifact_operator() {
            return Err(format!("{name} source plan has wrong operator").into());
        }
        let source_spec = args.work.join(format!("{name}.source.package.json"));
        materialize_native_operator_package_spec(&NativeOperatorPackageSpecRequest {
            definition_path: args.workspace.join(format!(
                "native-operators/cuda/package-definitions/{name}.json"
            )),
            g03_catalog_path: args.catalog.clone(),
            output_path: source_spec.clone(),
        })?;
        let target_spec = args.work.join(format!("{name}.target.package.json"));
        write_new(
            &target_spec,
            &serde_json::to_vec_pretty(&package_spec_for_target(
                read_json(&source_spec)?,
                args.architecture,
            ))?,
        )?;
        prepared.push((unit, source.clone(), plan.clone(), target_spec));
    }
    // Reject a missing historical source before spending time on any compiler.
    let mut receipts = Vec::new();
    for (unit, source, plan, spec) in prepared {
        receipts.push(build_package(
            args,
            unit.as_str(),
            &source,
            &plan,
            &spec,
            if matches!(
                unit,
                CudaNativeBuildUnit::CutlassFusedF32Attention
                    | CudaNativeBuildUnit::CutlassInt8FusedOutput
            ) {
                &args.workspace
            } else {
                &args.reference_workspace
            },
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
    if let Some((source, plan, spec)) = xqa {
        receipts.push(build_package(
            args,
            "flashinfer-xqa",
            &source,
            &plan,
            &spec,
            &args.workspace,
        )?);
    }
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
        let mut expected = old.lock.clone();
        if old.lock.operator == CudaNativeBuildUnit::UpstreamQ6F32Linear.artifact_operator() {
            let plan_path = args
                .workspace
                .join("native-operators/cuda/source-locks/upstream-q6-f32-linear.plan.json");
            let plan: NativeOperatorSourceBuildPlan = read_json(&plan_path)?;
            let spec: NativeOperatorPackageSpec =
                read_json(&args.work.join("upstream-q6.source.package.json"))?;
            // Permit exactly the selected locked source and its declared export
            // extension. Existing provider contracts must still match below.
            if old
                .lock
                .required_exports
                .iter()
                .any(|export| !spec.required_exports.contains(export))
            {
                return Err("Q6 prefill package removed an existing export".into());
            }
            expected.source_package_sha256 = plan.source_package.sha256;
            expected.source_build_plan.sha256 = digest_file(&plan_path)?;
            expected.required_exports = spec.required_exports;
        }
        let new = completed
            .artifacts
            .iter()
            .find(|new| new.lock.operator == old.lock.operator)
            .ok_or("a production artifact was lost")?;
        if new.lock.source_package_sha256 != expected.source_package_sha256
            || new.lock.source_build_plan.sha256 != expected.source_build_plan.sha256
            || new.lock.operation_bindings != expected.operation_bindings
            || new.lock.required_exports != expected.required_exports
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

fn package_spec_for_target(
    mut spec: NativeOperatorPackageSpec,
    architecture: Architecture,
) -> NativeOperatorPackageSpec {
    spec.compute_capabilities = vec![architecture.compute_capability().to_owned()];
    spec.cuda_toolkit = Some(architecture.cuda_toolkit().to_owned());
    spec.cuda_runtime_min = Some(architecture.cuda_toolkit().to_owned());
    spec
}

fn prepare_xqa(args: &Build) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let native = args
        .workspace
        .join("crates/ferrum-kernels/native/flashinfer-xqa");
    let source = args.work.join("flashinfer-xqa-source");
    materialize_native_operator_source_overlay(
        &native.join("source-overlay.json"),
        &args.source_bundle,
        &args.workspace,
        &source,
    )?;
    fs::create_dir(source.join("kernels"))?;
    for name in ["abi.h", "boundary.cu"] {
        write_new(
            &source.join("kernels").join(name),
            &fs::read(native.join(name))?,
        )?;
    }
    let plan = args.work.join("flashinfer-xqa.plan.json");
    lock_native_operator_source_definition(
        &native.join("flashinfer-xqa.seed.json"),
        &source,
        &plan,
    )?;
    let spec = args.work.join("flashinfer-xqa.package.json");
    materialize_native_operator_package_spec(&NativeOperatorPackageSpecRequest {
        definition_path: args
            .workspace
            .join("native-operators/cuda/package-definitions/flashinfer-xqa.json"),
        g03_catalog_path: args.catalog.clone(),
        output_path: spec.clone(),
    })?;
    Ok((source, plan, spec))
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
        CudaNativeBuildUnit::CutlassFusedF32Attention => {
            return Err("fused F32 attention uses its observed source closure".into())
        }
        CudaNativeBuildUnit::CutlassInt8FusedOutput => {
            return Err("fused INT8 output uses its observed source closure".into())
        }
        CudaNativeBuildUnit::FlashinferXqa => {
            return Err("XQA uses its observed source closure".into())
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
