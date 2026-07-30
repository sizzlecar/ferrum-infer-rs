use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant, UNIX_EPOCH};

use ferrum_native_ops::{
    legacy_signature_matches_without_numeric_line, load_manifest, CudaNativeBuildUnit,
    NativeBuildArtifactCache, NativeBuildArtifactLookup, NativeBuildArtifactSpec,
    NativeOperatorArtifactSetLock, NativeOperatorResolveRequest, NativeOperatorResolver,
    NativeOperatorSystemLibrary, ResolvedCudaNativeBuildCoverage,
};
use ferrum_types::{
    resolve_native_operator_manifest, NativeOperatorBackend, NativeOperatorLinkage,
    NativeOperatorRequirement, LEGACY_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

const FA2_NATIVE_MANIFEST_ENV: &str = "FERRUM_FA2_NATIVE_MANIFEST";
const FA2_NATIVE_ARTIFACT_ENV: &str = "FERRUM_FA2_NATIVE_ARTIFACT";
const FA2_NATIVE_SOURCE_SHA256_ENV: &str = "FERRUM_FA2_NATIVE_SOURCE_SHA256";
const FA2_NATIVE_INPUTS_SHA256_ENV: &str = "FERRUM_FA2_NATIVE_INPUTS_SHA256";
const FA2_NATIVE_ARTIFACT_COMPILE_ENV: &str = "FERRUM_FA2_NATIVE_ARTIFACT_COMPILE";
const NATIVE_OP_ARTIFACT_FEATURE_ENV: &str = "CARGO_FEATURE_NATIVE_OP_ARTIFACT";
const NATIVE_OPERATOR_SET_LOCK_ENV: &str = "FERRUM_NATIVE_OPERATOR_SET_LOCK";
const COMPILED_NATIVE_OPERATOR_SET_JSON_ENV: &str = "FERRUM_COMPILED_NATIVE_OPERATOR_SET_JSON";
const COMPILED_FA2_NATIVE_MANIFEST_ENV: &str = "FERRUM_COMPILED_FA2_NATIVE_MANIFEST";
const COMPILED_FA2_NATIVE_ARTIFACT_ENV: &str = "FERRUM_COMPILED_FA2_NATIVE_ARTIFACT";
const COMPILED_FA2_NATIVE_SOURCE_SHA256_ENV: &str = "FERRUM_COMPILED_FA2_NATIVE_SOURCE_SHA256";
const COMPILED_FA2_NATIVE_INPUTS_SHA256_ENV: &str = "FERRUM_COMPILED_FA2_NATIVE_INPUTS_SHA256";
const COMPILED_FA2_NATIVE_BINARY_SHA256_ENV: &str = "FERRUM_COMPILED_FA2_NATIVE_BINARY_SHA256";
const CUDA_NATIVE_BUILD_CACHE_ENV: &str = "FERRUM_CUDA_NATIVE_BUILD_CACHE";
const CUDA_NATIVE_IMPORT_DIRS_ENV: &str = "FERRUM_CUDA_NATIVE_IMPORT_DIRS";
const CUDA_NATIVE_SOURCE_POLICY_ENV: &str = "FERRUM_CUDA_NATIVE_SOURCE_POLICY";
const CUDA_NATIVE_SIGNATURE_SCHEMA: &str = "ferrum-cuda-native-input-v2";
const DEFAULT_NVCC_THREADS: u32 = 4;
const MAX_NVCC_THREADS: u32 = 8;
const HISTORICAL_NVCC_THREAD_VALUES: std::ops::RangeInclusive<u32> = 0..=MAX_NVCC_THREADS;

const CORE_PTX_KERNELS: &[&str] = &[
    "kernels/fused_add_rms_norm.cu",
    "kernels/fused_silu_mul.cu",
    "kernels/rms_norm.cu",
    "kernels/sandwich_norm.cu",
    "kernels/rope.cu",
    "kernels/decode_attention.cu",
    "kernels/residual_add.cu",
    "kernels/scaled_add_inplace.cu",
    "kernels/flash_decode_attention.cu",
    "kernels/paged_decode_attention.cu",
    "kernels/paged_varlen_attention.cu",
    "kernels/paged_varlen_attention_vllm.cu",
    "kernels/dequant_int4.cu",
    "kernels/batched_decode_attention.cu",
    "kernels/softmax.cu",
    "kernels/embedding_lookup.cu",
    "kernels/flash_attn_full.cu",
    "kernels/batched_flash_decode_attention.cu",
    "kernels/qk_norm_rope.cu",
    "kernels/split_qkv_norm_rope_into_paged_cache.cu",
    "kernels/transpose.cu",
    "kernels/kv_cache_append.cu",
    "kernels/split_qkv.cu",
    "kernels/add_bias.cu",
    "kernels/layer_norm.cu",
    "kernels/gelu.cu",
    "kernels/decode_attention_hm.cu",
    "kernels/gather_columns.cu",
    "kernels/moe_combine.cu",
    "kernels/moe_router.cu",
    "kernels/moe_align_block_size.cu",
    "kernels/moe_align_block_size_pair_ids.cu",
    "kernels/moe_build_pairs.cu",
    "kernels/int8_paged_decode_attention.cu",
    "kernels/argmax_rows.cu",
    "kernels/split_qkv_norm_rope_into_paged_cache_vllm.cu",
    "kernels/gated_delta_rule.cu",
    "kernels/linear_attention.cu",
    "kernels/vnext_causal_attention.cu",
    "kernels/qwen35_paged_qkv.cu",
];

const CORE_PTX_HEADERS: &[&str] = &["kernels/common.cuh"];

fn cuda_root_from_env() -> Option<PathBuf> {
    for key in [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        if let Some(value) = env::var_os(key) {
            let path = PathBuf::from(value);
            if path.join("include").join("cuda.h").is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn file_fingerprint(path: &str) -> String {
    let meta = fs::metadata(path).unwrap_or_else(|e| panic!("metadata {path}: {e}"));
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in &bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{path}:len={}:fnv1a64={hash:016x}", meta.len())
}

fn sha256_file_fingerprint(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    format!(
        "{}:len={}:sha256={:x}",
        path.display(),
        bytes.len(),
        Sha256::digest(&bytes)
    )
}

fn resolve_program(program: &Path) -> PathBuf {
    if program.components().count() > 1 {
        return program
            .canonicalize()
            .unwrap_or_else(|_| program.to_path_buf());
    }
    env::var_os("PATH")
        .into_iter()
        .flat_map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .map(|directory| directory.join(program))
        .find(|candidate| candidate.is_file())
        .and_then(|candidate| candidate.canonicalize().ok())
        .unwrap_or_else(|| program.to_path_buf())
}

fn command_version_fingerprint(program: &Path) -> String {
    let resolved = resolve_program(program);
    let metadata = fs::metadata(&resolved)
        .ok()
        .map(|metadata| {
            let modified = metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                .map(|value| format!("{}.{:09}", value.as_secs(), value.subsec_nanos()))
                .unwrap_or_else(|| "unknown".to_string());
            format!("len={}:mtime={modified}", metadata.len())
        })
        .unwrap_or_else(|| "metadata=unavailable".to_string());
    let output = std::process::Command::new(&resolved)
        .arg("--version")
        .output();
    match output {
        Ok(output) => {
            let mut bytes = output.stdout;
            bytes.extend_from_slice(&output.stderr);
            format!(
                "program={}:resolved={}:{}:status={:?}:sha256={:x}",
                program.display(),
                resolved.display(),
                metadata,
                output.status.code(),
                Sha256::digest(&bytes)
            )
        }
        Err(error) => format!(
            "program={}:resolved={}:{}:spawn_error={error}",
            program.display(),
            resolved.display(),
            metadata
        ),
    }
}

fn cuda_native_toolchain_identity() -> &'static str {
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        for key in [
            "TARGET",
            "HOST",
            "NVCC_CCBIN",
            "CC",
            "CXX",
            "NVCC_PREPEND_FLAGS",
            "NVCC_APPEND_FLAGS",
        ] {
            println!("cargo:rerun-if-env-changed={key}");
        }
        let cuda_root = cuda_root_from_env();
        let nvcc = cuda_root
            .as_ref()
            .map(|root| root.join("bin").join("nvcc"))
            .unwrap_or_else(|| PathBuf::from("nvcc"));
        if nvcc.is_absolute() && nvcc.is_file() {
            println!("cargo:rerun-if-changed={}", nvcc.display());
        }
        let ccbin = env::var_os("NVCC_CCBIN")
            .or_else(|| env::var_os("CC"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("cc"));
        let cxx = env::var_os("CXX")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("c++"));
        let mut lines = vec![
            format!("schema={CUDA_NATIVE_SIGNATURE_SCHEMA}"),
            format!("target={}", env::var("TARGET").unwrap_or_default()),
            format!("host={}", env::var("HOST").unwrap_or_default()),
            format!(
                "cuda_root={}",
                cuda_root
                    .as_deref()
                    .map(Path::display)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "<path-search>".to_string())
            ),
            format!("nvcc={}", command_version_fingerprint(&nvcc)),
            format!("ccbin={}", command_version_fingerprint(&ccbin)),
            format!("cxx={}", command_version_fingerprint(&cxx)),
            format!("ar={}", command_version_fingerprint(&PathBuf::from("ar"))),
        ];
        for key in [
            "NVCC_CCBIN",
            "CC",
            "CXX",
            "NVCC_PREPEND_FLAGS",
            "NVCC_APPEND_FLAGS",
        ] {
            lines.push(format!(
                "env.{key}={}",
                env::var(key).unwrap_or_else(|_| "<unset>".to_string())
            ));
        }
        if let Some(cuda_root) = cuda_root {
            for relative in ["include/cuda.h", "include/cuda_runtime.h"] {
                let path = cuda_root.join(relative);
                if path.is_file() {
                    println!("cargo:rerun-if-changed={}", path.display());
                    lines.push(sha256_file_fingerprint(&path));
                } else {
                    lines.push(format!("{}=<missing>", path.display()));
                }
            }
        }
        lines.join("\n")
    })
}

fn cuda_native_input_signature(content_signature: &str) -> String {
    format!("{content_signature}\n{}", cuda_native_toolchain_identity())
}

fn signature_hash(signature: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(signature.as_bytes()))
}

fn configured_nvcc_threads() -> String {
    static VALUE: OnceLock<String> = OnceLock::new();
    VALUE
        .get_or_init(|| {
            println!("cargo:rerun-if-env-changed=FERRUM_NVCC_THREADS");
            let raw = env::var("FERRUM_NVCC_THREADS")
                .unwrap_or_else(|_| DEFAULT_NVCC_THREADS.to_string());
            let value = raw.parse::<u32>().unwrap_or_else(|error| {
                panic!("FERRUM_NVCC_THREADS must be an integer in 1..={MAX_NVCC_THREADS}: {error}")
            });
            assert!(
                (1..=MAX_NVCC_THREADS).contains(&value),
                "FERRUM_NVCC_THREADS must be in 1..={MAX_NVCC_THREADS}, got {value}"
            );
            value.to_string()
        })
        .clone()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CudaNativeSourcePolicy {
    Allow,
    CacheOnly,
}

fn configured_cuda_native_source_policy() -> CudaNativeSourcePolicy {
    static POLICY: OnceLock<CudaNativeSourcePolicy> = OnceLock::new();
    *POLICY.get_or_init(|| {
        println!("cargo:rerun-if-env-changed={CUDA_NATIVE_SOURCE_POLICY_ENV}");
        match env::var(CUDA_NATIVE_SOURCE_POLICY_ENV)
            .unwrap_or_else(|_| "allow".to_string())
            .as_str()
        {
            "allow" => CudaNativeSourcePolicy::Allow,
            "cache-only" => CudaNativeSourcePolicy::CacheOnly,
            value => panic!(
                "{CUDA_NATIVE_SOURCE_POLICY_ENV} must be `allow` or `cache-only`, got {value:?}"
            ),
        }
    })
}

fn enforce_cuda_native_source_policy(artifact: &str, reason: &str, signature: &str) {
    if configured_cuda_native_source_policy() == CudaNativeSourcePolicy::CacheOnly {
        emit_cuda_build_summary(
            artifact,
            "rejected",
            reason,
            Duration::from_millis(0),
            signature,
        );
        panic!(
            "CUDA native cache-only policy rejected source compilation: \
artifact={artifact} reason={reason} inputs_hash={}",
            signature_hash(signature)
        );
    }
}

fn emit_cuda_build_summary(
    artifact: &str,
    status: &str,
    reason: &str,
    elapsed: std::time::Duration,
    signature: &str,
) {
    eprintln!(
        "[cuda-build-summary] artifact={artifact} status={status} reason={reason} \
elapsed_ms={} inputs_hash={}",
        elapsed.as_millis(),
        signature_hash(signature)
    );
}

fn optional_non_empty_env(key: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={key}");
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn required_native_env(key: &str, value: Option<String>) -> String {
    value.unwrap_or_else(|| {
        panic!("{key} is required when configuring an FA2 native operator artifact")
    })
}

fn normalize_compute_capability(raw: &str) -> String {
    let value = raw.trim();
    if let Some(rest) = value.strip_prefix("sm_") {
        return format!("sm_{}", rest.replace('.', ""));
    }
    format!("sm_{}", value.replace('.', ""))
}

fn native_static_link_name(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| {
            panic!(
                "native operator artifact has no UTF-8 file name: {}",
                path.display()
            )
        });
    let Some(stripped) = name
        .strip_prefix("lib")
        .and_then(|name| name.strip_suffix(".a"))
    else {
        panic!(
            "static native operator artifact must be named lib<name>.a, got {}",
            path.display()
        );
    };
    if stripped.is_empty() {
        panic!(
            "static native operator artifact link name is empty: {}",
            path.display()
        );
    }
    stripped.to_string()
}

fn native_dynamic_link_name(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| {
            panic!(
                "native operator artifact has no UTF-8 file name: {}",
                path.display()
            )
        });
    if let Some(stripped) = name
        .strip_prefix("lib")
        .and_then(|name| name.strip_suffix(".dylib"))
    {
        if !stripped.is_empty() {
            return stripped.to_string();
        }
    }
    if let Some(rest) = name.strip_prefix("lib") {
        if let Some((stripped, _version)) = rest.split_once(".so") {
            if !stripped.is_empty() {
                return stripped.to_string();
            }
        }
    }
    panic!(
        "dynamic native operator artifact must be named lib<name>.so* or lib<name>.dylib, got {}",
        path.display()
    );
}

fn required_cuda_native_build_units() -> Vec<CudaNativeBuildUnit> {
    [
        ("CARGO_FEATURE_MARLIN", CudaNativeBuildUnit::Marlin),
        ("CARGO_FEATURE_VLLM_MARLIN", CudaNativeBuildUnit::VllmMarlin),
        (
            "CARGO_FEATURE_VLLM_MOE_MARLIN",
            CudaNativeBuildUnit::VllmMoeMarlin,
        ),
        (
            "CARGO_FEATURE_VLLM_PAGED_ATTN_V2",
            CudaNativeBuildUnit::VllmPagedAttentionV2,
        ),
    ]
    .into_iter()
    .filter_map(|(feature, unit)| env::var_os(feature).is_some().then_some(unit))
    .collect()
}

fn reject_incomplete_native_artifact_set(lock_path: &Path, error: &dyn std::fmt::Display) -> ! {
    panic!(
        "native operator artifact set {} does not cover the active CUDA build graph: {error}",
        lock_path.display()
    )
}

fn link_native_operator_artifact_set() -> Option<ResolvedCudaNativeBuildCoverage> {
    let start = Instant::now();
    let feature_enabled = env::var_os(NATIVE_OP_ARTIFACT_FEATURE_ENV).is_some();
    println!("cargo:rerun-if-env-changed={NATIVE_OP_ARTIFACT_FEATURE_ENV}");
    let lock_path = optional_non_empty_env(NATIVE_OPERATOR_SET_LOCK_ENV);
    let Some(lock_path) = lock_path else {
        println!("cargo:rustc-env={COMPILED_NATIVE_OPERATOR_SET_JSON_ENV}=[]");
        return None;
    };
    if !feature_enabled {
        panic!("{NATIVE_OPERATOR_SET_LOCK_ENV} requires --features native-op-artifact");
    }
    for legacy_key in [
        FA2_NATIVE_MANIFEST_ENV,
        FA2_NATIVE_ARTIFACT_ENV,
        FA2_NATIVE_SOURCE_SHA256_ENV,
        FA2_NATIVE_INPUTS_SHA256_ENV,
    ] {
        if env::var_os(legacy_key).is_some() {
            panic!("{NATIVE_OPERATOR_SET_LOCK_ENV} cannot be combined with legacy {legacy_key}");
        }
    }

    let lock_path = PathBuf::from(lock_path);
    println!("cargo:rerun-if-changed={}", lock_path.display());
    let compute_capability = normalize_compute_capability(
        &env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| detect_cuda_compute_cap()),
    );
    let resolved_set =
        NativeOperatorArtifactSetLock::load_and_resolve(&lock_path, Some(&compute_capability))
            .unwrap_or_else(|error| {
                panic!(
                    "failed to resolve native operator artifact set {}: {error}",
                    lock_path.display()
                )
            });
    let required_build_units = required_cuda_native_build_units();
    let build_coverage =
        match ResolvedCudaNativeBuildCoverage::resolve(&resolved_set, required_build_units) {
            Ok(coverage) => coverage,
            Err(error) => reject_incomplete_native_artifact_set(&lock_path, &error),
        };

    let cuda_root = cuda_root_from_env();
    if let Some(cuda_root) = cuda_root.as_ref() {
        let lib64 = cuda_root.join("lib64");
        if lib64.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
        }
    }
    let mut system_libraries = BTreeSet::new();
    let mut inventory = Vec::with_capacity(resolved_set.artifacts.len());
    for artifact in &resolved_set.artifacts {
        let resolved = &artifact.resolved;
        if resolved.manifest.backend != NativeOperatorBackend::Cuda {
            panic!(
                "ferrum-kernels CUDA artifact set rejects non-CUDA operator {} ({:?})",
                resolved.manifest.operator, resolved.manifest.backend
            );
        }
        println!(
            "cargo:rerun-if-changed={}",
            resolved.manifest_path.display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            resolved.artifact_path.display()
        );
        let parent = resolved
            .artifact_path
            .parent()
            .unwrap_or_else(|| Path::new("."));
        println!("cargo:rustc-link-search=native={}", parent.display());
        match resolved.manifest.linkage {
            NativeOperatorLinkage::Static => println!(
                "cargo:rustc-link-lib=static={}",
                native_static_link_name(&resolved.artifact_path)
            ),
            NativeOperatorLinkage::Dynamic => println!(
                "cargo:rustc-link-lib=dylib={}",
                native_dynamic_link_name(&resolved.artifact_path)
            ),
        }
        system_libraries.extend(artifact.lock.system_libraries.iter().copied());
        inventory.push(serde_json::json!({
            "schema_version": resolved.manifest.schema_version,
            "operator": resolved.manifest.operator,
            "operator_abi_version": resolved.manifest.operator_abi_version,
            "ferrum_native_abi_version": resolved.manifest.ferrum_native_abi_version,
            "backend": resolved.manifest.backend,
            "linkage": resolved.manifest.linkage,
            "g03_catalog_sha256": resolved.manifest.g03_catalog_sha256,
            "abi_contract_sha256": resolved.manifest.abi_contract_sha256,
            "descriptor_export": resolved.manifest.descriptor_export,
            "operation_bindings": resolved.manifest.operation_bindings,
            "exports": resolved.manifest.exports,
            "source_package_sha256": resolved.manifest.source_package.sha256,
            "inputs_sha256": resolved.manifest.inputs_sha256,
            "binary_sha256": resolved.artifact_sha256,
        }));
    }
    for library in system_libraries {
        let link_name = match library {
            NativeOperatorSystemLibrary::CudaDriver => "cuda",
            NativeOperatorSystemLibrary::CudaRuntime => "cudart",
            NativeOperatorSystemLibrary::Cublas => "cublas",
            NativeOperatorSystemLibrary::CublasLt => "cublasLt",
            NativeOperatorSystemLibrary::StdCxx => {
                if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
                    "c++"
                } else {
                    "stdc++"
                }
            }
        };
        println!("cargo:rustc-link-lib=dylib={link_name}");
    }
    let inventory_json =
        serde_json::to_string(&inventory).expect("native operator inventory must serialize");
    println!("cargo:rustc-env={COMPILED_NATIVE_OPERATOR_SET_JSON_ENV}={inventory_json}");
    emit_cuda_build_summary(
        "native_operator_artifact_set",
        "linked",
        "manifest-v3-artifact-set-v5-validated",
        start.elapsed(),
        &format!(
            "lock={}:catalog={}:operators={}:build_units={}",
            resolved_set.lock_path.display(),
            resolved_set.g03_catalog_sha256,
            resolved_set.artifacts.len(),
            build_coverage
                .iter()
                .map(CudaNativeBuildUnit::as_str)
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
    Some(build_coverage)
}

fn link_fa2_native_operator_artifact() {
    let start = Instant::now();
    let feature_enabled = env::var_os(NATIVE_OP_ARTIFACT_FEATURE_ENV).is_some();
    println!("cargo:rerun-if-env-changed={NATIVE_OP_ARTIFACT_FEATURE_ENV}");

    let manifest = optional_non_empty_env(FA2_NATIVE_MANIFEST_ENV);
    let artifact = optional_non_empty_env(FA2_NATIVE_ARTIFACT_ENV);
    let source_sha256 = optional_non_empty_env(FA2_NATIVE_SOURCE_SHA256_ENV);
    let inputs_sha256 = optional_non_empty_env(FA2_NATIVE_INPUTS_SHA256_ENV);
    let configured = manifest.is_some() || artifact.is_some();
    let pinned_without_artifact =
        (source_sha256.is_some() || inputs_sha256.is_some()) && !configured;

    if !feature_enabled {
        if configured || pinned_without_artifact {
            panic!(
                "FA2 native operator artifact build config requires --features native-op-artifact"
            );
        }
        println!("cargo:rustc-env={FA2_NATIVE_ARTIFACT_COMPILE_ENV}=not_configured");
        return;
    }
    if !configured {
        if pinned_without_artifact {
            panic!(
                "FA2 native operator sha256 pins require {FA2_NATIVE_MANIFEST_ENV} and {FA2_NATIVE_ARTIFACT_ENV}"
            );
        }
        println!("cargo:rustc-env={FA2_NATIVE_ARTIFACT_COMPILE_ENV}=not_configured");
        emit_cuda_build_summary(
            "fa2_native_operator",
            "skipped",
            "native-op-artifact-feature-enabled-without-manifest",
            start.elapsed(),
            "fa2-native-operator=not-configured",
        );
        return;
    }

    let manifest = PathBuf::from(required_native_env(FA2_NATIVE_MANIFEST_ENV, manifest));
    let artifact = PathBuf::from(required_native_env(FA2_NATIVE_ARTIFACT_ENV, artifact));
    println!("cargo:rerun-if-changed={}", manifest.display());
    println!("cargo:rerun-if-changed={}", artifact.display());
    let legacy_manifest = load_manifest(&manifest).unwrap_or_else(|error| {
        panic!(
            "failed to load legacy FA2 native operator manifest {}: {error}",
            manifest.display()
        )
    });
    if legacy_manifest.schema_version != LEGACY_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION {
        panic!(
            "provider-bound native operators must use {NATIVE_OPERATOR_SET_LOCK_ENV}; \
             legacy {FA2_NATIVE_MANIFEST_ENV} only accepts schema v1"
        );
    }

    let compute_capability = normalize_compute_capability(
        &env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| detect_cuda_compute_cap()),
    );
    let request = NativeOperatorResolveRequest::new(
        "fa2",
        NativeOperatorBackend::Cuda,
        manifest.clone(),
        artifact.clone(),
    )
    .with_operator_abi_version(legacy_manifest.operator_abi_version.clone())
    .with_ferrum_native_abi_version(legacy_manifest.ferrum_native_abi_version.clone())
    .with_compute_capability(compute_capability.clone());
    let resolved = NativeOperatorResolver
        .resolve(&request)
        .unwrap_or_else(|err| {
            panic!(
                "failed to resolve FA2 native operator artifact manifest={} artifact={}: {err}",
                manifest.display(),
                artifact.display()
            )
        });
    let mut requirement = NativeOperatorRequirement::cuda("fa2", compute_capability);
    requirement.operator_abi_version = resolved.manifest.operator_abi_version.clone();
    requirement.ferrum_native_abi_version = resolved.manifest.ferrum_native_abi_version.clone();
    requirement.source_package_sha256 = source_sha256;
    requirement.inputs_sha256 = inputs_sha256;
    requirement.binary_sha256 = Some(resolved.artifact_sha256.clone());
    resolve_native_operator_manifest(Some(&resolved.manifest), &requirement).unwrap_or_else(|err| {
        panic!(
            "FA2 native operator artifact does not satisfy build requirement manifest={} artifact={}: {err}",
            manifest.display(),
            artifact.display()
        )
    });

    let parent = resolved
        .artifact_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    println!("cargo:rustc-link-search=native={}", parent.display());
    match resolved.manifest.linkage {
        NativeOperatorLinkage::Static => {
            println!(
                "cargo:rustc-link-lib=static={}",
                native_static_link_name(&resolved.artifact_path)
            );
        }
        NativeOperatorLinkage::Dynamic => {
            println!(
                "cargo:rustc-link-lib=dylib={}",
                native_dynamic_link_name(&resolved.artifact_path)
            );
        }
    }
    println!("cargo:rustc-env={FA2_NATIVE_ARTIFACT_COMPILE_ENV}=linked");
    println!(
        "cargo:rustc-env={COMPILED_FA2_NATIVE_MANIFEST_ENV}={}",
        resolved.manifest_path.display()
    );
    println!(
        "cargo:rustc-env={COMPILED_FA2_NATIVE_ARTIFACT_ENV}={}",
        resolved.artifact_path.display()
    );
    println!(
        "cargo:rustc-env={COMPILED_FA2_NATIVE_SOURCE_SHA256_ENV}={}",
        resolved.manifest.source_package.sha256
    );
    println!(
        "cargo:rustc-env={COMPILED_FA2_NATIVE_INPUTS_SHA256_ENV}={}",
        resolved.manifest.inputs_sha256
    );
    println!(
        "cargo:rustc-env={COMPILED_FA2_NATIVE_BINARY_SHA256_ENV}={}",
        resolved.artifact_sha256
    );
    emit_cuda_build_summary(
        "fa2_native_operator",
        "linked",
        "manifest-validated-artifact-linked",
        start.elapsed(),
        &format!(
            "manifest={}:artifact={}:binary_sha256={}",
            resolved.manifest_path.display(),
            resolved.artifact_path.display(),
            resolved.artifact_sha256
        ),
    );
}

fn metadata_hash_file_fingerprint(path: &str) -> String {
    let meta = fs::metadata(path).unwrap_or_else(|e| panic!("metadata {path}: {e}"));
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in &bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| format!("{}.{:09}", d.as_secs(), d.subsec_nanos()))
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "{path}:len={}:mtime={mtime}:fnv1a64={hash:016x}",
        meta.len()
    )
}

fn metadata_file_fingerprint(path: &str) -> String {
    let meta = fs::metadata(path).unwrap_or_else(|e| panic!("metadata {path}: {e}"));
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| format!("{}.{:09}", d.as_secs(), d.subsec_nanos()))
        .unwrap_or_else(|| "unknown".to_string());
    format!("{path}:len={}:mtime={mtime}", meta.len())
}

fn static_lib_signature(label: &str, deps: &[&str], flags: &[String]) -> String {
    let mut lines = Vec::with_capacity(2 + deps.len() + flags.len());
    lines.push(format!("label={label}"));
    lines.extend(flags.iter().map(|f| format!("flag={f}")));
    lines.extend(deps.iter().map(|p| file_fingerprint(p)));
    lines.join("\n")
}

fn content_static_lib_signature(label: &str, deps: &[&str], flags: &[String]) -> String {
    let mut lines = Vec::with_capacity(2 + deps.len() + flags.len());
    lines.push(format!("label={label}"));
    lines.extend(flags.iter().map(|flag| format!("flag={flag}")));
    lines.extend(
        deps.iter()
            .map(|path| sha256_file_fingerprint(Path::new(path))),
    );
    lines.join("\n")
}

fn historical_nvcc_scheduler_signatures(
    label: &str,
    deps: &[&str],
    canonical_flags: &[String],
) -> Vec<String> {
    HISTORICAL_NVCC_THREAD_VALUES
        .map(|threads| {
            let mut historical_flags = canonical_flags.to_vec();
            historical_flags.insert(2, format!("threads={threads}"));
            cuda_native_input_signature(&content_static_lib_signature(
                label,
                deps,
                &historical_flags,
            ))
        })
        .collect()
}

fn metadata_hash_static_lib_signature(label: &str, deps: &[&str], flags: &[String]) -> String {
    let mut lines = Vec::with_capacity(2 + deps.len() + flags.len());
    lines.push(format!("label={label}"));
    lines.extend(flags.iter().map(|f| format!("flag={f}")));
    lines.extend(deps.iter().map(|p| metadata_hash_file_fingerprint(p)));
    lines.join("\n")
}

fn metadata_static_lib_signature(label: &str, deps: &[&str], flags: &[String]) -> String {
    let mut lines = Vec::with_capacity(2 + deps.len() + flags.len());
    lines.push(format!("label={label}"));
    lines.extend(flags.iter().map(|f| format!("flag={f}")));
    lines.extend(deps.iter().map(|p| metadata_file_fingerprint(p)));
    lines.join("\n")
}

enum CacheState {
    Fresh(&'static str),
    Stale(&'static str),
}

struct CudaNativeBuildCache {
    cache: NativeBuildArtifactCache,
    import_dirs: Vec<PathBuf>,
}

fn configured_cuda_native_build_cache() -> Option<CudaNativeBuildCache> {
    println!("cargo:rerun-if-env-changed={CUDA_NATIVE_BUILD_CACHE_ENV}");
    println!("cargo:rerun-if-env-changed={CUDA_NATIVE_IMPORT_DIRS_ENV}");
    let cache_root = env::var_os(CUDA_NATIVE_BUILD_CACHE_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from);
    let import_dirs = env::var_os(CUDA_NATIVE_IMPORT_DIRS_ENV)
        .filter(|value| !value.is_empty())
        .map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();
    let Some(cache_root) = cache_root else {
        if !import_dirs.is_empty() {
            panic!("{CUDA_NATIVE_IMPORT_DIRS_ENV} requires {CUDA_NATIVE_BUILD_CACHE_ENV}");
        }
        return None;
    };
    if !cache_root.is_absolute() {
        panic!(
            "{CUDA_NATIVE_BUILD_CACHE_ENV} must be absolute, got {}",
            cache_root.display()
        );
    }
    for import_dir in &import_dirs {
        if !import_dir.is_absolute() || !import_dir.is_dir() {
            panic!(
                "{CUDA_NATIVE_IMPORT_DIRS_ENV} entries must be absolute directories, got {}",
                import_dir.display()
            );
        }
    }
    let cache = NativeBuildArtifactCache::new(&cache_root).unwrap_or_else(|error| {
        panic!(
            "failed to configure CUDA native build cache {}: {error}",
            cache_root.display()
        )
    });
    eprintln!(
        "[cuda-native-build-cache] root={} import_dirs={} toolchain_hash={}",
        cache.root().display(),
        import_dirs.len(),
        signature_hash(cuda_native_toolchain_identity())
    );
    Some(CudaNativeBuildCache { cache, import_dirs })
}

fn artifact_stamp_matches(
    stamp: &Path,
    signature: &str,
    cache_promotion_signatures: &[&str],
    import_migration_signatures: &[&str],
    obsolete_numeric_line_prefix: Option<&str>,
) -> bool {
    fs::read_to_string(stamp).is_ok_and(|existing| {
        std::iter::once(signature)
            .chain(
                cache_promotion_signatures
                    .iter()
                    .chain(import_migration_signatures)
                    .copied(),
            )
            .any(|candidate| {
                existing == candidate
                    || obsolete_numeric_line_prefix.is_some_and(|prefix| {
                        legacy_signature_matches_without_numeric_line(&existing, candidate, prefix)
                    })
            })
    })
}

fn publish_cuda_build_artifact(
    config: Option<&CudaNativeBuildCache>,
    artifact_id: &str,
    file_name: &str,
    signature: &str,
    source: &Path,
) {
    let Some(config) = config else {
        return;
    };
    let spec = NativeBuildArtifactSpec::new(artifact_id, file_name, signature)
        .unwrap_or_else(|error| panic!("invalid CUDA build artifact identity: {error}"));
    let receipt = config.cache.publish(&spec, source).unwrap_or_else(|error| {
        panic!(
            "failed to publish CUDA build artifact {artifact_id} from {}: {error}",
            source.display()
        )
    });
    eprintln!(
        "[cuda-native-build-cache] artifact={artifact_id} status=published \
sha256={} bytes={} entry={}",
        receipt.artifact_sha256,
        receipt.artifact_size_bytes,
        receipt.cache_entry.display()
    );
}

fn restore_cuda_build_artifact(
    config: Option<&CudaNativeBuildCache>,
    out_dir: &Path,
    artifact_id: &str,
    file_name: &str,
    stamp_file_name: &str,
    signature: &str,
    cache_promotion_signatures: &[&str],
    import_migration_signatures: &[&str],
) -> Option<String> {
    restore_cuda_build_artifact_with_import_migration(
        config,
        out_dir,
        artifact_id,
        file_name,
        stamp_file_name,
        signature,
        cache_promotion_signatures,
        import_migration_signatures,
        None,
    )
}

fn restore_cuda_build_artifact_with_import_migration(
    config: Option<&CudaNativeBuildCache>,
    out_dir: &Path,
    artifact_id: &str,
    file_name: &str,
    stamp_file_name: &str,
    signature: &str,
    cache_promotion_signatures: &[&str],
    import_migration_signatures: &[&str],
    obsolete_numeric_line_prefix: Option<&str>,
) -> Option<String> {
    let config = config?;
    let spec = NativeBuildArtifactSpec::new(artifact_id, file_name, signature)
        .unwrap_or_else(|error| panic!("invalid CUDA build artifact identity: {error}"));
    let destination = out_dir.join(file_name);
    match config
        .cache
        .restore(&spec, &destination)
        .unwrap_or_else(|error| {
            panic!("failed to restore CUDA build artifact {artifact_id}: {error}")
        }) {
        NativeBuildArtifactLookup::Hit(receipt) => {
            fs::write(out_dir.join(stamp_file_name), signature).unwrap_or_else(|error| {
                panic!("failed to write restored CUDA build artifact stamp: {error}")
            });
            eprintln!(
                "[cuda-native-build-cache] artifact={artifact_id} status=cache_hit \
sha256={} entry={}",
                receipt.artifact_sha256,
                receipt.cache_entry.display()
            );
            return Some("shared-native-build-cache".to_string());
        }
        NativeBuildArtifactLookup::Miss { .. } => {}
    }

    for migration_signature in cache_promotion_signatures {
        let migration_spec =
            NativeBuildArtifactSpec::new(artifact_id, file_name, *migration_signature)
                .unwrap_or_else(|error| {
                    panic!("invalid CUDA build artifact migration identity: {error}")
                });
        let migrated = match config
            .cache
            .restore(&migration_spec, &destination)
            .unwrap_or_else(|error| {
                panic!("failed to restore CUDA build artifact migration {artifact_id}: {error}")
            }) {
            NativeBuildArtifactLookup::Hit(receipt) => receipt,
            NativeBuildArtifactLookup::Miss { .. } => continue,
        };
        let published = config
            .cache
            .publish(&spec, &destination)
            .unwrap_or_else(|error| {
                panic!("failed to promote CUDA build artifact migration {artifact_id}: {error}")
            });
        assert_eq!(
            published.artifact_sha256, migrated.artifact_sha256,
            "promoted CUDA build artifact hash drifted"
        );
        assert_eq!(
            published.artifact_size_bytes, migrated.artifact_size_bytes,
            "promoted CUDA build artifact size drifted"
        );
        fs::write(out_dir.join(stamp_file_name), signature).unwrap_or_else(|error| {
            panic!("failed to write promoted CUDA build artifact stamp: {error}")
        });
        eprintln!(
            "[cuda-native-build-cache] artifact={artifact_id} status=promoted \
source_signature_sha256={} target_signature_sha256={} sha256={}",
            migration_spec.input_signature_sha256(),
            spec.input_signature_sha256(),
            published.artifact_sha256,
        );
        return Some("promoted-compatible-native-build-cache".to_string());
    }

    for import_dir in &config.import_dirs {
        let import_artifact = import_dir.join(file_name);
        let import_stamp = import_dir.join(stamp_file_name);
        if !import_artifact.is_file()
            || !artifact_stamp_matches(
                &import_stamp,
                signature,
                cache_promotion_signatures,
                import_migration_signatures,
                obsolete_numeric_line_prefix,
            )
        {
            continue;
        }
        let published = config
            .cache
            .publish(&spec, &import_artifact)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to import CUDA build artifact {artifact_id} from {}: {error}",
                    import_artifact.display()
                )
            });
        match config
            .cache
            .restore(&spec, &destination)
            .unwrap_or_else(|error| {
                panic!("failed to restore imported CUDA build artifact {artifact_id}: {error}")
            }) {
            NativeBuildArtifactLookup::Hit(_) => {}
            NativeBuildArtifactLookup::Miss { reason } => {
                panic!("imported CUDA build artifact {artifact_id} was not restorable: {reason}")
            }
        }
        fs::write(out_dir.join(stamp_file_name), signature).unwrap_or_else(|error| {
            panic!("failed to write imported CUDA build artifact stamp: {error}")
        });
        eprintln!(
            "[cuda-native-build-cache] artifact={artifact_id} status=imported \
sha256={} import_dir={}",
            published.artifact_sha256,
            import_dir.display()
        );
        return Some(format!("imported-native-output:{}", import_dir.display()));
    }
    None
}

fn static_lib_cache_state(out_dir: &Path, lib_name: &str, signature: &str) -> CacheState {
    let lib_file = out_dir.join(format!("lib{lib_name}.a"));
    let stamp_file = out_dir.join(format!("lib{lib_name}.stamp"));
    if !lib_file.is_file() {
        return CacheState::Stale("missing-lib");
    }
    if !stamp_file.is_file() {
        return CacheState::Stale("missing-stamp");
    }
    match fs::read_to_string(&stamp_file) {
        Ok(existing) if existing == signature => {
            eprintln!("[{lib_name}] cache hit: {}", lib_file.display());
            CacheState::Fresh("signature-match")
        }
        Ok(_) => CacheState::Stale("signature-changed"),
        Err(_) => CacheState::Stale("stamp-read-error"),
    }
}

fn write_static_lib_stamp(out_dir: &Path, lib_name: &str, signature: &str) {
    let stamp_file = out_dir.join(format!("lib{lib_name}.stamp"));
    fs::write(&stamp_file, signature)
        .unwrap_or_else(|e| panic!("[{lib_name}] failed to write {}: {e}", stamp_file.display()));
}

fn emit_cuda_static_link(
    out_dir: &Path,
    lib_name: &str,
    cuda_root: Option<&PathBuf>,
    link_stdcxx: bool,
) {
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");
    if let Some(cuda_root) = cuda_root {
        let lib64 = cuda_root.join("lib64");
        if lib64.exists() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    if link_stdcxx {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let native_artifact_coverage = link_native_operator_artifact_set();
    if native_artifact_coverage.is_none() {
        link_fa2_native_operator_artifact();
    }

    // Link Accelerate framework on macOS (provides cblas_sgemm, vDSP_*)
    if env::consts::OS == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set by cargo"));
    let _native_source_policy = configured_cuda_native_source_policy();
    let native_build_cache = configured_cuda_native_build_cache();
    let out_dir_clone = out_dir.clone();
    compile_core_ptx(&out_dir_clone, native_build_cache.as_ref());

    // Compile Marlin INT4xFP16 kernel separately (uses runtime API, not PTX).
    // Only when "marlin" feature is enabled. Requires SM >= 8.0 (Ampere).
    if env::var_os("CARGO_FEATURE_MARLIN").is_some() {
        if native_artifact_coverage
            .as_ref()
            .is_some_and(|coverage| coverage.contains(CudaNativeBuildUnit::Marlin))
        {
            emit_cuda_build_summary(
                "marlin",
                "artifact",
                "native-operator-artifact-set",
                Duration::ZERO,
                CudaNativeBuildUnit::Marlin.artifact_operator(),
            );
        } else {
            compile_marlin(&out_dir_clone, native_build_cache.as_ref());
        }
    }

    // vLLM gptq_marlin port (Phase 12). Heavier C++ template instantiations
    // than the IST-DASLab port — compile time ~30 min on first build. Opt-in
    // via `--features vllm-marlin`.
    if env::var_os("CARGO_FEATURE_VLLM_MARLIN").is_some() {
        if native_artifact_coverage
            .as_ref()
            .is_some_and(|coverage| coverage.contains(CudaNativeBuildUnit::VllmMarlin))
        {
            emit_cuda_build_summary(
                "vllm_marlin",
                "artifact",
                "native-operator-artifact-set",
                Duration::ZERO,
                CudaNativeBuildUnit::VllmMarlin.artifact_operator(),
            );
        } else {
            compile_vllm_marlin(&out_dir_clone, native_build_cache.as_ref());
        }
    }

    // vLLM moe_marlin_wna16 port (Stage 14). Vendored from
    // vllm/csrc/moe/marlin_moe_wna16/ at v0.10.2. Single .cu file with
    // many template instantiations via COMMON_GET_IF macros — compile
    // time ~15-20 min on first build. Opt-in via `--features vllm-moe-marlin`.
    if env::var_os("CARGO_FEATURE_VLLM_MOE_MARLIN").is_some() {
        if native_artifact_coverage
            .as_ref()
            .is_some_and(|coverage| coverage.contains(CudaNativeBuildUnit::VllmMoeMarlin))
        {
            emit_cuda_build_summary(
                "vllm_moe_marlin",
                "artifact",
                "native-operator-artifact-set",
                Duration::ZERO,
                CudaNativeBuildUnit::VllmMoeMarlin.artifact_operator(),
            );
        } else {
            compile_vllm_moe_marlin(&out_dir_clone, native_build_cache.as_ref());
        }
    }

    // vLLM paged_attention_v2 port (2026-05-12). Vendored from vllm v0.20.2
    // (csrc/attention/{paged_attention_v2.cu,attention_kernels.cuh,...})
    // with torch headers stripped. Opt-in via `--features vllm-paged-attn-v2`.
    // Builds a static lib of the single (HEAD=128, BLOCK=16, FP16, no-FP8,
    // no-blocksparse) instantiation — ~1-2 min compile.
    if env::var_os("CARGO_FEATURE_VLLM_PAGED_ATTN_V2").is_some() {
        if native_artifact_coverage
            .as_ref()
            .is_some_and(|coverage| coverage.contains(CudaNativeBuildUnit::VllmPagedAttentionV2))
        {
            emit_cuda_build_summary(
                "vllm_paged_attn",
                "artifact",
                "native-operator-artifact-set",
                Duration::ZERO,
                CudaNativeBuildUnit::VllmPagedAttentionV2.artifact_operator(),
            );
        } else {
            compile_vllm_paged_attn(&out_dir_clone, native_build_cache.as_ref());
        }
    }

    // Legacy compatibility switch. The source-linked FA2 path has moved out of
    // the main repository; enabling this feature must not compile vendored
    // FlashAttention/CUTLASS input from crates/.
    if env::var_os("CARGO_FEATURE_FA2_SOURCE").is_some() {
        report_fa2_source_obsolete();
    }
}

fn detect_cuda_compute_cap() -> String {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if let Ok(value) = env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={value}");
        return value;
    }

    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
        .expect("nvidia-smi failed while detecting CUDA compute capability");
    if !output.status.success() {
        panic!("nvidia-smi failed while detecting CUDA compute capability");
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let cap = stdout
        .lines()
        .next()
        .expect("missing nvidia-smi compute_cap output")
        .trim()
        .replace('.', "");
    if cap.is_empty() {
        panic!("empty CUDA compute capability from nvidia-smi");
    }
    println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
    cap
}

fn core_ptx_signature(kernel: &str, flags: &[String]) -> String {
    let mut lines = Vec::with_capacity(2 + flags.len() + CORE_PTX_HEADERS.len());
    lines.push(format!("label=core-ptx"));
    lines.push(format!("kernel={kernel}"));
    lines.extend(flags.iter().map(|f| format!("flag={f}")));
    lines.push(file_fingerprint(kernel));
    lines.extend(CORE_PTX_HEADERS.iter().map(|p| file_fingerprint(p)));
    lines.join("\n")
}

fn content_core_ptx_signature(kernel: &str, flags: &[String]) -> String {
    let mut lines = Vec::with_capacity(2 + flags.len() + CORE_PTX_HEADERS.len());
    lines.push("label=core-ptx".to_string());
    lines.push(format!("kernel={kernel}"));
    lines.extend(flags.iter().map(|flag| format!("flag={flag}")));
    lines.push(sha256_file_fingerprint(Path::new(kernel)));
    lines.extend(
        CORE_PTX_HEADERS
            .iter()
            .map(|path| sha256_file_fingerprint(Path::new(path))),
    );
    lines.join("\n")
}

fn core_ptx_cache_state(out_dir: &Path, kernel: &str, signature: &str) -> CacheState {
    let stem = Path::new(kernel)
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("kernel filename");
    let ptx = out_dir.join(format!("{stem}.ptx"));
    let stamp = out_dir.join(format!("{stem}.ptx.stamp"));
    if !ptx.is_file() {
        return CacheState::Stale("missing-ptx");
    }
    if !stamp.is_file() {
        return CacheState::Stale("missing-stamp");
    }
    match fs::read_to_string(&stamp) {
        Ok(existing) if existing == signature => CacheState::Fresh("signature-match"),
        Ok(_) => CacheState::Stale("signature-changed"),
        Err(_) => CacheState::Stale("stamp-read-error"),
    }
}

fn write_core_ptx_stamp(out_dir: &Path, kernel: &str, signature: &str) {
    let stem = Path::new(kernel)
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("kernel filename");
    let stamp = out_dir.join(format!("{stem}.ptx.stamp"));
    fs::write(&stamp, signature)
        .unwrap_or_else(|e| panic!("[core-ptx] failed to write stamp {}: {e}", stamp.display()));
}

fn write_core_ptx_bindings(out_dir: &Path) {
    let mut content = String::new();
    for kernel in CORE_PTX_KERNELS {
        let stem = Path::new(kernel)
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("kernel filename");
        content.push_str(&format!(
            "pub const {}: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/{}.ptx\"));\n",
            stem.to_uppercase().replace('.', "_"),
            stem
        ));
    }
    let ptx_rs = out_dir.join("ptx.rs");
    if fs::read_to_string(&ptx_rs).ok().as_deref() != Some(content.as_str()) {
        fs::write(&ptx_rs, content)
            .unwrap_or_else(|e| panic!("[core-ptx] failed to write {}: {e}", ptx_rs.display()));
    }
}

fn compile_core_ptx(out_dir: &Path, native_build_cache: Option<&CudaNativeBuildCache>) {
    for path in CORE_PTX_KERNELS.iter().chain(CORE_PTX_HEADERS.iter()) {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");

    let cuda_root = cuda_root_from_env();
    let cuda_include = cuda_root.as_ref().map(|root| root.join("include"));
    if let Some(cuda_include) = &cuda_include {
        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            cuda_include.display()
        );
    }
    let nvcc = cuda_root
        .as_ref()
        .map(|r| r.join("bin").join("nvcc"))
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    let compute_cap = detect_cuda_compute_cap();
    let ccbin = env::var("NVCC_CCBIN").ok();
    let mut flags = vec![
        format!("nvcc={}", nvcc.display()),
        format!("arch=sm_{compute_cap}"),
        "-Ikernels".to_string(),
        "--expt-relaxed-constexpr".to_string(),
        "-std=c++17".to_string(),
        "-O3".to_string(),
        "--use_fast_math".to_string(),
    ];
    if let Some(cuda_include) = &cuda_include {
        flags.push(format!("-I{}", cuda_include.display()));
    }
    if let Some(ccbin) = &ccbin {
        flags.push(format!("ccbin={ccbin}"));
    }

    for kernel in CORE_PTX_KERNELS {
        let start = Instant::now();
        let legacy_signature = core_ptx_signature(kernel, &flags);
        let content_signature = content_core_ptx_signature(kernel, &flags);
        let signature = cuda_native_input_signature(&content_signature);
        let stem = Path::new(kernel)
            .file_stem()
            .and_then(|value| value.to_str())
            .expect("kernel filename");
        let file_name = format!("{stem}.ptx");
        let stamp_file_name = format!("{stem}.ptx.stamp");
        let artifact_id = format!("core_ptx.{stem}");
        match core_ptx_cache_state(out_dir, kernel, &signature) {
            CacheState::Fresh(reason) => {
                publish_cuda_build_artifact(
                    native_build_cache,
                    &artifact_id,
                    &file_name,
                    &signature,
                    &out_dir.join(&file_name),
                );
                emit_cuda_build_summary(
                    &format!("core-ptx:{}", Path::new(kernel).display()),
                    "cache_hit",
                    reason,
                    start.elapsed(),
                    &signature,
                );
            }
            CacheState::Stale(reason) => {
                if let Some(cache_reason) = restore_cuda_build_artifact(
                    native_build_cache,
                    out_dir,
                    &artifact_id,
                    &file_name,
                    &stamp_file_name,
                    &signature,
                    &[],
                    &[&legacy_signature],
                ) {
                    emit_cuda_build_summary(
                        &format!("core-ptx:{}", Path::new(kernel).display()),
                        "cache_hit",
                        &cache_reason,
                        start.elapsed(),
                        &signature,
                    );
                    continue;
                }
                enforce_cuda_native_source_policy(
                    &format!("core-ptx:{}", Path::new(kernel).display()),
                    reason,
                    &signature,
                );
                let mut command = std::process::Command::new(&nvcc);
                command
                    .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                    .arg("--ptx")
                    .args(["--default-stream", "per-thread"])
                    .args([
                        "--output-directory",
                        out_dir.to_str().expect("OUT_DIR utf8"),
                    ])
                    .arg("-Ikernels")
                    .arg("--expt-relaxed-constexpr")
                    .arg("-std=c++17")
                    .arg("-O3")
                    .arg("--use_fast_math");
                if let Some(cuda_include) = &cuda_include {
                    command.arg(format!("-I{}", cuda_include.display()));
                }
                if let Some(ccbin) = &ccbin {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin]);
                }
                command.arg(kernel);
                let output = command
                    .output()
                    .unwrap_or_else(|e| panic!("[core-ptx] nvcc spawn failed for {kernel}: {e}"));
                if !output.status.success() {
                    panic!(
                        "[core-ptx] nvcc failed compiling {kernel}: {:?}\n\n# stdout\n{}\n\n# stderr\n{}",
                        command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                write_core_ptx_stamp(out_dir, kernel, &signature);
                publish_cuda_build_artifact(
                    native_build_cache,
                    &artifact_id,
                    &file_name,
                    &signature,
                    &out_dir.join(&file_name),
                );
                emit_cuda_build_summary(
                    &format!("core-ptx:{}", Path::new(kernel).display()),
                    "built",
                    reason,
                    start.elapsed(),
                    &signature,
                );
            }
        }
    }
    write_core_ptx_bindings(out_dir);
}

fn report_fa2_source_obsolete() {
    println!("cargo:warning=feature fa2-source is obsolete; use a Ferrum native operator artifact for FA2");
    println!("cargo:rustc-env=FERRUM_FA2_SOURCE_COMPILE=obsolete");
    emit_cuda_build_summary(
        "fa2_source",
        "skipped",
        "obsolete-native-operator-artifact-required",
        Duration::from_millis(0),
        "fa2-source=obsolete",
    );
}

fn compile_vllm_paged_attn(out_dir: &PathBuf, native_build_cache: Option<&CudaNativeBuildCache>) {
    let cu_files: &[&str] = &["kernels/vllm_attn/launcher.cu"];
    let header_files: &[&str] = &[
        "kernels/vllm_attn/attention_kernels.cuh",
        "kernels/vllm_attn/attention_dtypes.h",
        "kernels/vllm_attn/attention_utils.cuh",
        "kernels/vllm_attn/attention_generic.cuh",
        "kernels/vllm_attn/dtype_float16.cuh",
        "kernels/vllm_attn/dtype_float32.cuh",
        "kernels/vllm_attn/dtype_bfloat16.cuh",
        "kernels/vllm_attn/dtype_fp8.cuh",
        "kernels/vllm_attn/ferrum_shim.h",
        "kernels/vllm_attn/quant_utils_stub.cuh",
        "kernels/vllm_attn/include/cuda_compat.h",
    ];
    for f in cu_files.iter().chain(header_files.iter()) {
        println!("cargo:rerun-if-changed={f}");
    }

    let cuda_root = cuda_root_from_env();
    let nvcc = cuda_root
        .as_ref()
        .map(|r| r.join("bin").join("nvcc"))
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    if !nvcc.exists() && cuda_root.is_some() {
        eprintln!("nvcc not found at {nvcc:?}, skipping vllm-paged-attn-v2");
        return;
    }

    let compute_cap = env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".to_string());
    let nvcc_threads = configured_nvcc_threads();
    let flags = vec![
        format!("nvcc={}", nvcc.display()),
        format!("arch=sm_{compute_cap}"),
        "-Ikernels/vllm_attn".to_string(),
        "-std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda"
            .to_string(),
        "-Xcompiler -fPIC".to_string(),
    ];
    let deps: Vec<&str> = cu_files
        .iter()
        .chain(header_files.iter())
        .copied()
        .collect();
    let pre_quant_header_deps: Vec<&str> = deps
        .iter()
        .copied()
        .filter(|path| *path != "kernels/vllm_attn/quant_utils_stub.cuh")
        .collect();
    let pre_quant_header_signature =
        static_lib_signature("vllm-paged-attn-v2", &pre_quant_header_deps, &flags);
    let pre_quant_header_metadata_hash_signature =
        metadata_hash_static_lib_signature("vllm-paged-attn-v2", &pre_quant_header_deps, &flags);
    let pre_quant_header_metadata_signature =
        metadata_static_lib_signature("vllm-paged-attn-v2", &pre_quant_header_deps, &flags);
    let legacy_signature = static_lib_signature("vllm-paged-attn-v2", &deps, &flags);
    let content_signature = content_static_lib_signature("vllm-paged-attn-v2", &deps, &flags);
    let signature = cuda_native_input_signature(&content_signature);
    let scheduler_migration_signatures =
        historical_nvcc_scheduler_signatures("vllm-paged-attn-v2", &deps, &flags);
    let metadata_hash_signature =
        metadata_hash_static_lib_signature("vllm-paged-attn-v2", &deps, &flags);
    let metadata_signature = metadata_static_lib_signature("vllm-paged-attn-v2", &deps, &flags);
    let build_start = Instant::now();
    let cache_state = static_lib_cache_state(out_dir, "vllm_paged_attn", &signature);
    let build_reason = match cache_state {
        CacheState::Fresh(reason) => {
            publish_cuda_build_artifact(
                native_build_cache,
                "static.vllm_paged_attn",
                "libvllm_paged_attn.a",
                &signature,
                &out_dir.join("libvllm_paged_attn.a"),
            );
            emit_cuda_build_summary(
                "vllm_paged_attn",
                "cache_hit",
                reason,
                build_start.elapsed(),
                &signature,
            );
            emit_cuda_static_link(out_dir, "vllm_paged_attn", cuda_root.as_ref(), true);
            return;
        }
        CacheState::Stale(reason) => {
            let cache_promotion_signatures = scheduler_migration_signatures
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            if let Some(cache_reason) = restore_cuda_build_artifact(
                native_build_cache,
                out_dir,
                "static.vllm_paged_attn",
                "libvllm_paged_attn.a",
                "libvllm_paged_attn.stamp",
                &signature,
                &cache_promotion_signatures,
                &[
                    &legacy_signature,
                    &metadata_hash_signature,
                    &metadata_signature,
                    &pre_quant_header_signature,
                    &pre_quant_header_metadata_hash_signature,
                    &pre_quant_header_metadata_signature,
                ],
            ) {
                emit_cuda_build_summary(
                    "vllm_paged_attn",
                    "cache_hit",
                    &cache_reason,
                    build_start.elapsed(),
                    &signature,
                );
                emit_cuda_static_link(out_dir, "vllm_paged_attn", cuda_root.as_ref(), true);
                return;
            }
            reason
        }
    };
    enforce_cuda_native_source_policy("vllm_paged_attn", build_reason, &signature);

    let mut object_files: Vec<PathBuf> = Vec::new();
    for src in cu_files {
        let stem = std::path::Path::new(src)
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("cu filename");
        let obj = out_dir.join(format!("vllm_paged_attn_{stem}.o"));
        eprintln!("[vllm-paged-attn-v2] compiling {src} -> {}", obj.display());

        let status = std::process::Command::new(&nvcc)
            .args(["-c", src, "-o"])
            .arg(obj.to_str().unwrap())
            .args([
                &format!("-arch=sm_{compute_cap}"),
                "-Ikernels/vllm_attn",
                "-std=c++17",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-Xcompiler",
                "-fPIC",
                "--threads",
                nvcc_threads.as_str(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("[vllm-paged-attn-v2] nvcc spawn failed for {src}: {e}"));
        if !status.success() {
            panic!(
                "[vllm-paged-attn-v2] nvcc failed compiling {src}. Disable \
                 the feature or fix CUDA setup."
            );
        }
        object_files.push(obj);
    }

    let lib_file = out_dir.join("libvllm_paged_attn.a");
    let mut ar_args: Vec<String> = vec!["rcs".to_string(), lib_file.display().to_string()];
    for o in &object_files {
        ar_args.push(o.display().to_string());
    }
    let ar_status = std::process::Command::new("ar")
        .args(&ar_args)
        .status()
        .unwrap_or_else(|e| panic!("[vllm-paged-attn-v2] ar spawn failed: {e}"));
    if !ar_status.success() {
        panic!("[vllm-paged-attn-v2] ar failed to bundle {lib_file:?}");
    }

    write_static_lib_stamp(out_dir, "vllm_paged_attn", &signature);
    publish_cuda_build_artifact(
        native_build_cache,
        "static.vllm_paged_attn",
        "libvllm_paged_attn.a",
        &signature,
        &lib_file,
    );
    emit_cuda_static_link(out_dir, "vllm_paged_attn", cuda_root.as_ref(), true);
    eprintln!(
        "[vllm-paged-attn-v2] static lib built: {}",
        lib_file.display()
    );
    emit_cuda_build_summary(
        "vllm_paged_attn",
        "built",
        build_reason,
        build_start.elapsed(),
        &signature,
    );
}

fn compile_vllm_moe_marlin(out_dir: &PathBuf, native_build_cache: Option<&CudaNativeBuildCache>) {
    // CUDA 13 hidden-default-visibility workaround: implicit Marlin<...>
    // instantiations inside ops.cu's dispatcher are emitted with hidden
    // ELF visibility and `ar`-bundling rejects them at the final rust-lld
    // link. kernel_instantiations.cu explicitly instantiates the same
    // configurations at namespace scope to force external linkage. See
    // the file header for the upstream vLLM reference.
    let cu_files: &[&str] = &[
        "kernels/vllm_marlin_moe/ops.cu",
        "kernels/vllm_marlin_moe/kernel_instantiations.cu",
    ];
    let header_files: &[&str] = &[
        "kernels/vllm_marlin_moe/kernel.h",
        "kernels/vllm_marlin_moe/marlin_template.h",
        "kernels/vllm_marlin_moe/vllm_torch_shim.h",
        "kernels/vllm_marlin_moe/core/scalar_type.hpp",
        "kernels/vllm_marlin_moe/quantization/gptq_marlin/marlin.cuh",
        "kernels/vllm_marlin_moe/quantization/gptq_marlin/marlin_dtypes.cuh",
        "kernels/vllm_marlin_moe/quantization/gptq_marlin/dequant.h",
    ];
    for f in cu_files.iter().chain(header_files.iter()) {
        println!("cargo:rerun-if-changed={f}");
    }

    let cuda_root = cuda_root_from_env();
    let nvcc = cuda_root
        .as_ref()
        .map(|r| r.join("bin").join("nvcc"))
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    if !nvcc.exists() && cuda_root.is_some() {
        eprintln!("nvcc not found at {nvcc:?}, skipping vllm-moe-marlin");
        return;
    }

    let compute_cap = env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".to_string());
    let nvcc_threads = configured_nvcc_threads();
    let flags = vec![
        format!("nvcc={}", nvcc.display()),
        format!("arch=sm_{compute_cap}"),
        "-Ikernels/vllm_marlin_moe".to_string(),
        "-DMARLIN_NAMESPACE_NAME=marlin_moe_wna16".to_string(),
        "-std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda"
            .to_string(),
        "-Xcompiler -fPIC -Xcompiler -fvisibility=default".to_string(),
    ];
    let deps: Vec<&str> = cu_files
        .iter()
        .chain(header_files.iter())
        .copied()
        .collect();
    let legacy_signature = static_lib_signature("vllm-moe-marlin", &deps, &flags);
    let content_signature = content_static_lib_signature("vllm-moe-marlin", &deps, &flags);
    let signature = cuda_native_input_signature(&content_signature);
    let scheduler_migration_signatures =
        historical_nvcc_scheduler_signatures("vllm-moe-marlin", &deps, &flags);
    let metadata_hash_signature =
        metadata_hash_static_lib_signature("vllm-moe-marlin", &deps, &flags);
    let metadata_signature = metadata_static_lib_signature("vllm-moe-marlin", &deps, &flags);
    let build_start = Instant::now();
    let cache_state = static_lib_cache_state(out_dir, "vllm_moe_marlin", &signature);
    let build_reason = match cache_state {
        CacheState::Fresh(reason) => {
            publish_cuda_build_artifact(
                native_build_cache,
                "static.vllm_moe_marlin",
                "libvllm_moe_marlin.a",
                &signature,
                &out_dir.join("libvllm_moe_marlin.a"),
            );
            emit_cuda_build_summary(
                "vllm_moe_marlin",
                "cache_hit",
                reason,
                build_start.elapsed(),
                &signature,
            );
            emit_cuda_static_link(out_dir, "vllm_moe_marlin", cuda_root.as_ref(), true);
            return;
        }
        CacheState::Stale(reason) => {
            let cache_promotion_signatures = scheduler_migration_signatures
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            if let Some(cache_reason) = restore_cuda_build_artifact(
                native_build_cache,
                out_dir,
                "static.vllm_moe_marlin",
                "libvllm_moe_marlin.a",
                "libvllm_moe_marlin.stamp",
                &signature,
                &cache_promotion_signatures,
                &[
                    &legacy_signature,
                    &metadata_hash_signature,
                    &metadata_signature,
                ],
            ) {
                emit_cuda_build_summary(
                    "vllm_moe_marlin",
                    "cache_hit",
                    &cache_reason,
                    build_start.elapsed(),
                    &signature,
                );
                emit_cuda_static_link(out_dir, "vllm_moe_marlin", cuda_root.as_ref(), true);
                return;
            }
            reason
        }
    };
    enforce_cuda_native_source_policy("vllm_moe_marlin", build_reason, &signature);

    let mut object_files: Vec<PathBuf> = Vec::new();
    for src in cu_files {
        let stem = std::path::Path::new(src)
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("cu filename");
        let obj = out_dir.join(format!("vllm_moe_{stem}.o"));
        eprintln!("[vllm-moe-marlin] compiling {src} -> {}", obj.display());

        let status = std::process::Command::new(&nvcc)
            .args(["-c", src, "-o"])
            .arg(obj.to_str().unwrap())
            .args([
                &format!("-arch=sm_{compute_cap}"),
                "-Ikernels/vllm_marlin_moe",
                "-DMARLIN_NAMESPACE_NAME=marlin_moe_wna16",
                "-std=c++17",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-Xcompiler",
                "-fPIC",
                // CUDA 13's nvcc defaults templated kernel instantiations
                // to hidden ELF visibility — the resulting static archive
                // doesn't expose Marlin<...> instances at link time, and
                // rust-lld then fails to resolve them. Explicit default
                // visibility is safe on 12.x too.
                "-Xcompiler",
                "-fvisibility=default",
                "--threads",
                nvcc_threads.as_str(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("[vllm-moe-marlin] nvcc spawn failed for {src}: {e}"));
        if !status.success() {
            panic!(
                "[vllm-moe-marlin] nvcc failed compiling {src}. \
                 Disable with `--features vllm-moe-marlin` removed, \
                 or fix CUDA setup."
            );
        }
        object_files.push(obj);
    }

    let lib_file = out_dir.join("libvllm_moe_marlin.a");
    let mut ar_args: Vec<String> = vec!["rcs".to_string(), lib_file.display().to_string()];
    for o in &object_files {
        ar_args.push(o.display().to_string());
    }
    let ar_status = std::process::Command::new("ar")
        .args(&ar_args)
        .status()
        .unwrap_or_else(|e| panic!("[vllm-moe-marlin] ar spawn failed: {e}"));
    if !ar_status.success() {
        panic!("[vllm-moe-marlin] ar failed to bundle {lib_file:?}");
    }

    write_static_lib_stamp(out_dir, "vllm_moe_marlin", &signature);
    publish_cuda_build_artifact(
        native_build_cache,
        "static.vllm_moe_marlin",
        "libvllm_moe_marlin.a",
        &signature,
        &lib_file,
    );
    emit_cuda_static_link(out_dir, "vllm_moe_marlin", cuda_root.as_ref(), true);
    eprintln!("[vllm-moe-marlin] static lib built: {}", lib_file.display());
    emit_cuda_build_summary(
        "vllm_moe_marlin",
        "built",
        build_reason,
        build_start.elapsed(),
        &signature,
    );
}

fn compile_vllm_marlin(out_dir: &PathBuf, native_build_cache: Option<&CudaNativeBuildCache>) {
    // The dispatcher references the full generated specialization set even
    // though the versioned FFI exposes only typed, validated combinations.
    // Compile the complete selector closure so adding a supported dtype mapping
    // never creates a hidden link-time dependency.
    let cu_files: &[&str] = &[
        "vllm_marlin/marlin.cu",
        "vllm_marlin/gptq_marlin_repack.cu",
        "vllm_marlin/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu",
        "vllm_marlin/sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu",
        "vllm_marlin/sm80_kernel_bfloat16_u4_bfloat16.cu",
        "vllm_marlin/sm80_kernel_bfloat16_u4b8_bfloat16.cu",
        "vllm_marlin/sm80_kernel_bfloat16_u8b128_bfloat16.cu",
        "vllm_marlin/sm80_kernel_float16_fe2m1f_float16.cu",
        "vllm_marlin/sm80_kernel_float16_fe4m3fn_float16.cu",
        "vllm_marlin/sm80_kernel_float16_u4_float16.cu",
        "vllm_marlin/sm80_kernel_float16_u4b8_float16.cu",
        "vllm_marlin/sm80_kernel_float16_u8b128_float16.cu",
        "vllm_marlin/sm80_kernel_s8_u4_bfloat16.cu",
        "vllm_marlin/sm80_kernel_s8_u4_float16.cu",
        "vllm_marlin/sm80_kernel_s8_u4b8_bfloat16.cu",
        "vllm_marlin/sm80_kernel_s8_u4b8_float16.cu",
        "vllm_marlin/sm89_kernel_fe4m3fn_fe2m1f_bfloat16.cu",
        "vllm_marlin/sm89_kernel_fe4m3fn_u4_bfloat16.cu",
        "vllm_marlin/sm89_kernel_fe4m3fn_u4_float16.cu",
        "vllm_marlin/sm89_kernel_fe4m3fn_u4b8_bfloat16.cu",
        "vllm_marlin/sm89_kernel_fe4m3fn_u4b8_float16.cu",
    ];
    let header_files: &[&str] = &[
        "vllm_marlin/marlin_template.h",
        "vllm_marlin/marlin_mma.h",
        "vllm_marlin/marlin_dtypes.cuh",
        "vllm_marlin/marlin.cuh",
        "vllm_marlin/dequant.h",
        "vllm_marlin/ferrum_marlin_ffi.h",
        "vllm_marlin/kernel.h",
        "vllm_marlin/kernel_selector.h",
        "vllm_marlin/scalar_type.hpp",
        "vllm_marlin/torch_stubs.h",
    ];
    for f in cu_files.iter().chain(header_files.iter()) {
        println!("cargo:rerun-if-changed={f}");
    }

    let cuda_root = cuda_root_from_env();
    let nvcc = cuda_root
        .as_ref()
        .map(|r| r.join("bin").join("nvcc"))
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    if !nvcc.exists() && cuda_root.is_some() {
        eprintln!("nvcc not found at {nvcc:?}, skipping vllm-marlin");
        return;
    }

    let compute_cap = env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".to_string());
    let nvcc_threads = configured_nvcc_threads();
    let flags = vec![
        format!("nvcc={}", nvcc.display()),
        format!("arch=sm_{compute_cap}"),
        "-Ivllm_marlin".to_string(),
        "-DMARLIN_NAMESPACE_NAME=marlin".to_string(),
        "-std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda"
            .to_string(),
        "-Xcompiler -fPIC -Xcompiler -fvisibility=default".to_string(),
    ];
    let deps: Vec<&str> = cu_files
        .iter()
        .chain(header_files.iter())
        .copied()
        .collect();
    let legacy_signature = static_lib_signature("vllm-marlin", &deps, &flags);
    let content_signature = content_static_lib_signature("vllm-marlin", &deps, &flags);
    let signature = cuda_native_input_signature(&content_signature);
    let scheduler_migration_signatures =
        historical_nvcc_scheduler_signatures("vllm-marlin", &deps, &flags);
    let metadata_hash_signature = metadata_hash_static_lib_signature("vllm-marlin", &deps, &flags);
    let metadata_signature = metadata_static_lib_signature("vllm-marlin", &deps, &flags);
    let build_start = Instant::now();
    let cache_state = static_lib_cache_state(out_dir, "vllm_marlin", &signature);
    let build_reason = match cache_state {
        CacheState::Fresh(reason) => {
            publish_cuda_build_artifact(
                native_build_cache,
                "static.vllm_marlin",
                "libvllm_marlin.a",
                &signature,
                &out_dir.join("libvllm_marlin.a"),
            );
            emit_cuda_build_summary(
                "vllm_marlin",
                "cache_hit",
                reason,
                build_start.elapsed(),
                &signature,
            );
            emit_cuda_static_link(out_dir, "vllm_marlin", cuda_root.as_ref(), true);
            return;
        }
        CacheState::Stale(reason) => {
            let cache_promotion_signatures = scheduler_migration_signatures
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            if let Some(cache_reason) = restore_cuda_build_artifact(
                native_build_cache,
                out_dir,
                "static.vllm_marlin",
                "libvllm_marlin.a",
                "libvllm_marlin.stamp",
                &signature,
                &cache_promotion_signatures,
                &[
                    &legacy_signature,
                    &metadata_hash_signature,
                    &metadata_signature,
                ],
            ) {
                emit_cuda_build_summary(
                    "vllm_marlin",
                    "cache_hit",
                    &cache_reason,
                    build_start.elapsed(),
                    &signature,
                );
                emit_cuda_static_link(out_dir, "vllm_marlin", cuda_root.as_ref(), true);
                return;
            }
            reason
        }
    };
    enforce_cuda_native_source_policy("vllm_marlin", build_reason, &signature);

    // Compile each .cu to its own .o
    let mut object_files: Vec<PathBuf> = Vec::new();
    for src in cu_files {
        let stem = std::path::Path::new(src)
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("cu filename");
        let obj = out_dir.join(format!("{stem}.o"));
        eprintln!("[vllm-marlin] compiling {src} -> {}", obj.display());

        let status = std::process::Command::new(&nvcc)
            .args(["-c", src, "-o"])
            .arg(obj.to_str().unwrap())
            .args([
                &format!("-arch=sm_{compute_cap}"),
                "-Ivllm_marlin",
                "-DMARLIN_NAMESPACE_NAME=marlin",
                "-std=c++17",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-Xcompiler",
                "-fPIC",
                // CUDA 13 default-hidden-visibility workaround. The
                // marlin_template.h Marlin template carries a
                // `__attribute__((visibility("default")))` to mark the
                // kernel exportable, but nvcc 13 still emits the host
                // stub with hidden ELF visibility unless the host
                // compiler is told otherwise. Without this, the
                // sm80_kernel_*.cu explicit instantiations end up as
                // hidden symbols inside libvllm_marlin.a and rust-lld
                // refuses them at the final link. Mirrors the same
                // flag added to compile_vllm_moe_marlin.
                "-Xcompiler",
                "-fvisibility=default",
                // vLLM kernels read CUDA_ARCH at compile time; emit it for nvcc
                "--threads",
                nvcc_threads.as_str(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("[vllm-marlin] nvcc spawn failed for {src}: {e}"));
        if !status.success() {
            panic!(
                "[vllm-marlin] nvcc failed compiling {src}. Disable with \
                `--features vllm-marlin` removed, or fix CUDA setup."
            );
        }
        object_files.push(obj);
    }

    // Pack into a static library
    let lib_file = out_dir.join("libvllm_marlin.a");
    let mut ar_args: Vec<String> = vec!["rcs".to_string(), lib_file.display().to_string()];
    for o in &object_files {
        ar_args.push(o.display().to_string());
    }
    let ar_status = std::process::Command::new("ar")
        .args(&ar_args)
        .status()
        .unwrap_or_else(|e| panic!("[vllm-marlin] ar spawn failed: {e}"));
    if !ar_status.success() {
        panic!("[vllm-marlin] ar failed to bundle {lib_file:?}");
    }

    write_static_lib_stamp(out_dir, "vllm_marlin", &signature);
    publish_cuda_build_artifact(
        native_build_cache,
        "static.vllm_marlin",
        "libvllm_marlin.a",
        &signature,
        &lib_file,
    );
    emit_cuda_static_link(out_dir, "vllm_marlin", cuda_root.as_ref(), true);
    eprintln!("[vllm-marlin] static lib built: {}", lib_file.display());
    emit_cuda_build_summary(
        "vllm_marlin",
        "built",
        build_reason,
        build_start.elapsed(),
        &signature,
    );
}

fn compile_marlin(out_dir: &PathBuf, native_build_cache: Option<&CudaNativeBuildCache>) {
    println!("cargo:rerun-if-changed=kernels/marlin_cuda_kernel.cu");

    let cuda_root = cuda_root_from_env();
    let nvcc = cuda_root
        .as_ref()
        .map(|r| r.join("bin").join("nvcc"))
        .unwrap_or_else(|| PathBuf::from("nvcc"));

    if !nvcc.exists() && cuda_root.is_some() {
        eprintln!("nvcc not found at {:?}, skipping Marlin kernel", nvcc);
        return;
    }

    // This kernel always emits compute_80 PTX, so runtime device capability is
    // not an output-affecting cache input.
    let flags = vec![
        format!("nvcc={}", nvcc.display()),
        "arch=compute_80".to_string(),
        "-std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -Xcompiler -fPIC".to_string(),
    ];
    let legacy_signature =
        static_lib_signature("marlin", &["kernels/marlin_cuda_kernel.cu"], &flags);
    let content_signature =
        content_static_lib_signature("marlin", &["kernels/marlin_cuda_kernel.cu"], &flags);
    let signature = cuda_native_input_signature(&content_signature);
    let metadata_hash_signature =
        metadata_hash_static_lib_signature("marlin", &["kernels/marlin_cuda_kernel.cu"], &flags);
    let metadata_signature =
        metadata_static_lib_signature("marlin", &["kernels/marlin_cuda_kernel.cu"], &flags);
    let build_start = Instant::now();
    let cache_state = static_lib_cache_state(out_dir, "marlin", &signature);
    let build_reason = match cache_state {
        CacheState::Fresh(reason) => {
            publish_cuda_build_artifact(
                native_build_cache,
                "static.marlin",
                "libmarlin.a",
                &signature,
                &out_dir.join("libmarlin.a"),
            );
            emit_cuda_build_summary(
                "marlin",
                "cache_hit",
                reason,
                build_start.elapsed(),
                &signature,
            );
            emit_cuda_static_link(out_dir, "marlin", cuda_root.as_ref(), false);
            return;
        }
        CacheState::Stale(reason) => {
            if let Some(cache_reason) = restore_cuda_build_artifact_with_import_migration(
                native_build_cache,
                out_dir,
                "static.marlin",
                "libmarlin.a",
                "libmarlin.stamp",
                &signature,
                &[],
                &[
                    &legacy_signature,
                    &metadata_hash_signature,
                    &metadata_signature,
                ],
                Some("flag=reported_compute_cap="),
            ) {
                emit_cuda_build_summary(
                    "marlin",
                    "cache_hit",
                    &cache_reason,
                    build_start.elapsed(),
                    &signature,
                );
                emit_cuda_static_link(out_dir, "marlin", cuda_root.as_ref(), false);
                return;
            }
            reason
        }
    };
    enforce_cuda_native_source_policy("marlin", build_reason, &signature);

    let obj_file = out_dir.join("marlin_cuda_kernel.o");
    let status = std::process::Command::new(&nvcc)
        .args(["-c", "kernels/marlin_cuda_kernel.cu", "-o"])
        .arg(obj_file.to_str().unwrap())
        .args([
            // Generate PTX for compute_80, embed as PTX (not SASS).
            // The GPU driver JIT-compiles to native code at runtime.
            // This provides forward compatibility across GPU architectures.
            "-arch=compute_80",
            "-std=c++17",
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-Xcompiler",
            "-fPIC",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            // Create static library from object file
            let lib_file = out_dir.join("libmarlin.a");
            let ar_status = std::process::Command::new("ar")
                .args(["rcs"])
                .arg(lib_file.to_str().unwrap())
                .arg(obj_file.to_str().unwrap())
                .status();
            if let Ok(s) = ar_status {
                if s.success() {
                    write_static_lib_stamp(out_dir, "marlin", &signature);
                    publish_cuda_build_artifact(
                        native_build_cache,
                        "static.marlin",
                        "libmarlin.a",
                        &signature,
                        &lib_file,
                    );
                    emit_cuda_static_link(out_dir, "marlin", cuda_root.as_ref(), false);
                    eprintln!("Marlin kernel compiled successfully (compute_80 PTX)");
                    emit_cuda_build_summary(
                        "marlin",
                        "built",
                        build_reason,
                        build_start.elapsed(),
                        &signature,
                    );
                    return;
                }
            }
            eprintln!("Failed to create libmarlin.a, Marlin disabled");
        }
        Ok(s) => {
            panic!(
                "nvcc failed with {s} compiling Marlin kernel. \
                    Remove --features marlin or fix CUDA setup."
            );
        }
        Err(e) => {
            panic!(
                "nvcc not available ({e}). \
                    Remove --features marlin or install CUDA toolkit."
            );
        }
    }
}
