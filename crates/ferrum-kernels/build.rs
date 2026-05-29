use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

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

fn static_lib_is_fresh(
    out_dir: &Path,
    lib_name: &str,
    signature: &str,
    migration_signatures: &[&str],
) -> bool {
    let lib_file = out_dir.join(format!("lib{lib_name}.a"));
    let stamp_file = out_dir.join(format!("lib{lib_name}.stamp"));
    if !lib_file.is_file() || !stamp_file.is_file() {
        return false;
    }
    match fs::read_to_string(&stamp_file) {
        Ok(existing) if existing == signature => {
            eprintln!("[{lib_name}] cache hit: {}", lib_file.display());
            true
        }
        Ok(existing) if migration_signatures.iter().any(|s| existing == **s) => {
            write_static_lib_stamp(out_dir, lib_name, signature);
            eprintln!(
                "[{lib_name}] cache hit: {} (migrated stamp)",
                lib_file.display()
            );
            true
        }
        _ => false,
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

    // Link Accelerate framework on macOS (provides cblas_sgemm, vDSP_*)
    if env::consts::OS == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    println!("cargo:rerun-if-changed=kernels/fused_add_rms_norm.cu");
    println!("cargo:rerun-if-changed=kernels/fused_silu_mul.cu");
    println!("cargo:rerun-if-changed=kernels/rms_norm.cu");
    println!("cargo:rerun-if-changed=kernels/rope.cu");
    println!("cargo:rerun-if-changed=kernels/decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/residual_add.cu");
    println!("cargo:rerun-if-changed=kernels/flash_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/paged_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/paged_varlen_attention.cu");
    println!("cargo:rerun-if-changed=kernels/paged_varlen_attention_vllm.cu");
    println!("cargo:rerun-if-changed=kernels/dequant_int4.cu");
    println!("cargo:rerun-if-changed=kernels/batched_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/common.cuh");
    println!("cargo:rerun-if-changed=kernels/softmax.cu");
    println!("cargo:rerun-if-changed=kernels/embedding_lookup.cu");
    println!("cargo:rerun-if-changed=kernels/flash_attn_full.cu");
    println!("cargo:rerun-if-changed=kernels/batched_flash_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/qk_norm_rope.cu");
    println!("cargo:rerun-if-changed=kernels/split_qkv_norm_rope_into_paged_cache.cu");
    println!("cargo:rerun-if-changed=kernels/transpose.cu");
    println!("cargo:rerun-if-changed=kernels/kv_cache_append.cu");
    println!("cargo:rerun-if-changed=kernels/split_qkv.cu");
    println!("cargo:rerun-if-changed=kernels/add_bias.cu");
    println!("cargo:rerun-if-changed=kernels/layer_norm.cu");
    println!("cargo:rerun-if-changed=kernels/gelu.cu");
    println!("cargo:rerun-if-changed=kernels/decode_attention_hm.cu");
    println!("cargo:rerun-if-changed=kernels/gather_columns.cu");
    println!("cargo:rerun-if-changed=kernels/moe_combine.cu");
    println!("cargo:rerun-if-changed=kernels/moe_router.cu");
    println!("cargo:rerun-if-changed=kernels/moe_align_block_size.cu");
    println!("cargo:rerun-if-changed=kernels/moe_align_block_size_pair_ids.cu");
    println!("cargo:rerun-if-changed=kernels/moe_build_pairs.cu");
    println!("cargo:rerun-if-changed=kernels/int8_paged_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/argmax_rows.cu");
    println!("cargo:rerun-if-changed=kernels/split_qkv_norm_rope_into_paged_cache_vllm.cu");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set by cargo"));
    let ptx_rs = out_dir.join("ptx.rs");

    let out_dir_clone = out_dir.clone();
    let mut builder = bindgen_cuda::Builder::default()
        .kernel_paths(vec![
            "kernels/fused_add_rms_norm.cu",
            "kernels/fused_silu_mul.cu",
            "kernels/rms_norm.cu",
            "kernels/rope.cu",
            "kernels/decode_attention.cu",
            "kernels/residual_add.cu",
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
        ])
        .out_dir(out_dir)
        .arg("-Ikernels") // for common.cuh
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math");

    if let Some(cuda_root) = cuda_root_from_env() {
        builder.cuda_root(cuda_root);
    }

    let bindings = builder
        .build_ptx()
        .expect("failed to compile ferrum CUDA kernels to PTX");

    bindings
        .write(&ptx_rs)
        .expect("failed to write ferrum CUDA PTX bindings");

    // Compile Marlin INT4xFP16 kernel separately (uses runtime API, not PTX).
    // Only when "marlin" feature is enabled. Requires SM >= 8.0 (Ampere).
    if env::var_os("CARGO_FEATURE_MARLIN").is_some() {
        compile_marlin(&out_dir_clone);
    }

    // vLLM gptq_marlin port (Phase 12). Heavier C++ template instantiations
    // than the IST-DASLab port — compile time ~30 min on first build. Opt-in
    // via `--features vllm-marlin`.
    if env::var_os("CARGO_FEATURE_VLLM_MARLIN").is_some() {
        compile_vllm_marlin(&out_dir_clone);
    }

    // vLLM moe_marlin_wna16 port (Stage 14). Vendored from
    // vllm/csrc/moe/marlin_moe_wna16/ at v0.10.2. Single .cu file with
    // many template instantiations via COMMON_GET_IF macros — compile
    // time ~15-20 min on first build. Opt-in via `--features vllm-moe-marlin`.
    if env::var_os("CARGO_FEATURE_VLLM_MOE_MARLIN").is_some() {
        compile_vllm_moe_marlin(&out_dir_clone);
    }

    // vLLM paged_attention_v2 port (2026-05-12). Vendored from vllm v0.20.2
    // (csrc/attention/{paged_attention_v2.cu,attention_kernels.cuh,...})
    // with torch headers stripped. Opt-in via `--features vllm-paged-attn-v2`.
    // Builds a static lib of the single (HEAD=128, BLOCK=16, FP16, no-FP8,
    // no-blocksparse) instantiation — ~1-2 min compile.
    if env::var_os("CARGO_FEATURE_VLLM_PAGED_ATTN_V2").is_some() {
        compile_vllm_paged_attn(&out_dir_clone);
    }
}

fn compile_vllm_paged_attn(out_dir: &PathBuf) {
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
    let nvcc_threads = env::var("FERRUM_NVCC_THREADS").unwrap_or_else(|_| "0".to_string());
    let flags = vec![
        format!("nvcc={}", nvcc.display()),
        format!("arch=sm_{compute_cap}"),
        format!("threads={nvcc_threads}"),
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
    let signature = static_lib_signature("vllm-paged-attn-v2", &deps, &flags);
    let metadata_hash_signature =
        metadata_hash_static_lib_signature("vllm-paged-attn-v2", &deps, &flags);
    let metadata_signature = metadata_static_lib_signature("vllm-paged-attn-v2", &deps, &flags);
    if static_lib_is_fresh(
        out_dir,
        "vllm_paged_attn",
        &signature,
        &[&metadata_hash_signature, &metadata_signature],
    ) {
        emit_cuda_static_link(out_dir, "vllm_paged_attn", cuda_root.as_ref(), true);
        return;
    }

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
    emit_cuda_static_link(out_dir, "vllm_paged_attn", cuda_root.as_ref(), true);
    eprintln!(
        "[vllm-paged-attn-v2] static lib built: {}",
        lib_file.display()
    );
}

fn compile_vllm_moe_marlin(out_dir: &PathBuf) {
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
    let nvcc_threads = env::var("FERRUM_NVCC_THREADS").unwrap_or_else(|_| "0".to_string());
    let flags = vec![
        format!("nvcc={}", nvcc.display()),
        format!("arch=sm_{compute_cap}"),
        format!("threads={nvcc_threads}"),
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
    let signature = static_lib_signature("vllm-moe-marlin", &deps, &flags);
    let metadata_hash_signature =
        metadata_hash_static_lib_signature("vllm-moe-marlin", &deps, &flags);
    let metadata_signature = metadata_static_lib_signature("vllm-moe-marlin", &deps, &flags);
    if static_lib_is_fresh(
        out_dir,
        "vllm_moe_marlin",
        &signature,
        &[&metadata_hash_signature, &metadata_signature],
    ) {
        emit_cuda_static_link(out_dir, "vllm_moe_marlin", cuda_root.as_ref(), true);
        return;
    }

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
    emit_cuda_static_link(out_dir, "vllm_moe_marlin", cuda_root.as_ref(), true);
    eprintln!("[vllm-moe-marlin] static lib built: {}", lib_file.display());
}

fn compile_vllm_marlin(out_dir: &PathBuf) {
    // Source files: marlin.cu (dispatcher + extern C wrapper),
    // gptq_marlin_repack.cu (currently disabled torch wrapper, kernel still
    // present), and per-shape kernel template instantiations.
    // For a first pass we only build the FP16 + kU4B8 + FP16 variant —
    // matches what our M2 (Llama-3.1-8B GPTQ-INT4) workload exercises.
    // Other dtype variants are present in the kernel_selector.h dispatch but
    // never reached at runtime; if linker complains about unresolved symbols
    // we'll add more variants.
    // We have to compile ALL kernel template instantiations referenced by
    // kernel_selector.h, even unreachable ones at runtime — the compiler still
    // needs to resolve every `Marlin<...>` template in the if/else dispatch.
    // Pruning is brittle (if we miss a variant the link fails), so we just
    // build the full set generated by generate_kernels.py 8.9.
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
    let nvcc_threads = env::var("FERRUM_NVCC_THREADS").unwrap_or_else(|_| "0".to_string());
    let flags = vec![
        format!("nvcc={}", nvcc.display()),
        format!("arch=sm_{compute_cap}"),
        format!("threads={nvcc_threads}"),
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
    let signature = static_lib_signature("vllm-marlin", &deps, &flags);
    let metadata_hash_signature = metadata_hash_static_lib_signature("vllm-marlin", &deps, &flags);
    let metadata_signature = metadata_static_lib_signature("vllm-marlin", &deps, &flags);
    if static_lib_is_fresh(
        out_dir,
        "vllm_marlin",
        &signature,
        &[&metadata_hash_signature, &metadata_signature],
    ) {
        emit_cuda_static_link(out_dir, "vllm_marlin", cuda_root.as_ref(), true);
        return;
    }

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
    emit_cuda_static_link(out_dir, "vllm_marlin", cuda_root.as_ref(), true);
    eprintln!("[vllm-marlin] static lib built: {}", lib_file.display());
}

fn compile_marlin(out_dir: &PathBuf) {
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

    // Determine compute capability: use CUDA_COMPUTE_CAP env or default to 80
    let compute_cap = env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "80".to_string());
    let flags = vec![
        format!("nvcc={}", nvcc.display()),
        "arch=compute_80".to_string(),
        format!("reported_compute_cap={compute_cap}"),
        "-std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr -Xcompiler -fPIC".to_string(),
    ];
    let signature = static_lib_signature("marlin", &["kernels/marlin_cuda_kernel.cu"], &flags);
    let metadata_hash_signature =
        metadata_hash_static_lib_signature("marlin", &["kernels/marlin_cuda_kernel.cu"], &flags);
    let metadata_signature =
        metadata_static_lib_signature("marlin", &["kernels/marlin_cuda_kernel.cu"], &flags);
    if static_lib_is_fresh(
        out_dir,
        "marlin",
        &signature,
        &[&metadata_hash_signature, &metadata_signature],
    ) {
        emit_cuda_static_link(out_dir, "marlin", cuda_root.as_ref(), false);
        return;
    }

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
                    emit_cuda_static_link(out_dir, "marlin", cuda_root.as_ref(), false);
                    eprintln!("Marlin kernel compiled successfully (sm_{compute_cap})");
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
