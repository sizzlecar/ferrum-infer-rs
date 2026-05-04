use std::env;
use std::path::PathBuf;

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
    println!("cargo:rerun-if-changed=kernels/dequant_int4.cu");
    println!("cargo:rerun-if-changed=kernels/batched_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/common.cuh");
    println!("cargo:rerun-if-changed=kernels/softmax.cu");
    println!("cargo:rerun-if-changed=kernels/embedding_lookup.cu");
    println!("cargo:rerun-if-changed=kernels/flash_attn_full.cu");
    println!("cargo:rerun-if-changed=kernels/batched_flash_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/qk_norm_rope.cu");
    println!("cargo:rerun-if-changed=kernels/transpose.cu");
    println!("cargo:rerun-if-changed=kernels/kv_cache_append.cu");
    println!("cargo:rerun-if-changed=kernels/split_qkv.cu");
    println!("cargo:rerun-if-changed=kernels/add_bias.cu");
    println!("cargo:rerun-if-changed=kernels/layer_norm.cu");
    println!("cargo:rerun-if-changed=kernels/gelu.cu");
    println!("cargo:rerun-if-changed=kernels/decode_attention_hm.cu");
    println!("cargo:rerun-if-changed=kernels/gather_columns.cu");

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
            "kernels/dequant_int4.cu",
            "kernels/batched_decode_attention.cu",
            "kernels/softmax.cu",
            "kernels/embedding_lookup.cu",
            "kernels/flash_attn_full.cu",
            "kernels/batched_flash_decode_attention.cu",
            "kernels/qk_norm_rope.cu",
            "kernels/transpose.cu",
            "kernels/kv_cache_append.cu",
            "kernels/split_qkv.cu",
            "kernels/add_bias.cu",
            "kernels/layer_norm.cu",
            "kernels/gelu.cu",
            "kernels/decode_attention_hm.cu",
            "kernels/gather_columns.cu",
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
    for f in cu_files {
        println!("cargo:rerun-if-changed={f}");
    }
    // Headers (any change re-triggers build)
    println!("cargo:rerun-if-changed=vllm_marlin/marlin_template.h");
    println!("cargo:rerun-if-changed=vllm_marlin/marlin_mma.h");
    println!("cargo:rerun-if-changed=vllm_marlin/marlin_dtypes.cuh");
    println!("cargo:rerun-if-changed=vllm_marlin/marlin.cuh");
    println!("cargo:rerun-if-changed=vllm_marlin/dequant.h");
    println!("cargo:rerun-if-changed=vllm_marlin/kernel.h");
    println!("cargo:rerun-if-changed=vllm_marlin/kernel_selector.h");
    println!("cargo:rerun-if-changed=vllm_marlin/scalar_type.hpp");
    println!("cargo:rerun-if-changed=vllm_marlin/torch_stubs.h");

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
                // vLLM kernels read CUDA_ARCH at compile time; emit it for nvcc
                "--threads",
                "0",
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

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vllm_marlin");
    if let Some(ref cuda_root) = cuda_root {
        let lib64 = cuda_root.join("lib64");
        if lib64.exists() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
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
                    println!("cargo:rustc-link-search=native={}", out_dir.display());
                    println!("cargo:rustc-link-lib=static=marlin");
                    // Link CUDA runtime
                    if let Some(ref cuda_root) = cuda_root {
                        let lib64 = cuda_root.join("lib64");
                        if lib64.exists() {
                            println!("cargo:rustc-link-search=native={}", lib64.display());
                        }
                    }
                    println!("cargo:rustc-link-lib=dylib=cudart");
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
