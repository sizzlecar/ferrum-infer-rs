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
    println!("cargo:rerun-if-changed=kernels/fused_add_rms_norm.cu");
    println!("cargo:rerun-if-changed=kernels/fused_silu_mul.cu");
    println!("cargo:rerun-if-changed=kernels/rms_norm.cu");
    println!("cargo:rerun-if-changed=kernels/rope.cu");
    println!("cargo:rerun-if-changed=kernels/decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/residual_add.cu");
    println!("cargo:rerun-if-changed=kernels/flash_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/paged_decode_attention.cu");
    println!("cargo:rerun-if-changed=kernels/dequant_int4.cu");

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
            "kernels/dequant_int4.cu",
        ])
        .out_dir(out_dir)
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
            &format!("-arch=sm_{compute_cap}"),
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
