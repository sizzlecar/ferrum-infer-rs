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
}
