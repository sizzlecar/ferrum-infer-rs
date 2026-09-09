//! Use one source of IQ reconstruction tables for CPU, Metal and compiled CUDA.
use std::{fmt::Write, fs, path::Path};

#[path = "../src/gguf_blocks/iq3s_grid.rs"]
mod iq3s_grid;
#[path = "../src/gguf_blocks/iq4nl_values.rs"]
mod iq4nl_values;

pub const KERNEL: &str = "kernels/vnext_gguf.cu";
pub const INPUTS: &[&str] = &[
    "build_support/gguf.rs",
    "src/gguf_blocks/iq3s_grid.rs",
    "src/gguf_blocks/iq4nl_values.rs",
];

pub fn write_tables(out_dir: &Path) {
    let mut source = String::from(
        "// Generated from Ferrum's shared GGML codebooks.\n#pragma once\n\
         __device__ __constant__ unsigned int iq3_s_grid[512] = {\n",
    );
    for value in iq3s_grid::IQ3_S_GRID {
        write!(&mut source, "0x{value:08x},").unwrap();
    }
    source.push_str("\n};\n__device__ __constant__ signed char iq4_nl_values[16] = {");
    for value in iq4nl_values::IQ4_NL_VALUES {
        write!(&mut source, "{value},").unwrap();
    }
    source.push_str("\n};\n");
    let path = out_dir.join("vnext_gguf_codebooks.cuh");
    if fs::read_to_string(&path).ok().as_deref() != Some(source.as_str()) {
        fs::write(&path, source).expect("write native GGUF codebooks");
    }
}
