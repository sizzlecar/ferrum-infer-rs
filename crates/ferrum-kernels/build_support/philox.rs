//! Discover actual transitive CUDA/cuRAND inputs rather than trusting a toolkit
//! version string. Used only for the shared BF16 native translation unit.
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub fn identity(
    nvcc: &Path,
    cuda_root: Option<&Path>,
    kernel: &str,
    flags: &[String],
    allow_unsupported: bool,
) -> Result<String, String> {
    let mut command = Command::new(nvcc);
    command.args([
        "--generate-dependencies",
        "--dependency-target-name",
        "ferrum_philox",
        "--default-stream",
        "per-thread",
    ]);
    for flag in flags {
        if flag.starts_with("nvcc=") {
            continue;
        }
        if let Some(arch) = flag.strip_prefix("arch=") {
            command.arg(format!("--gpu-architecture={arch}"));
        } else if let Some(ccbin) = flag.strip_prefix("ccbin=") {
            if allow_unsupported {
                command.arg("-allow-unsupported-compiler");
            }
            command.args(["-ccbin", ccbin]);
        } else {
            command.arg(flag);
        }
    }
    let output = command
        .arg(kernel)
        .output()
        .map_err(|e| format!("Philox dependency scan: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "Philox nvcc dependency scan failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let source = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let mut paths = BTreeSet::new();
    for path in parse_dependencies(&source)? {
        let canonical = path
            .canonicalize()
            .map_err(|e| format!("Philox dependency {}: {e}", path.display()))?;
        paths.insert(canonical);
    }
    for required in [
        "curand_kernel.h",
        "curand_philox4x32_x.h",
        "curand_normal.h",
    ] {
        if !paths
            .iter()
            .any(|p| p.file_name().is_some_and(|name| name == required))
        {
            return Err(format!(
                "Philox dependency scan omitted required actual header {required}"
            ));
        }
    }
    let compiler = nvcc.canonicalize().map_err(|e| e.to_string())?;
    let compiler_root = compiler.parent().and_then(Path::parent);
    let libdevice_dirs = cuda_root
        .into_iter()
        .chain(compiler_root)
        .map(|p| p.join("nvvm/libdevice"));
    let mut found_bc = false;
    for directory in libdevice_dirs {
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&directory).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.extension().is_some_and(|ext| ext == "bc") {
                paths.insert(path.canonicalize().map_err(|e| e.to_string())?);
                found_bc = true;
            }
        }
    }
    if !found_bc {
        return Err("Philox cannot identify actual toolkit libdevice bitcode".into());
    }
    paths.insert(compiler);
    let version = Command::new(nvcc)
        .arg("--version")
        .output()
        .map_err(|e| e.to_string())?;
    if !version.status.success() {
        return Err("Philox nvcc --version failed".into());
    }
    let mut digest = Sha256::new();
    digest.update(b"ferrum-philox-build-input-v1\0");
    digest.update(&version.stdout);
    for flag in flags {
        digest.update(flag.as_bytes());
        digest.update([0]);
    }
    digest.update(b"default-stream=per-thread\0");
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let bytes = fs::read(&path).map_err(|e| format!("Philox input {}: {e}", path.display()))?;
        let name = path
            .to_str()
            .ok_or("Philox build dependency path is not UTF-8")?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(Sha256::digest(&bytes));
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// NVCC emits one fixed target and Make-escaped prerequisites. Retain native
/// Windows backslashes unless they escape whitespace, '#', ':' or '\\'.
pub fn parse_dependencies(source: &str) -> Result<Vec<PathBuf>, String> {
    let normalized = source.replace("\\\r\n", " ").replace("\\\n", " ");
    let (target, body) = normalized
        .split_once(':')
        .ok_or("unexpected Philox dependency target")?;
    // NVCC's Make output permits whitespace before the rule delimiter.
    // Split only the target delimiter, preserving drive letters in paths.
    if target.trim() != "ferrum_philox" {
        return Err("unexpected Philox dependency target".into());
    }
    let mut paths = Vec::new();
    let mut word = String::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if chars
                .peek()
                .is_some_and(|next| next.is_whitespace() || matches!(*next, '#' | ':' | '\\'))
            {
                word.push(chars.next().unwrap());
            } else {
                word.push(c);
            }
        } else if c.is_whitespace() {
            if !word.is_empty() {
                paths.push(PathBuf::from(std::mem::take(&mut word)));
            }
        } else if c == '$' && chars.peek() == Some(&'$') {
            chars.next();
            word.push('$');
        } else {
            word.push(c);
        }
    }
    if !word.is_empty() {
        paths.push(PathBuf::from(word));
    }
    if paths.is_empty() {
        return Err("empty Philox dependency list".into());
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dependencies_preserve_escaped_paths_and_line_continuations() {
        assert_eq!(parse_dependencies("ferrum_philox: kernels/main.cu \\\n /CUDA\\ toolkit/curand_kernel.h C:\\CUDA\\include\\curand_normal.h\n").unwrap(),
            vec![PathBuf::from("kernels/main.cu"),PathBuf::from("/CUDA toolkit/curand_kernel.h"),PathBuf::from(r"C:\CUDA\include\curand_normal.h")]);
        assert_eq!(
            parse_dependencies("ferrum_philox: a$$b\\#c.h").unwrap(),
            vec![PathBuf::from("a$b#c.h")]
        );
    }
    #[test]
    fn nvcc_dependency_target_accepts_make_whitespace() {
        assert_eq!(
            parse_dependencies(
                "\nferrum_philox : kernels/vnext_bf16_silu.cu \\\n    /usr/include/stdc-predef.h\n"
            )
            .unwrap(),
            vec![
                PathBuf::from("kernels/vnext_bf16_silu.cu"),
                PathBuf::from("/usr/include/stdc-predef.h")
            ]
        );
        assert_eq!(
            parse_dependencies(" \tferrum_philox\t: C:\\CUDA\\include\\curand_normal.h").unwrap(),
            vec![PathBuf::from(r"C:\CUDA\include\curand_normal.h")]
        );
    }
    #[test]
    fn malformed_or_empty_dependency_output_is_rejected() {
        assert!(parse_dependencies("other: a.h").is_err());
        assert!(parse_dependencies("ferrum_philox_suffix : a.h").is_err());
        assert!(parse_dependencies("ferrum_philox other : a.h").is_err());
        assert!(parse_dependencies("ferrum_philox:\n").is_err());
    }
}
