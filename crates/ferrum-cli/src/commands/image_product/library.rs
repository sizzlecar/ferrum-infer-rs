//! Pure path discovery. Native loading remains inside the admitted CUDA owner.
use anyhow::{ensure, Context, Result};
use std::path::{Path, PathBuf};

pub(super) fn resolve(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return canonical_library(path);
    }
    let environment = std::env::var_os("LD_LIBRARY_PATH")
        .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default();
    if let Some(path) = first_library(&environment)? {
        return Ok(path);
    }
    #[cfg(target_os = "linux")]
    if let Some(path) = cached_library()? {
        return Ok(path);
    }
    let standard = [
        "/lib",
        "/lib64",
        "/usr/lib",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/local/cuda/lib64",
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ];
    if let Some(path) = first_library(&standard.map(PathBuf::from))? {
        return Ok(path);
    }
    anyhow::bail!("cuDNN 9 was not found in LD_LIBRARY_PATH, the system loader cache, or standard library directories; install cuDNN 9 or pass --image-cudnn-library /absolute/path/libcudnn.so.9")
}
fn canonical_library(path: &Path) -> Result<PathBuf> {
    let path = std::fs::canonicalize(path)
        .with_context(|| format!("resolve cuDNN library {}", path.display()))?;
    ensure!(
        path.is_file(),
        "cuDNN library is not a regular file: {}",
        path.display()
    );
    Ok(path)
}
fn first_library(directories: &[PathBuf]) -> Result<Option<PathBuf>> {
    for directory in directories {
        let path = directory.join("libcudnn.so.9");
        if path.try_exists()? {
            return canonical_library(&path).map(Some);
        }
    }
    Ok(None)
}
fn cache_paths(cache: &str) -> impl Iterator<Item = PathBuf> + '_ {
    cache.lines().filter_map(|line| {
        let (description, path) = line.split_once("=>")?;
        (description.split_whitespace().next()? == "libcudnn.so.9")
            .then(|| PathBuf::from(path.trim()))
            .filter(|path| path.is_absolute())
    })
}
#[cfg(target_os = "linux")]
fn cached_library() -> Result<Option<PathBuf>> {
    use std::{
        io::Read,
        process::{Command, Stdio},
    };
    const LIMIT: u64 = 4 * 1024 * 1024;
    for program in ["/sbin/ldconfig", "/usr/sbin/ldconfig"] {
        if !Path::new(program).is_file() {
            continue;
        }
        let mut child = Command::new(program)
            .arg("-p")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("read system loader cache")?;
        let mut bytes = Vec::new();
        let read = child
            .stdout
            .take()
            .context("missing loader cache stdout")?
            .take(LIMIT + 1)
            .read_to_end(&mut bytes);
        if read.is_err() || bytes.len() as u64 > LIMIT {
            let _ = child.kill();
            let _ = child.wait();
            read?;
            anyhow::bail!("system loader cache exceeds bounded discovery size");
        }
        if !child.wait()?.success() {
            continue;
        }
        for path in cache_paths(std::str::from_utf8(&bytes).context("loader cache is not UTF-8")?) {
            if path.try_exists()? {
                return canonical_library(&path).map(Some);
            }
        }
        break;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_paths_and_search_order_do_not_load_native_code() {
        let directory = tempfile::tempdir().unwrap();
        let a = directory.path().join("a");
        let b = directory.path().join("b");
        std::fs::create_dir(&a).unwrap();
        std::fs::create_dir(&b).unwrap();
        // Not a native library: discovery must not call dlopen or a symbol.
        std::fs::write(a.join("libcudnn.so.9"), b"a").unwrap();
        std::fs::write(b.join("libcudnn.so.9"), b"b").unwrap();
        assert_eq!(
            first_library(&[b.clone(), a.clone()]).unwrap().unwrap(),
            std::fs::canonicalize(b.join("libcudnn.so.9")).unwrap()
        );
        assert_eq!(
            resolve(Some(&a.join("libcudnn.so.9"))).unwrap(),
            std::fs::canonicalize(a.join("libcudnn.so.9")).unwrap()
        );
        assert!(resolve(Some(&a.join("missing"))).is_err());
        assert!(resolve(Some(&a)).is_err());
        assert_eq!(cache_paths("libcudnn.so.8 (libc6) => /old\nlibcudnn.so.9 (libc6) => /actual/libcudnn.so.9\nlibcudnn.so.9 => relative").collect::<Vec<_>>(), vec![PathBuf::from("/actual/libcudnn.so.9")]);
    }
}
