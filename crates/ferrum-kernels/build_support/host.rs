//! Host tools and target library paths used by the CUDA build script.

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostTools {
    Unix,
    WindowsMsvc,
}

impl HostTools {
    pub fn current() -> Self {
        if cfg!(windows) {
            Self::WindowsMsvc
        } else {
            Self::Unix
        }
    }

    pub fn nvcc(self, cuda_root: Option<&Path>) -> PathBuf {
        let name = match self {
            Self::Unix => "nvcc",
            Self::WindowsMsvc => "nvcc.exe",
        };
        cuda_root
            .map(|root| root.join("bin").join(name))
            .unwrap_or_else(|| PathBuf::from(name))
    }

    pub fn c_compiler(self) -> &'static str {
        match self {
            Self::Unix => "cc",
            Self::WindowsMsvc => "cl.exe",
        }
    }

    pub fn cpp_compiler(self) -> &'static str {
        match self {
            Self::Unix => "c++",
            Self::WindowsMsvc => "cl.exe",
        }
    }

    pub fn archiver(self) -> &'static str {
        match self {
            Self::Unix => "ar",
            Self::WindowsMsvc => "lib.exe",
        }
    }

    pub fn executable_candidates(self, program: &Path) -> Vec<PathBuf> {
        let mut candidates = vec![program.to_path_buf()];
        if self == Self::WindowsMsvc && program.extension().is_none() {
            candidates.push(program.with_extension("exe"));
        }
        candidates
    }
}

pub fn cuda_library_directory(root: &Path, target: &str) -> PathBuf {
    if target.ends_with("windows-msvc") {
        root.join("lib").join("x64")
    } else {
        root.join("lib64")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_tools_keep_paths_with_spaces_as_one_path() {
        let root = Path::new("toolchains/CUDA Toolkit");
        assert_eq!(
            HostTools::WindowsMsvc.nvcc(Some(root)),
            root.join("bin/nvcc.exe")
        );
        assert_eq!(HostTools::Unix.nvcc(Some(root)), root.join("bin/nvcc"));
        assert_eq!(
            HostTools::WindowsMsvc.executable_candidates(Path::new("toolchains/Visual Studio/cl")),
            [
                PathBuf::from("toolchains/Visual Studio/cl"),
                PathBuf::from("toolchains/Visual Studio/cl.exe")
            ]
        );
        assert_eq!(
            HostTools::WindowsMsvc.executable_candidates(Path::new("nvcc.exe")),
            [PathBuf::from("nvcc.exe")]
        );
    }

    #[test]
    fn cuda_library_paths_follow_the_link_target() {
        let windows = "x86_64-pc-windows-msvc";
        let linux = "x86_64-unknown-linux-gnu";
        assert_eq!(
            cuda_library_directory(Path::new("cuda"), windows),
            Path::new("cuda/lib/x64")
        );
        assert_eq!(
            cuda_library_directory(Path::new("cuda"), linux),
            Path::new("cuda/lib64")
        );
    }
}
