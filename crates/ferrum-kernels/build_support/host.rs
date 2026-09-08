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

    /// Keep the selected file identity separate from NVCC's argv[0]. On Windows,
    /// an extended-length argv[0] prevents NVCC from loading its adjacent profile.
    pub fn nvcc_invocation(self, selected: &Path) -> Result<PathBuf, String> {
        if self == Self::Unix {
            return Ok(selected.to_path_buf());
        }
        let selected = selected
            .to_str()
            .ok_or("Windows NVCC path must be valid Unicode")?
            .replace('\\', "/");
        let ordinary = if let Some(tail) = selected.strip_prefix("//?/UNC/") {
            format!("//{tail}")
        } else {
            selected
                .strip_prefix("//?/")
                .unwrap_or(&selected)
                .to_string()
        };
        let bytes = ordinary.as_bytes();
        let drive = bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && &bytes[1..3] == b":/";
        let unc = ordinary.strip_prefix("//").is_some_and(|tail| {
            let mut components = tail.split('/');
            [components.next(), components.next()]
                .iter()
                .all(|part| part.is_some_and(|part| !matches!(part, "" | "." | ".." | "?")))
        });
        if (!drive && !unc)
            || ordinary.contains(['\0', '\r', '\n'])
            || !ordinary
                .rsplit('/')
                .next()
                .is_some_and(|name| name.eq_ignore_ascii_case("nvcc.exe"))
        {
            return Err("Windows NVCC invocation requires an absolute nvcc.exe path".to_string());
        }
        Ok(PathBuf::from(ordinary))
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

    pub fn nvcc_environment_option(self) -> Option<&'static str> {
        (self == Self::WindowsMsvc).then_some("--use-local-env")
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
        assert_eq!(
            HostTools::WindowsMsvc.nvcc_environment_option(),
            Some("--use-local-env")
        );
        assert_eq!(HostTools::Unix.nvcc_environment_option(), None);
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

    #[test]
    fn nvcc_invocation_preserves_the_tool_location_without_the_windows_prefix() {
        for (selected, invocation) in [
            (
                r"\\?\C:\CUDA Toolkit\中文\bin\nvcc.exe",
                "C:/CUDA Toolkit/中文/bin/nvcc.exe",
            ),
            (
                r"\\?\UNC\server\CUDA Toolkit\bin\nvcc.exe",
                "//server/CUDA Toolkit/bin/nvcc.exe",
            ),
            (
                r"C:\CUDA Toolkit\bin\nvcc.exe",
                "C:/CUDA Toolkit/bin/nvcc.exe",
            ),
        ] {
            assert_eq!(
                HostTools::WindowsMsvc
                    .nvcc_invocation(Path::new(selected))
                    .unwrap(),
                Path::new(invocation)
            );
        }
        for invalid in [
            "nvcc.exe",
            "relative/nvcc.exe",
            "C:/CUDA/cl.exe",
            r"\\.\CUDA\nvcc.exe",
        ] {
            assert!(HostTools::WindowsMsvc
                .nvcc_invocation(Path::new(invalid))
                .is_err());
        }
    }

    #[test]
    fn unix_nvcc_invocation_preserves_the_selected_spelling() {
        for selected in [
            "nvcc",
            "/usr/local/cuda/../cuda/bin/nvcc",
            r"\\?\literal-nvcc-name",
        ] {
            assert_eq!(
                HostTools::Unix
                    .nvcc_invocation(Path::new(selected))
                    .unwrap(),
                Path::new(selected)
            );
        }
    }
}
