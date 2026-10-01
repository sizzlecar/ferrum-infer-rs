use super::*;
use std::path::Path;

/// Product override for the shared run/serve cache. PlatformDefault uses only
/// standard OS cache-location variables, never a Ferrum-specific environment
/// control, model cache, working directory, or temporary-directory fallback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloAutomaticCalibrationCacheLocationV1 {
    PlatformDefault {},
    /// Relative paths in a SLO config file are resolved against that file by
    /// the shared loader. Programmatic callers must supply an absolute path.
    Directory {
        path: PathBuf,
    },
}

impl Default for SloAutomaticCalibrationCacheLocationV1 {
    fn default() -> Self {
        Self::PlatformDefault {}
    }
}

/// A location failure disables reuse for this attempt, not in-memory training.
/// No directory is created or accessed while resolving the location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SloAutomaticCalibrationCacheLocationErrorV1 {
    #[error("automatic cache location is unsupported on this platform")]
    UnsupportedPlatform,
    #[error("automatic cache location has no absolute user cache or home directory")]
    UserCacheUnavailable,
    #[error("automatic cache directory must be resolved to an absolute path")]
    UnresolvedDirectory,
}

impl SloAutomaticCalibrationCacheLocationV1 {
    pub fn validate(&self) -> Result<(), String> {
        if let Self::Directory { path } = self {
            if path.as_os_str().is_empty() {
                return Err("automatic reuse directory must not be empty".into());
            }
        }
        Ok(())
    }

    /// Resolves a path without opening storage, creating directories or
    /// claiming cache ownership. The consumer must still enforce permissions,
    /// locking, global retention and clean-manifest validation before reuse.
    ///
    /// Linux: absolute XDG_CACHE_HOME, otherwise HOME/.cache.
    /// macOS: HOME/Library/Caches. Windows: LOCALAPPDATA.
    /// Every platform appends ferrum/automatic-calibration-v1.
    pub fn resolve_directory(
        &self,
    ) -> Result<PathBuf, SloAutomaticCalibrationCacheLocationErrorV1> {
        match self {
            Self::Directory { path } if path.is_absolute() => Ok(path.clone()),
            Self::Directory { .. } => {
                Err(SloAutomaticCalibrationCacheLocationErrorV1::UnresolvedDirectory)
            }
            Self::PlatformDefault {} => platform_default(),
        }
    }
}

#[derive(Clone, Copy)]
enum Platform {
    Linux,
    MacOs,
    Windows,
    Unsupported,
}

#[cfg(target_os = "linux")]
fn platform_default() -> Result<PathBuf, SloAutomaticCalibrationCacheLocationErrorV1> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let cache = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from);
    resolve_platform(Platform::Linux, home.as_deref(), cache.as_deref())
}

#[cfg(target_os = "macos")]
fn platform_default() -> Result<PathBuf, SloAutomaticCalibrationCacheLocationErrorV1> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    resolve_platform(Platform::MacOs, home.as_deref(), None)
}

#[cfg(target_os = "windows")]
fn platform_default() -> Result<PathBuf, SloAutomaticCalibrationCacheLocationErrorV1> {
    let cache = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    resolve_platform(Platform::Windows, None, cache.as_deref())
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn platform_default() -> Result<PathBuf, SloAutomaticCalibrationCacheLocationErrorV1> {
    resolve_platform(Platform::Unsupported, None, None)
}

fn resolve_platform(
    platform: Platform,
    home: Option<&Path>,
    cache: Option<&Path>,
) -> Result<PathBuf, SloAutomaticCalibrationCacheLocationErrorV1> {
    let home = home.filter(|path| path.is_absolute());
    let cache = cache.filter(|path| path.is_absolute());
    let root = match platform {
        Platform::Linux => cache
            .map(Path::to_path_buf)
            .or_else(|| home.map(|p| p.join(".cache"))),
        Platform::MacOs => home.map(|p| p.join("Library").join("Caches")),
        Platform::Windows => cache.map(Path::to_path_buf),
        Platform::Unsupported => {
            return Err(SloAutomaticCalibrationCacheLocationErrorV1::UnsupportedPlatform)
        }
    }
    .ok_or(SloAutomaticCalibrationCacheLocationErrorV1::UserCacheUnavailable)?;
    Ok(root.join("ferrum").join("automatic-calibration-v1"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn absolute(path: &str) -> PathBuf {
        // Deterministic test inputs, without changing process-global variables
        // or requiring the named directory to exist.
        if cfg!(windows) {
            PathBuf::from("C:\\").join(path)
        } else {
            PathBuf::from("/").join(path)
        }
    }

    #[test]
    fn automatic_reuse_platform_locations_share_a_versioned_product_directory() {
        let home = absolute("users/alice");
        let xdg = absolute("cache/alice");
        let suffix = Path::new("ferrum").join("automatic-calibration-v1");
        assert_eq!(
            resolve_platform(Platform::Linux, Some(&home), Some(&xdg)).unwrap(),
            xdg.join(&suffix)
        );
        for invalid in [None, Some(Path::new("")), Some(Path::new("relative/cache"))] {
            assert_eq!(
                resolve_platform(Platform::Linux, Some(&home), invalid).unwrap(),
                home.join(".cache").join(&suffix)
            );
        }
        assert_eq!(
            resolve_platform(Platform::MacOs, Some(&home), Some(&xdg)).unwrap(),
            home.join("Library").join("Caches").join(&suffix)
        );
        assert_eq!(
            resolve_platform(Platform::Windows, Some(&home), Some(&xdg)).unwrap(),
            xdg.join(&suffix)
        );
    }

    #[test]
    fn automatic_reuse_location_failures_do_not_fall_back_to_working_or_temp_directories() {
        for platform in [Platform::Linux, Platform::MacOs, Platform::Windows] {
            for home in [None, Some(Path::new("")), Some(Path::new("relative/home"))] {
                assert_eq!(
                    resolve_platform(platform, home, Some(Path::new("relative/cache"))),
                    Err(SloAutomaticCalibrationCacheLocationErrorV1::UserCacheUnavailable)
                );
            }
        }
        assert_eq!(
            resolve_platform(Platform::Unsupported, Some(&absolute("home")), None),
            Err(SloAutomaticCalibrationCacheLocationErrorV1::UnsupportedPlatform)
        );
        let relative = SloAutomaticCalibrationCacheLocationV1::Directory {
            path: "cache".into(),
        };
        relative.validate().unwrap(); // File loader may resolve this later.
        assert_eq!(
            relative.resolve_directory(),
            Err(SloAutomaticCalibrationCacheLocationErrorV1::UnresolvedDirectory)
        );
        let path = absolute("explicit/cache");
        assert_eq!(
            SloAutomaticCalibrationCacheLocationV1::Directory { path: path.clone() }
                .resolve_directory()
                .unwrap(),
            path
        );
    }
}
