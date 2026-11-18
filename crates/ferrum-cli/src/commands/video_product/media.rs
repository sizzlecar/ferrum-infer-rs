//! Resolve host executables once, keeping input decoding optional.
use anyhow::{ensure, Context, Result};
use ferrum_engine::video_generation::VideoReferenceKind;
use std::{
    env,
    ffi::OsStr,
    path::{Path, PathBuf},
};

fn executable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        CString::new(path.as_os_str().as_bytes()).is_ok_and(|name| {
            // access reads the NUL-terminated path only for this call.
            unsafe { libc::access(name.as_ptr(), libc::X_OK) == 0 }
        })
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn find(name: &str, search: Option<&OsStr>) -> Option<PathBuf> {
    env::split_paths(search?)
        .filter(|p| p.is_absolute())
        .map(|p| p.join(name))
        .find(|p| executable(p))
}

const FFMPEG: &str = if cfg!(windows) {
    "ffmpeg.exe"
} else {
    "ffmpeg"
};
const FFPROBE: &str = if cfg!(windows) {
    "ffprobe.exe"
} else {
    "ffprobe"
};

pub(super) fn output_ffmpeg(explicit: Option<&Path>, search: Option<&OsStr>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        ensure!(path.is_absolute(), "output.ffmpeg must be an absolute path");
        ensure!(
            executable(path),
            "output.ffmpeg is not executable: {}",
            path.display()
        );
        return Ok(path.to_owned());
    }
    find(FFMPEG, search).context("MP4 output requires FFmpeg with libx264 and AAC; install it on PATH or set output.ffmpeg to its absolute executable path")
}

pub(super) fn reference_tools(
    output_ffmpeg: &Path,
    input_ffmpeg: Option<&Path>,
    input_ffprobe: Option<&Path>,
    search: Option<&OsStr>,
) -> Result<(PathBuf, PathBuf)> {
    let ffmpeg = input_ffmpeg.unwrap_or(output_ffmpeg);
    ensure!(
        ffmpeg.is_absolute(),
        "input.ffmpeg must be an absolute path"
    );
    let ffprobe = if let Some(path) = input_ffprobe {
        ensure!(path.is_absolute(), "input.ffprobe must be an absolute path");
        path.to_owned()
    } else {
        // Some hosts have a standalone output encoder and the input tools in
        // another directory. Preserve the original sibling preference.
        let sibling = ffmpeg.with_file_name(FFPROBE);
        if executable(&sibling) {
            sibling
        } else {
            // Keep the conventional path for a later video-reference request.
            // Images and WAVE don't invoke either input tool.
            find(FFPROBE, search).unwrap_or(sibling)
        }
    };
    Ok((ffmpeg.to_owned(), ffprobe))
}

/// CLI reference kinds are known before loading weights. A server discovers
/// each reference at request time, where the decoder reports spawn failures.
pub(super) fn check_run_input(
    mut kinds: impl Iterator<Item = VideoReferenceKind>,
    ffmpeg: &Path,
    ffprobe: &Path,
) -> Result<()> {
    if kinds.any(|kind| kind == VideoReferenceKind::Video) {
        ensure!(
            executable(ffmpeg),
            "video reference requires input.ffmpeg to be executable: {}",
            ffmpeg.display()
        );
        ensure!(executable(ffprobe), "video reference requires ffprobe; install it beside input FFmpeg or on PATH, or set input.ffprobe ({})", ffprobe.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installed(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::File::create(&path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        path
    }

    #[test]
    fn output_finds_executable_and_respects_explicit_choice() {
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("first");
        let other = root.path().join("other");
        std::fs::create_dir_all(first.join(FFMPEG)).unwrap();
        std::fs::create_dir(&other).unwrap();
        let working = installed(&other, FFMPEG);
        let search = env::join_paths([first.as_path(), other.as_path()]).unwrap();
        assert_eq!(output_ffmpeg(None, Some(&search)).unwrap(), working);
        let override_path = installed(&first, "output-encoder");
        assert_eq!(
            output_ffmpeg(Some(&override_path), Some(&search)).unwrap(),
            override_path
        );
        assert!(output_ffmpeg(Some(&first.join("missing")), Some(&search)).is_err());
        assert!(output_ffmpeg(Some(Path::new(FFMPEG)), Some(&search)).is_err());
        let error = output_ffmpeg(None, Some(first.as_os_str())).unwrap_err();
        assert!(error.to_string().contains("output.ffmpeg"));
    }

    #[test]
    fn reference_probe_is_optional_and_prefers_the_selected_decoder_sibling() {
        let root = tempfile::tempdir().unwrap();
        let encoder = installed(root.path(), "output-encoder");
        let decode_dir = tempfile::tempdir().unwrap();
        let decoder = installed(decode_dir.path(), FFMPEG);
        let empty_path = Some(root.path().as_os_str());
        let (input, missing) = reference_tools(&encoder, Some(&decoder), None, empty_path).unwrap();
        assert_eq!(input, decoder);
        assert!(!missing.exists());
        check_run_input(
            [VideoReferenceKind::Image, VideoReferenceKind::Audio].into_iter(),
            &input,
            &missing,
        )
        .unwrap();
        let error =
            check_run_input([VideoReferenceKind::Video].into_iter(), &input, &missing).unwrap_err();
        assert!(error.to_string().contains("input.ffprobe"));
        let system = installed(root.path(), FFPROBE);
        let (_, fallback) = reference_tools(&encoder, Some(&decoder), None, empty_path).unwrap();
        assert_eq!(fallback, system);
        let sibling = installed(decode_dir.path(), FFPROBE);
        let (_, preferred) = reference_tools(&encoder, Some(&decoder), None, empty_path).unwrap();
        assert_eq!(preferred, sibling);
        check_run_input([VideoReferenceKind::Video].into_iter(), &input, &preferred).unwrap();
        let explicit = root.path().join("custom-probe");
        let (_, selected) =
            reference_tools(&encoder, Some(&decoder), Some(&explicit), empty_path).unwrap();
        assert_eq!(selected, explicit);
        assert!(reference_tools(&encoder, None, Some(Path::new(FFPROBE)), empty_path).is_err());
    }
}
