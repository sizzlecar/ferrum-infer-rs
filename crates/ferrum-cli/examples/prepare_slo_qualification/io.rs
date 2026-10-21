use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    path::Path,
};

pub(super) fn read_bounded(path: &Path, max_bytes: usize) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > max_bytes as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source is not a bounded regular file",
        ));
    }
    let bound = max_bytes
        .checked_add(1)
        .ok_or_else(|| io::Error::other("file byte limit overflow"))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(bound as u64).read_to_end(&mut bytes)?;
    if bytes.len() > max_bytes {
        return Err(io::Error::other("source grew past byte limit"));
    }
    Ok(bytes)
}

struct BoundedWriter<'a, W: Write> {
    inner: W,
    remaining: &'a mut usize,
    digest: Sha256,
}
impl<W: Write> Write for BoundedWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > *self.remaining {
            return Err(io::Error::other("total serialized output byte limit"));
        }
        let written = self.inner.write(bytes)?;
        *self.remaining -= written;
        self.digest.update(&bytes[..written]);
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
pub(super) fn write_json_bounded(
    path: &Path,
    value: &impl Serialize,
    remaining: &mut usize,
) -> anyhow::Result<String> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BoundedWriter {
        inner: file,
        remaining,
        digest: Sha256::new(),
    };
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.inner.sync_all()?;
    Ok(format!("{:x}", writer.digest.finalize()))
}

/// Canonical compact JSON hash without an intermediate serialized allocation.
pub(super) fn json_hash_bounded(
    value: &impl Serialize,
    max_bytes: usize,
) -> anyhow::Result<String> {
    let mut remaining = max_bytes;
    let mut writer = BoundedWriter {
        inner: io::sink(),
        remaining: &mut remaining,
        digest: Sha256::new(),
    };
    serde_json::to_writer(&mut writer, value)?;
    Ok(format!("{:x}", writer.digest.finalize()))
}

/// One OS operation; unlike exists()+rename, an existing empty directory or
/// symlink is never replaced. Staging must be on the destination filesystem.
pub(super) fn publish_new_directory(staging: &Path, destination: &Path) -> io::Result<()> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let source = CString::new(staging.as_os_str().as_bytes())?;
        let target = CString::new(destination.as_os_str().as_bytes())?;
        #[cfg(target_os = "linux")]
        // SAFETY: both pointers refer to live NUL-terminated path strings; the
        // kernel owns path resolution and the no-replace operation.
        let status = unsafe {
            libc::syscall(
                libc::SYS_renameat2,
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                target.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        #[cfg(target_os = "macos")]
        // SAFETY: same path-lifetime contract as above; RENAME_EXCL is atomic.
        let status =
            unsafe { libc::renamex_np(source.as_ptr(), target.as_ptr(), libc::RENAME_EXCL) };
        if status == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (staging, destination);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "atomic no-replace directory publication is supported on Linux and macOS",
        ))
    }
}
