use super::CpuRuntimeError;

/// Hardware/process capacity for the existing typed runtime memory policy.
/// Provider allocations are separately charged against its admitted ceiling.
pub(crate) fn host_memory_capacity() -> Result<u64, CpuRuntimeError> {
    let mut bytes = platform_capacity()?;
    #[cfg(unix)]
    {
        let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
        // SAFETY: getrlimit initializes the complete output on success.
        if unsafe { libc::getrlimit(libc::RLIMIT_AS, limit.as_mut_ptr()) } != 0 {
            return Err(CpuRuntimeError::new(format!(
                "read CPU address-space limit: {}",
                std::io::Error::last_os_error()
            )));
        }
        let limit = unsafe { limit.assume_init() };
        if limit.rlim_cur != libc::RLIM_INFINITY {
            bytes = bytes.min(limit.rlim_cur as u64);
        }
    }
    #[cfg(target_os = "linux")]
    {
        bytes = linux_cgroup_capacity(bytes)?;
    }
    if bytes == 0 || bytes > usize::MAX as u64 {
        return Err(CpuRuntimeError::new(
            "CPU memory capacity is zero or exceeds the process address space",
        ));
    }
    Ok(bytes)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn platform_capacity() -> Result<u64, CpuRuntimeError> {
    let mut bytes = 0_u64;
    let mut length = std::mem::size_of_val(&bytes);
    // SAFETY: the output points to writable storage of the advertised length;
    // a null new-value pointer makes this a read-only sysctl query.
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 || length != std::mem::size_of_val(&bytes) {
        return Err(CpuRuntimeError::new(format!(
            "read CPU physical memory: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(bytes)
}

#[cfg(target_os = "linux")]
fn platform_capacity() -> Result<u64, CpuRuntimeError> {
    // SAFETY: these read-only sysconf keys require no pointers.
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page_bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages <= 0 || page_bytes <= 0 {
        return Err(CpuRuntimeError::new(
            "read CPU physical page capacity failed",
        ));
    }
    (pages as u64)
        .checked_mul(page_bytes as u64)
        .ok_or_else(|| CpuRuntimeError::new("CPU physical page capacity overflows"))
}

#[cfg(target_os = "windows")]
fn platform_capacity() -> Result<u64, CpuRuntimeError> {
    // Native ABI: https://learn.microsoft.com/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }
    let mut status: MemoryStatusEx = unsafe { std::mem::zeroed() };
    status.length = std::mem::size_of::<MemoryStatusEx>() as u32;
    // SAFETY: status has the documented C layout and its initialized dwLength.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err(CpuRuntimeError::new(format!(
            "read CPU physical memory: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(status
        .total_physical
        .min(status.total_virtual)
        .min(status.total_page_file))
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "windows"
)))]
fn platform_capacity() -> Result<u64, CpuRuntimeError> {
    Err(CpuRuntimeError::new(
        "CPU memory discovery is unavailable on this operating system",
    ))
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_limit(value: &str) -> Result<Option<u64>, CpuRuntimeError> {
    if value.trim() == "max" {
        return Ok(None);
    }
    value.trim().parse::<u64>().map(Some).map_err(|_| {
        CpuRuntimeError::new("Linux cgroup memory limit is not an unsigned byte count or max")
    })
}

#[cfg(target_os = "linux")]
fn linux_cgroup_capacity(mut bytes: u64) -> Result<u64, CpuRuntimeError> {
    use std::path::{Component, Path};
    let membership = std::fs::read_to_string("/proc/self/cgroup")
        .map_err(|error| CpuRuntimeError::new(format!("read CPU cgroup membership: {error}")))?;
    for row in membership.lines() {
        let mut columns = row.splitn(3, ':');
        let _hierarchy = columns.next();
        let controllers = columns.next().unwrap_or("");
        let Some(group) = columns.next() else {
            continue;
        };
        let (root, filename) = if controllers.is_empty() {
            (Path::new("/sys/fs/cgroup"), "memory.max")
        } else if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            (Path::new("/sys/fs/cgroup/memory"), "memory.limit_in_bytes")
        } else {
            continue;
        };
        let relative = Path::new(group.trim_start_matches('/'));
        // Some container namespaces expose an outer membership path. Never
        // escape the visible cgroup mount; its root limit still applies.
        let mut directory = if relative
            .components()
            .all(|part| matches!(part, Component::Normal(_) | Component::CurDir))
        {
            root.join(relative)
        } else {
            root.to_path_buf()
        };
        loop {
            match std::fs::read_to_string(directory.join(filename)) {
                Ok(value) => {
                    if let Some(limit) = parse_cgroup_limit(&value)? {
                        bytes = bytes.min(limit);
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(CpuRuntimeError::new(format!(
                        "read CPU cgroup memory ceiling: {error}"
                    )))
                }
            }
            if directory == root || !directory.pop() {
                break;
            }
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_capacity_is_a_real_addressable_ceiling() {
        let bytes = host_memory_capacity().unwrap();
        assert!(bytes > 0 && bytes <= usize::MAX as u64);
        assert!(bytes <= platform_capacity().unwrap());
    }
    #[test]
    fn cgroup_limit_parsing_preserves_zero_and_rejects_invalid_limits() {
        assert_eq!(parse_cgroup_limit("max\n").unwrap(), None);
        assert_eq!(parse_cgroup_limit("0").unwrap(), Some(0));
        assert_eq!(parse_cgroup_limit(" 1073741824\n").unwrap(), Some(1 << 30));
        for value in ["-1", "1GB", "", "18446744073709551616"] {
            assert!(parse_cgroup_limit(value).is_err());
        }
    }
}
