use super::CpuRuntimeError;

/// Remaining host memory, including allocations by this and other processes.
/// Physical availability, address-space headroom and cgroup headroom are
/// independent ceilings: combine them with min, never subtract usage twice.
pub(crate) fn host_memory_available() -> Result<u64, CpuRuntimeError> {
    let bytes = platform_available()?;
    #[cfg(unix)]
    let bytes = match super::address_space_limit()? {
        Some(limit) => bytes.min(limit.saturating_sub(process_virtual_bytes()?)),
        None => bytes,
    };
    #[cfg(target_os = "linux")]
    let bytes = super::linux_cgroup_bound(bytes, true)?;
    Ok(bytes)
}

#[cfg(target_os = "linux")]
fn platform_available() -> Result<u64, CpuRuntimeError> {
    let memory = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| CpuRuntimeError::new(format!("read host available memory: {error}")))?;
    parse_proc_kib(&memory, "MemAvailable")
}

#[cfg(any(target_os = "linux", test))]
fn parse_proc_kib(contents: &str, field: &str) -> Result<u64, CpuRuntimeError> {
    let value = contents
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find_map(|(name, value)| (name == field).then_some(value))
        .ok_or_else(|| {
            CpuRuntimeError::new(format!("missing {field} in proc memory observation"))
        })?;
    let mut parts = value.split_whitespace();
    let bytes = parts
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|value| value.checked_mul(1024));
    if parts.next() != Some("kB") || parts.next().is_some() {
        return Err(CpuRuntimeError::new(format!("invalid {field} memory unit")));
    }
    bytes.ok_or_else(|| CpuRuntimeError::new(format!("invalid or overflowing {field} byte count")))
}

#[cfg(target_os = "linux")]
fn process_virtual_bytes() -> Result<u64, CpuRuntimeError> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|error| CpuRuntimeError::new(format!("read process virtual memory: {error}")))?;
    parse_proc_kib(&status, "VmSize")
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(deprecated)] // libc exposes the native Mach ABI used by these queries.
fn platform_available() -> Result<u64, CpuRuntimeError> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    // Zeroing also initializes fields newer than the running kernel; only the
    // long-established leading page counts are consumed below.
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: host_statistics64 writes at most count integer words to stats.
    // Release the send right returned by mach_host_self after the query.
    let result = unsafe {
        let host = libc::mach_host_self();
        let result = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut stats as *mut libc::vm_statistics64).cast(),
            &mut count,
        );
        mach_port_deallocate(libc::mach_task_self(), host);
        result
    };
    if result != libc::KERN_SUCCESS || count < 3 {
        return Err(CpuRuntimeError::new(format!(
            "read host free/inactive memory pages failed: {result}"
        )));
    }
    // SAFETY: this read-only sysconf key requires no pointer.
    let page_bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_bytes <= 0 {
        return Err(CpuRuntimeError::new("read host page size failed"));
    }
    // Free + reclaimable inactive pages is a snapshot estimate, not swap or a
    // promise of future allocation success. Speculative pages are already
    // included in free_count, and must not be added again.
    (u64::from(stats.free_count) + u64::from(stats.inactive_count))
        .checked_mul(page_bytes as u64)
        .ok_or_else(|| CpuRuntimeError::new("host available page count overflows"))
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(deprecated)]
fn process_virtual_bytes() -> Result<u64, CpuRuntimeError> {
    let mut info: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    // SAFETY: task_info receives the documented layout and size for this task.
    let result = unsafe {
        libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            (&mut info as *mut libc::mach_task_basic_info).cast(),
            &mut count,
        )
    };
    if result != libc::KERN_SUCCESS || count < libc::MACH_TASK_BASIC_INFO_COUNT {
        return Err(CpuRuntimeError::new(format!(
            "read process virtual memory failed: {result}"
        )));
    }
    Ok(info.virtual_size)
}

#[cfg(target_os = "windows")]
fn platform_available() -> Result<u64, CpuRuntimeError> {
    let status = super::windows_memory_status()?;
    Ok(status
        .available_physical
        .min(status.available_virtual)
        .min(status.available_page_file))
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "windows"
)))]
fn platform_available() -> Result<u64, CpuRuntimeError> {
    Err(CpuRuntimeError::new(
        "host available memory discovery is unavailable on this operating system",
    ))
}

#[cfg(all(
    unix,
    not(any(target_os = "macos", target_os = "ios", target_os = "linux"))
))]
fn process_virtual_bytes() -> Result<u64, CpuRuntimeError> {
    Err(CpuRuntimeError::new(
        "process virtual memory discovery is unavailable on this operating system",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_memory_uses_available_instead_of_total_or_free() {
        assert_eq!(
            parse_proc_kib(
                "MemTotal: 4096 kB\nMemFree: 12 kB\nMemAvailable: 123 kB\n",
                "MemAvailable"
            )
            .unwrap(),
            123 * 1024
        );
        assert_eq!(
            parse_proc_kib("MemAvailable: 0 kB", "MemAvailable").unwrap(),
            0
        );
        assert_eq!(
            parse_proc_kib("VmSize: 12 kB", "VmSize").unwrap(),
            12 * 1024
        );
    }

    #[test]
    fn missing_or_invalid_available_memory_does_not_fall_back_to_capacity() {
        for observation in [
            "MemTotal: 4096 kB",
            "MemAvailable: -1 kB",
            "MemAvailable: 1 MB",
            "MemAvailable: 1 kB junk",
            "MemAvailable: 18446744073709551615 kB",
        ] {
            assert!(parse_proc_kib(observation, "MemAvailable").is_err());
        }
    }
}
