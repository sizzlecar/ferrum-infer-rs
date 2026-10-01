//! Cold OS identity capture; hot reads never do filesystem I/O or hashing.

use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;

pub(super) struct SystemCostClock {
    domain: CostMonotonicDomainV1,
    #[cfg(target_os = "linux")]
    _namespace: std::fs::File,
    #[cfg(target_os = "macos")]
    numerator: u32,
    #[cfg(target_os = "macos")]
    denominator: u32,
}

impl SystemCostClock {
    pub(super) fn domain(&self) -> &CostMonotonicDomainV1 {
        &self.domain
    }

    #[cfg(target_os = "linux")]
    pub(super) fn capture() -> Option<Self> {
        use std::{num::NonZeroU64, os::unix::fs::MetadataExt};

        // Linux exposes timens_offsets in /proc/<tid>, but not in the
        // /proc/<tgid>/task/<tid> directory used by thread-self. Resolve the
        // calling thread in this procfs mount's PID namespace, which need not
        // be the namespace used by gettid(). /proc/self would name the leader.
        let thread = linux_thread_proc_root(&std::fs::read_link("/proc/thread-self").ok()?)?;
        // thread-self describes the calling clock producer, not a different
        // task's namespace. Retain the handle to prevent live inode reuse.
        let namespace = std::fs::File::open("/proc/thread-self/ns/time").ok()?;
        let metadata = namespace.metadata().ok()?;
        let identity = (metadata.dev(), metadata.ino());
        let children = std::fs::metadata("/proc/thread-self/ns/time_for_children").ok()?;
        if identity != (children.dev(), children.ino()) {
            // timens_offsets describes time_for_children. Do not bind those
            // offsets to a parent that has not entered that namespace.
            return None;
        }
        let boot = parse_uuid(read_bounded::<64>("/proc/sys/kernel/random/boot_id")?.as_slice())?;
        let thread_time = thread.join("ns/time");
        let thread_children = thread.join("ns/time_for_children");
        for path in [&thread_time, &thread_children] {
            let before = std::fs::metadata(path).ok()?;
            if identity != (before.dev(), before.ino()) {
                return None;
            }
        }
        let offsets = read_bounded::<256>(thread.join("timens_offsets").to_str()?)?;
        let (seconds, nanoseconds) = parse_boottime_offset(offsets.as_slice())?;
        // Fail closed if cold capture crossed a namespace transition. Ferrum
        // never changes time namespace while its cost clock is live.
        for path in [
            std::path::Path::new("/proc/thread-self/ns/time"),
            std::path::Path::new("/proc/thread-self/ns/time_for_children"),
            thread_time.as_path(),
            thread_children.as_path(),
        ] {
            let after = std::fs::metadata(path).ok()?;
            if identity != (after.dev(), after.ino()) {
                return None;
            }
        }
        let value = Self {
            domain: CostMonotonicDomainV1::new_linux_boottime(
                boot,
                identity.0,
                NonZeroU64::new(identity.1)?,
                seconds,
                nanoseconds,
            )
            .ok()?,
            _namespace: namespace,
        };
        value.now_ns()?;
        Some(value)
    }

    #[cfg(target_os = "linux")]
    pub(super) fn now_ns(&self) -> Option<u64> {
        let mut value = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: clock_gettime writes exactly one valid timespec. BOOTTIME
        // includes suspend; unlike REALTIME it cannot be wall-clock adjusted.
        if unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut value) } != 0 {
            return None;
        }
        timespec_ns(
            value.tv_sec.try_into().ok()?,
            value.tv_nsec.try_into().ok()?,
        )
    }

    #[cfg(target_os = "macos")]
    pub(super) fn capture() -> Option<Self> {
        let mut bytes = [0_u8; 64];
        let mut length = bytes.len();
        // SAFETY: fixed NUL-terminated query, writable bounded output and exact
        // capacity; no sysctl mutation or allocation from an OS size reply.
        let status = unsafe {
            libc::sysctlbyname(
                c"kern.bootsessionuuid".as_ptr(),
                bytes.as_mut_ptr().cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 || length > bytes.len() {
            return None;
        }
        let boot = parse_uuid(bytes[..length].strip_suffix(&[0])?)?;
        let mut scale = libc::mach_timebase_info { numer: 0, denom: 0 };
        // SAFETY: the pointer refers to the complete writable ABI structure.
        if unsafe { libc::mach_timebase_info(&mut scale) } != 0
            || scale.numer == 0
            || scale.denom == 0
        {
            return None;
        }
        let value = Self {
            domain: CostMonotonicDomainV1::new_macos_continuous(boot).ok()?,
            numerator: scale.numer,
            denominator: scale.denom,
        };
        value.now_ns()?;
        Some(value)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn now_ns(&self) -> Option<u64> {
        // SAFETY: this zero-argument OS read returns unsigned continuous ticks
        // since boot, including sleep. It has the mach_timebase_info scale.
        ticks_ns(
            unsafe { mach_continuous_time() },
            self.numerator,
            self.denominator,
        )
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub(super) fn capture() -> Option<Self> {
        None
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub(super) fn now_ns(&self) -> Option<u64> {
        None
    }
}

#[cfg(any(target_os = "linux", test))]
fn linux_thread_proc_root(thread_self: &std::path::Path) -> Option<std::path::PathBuf> {
    // fs/proc/thread_self.c emits <tgid>/task/<tid> in the procfs mount's
    // namespace. Accept only that kernel form; never normalize traversal or
    // substitute the thread-group leader for a missing calling-thread ID.
    fn pid(value: &str) -> Option<&str> {
        if value.is_empty()
            || value.len() > 10
            || value.starts_with('0')
            || !value.bytes().all(|byte| byte.is_ascii_digit())
            || value.parse::<i32>().ok()? <= 0
        {
            return None;
        }
        Some(value)
    }
    let mut parts = thread_self.to_str()?.split('/');
    pid(parts.next()?)?;
    if parts.next()? != "task" {
        return None;
    }
    let tid = pid(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some(std::path::Path::new("/proc").join(tid))
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    // Declared in mach/mach_time.h (macOS 10.12+); libc currently omits this
    // symbol. libSystem is already linked by std/libc.
    fn mach_continuous_time() -> u64;
}

#[cfg(any(target_os = "linux", test))]
fn timespec_ns(seconds: i64, nanoseconds: i64) -> Option<u64> {
    if !(0..1_000_000_000).contains(&nanoseconds) {
        return None;
    }
    u64::try_from(seconds)
        .ok()?
        .checked_mul(1_000_000_000)?
        .checked_add(u64::try_from(nanoseconds).ok()?)
}

#[cfg(any(target_os = "macos", test))]
fn ticks_ns(ticks: u64, numerator: u32, denominator: u32) -> Option<u64> {
    if numerator == 0 || denominator == 0 {
        return None;
    }
    let ns = u128::from(ticks)
        .checked_mul(u128::from(numerator))?
        .checked_div(u128::from(denominator))?;
    u64::try_from(ns).ok()
}

#[cfg(any(target_os = "linux", target_os = "macos", test))]
fn parse_uuid(raw: &[u8]) -> Option<[u8; 16]> {
    let raw = raw.strip_suffix(b"\n").unwrap_or(raw);
    if raw.len() != 36 {
        return None;
    }
    let mut value = [0_u8; 16];
    let mut nibbles = 0;
    for (index, byte) in raw.iter().copied().enumerate() {
        if matches!(index, 8 | 13 | 18 | 23) {
            if byte != b'-' {
                return None;
            }
        } else {
            let digit = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => return None,
            };
            value[nibbles / 2] = (value[nibbles / 2] << 4) | digit;
            nibbles += 1;
        }
    }
    (value != [0; 16]).then_some(value)
}

#[cfg(target_os = "linux")]
struct BoundedRead<const N: usize> {
    bytes: [u8; N],
    length: usize,
}
#[cfg(target_os = "linux")]
impl<const N: usize> BoundedRead<N> {
    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.length]
    }
}
#[cfg(target_os = "linux")]
fn read_bounded<const N: usize>(path: &str) -> Option<BoundedRead<N>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut result = BoundedRead {
        bytes: [0; N],
        length: 0,
    };
    loop {
        if result.length == N {
            let mut extra = [0_u8; 1];
            return (file.read(&mut extra).ok()? == 0).then_some(result);
        }
        let count = file.read(&mut result.bytes[result.length..]).ok()?;
        if count == 0 {
            return Some(result);
        }
        result.length += count;
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_boottime_offset(bytes: &[u8]) -> Option<(i64, u32)> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut monotonic = None;
    let mut boottime = None;
    for line in text.lines() {
        let mut fields = line.split_ascii_whitespace();
        let slot = match fields.next()? {
            "monotonic" => &mut monotonic,
            "boottime" => &mut boottime,
            _ => return None,
        };
        let seconds = fields.next()?.parse::<i64>().ok()?;
        let nanoseconds = fields.next()?.parse::<u32>().ok()?;
        if nanoseconds >= 1_000_000_000 || fields.next().is_some() || slot.is_some() {
            return None;
        }
        *slot = Some((seconds, nanoseconds));
    }
    monotonic?;
    boottime
}

#[cfg(test)]
mod tests;
