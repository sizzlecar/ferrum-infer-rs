use super::*;

#[test]
fn system_cost_clock_procfs_thread_path_preserves_the_calling_thread() {
    use std::path::{Path, PathBuf};

    assert_eq!(
        linux_thread_proc_root(Path::new("123/task/123")),
        Some(PathBuf::from("/proc/123"))
    );
    assert_eq!(
        linux_thread_proc_root(Path::new("123/task/456")),
        Some(PathBuf::from("/proc/456"))
    );
    assert_eq!(
        linux_thread_proc_root(Path::new("2147483647/task/1")),
        Some(PathBuf::from("/proc/1"))
    );
    for raw in [
        "",
        "123",
        "123/task",
        "/123/task/456",
        "123/task/456/",
        "123//task/456",
        "123/./task/456",
        "123/task/../456",
        "123/tasks/456",
        "0/task/456",
        "123/task/0",
        "0123/task/456",
        "123/task/0456",
        "123/task/+456",
        "123/task/-456",
        "123/task/2147483648",
        "2147483648/task/456",
        "123/task/9999999999999999999999999",
        "123/task/self",
    ] {
        assert_eq!(linux_thread_proc_root(Path::new(raw)), None, "{raw}");
    }
}

#[test]
fn system_cost_clock_timespec_checks_sign_subsecond_and_overflow() {
    assert_eq!(timespec_ns(0, 0), Some(0));
    assert_eq!(timespec_ns(1, 999_999_999), Some(1_999_999_999));
    assert_eq!(timespec_ns(-1, 0), None);
    assert_eq!(timespec_ns(1, -1), None);
    assert_eq!(timespec_ns(1, 1_000_000_000), None);
    let whole = (u64::MAX / 1_000_000_000) as i64;
    let rest = (u64::MAX % 1_000_000_000) as i64;
    assert_eq!(timespec_ns(whole, rest), Some(u64::MAX));
    assert_eq!(timespec_ns(whole, rest + 1), None);
    assert_eq!(timespec_ns(whole + 1, 0), None);
}

#[test]
fn system_cost_clock_tick_conversion_preserves_large_inputs_and_rejects_overflow() {
    assert_eq!(ticks_ns(0, 1, 1), Some(0));
    assert_eq!(ticks_ns(7, 125, 3), Some(291));
    assert_eq!(ticks_ns(u64::MAX, 125, 125), Some(u64::MAX));
    assert_eq!(ticks_ns(u64::MAX, 2, 1), None);
    assert_eq!(ticks_ns(1, 0, 1), None);
    assert_eq!(ticks_ns(1, 1, 0), None);
    // Both active time and suspend time are OS continuous ticks. Conversion
    // keeps the elapsed interval; it never re-anchors at process creation.
    let before = ticks_ns(400, 125, 3).unwrap();
    let after = ticks_ns(400 + 6_000, 125, 3).unwrap();
    assert_eq!(after - before, 250_000);
}

#[test]
fn system_cost_clock_boot_identity_and_namespace_parser_fail_closed() {
    let uuid = b"01234567-89ab-cdef-0123-456789abcdef";
    assert_eq!(
        parse_uuid(uuid),
        parse_uuid(b"01234567-89AB-CDEF-0123-456789ABCDEF\n")
    );
    assert!(parse_uuid(uuid).is_some());
    for raw in [
        &b"00000000-0000-0000-0000-000000000000"[..],
        &b"01234567-89ab-cdef-0123-456789abcdeg"[..],
        &b"0123456789abcdef0123456789abcdef"[..],
        &b"01234567-89ab-cdef-0123-456789abcdef\n\n"[..],
    ] {
        assert_eq!(parse_uuid(raw), None);
    }
    assert_eq!(
        parse_boottime_offset(b"monotonic 0 0\nboottime -2 999999999\n"),
        Some((-2, 999_999_999))
    );
    for raw in [
        &b"monotonic 0 0\n"[..],
        &b"boottime 0 0\n"[..],
        &b"monotonic 0 0\nboottime 0 1000000000\n"[..],
        &b"monotonic 0 0\nboottime 0 -1\n"[..],
        &b"monotonic 0 0\nboottime 0 0\nboottime 0 0\n"[..],
        &b"monotonic 0 0\nboottime 0 0 trailing\n"[..],
    ] {
        assert_eq!(parse_boottime_offset(raw), None);
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
#[ignore = "requires readable OS boot identity and clock namespace; run explicitly on the target host"]
fn system_cost_clock_platform_instances_share_origin_and_keep_elapsed_age() {
    let first =
        SystemCostClock::capture().expect("supported OS boot/namespace identity unavailable");
    let observed = first.now_ns().expect("read first OS continuous clock");
    let second = SystemCostClock::capture().expect("read independent OS clock instance");
    let current = second.now_ns().expect("read second OS continuous clock");
    assert_eq!(first.domain(), second.domain());
    assert_eq!(first.domain().sha256(), second.domain().sha256());
    assert!(current >= observed);
    let later = first.now_ns().unwrap();
    assert!(later >= current);
    assert!(later.checked_sub(observed).unwrap() >= current - observed);
    #[cfg(target_os = "linux")]
    assert_eq!(
        first.domain().clock_kind(),
        ferrum_interfaces::execution_cost::CostMonotonicClockKindV1::LinuxBoottimeV1
    );
    #[cfg(target_os = "macos")]
    assert_eq!(
        first.domain().clock_kind(),
        ferrum_interfaces::execution_cost::CostMonotonicClockKindV1::MacOsContinuousV1
    );
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires readable Linux procfs boot/time namespace identity; run explicitly on the target host"]
fn system_cost_clock_linux_worker_uses_calling_thread_and_preserves_origin() {
    use std::os::unix::fs::MetadataExt;

    let first = SystemCostClock::capture().expect("capture parent OS clock");
    let before = first.now_ns().unwrap();
    let (worker_domain, worker_now) = std::thread::spawn(|| {
        let target = std::fs::read_link("/proc/thread-self").unwrap();
        let target = target.to_str().unwrap();
        let mut parts = target.split('/');
        let tgid = parts.next().unwrap();
        assert_eq!(parts.next(), Some("task"));
        let tid = parts.next().unwrap();
        assert_ne!(tgid, tid, "test must exercise a non-leader thread");

        let root = linux_thread_proc_root(std::path::Path::new(target)).unwrap();
        let current = std::fs::metadata("/proc/thread-self/ns/time").unwrap();
        for path in [root.join("ns/time"), root.join("ns/time_for_children")] {
            let actual = std::fs::metadata(path).unwrap();
            assert_eq!((current.dev(), current.ino()), (actual.dev(), actual.ino()));
        }
        let offsets = read_bounded::<256>(root.join("timens_offsets").to_str().unwrap())
            .expect("read calling thread offsets via top-level procfs task directory");
        assert!(parse_boottime_offset(offsets.as_slice()).is_some());
        let worker = SystemCostClock::capture().expect("capture non-leader OS clock");
        (worker.domain().clone(), worker.now_ns().unwrap())
    })
    .join()
    .unwrap();
    let after = first.now_ns().unwrap();
    assert_eq!(first.domain(), &worker_domain);
    assert_eq!(first.domain().sha256(), worker_domain.sha256());
    assert!(before <= worker_now && worker_now <= after);
}
