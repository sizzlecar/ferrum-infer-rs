use super::*;
use clap::Parser;
use tempfile::TempDir;

fn arguments(root: &Path, pid: Option<u32>) -> Arguments {
    Arguments {
        pid,
        cgroup_directory: pid.is_none().then(|| root.join("cgroup")),
        output: root.join("samples.jsonl"),
        interval_ms: 200,
        duration_ms: 1_000,
        maximum_pids: 16,
        maximum_traversal_nodes: 64,
        maximum_input_bytes: 64 * 1024,
        stop_file: None,
    }
}

fn stat(pid: u32, start: u64) -> String {
    // Fields 3 through 22, including a deliberately awkward comm field.
    let mut fields = vec!["S".to_owned()];
    fields.extend((4..22).map(|_| "0".to_owned()));
    fields.push(start.to_string());
    format!("{pid} (worker ) with spaces) {}\n", fields.join(" "))
}

fn write_process(root: &Path, pid: u32, start: u64, rss_kib: u64) {
    let dir = root.join(pid.to_string());
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("stat"), stat(pid, start)).unwrap();
    fs::write(
        dir.join("status"),
        format!("Name:\tworker\nVmRSS:\t{rss_kib} kB\nVmHWM:\t999 kB\n"),
    )
    .unwrap();
}

fn write_members(path: &Path, members: &str) {
    fs::create_dir_all(path).unwrap();
    fs::write(path.join("cgroup.procs"), members).unwrap();
}

#[test]
fn explicit_target_arguments_and_new_output_are_required() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("output.jsonl");
    let base = [
        "sample_host_rss",
        "--output",
        path.to_str().unwrap(),
        "--duration-ms",
        "1000",
    ];
    assert!(Arguments::try_parse_from(base).is_err());
    let both = base
        .into_iter()
        .chain(["--pid", "42", "--cgroup-directory", "/a"]);
    assert!(Arguments::try_parse_from(both).is_err());
    let mut args = arguments(temp.path(), Some(42));
    args.validate().unwrap();
    args.interval_ms = 0;
    assert!(args.validate().is_err());
    args.interval_ms = 200;
    args.maximum_pids = 0;
    assert!(args.validate().is_err());
    let mut file = super::super::create_output(&path).unwrap();
    std::io::Write::write_all(&mut file, b"preserved").unwrap();
    assert_eq!(
        super::super::create_output(&path).unwrap_err().kind(),
        io::ErrorKind::AlreadyExists
    );
    assert_eq!(fs::read(&path).unwrap(), b"preserved");
}

#[test]
fn target_pid_samples_its_own_identity_and_stops_on_reuse_or_exit() {
    let temp = TempDir::new().unwrap();
    let proc_root = temp.path().join("proc");
    write_process(&proc_root, 42, 123, 7);
    let args = arguments(temp.path(), Some(42));
    let mut sampler = Sampler::new(&args, proc_root.clone()).unwrap();
    let first = sampler.sample(&args);
    assert_eq!(first.rss_sum_bytes, Some(7 * 1024));
    assert_eq!(first.processes[0].pid, 42);
    assert_eq!(first.processes[0].start_ticks, 123);
    assert_eq!(
        first.processes[0].lifetime_rss_high_water_bytes,
        Some(999 * 1024)
    );
    assert!(first.errors.is_empty());
    write_process(&proc_root, 42, 124, 8);
    let reused = sampler.sample(&args);
    assert_eq!(reused.stop_reason, Some("root_pid_reused"));
    assert_eq!(reused.rss_sum_bytes, None);
    fs::remove_dir_all(proc_root.join("42")).unwrap();
    let exited = sampler.sample(&args);
    assert_eq!(
        exited.stop_reason,
        Some("root_pid_unavailable_or_read_error")
    );
    assert_eq!(exited.rss_sum_bytes, None);
    assert!(!exited.errors.is_empty());
}

#[test]
fn recursive_members_are_deduplicated_and_read_errors_remain_incomplete() {
    let temp = TempDir::new().unwrap();
    let args = arguments(temp.path(), None);
    let cgroup = args.cgroup_directory.as_ref().unwrap();
    let proc_root = temp.path().join("proc");
    write_members(cgroup, "42\n");
    write_members(&cgroup.join("worker"), "43\n42\n");
    write_process(&proc_root, 42, 100, 7);
    write_process(&proc_root, 43, 200, 13);
    let mut sampler = Sampler::new(&args, proc_root.clone()).unwrap();
    let complete = sampler.sample(&args);
    assert_eq!(complete.members_discovered, Some(2));
    assert_eq!(complete.members_sampled, 2);
    assert_eq!(complete.rss_sum_bytes, Some(20 * 1024));
    assert_eq!(complete.processes[1].start_ticks, 200);
    fs::remove_file(proc_root.join("43/status")).unwrap();
    let partial = sampler.sample(&args);
    assert_eq!(partial.rss_sum_bytes, None);
    assert_eq!(partial.partial_rss_sum_bytes, 7 * 1024);
    assert_eq!(partial.members_sampled, 1);
    assert_eq!(partial.errors[0].pid, Some(43));
    assert!(partial.stop_reason.is_none());
    fs::rename(cgroup, temp.path().join("old_cgroup")).unwrap();
    write_members(cgroup, "42\n");
    let replaced = sampler.sample(&args);
    assert_eq!(replaced.stop_reason, Some("root_cgroup_replaced"));
    assert_eq!(replaced.rss_sum_bytes, None);
}

#[test]
fn empty_or_disappeared_cgroup_is_not_a_successful_zero_sample() {
    let temp = TempDir::new().unwrap();
    let args = arguments(temp.path(), None);
    let cgroup = args.cgroup_directory.as_ref().unwrap();
    write_members(cgroup, "");
    let mut sampler = Sampler::new(&args, temp.path().join("proc")).unwrap();
    let empty = sampler.sample(&args);
    assert_eq!(empty.rss_sum_bytes, None);
    assert_eq!(empty.stop_reason, Some("target_cgroup_empty"));
    fs::remove_dir_all(cgroup).unwrap();
    let gone = sampler.sample(&args);
    assert_eq!(gone.rss_sum_bytes, None);
    assert_eq!(gone.stop_reason, Some("root_cgroup_unavailable"));
}

#[test]
fn pid_traversal_and_read_limits_fail_closed_without_reading_past_byte_budget() {
    let temp = TempDir::new().unwrap();
    let mut args = arguments(temp.path(), None);
    let cgroup = args.cgroup_directory.as_ref().unwrap().clone();
    let proc_root = temp.path().join("proc");
    write_members(&cgroup, "42\n43\n");
    write_process(&proc_root, 42, 100, 7);
    write_process(&proc_root, 43, 200, 13);
    args.maximum_pids = 1;
    let mut sampler = Sampler::new(&args, proc_root.clone()).unwrap();
    let limited = sampler.sample(&args);
    assert_eq!(limited.rss_sum_bytes, None);
    assert!(limited.errors[0].message.contains("PID limit"));
    args.maximum_pids = 16;
    args.maximum_traversal_nodes = 1;
    let limited = sampler.sample(&args);
    assert_eq!(limited.rss_sum_bytes, None);
    assert!(limited.errors[0].message.contains("traversal limit"));
    args.maximum_traversal_nodes = 64;
    args.maximum_input_bytes = 5;
    let limited = sampler.sample(&args);
    assert_eq!(limited.rss_sum_bytes, None);
    assert_eq!(limited.input_bytes_read, 5);
    assert!(limited.errors[0].message.contains("byte limit"));
    args.maximum_input_bytes = 64 * 1024;
    std::os::unix::fs::symlink(temp.path().join("elsewhere"), cgroup.join("escape")).unwrap();
    let linked = sampler.sample(&args);
    assert_eq!(linked.rss_sum_bytes, None);
    assert!(linked.errors[0].message.contains("symlink"));
}

#[test]
fn proc_fields_are_checked_not_assumed_to_be_current_process_or_bytes() {
    assert_eq!(start_ticks(&stat(42, 900), 42).unwrap(), 900);
    assert!(start_ticks(&stat(43, 900), 42).is_err());
    assert_eq!(memory_bytes("VmRSS:\t0 kB\n", "VmRSS:").unwrap(), Some(0));
    assert!(memory_bytes("VmRSS: 3 bytes\n", "VmRSS:").is_err());
    assert!(memory_bytes("VmRSS: 3 kB\nVmRSS: 4 kB\n", "VmRSS:").is_err());
    assert!(memory_bytes("VmRSS: 18446744073709551615 kB\n", "VmRSS:").is_err());
}
