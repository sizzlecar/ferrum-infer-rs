use super::Arguments;
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    io::{self, Read},
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

struct ReadBudget {
    remaining: usize,
}
impl ReadBudget {
    fn read(&mut self, path: &Path) -> io::Result<String> {
        let mut file = fs::File::open(path)?;
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        loop {
            // Conservatively reject exact exhaustion rather than reading an
            // extra byte to probe EOF outside the declared aggregate bound.
            if self.remaining == 0 {
                return Err(invalid("sample input byte limit exhausted"));
            }
            let limit = chunk.len().min(self.remaining);
            let n = file.read(&mut chunk[..limit])?;
            if n == 0 {
                break;
            }
            self.remaining -= n;
            bytes.try_reserve_exact(n).map_err(io::Error::other)?;
            bytes.extend_from_slice(&chunk[..n]);
        }
        String::from_utf8(bytes).map_err(io::Error::other)
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ProcessSample {
    pub pid: u32,
    pub start_ticks: u64,
    pub rss_bytes: u64,
    pub lifetime_rss_high_water_bytes: Option<u64>,
}

fn start_ticks(stat: &str, pid: u32) -> io::Result<u64> {
    if stat
        .split_once(' ')
        .and_then(|(p, _)| p.parse::<u32>().ok())
        != Some(pid)
    {
        return Err(invalid("proc stat PID differs from requested process"));
    }
    stat.rsplit_once(") ")
        .and_then(|(_, tail)| tail.split_whitespace().nth(19))
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| invalid("invalid proc stat starttime"))
}
fn memory_bytes(status: &str, name: &str) -> io::Result<Option<u64>> {
    let mut value = None;
    for line in status.lines().filter(|line| line.starts_with(name)) {
        if value.is_some() {
            return Err(invalid("duplicate proc memory field"));
        }
        let mut fields = line.split_whitespace();
        if fields.next() != Some(name) {
            return Err(invalid("invalid proc memory field"));
        }
        let n = fields
            .next()
            .and_then(|n| n.parse::<u64>().ok())
            .and_then(|n| n.checked_mul(1024));
        if fields.next() != Some("kB") || fields.next().is_some() || n.is_none() {
            return Err(invalid("invalid proc memory count or unit"));
        }
        value = n;
    }
    Ok(value)
}
fn process(proc_root: &Path, pid: u32, budget: &mut ReadBudget) -> io::Result<ProcessSample> {
    let path = proc_root.join(pid.to_string());
    let before = start_ticks(&budget.read(&path.join("stat"))?, pid)?;
    let status = budget.read(&path.join("status"))?;
    let after = start_ticks(&budget.read(&path.join("stat"))?, pid)?;
    if before != after {
        return Err(invalid("PID incarnation changed during sample"));
    }
    Ok(ProcessSample {
        pid,
        start_ticks: before,
        rss_bytes: memory_bytes(&status, "VmRSS:")?
            .ok_or_else(|| invalid("VmRSS is unavailable"))?,
        lifetime_rss_high_water_bytes: memory_bytes(&status, "VmHWM:")?,
    })
}

fn directory_identity(path: &Path) -> io::Result<(u64, u64)> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid("cgroup root must be a real directory"));
    }
    Ok((metadata.dev(), metadata.ino()))
}

fn members(root: &Path, args: &Arguments, budget: &mut ReadBudget) -> io::Result<BTreeSet<u32>> {
    let mut pending = vec![root.to_owned()];
    let mut traversed = 1usize;
    let mut pids = BTreeSet::new();
    while let Some(path) = pending.pop() {
        directory_identity(&path)?;
        for line in budget.read(&path.join("cgroup.procs"))?.lines() {
            let pid = line
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|&pid| pid > 0 && pid <= i32::MAX as u32)
                .ok_or_else(|| invalid("invalid cgroup.procs PID"))?;
            pids.insert(pid);
            if pids.len() > args.maximum_pids {
                return Err(invalid("cgroup PID limit exceeded"));
            }
        }
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            traversed = traversed
                .checked_add(1)
                .ok_or_else(|| invalid("traversal count overflow"))?;
            if traversed > args.maximum_traversal_nodes {
                return Err(invalid("cgroup traversal limit exceeded"));
            }
            let kind = entry.file_type()?;
            if kind.is_symlink() {
                return Err(invalid("symlink encountered in cgroup traversal"));
            }
            if kind.is_dir() {
                pending.push(entry.path());
            }
        }
    }
    Ok(pids)
}

#[derive(Debug, Serialize)]
pub(super) struct SampleError {
    operation: &'static str,
    pid: Option<u32>,
    message: String,
}
#[derive(Debug, Default, Serialize)]
pub(super) struct Sample {
    pub members_discovered: Option<usize>,
    pub members_sampled: usize,
    pub rss_sum_bytes: Option<u64>,
    pub partial_rss_sum_bytes: u64,
    pub input_bytes_read: usize,
    pub processes: Vec<ProcessSample>,
    pub errors: Vec<SampleError>,
    pub stop_reason: Option<&'static str>,
}
impl Sample {
    fn error(&mut self, operation: &'static str, pid: Option<u32>, error: io::Error) {
        self.errors.push(SampleError {
            operation,
            pid,
            message: error.to_string(),
        });
    }
    fn add(&mut self, process: ProcessSample) {
        if let Some(sum) = self.partial_rss_sum_bytes.checked_add(process.rss_bytes) {
            self.partial_rss_sum_bytes = sum;
        } else {
            self.error("rss_sum", None, invalid("RSS sum overflows"));
        }
        self.processes.push(process);
        self.members_sampled += 1;
    }
}

pub(super) struct Sampler {
    proc_root: PathBuf,
    pid: Option<u32>,
    pid_start: Option<u64>,
    cgroup: Option<(PathBuf, (u64, u64))>,
}
impl Sampler {
    pub fn new(args: &Arguments, proc_root: PathBuf) -> io::Result<Self> {
        args.validate()?;
        let cgroup = args
            .cgroup_directory
            .as_ref()
            .map(|p| directory_identity(p).map(|id| (p.clone(), id)))
            .transpose()?;
        Ok(Self {
            proc_root,
            pid: args.pid,
            pid_start: None,
            cgroup,
        })
    }
    pub fn sample(&mut self, args: &Arguments) -> Sample {
        let mut budget = ReadBudget {
            remaining: args.maximum_input_bytes,
        };
        let mut result = Sample::default();
        self.collect(args, &mut budget, &mut result);
        result.input_bytes_read = args.maximum_input_bytes - budget.remaining;
        if result.errors.is_empty()
            && result.stop_reason.is_none()
            && result.members_sampled > 0
            && result.members_discovered == Some(result.members_sampled)
        {
            result.rss_sum_bytes = Some(result.partial_rss_sum_bytes);
        }
        result
    }
    fn collect(&mut self, args: &Arguments, budget: &mut ReadBudget, result: &mut Sample) {
        if let Some(pid) = self.pid {
            result.members_discovered = Some(1);
            match process(&self.proc_root, pid, budget) {
                Ok(sample) => {
                    if self
                        .pid_start
                        .is_some_and(|start| start != sample.start_ticks)
                    {
                        result.error("pid_identity", Some(pid), invalid("root PID was reused"));
                        result.stop_reason = Some("root_pid_reused");
                    } else {
                        self.pid_start = Some(sample.start_ticks);
                        result.add(sample);
                    }
                }
                Err(error) => {
                    result.error("process_read", Some(pid), error);
                    result.stop_reason = Some("root_pid_unavailable_or_read_error");
                }
            }
            return;
        }
        let (path, identity) = self.cgroup.as_ref().expect("validated target mode");
        if !Self::check_root(path, *identity, result) {
            return;
        }
        let before = match members(path, args, budget) {
            Ok(pids) => pids,
            Err(error) => {
                result.error("membership_before", None, error);
                Self::check_root(path, *identity, result);
                return;
            }
        };
        result.members_discovered = Some(before.len());
        if before.is_empty() {
            result.error(
                "membership_before",
                None,
                invalid("target cgroup has no processes"),
            );
            result.stop_reason = Some("target_cgroup_empty");
            return;
        }
        for &pid in &before {
            match process(&self.proc_root, pid, budget) {
                Ok(sample) => result.add(sample),
                Err(error) => result.error("process_read", Some(pid), error),
            }
            if budget.remaining == 0 {
                break;
            }
        }
        match members(path, args, budget) {
            Ok(after) if before == after => {}
            Ok(_) => result.error(
                "membership_after",
                None,
                invalid("cgroup membership changed during sample"),
            ),
            Err(error) => result.error("membership_after", None, error),
        }
        Self::check_root(path, *identity, result);
    }
    fn check_root(path: &Path, expected: (u64, u64), result: &mut Sample) -> bool {
        match directory_identity(path) {
            Ok(identity) if identity == expected => true,
            Ok(_) => {
                result.error(
                    "cgroup_identity",
                    None,
                    invalid("root cgroup directory was replaced"),
                );
                result.stop_reason = Some("root_cgroup_replaced");
                false
            }
            Err(error) => {
                result.error("cgroup_identity", None, error);
                result.stop_reason = Some("root_cgroup_unavailable");
                false
            }
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
