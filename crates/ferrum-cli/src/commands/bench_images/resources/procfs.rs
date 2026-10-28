use super::{ImageProcessScope, ImageResourceSamplingConfig};
use ferrum_bench_core::images::resource_usage::{proc_memory_bytes, proc_stat, ProcessIdentity};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
};

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

struct Budget {
    bytes: usize,
    nodes: usize,
}
impl Budget {
    fn read(&mut self, path: &Path) -> io::Result<String> {
        if self.bytes == 0 {
            return Err(invalid("proc sample input byte limit exhausted"));
        }
        let mut bytes = Vec::new();
        fs::File::open(path)?
            .take(self.bytes.min(64 * 1024) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.len() == self.bytes.min(64 * 1024) {
            return Err(invalid("proc file or sample byte limit exhausted"));
        }
        self.bytes -= bytes.len();
        String::from_utf8(bytes).map_err(io::Error::other)
    }
    fn node(&mut self) -> io::Result<()> {
        self.nodes = self
            .nodes
            .checked_sub(1)
            .ok_or_else(|| invalid("proc traversal node limit exhausted"))?;
        Ok(())
    }
}

pub(super) struct ProcessSampler {
    root: PathBuf,
    pid: u32,
    identity: Result<ProcessIdentity, String>,
    stopped: Option<String>,
}

impl ProcessSampler {
    pub(super) fn new(root: PathBuf, pid: u32) -> Self {
        let mut budget = Budget {
            bytes: 64 * 1024,
            nodes: 1,
        };
        let identity = budget
            .read(&root.join(pid.to_string()).join("stat"))
            .and_then(|stat| proc_stat(&stat, pid).map(|stat| stat.0))
            .map_err(|error| error.to_string());
        Self {
            root,
            pid,
            identity,
            stopped: None,
        }
    }
    pub(super) fn identity(&self) -> Option<ProcessIdentity> {
        self.identity.as_ref().ok().copied()
    }

    fn identity_at(&self, pid: u32, budget: &mut Budget) -> io::Result<(ProcessIdentity, u32)> {
        proc_stat(
            &budget.read(&self.root.join(pid.to_string()).join("stat"))?,
            pid,
        )
    }

    fn members(
        &self,
        config: &ImageResourceSamplingConfig,
        budget: &mut Budget,
    ) -> io::Result<BTreeMap<u32, ProcessIdentity>> {
        let mut members = BTreeMap::new();
        let mut pending = vec![(self.pid, None)];
        while let Some((pid, parent)) = pending.pop() {
            budget.node()?;
            let (identity, actual_parent) = self.identity_at(pid, budget)?;
            if parent.is_some_and(|parent| parent != actual_parent) {
                return Err(invalid("process tree parent changed"));
            }
            if members.insert(pid, identity).is_some() {
                return Err(invalid("process tree duplicate or cycle"));
            }
            if members.len() > config.maximum_processes {
                return Err(invalid("process tree process limit exceeded"));
            }
            if config.process_scope == ImageProcessScope::Process {
                break;
            }
            let mut children = BTreeSet::new();
            // A child may belong to any thread, not just the process leader.
            for task in fs::read_dir(self.root.join(pid.to_string()).join("task"))? {
                budget.node()?;
                let task = task?;
                if task.file_name().to_string_lossy().parse::<u32>().is_err() {
                    continue;
                }
                for child in budget
                    .read(&task.path().join("children"))?
                    .split_whitespace()
                {
                    let child = child
                        .parse::<u32>()
                        .ok()
                        .filter(|pid| *pid > 0 && *pid <= i32::MAX as u32)
                        .ok_or_else(|| invalid("invalid process child PID"))?;
                    children.insert(child);
                    if children.len() > config.maximum_processes {
                        return Err(invalid("process child limit exceeded"));
                    }
                }
            }
            pending.extend(children.into_iter().map(|child| (child, Some(pid))));
            if pending.len() > config.maximum_processes {
                return Err(invalid("pending process limit exceeded"));
            }
        }
        Ok(members)
    }

    pub(super) fn sample(&mut self, config: &ImageResourceSamplingConfig) -> Result<u64, String> {
        if let Some(error) = &self.stopped {
            return Err(error.clone());
        }
        let expected = *self.identity.as_ref().map_err(Clone::clone)?;
        let mut budget = Budget {
            bytes: config.maximum_proc_input_bytes,
            nodes: config.maximum_tree_nodes,
        };
        let result = (|| -> io::Result<u64> {
            if self.identity_at(self.pid, &mut budget)?.0 != expected {
                return Err(invalid("root PID was reused"));
            }
            let before = self.members(config, &mut budget)?;
            let mut sum = 0u64;
            for (&pid, identity) in &before {
                if self.identity_at(pid, &mut budget)?.0 != *identity {
                    return Err(invalid("PID changed before RSS read"));
                }
                let status = budget.read(&self.root.join(pid.to_string()).join("status"))?;
                let rss = proc_memory_bytes(&status, "VmRSS:")?
                    .ok_or_else(|| invalid("VmRSS is unavailable"))?;
                if self.identity_at(pid, &mut budget)?.0 != *identity {
                    return Err(invalid("PID changed during RSS read"));
                }
                sum = sum
                    .checked_add(rss)
                    .ok_or_else(|| invalid("process RSS sum overflow"))?;
            }
            if self.members(config, &mut budget)? != before {
                return Err(invalid(
                    "process tree membership or identity changed during sample",
                ));
            }
            if self.identity_at(self.pid, &mut budget)?.0 != expected {
                return Err(invalid("root PID changed during sample"));
            }
            Ok(sum)
        })();
        // Once the pinned root disappears or changes, never attach to a later
        // incarnation even if a future sample sees the same numeric PID.
        match self.identity_at(
            self.pid,
            &mut Budget {
                bytes: 64 * 1024,
                nodes: 1,
            },
        ) {
            Ok((identity, _)) if identity == expected => {}
            _ => {
                self.stopped =
                    Some("pinned root process is unavailable or its PID was reused".into())
            }
        }
        result.map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn process(root: &Path, pid: u32, parent: u32, start: u64, rss: u64, children: &str) {
        let dir = root.join(pid.to_string());
        fs::create_dir_all(dir.join("task").join(pid.to_string())).unwrap();
        let mut fields = vec!["S".into(), parent.to_string()];
        fields.extend((5..22).map(|_| "0".into()));
        fields.push(start.to_string());
        fs::write(
            dir.join("stat"),
            format!("{pid} (worker ) name) {}", fields.join(" ")),
        )
        .unwrap();
        fs::write(dir.join("status"), format!("VmRSS: {rss} kB\n")).unwrap();
        fs::write(
            dir.join("task").join(pid.to_string()).join("children"),
            children,
        )
        .unwrap();
    }
    #[test]
    fn tree_counts_descendants_and_pins_root_identity_across_windows() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        process(root, 42, 1, 100, 3, "43");
        process(root, 43, 42, 200, 5, "");
        let mut sampler = ProcessSampler::new(root.into(), 42);
        let mut config = ImageResourceSamplingConfig::default();
        assert_eq!(sampler.sample(&config).unwrap(), 8 * 1024);
        config.process_scope = ImageProcessScope::Process;
        assert_eq!(sampler.sample(&config).unwrap(), 3 * 1024);
        process(root, 42, 1, 101, 99, "");
        assert!(sampler.sample(&config).is_err());
        process(root, 42, 1, 100, 3, "");
        assert!(sampler.sample(&config).is_err());
    }
    #[test]
    fn incomplete_child_or_capacity_is_never_reported_as_partial_success() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        process(root, 42, 1, 100, 3, "43");
        let mut sampler = ProcessSampler::new(root.into(), 42);
        let mut config = ImageResourceSamplingConfig::default();
        assert!(sampler.sample(&config).is_err());
        process(root, 43, 42, 200, 5, "");
        config.maximum_processes = 1;
        assert!(sampler.sample(&config).is_err());
        config.maximum_processes = 16;
        config.maximum_proc_input_bytes = 1;
        assert!(sampler.sample(&config).is_err());
    }
}
