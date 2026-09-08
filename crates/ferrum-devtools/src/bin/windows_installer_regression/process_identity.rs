//! Read-only process identity, held by a handle so PID reuse cannot pass liveness.
use anyhow::{ensure, Result};
use serde::Serialize;
use std::{
    ffi::c_void,
    mem::size_of,
    path::{Path, PathBuf},
};

type Handle = *mut c_void;
#[repr(C)]
struct ProcessEntry {
    size: u32,
    usage: u32,
    pid: u32,
    default_heap: usize,
    module: u32,
    threads: u32,
    parent_pid: u32,
    priority: i32,
    flags: u32,
    executable: [u16; 260],
}
#[repr(C)]
#[derive(Default)]
struct FileTime {
    low: u32,
    high: u32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateToolhelp32Snapshot(flags: u32, pid: u32) -> Handle;
    fn Process32FirstW(snapshot: Handle, entry: *mut ProcessEntry) -> i32;
    fn Process32NextW(snapshot: Handle, entry: *mut ProcessEntry) -> i32;
    fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
    fn QueryFullProcessImageNameW(
        process: Handle,
        flags: u32,
        path: *mut u16,
        size: *mut u32,
    ) -> i32;
    fn GetProcessTimes(
        process: Handle,
        created: *mut FileTime,
        exit: *mut FileTime,
        kernel: *mut FileTime,
        user: *mut FileTime,
    ) -> i32;
    fn WaitForSingleObject(handle: Handle, milliseconds: u32) -> u32;
    fn CloseHandle(handle: Handle) -> i32;
    fn GetLastError() -> u32;
}
struct OwnedHandle(Handle);
impl Drop for OwnedHandle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Identity {
    pub pid: u32,
    pub parent_pid: u32,
    pub creation_time_100ns_since_1601: u64,
    pub image: PathBuf,
}
pub struct Process {
    handle: OwnedHandle,
    pub identity: Identity,
}
impl Process {
    pub fn alive(&self) -> Result<()> {
        let status = unsafe { WaitForSingleObject(self.handle.0, 0) };
        ensure!(
            status == 258,
            "original core process ended or cannot be observed (wait={status}, PID={})",
            self.identity.pid
        );
        Ok(())
    }
    pub fn wait_for_exit(&self, milliseconds: u32) -> Result<()> {
        let status = unsafe { WaitForSingleObject(self.handle.0, milliseconds) };
        ensure!(
            status == 0,
            "owned core did not exit after launcher cleanup (wait={status}, PID={})",
            self.identity.pid
        );
        Ok(())
    }
}

pub fn child(parent_pid: u32, expected_image: &Path) -> Result<Option<Process>> {
    let snapshot = unsafe { CreateToolhelp32Snapshot(2, 0) };
    ensure!(
        snapshot as isize != -1,
        "cannot snapshot process identities: {}",
        std::io::Error::last_os_error()
    );
    let snapshot = OwnedHandle(snapshot);
    let mut entry = ProcessEntry {
        size: size_of::<ProcessEntry>() as u32,
        usage: 0,
        pid: 0,
        default_heap: 0,
        module: 0,
        threads: 0,
        parent_pid: 0,
        priority: 0,
        flags: 0,
        executable: [0; 260],
    };
    let mut present = unsafe { Process32FirstW(snapshot.0, &mut entry) };
    let mut result = None;
    while present != 0 {
        if entry.parent_pid == parent_pid {
            let raw = unsafe { OpenProcess(0x0010_1000, 0, entry.pid) };
            ensure!(
                !raw.is_null(),
                "cannot observe launcher child: {}",
                std::io::Error::last_os_error()
            );
            let handle = OwnedHandle(raw);
            let mut path = vec![0u16; 32768];
            let mut size = path.len() as u32;
            ensure!(
                unsafe { QueryFullProcessImageNameW(raw, 0, path.as_mut_ptr(), &mut size) } != 0,
                "cannot read launcher child image"
            );
            let image = PathBuf::from(String::from_utf16(&path[..size as usize])?);
            if image.canonicalize()? == expected_image.canonicalize()? {
                let (mut created, mut exit, mut kernel, mut user) = (
                    FileTime::default(),
                    FileTime::default(),
                    FileTime::default(),
                    FileTime::default(),
                );
                ensure!(
                    unsafe {
                        GetProcessTimes(raw, &mut created, &mut exit, &mut kernel, &mut user)
                    } != 0,
                    "cannot read child creation identity"
                );
                let process = Process {
                    handle,
                    identity: Identity {
                        pid: entry.pid,
                        parent_pid,
                        creation_time_100ns_since_1601: ((created.high as u64) << 32)
                            | created.low as u64,
                        image,
                    },
                };
                process.alive()?;
                // Bind the historical snapshot's parent relationship to this live handle.
                // Otherwise a PID reused between the first snapshot and OpenProcess could
                // refer to an unrelated process running the same executable.
                confirm_live_parent(process.identity.pid, parent_pid)?;
                process.alive()?;
                ensure!(
                    result.is_none(),
                    "multiple core children belong to one launcher"
                );
                result = Some(process);
            }
        }
        present = unsafe { Process32NextW(snapshot.0, &mut entry) };
    }
    ensure!(
        unsafe { GetLastError() } == 18,
        "process snapshot enumeration failed"
    );
    Ok(result)
}

fn confirm_live_parent(pid: u32, parent_pid: u32) -> Result<()> {
    let raw = unsafe { CreateToolhelp32Snapshot(2, 0) };
    ensure!(raw as isize != -1, "cannot refresh process parent identity");
    let snapshot = OwnedHandle(raw);
    // PROCESSENTRY32W is entirely integer fields, so its all-zero value is valid.
    let mut entry: ProcessEntry = unsafe { std::mem::zeroed() };
    entry.size = size_of::<ProcessEntry>() as u32;
    let mut present = unsafe { Process32FirstW(snapshot.0, &mut entry) };
    while present != 0 {
        if entry.pid == pid {
            ensure!(
                entry.parent_pid == parent_pid,
                "observed core no longer belongs to the owned launcher"
            );
            return Ok(());
        }
        present = unsafe { Process32NextW(snapshot.0, &mut entry) };
    }
    anyhow::bail!("live core is absent from the refreshed process snapshot")
}
