//! Read-only Win32 registry and known-folder calls. No registry writer is linked.
use super::evidence::RegistryValue;
use anyhow::{ensure, Context, Result};
use std::{ffi::c_void, path::PathBuf, ptr};
type Key = *mut c_void;
const HKCU: Key = -2147483647isize as Key;
const UNINSTALL: &str =
    "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Ferrum.CLI.Windows_is1";
const OWNERSHIP: &str = "Software\\Ferrum\\Installer";
#[link(name = "advapi32")]
unsafe extern "system" {
    fn RegOpenKeyExW(
        key: Key,
        subkey: *const u16,
        options: u32,
        access: u32,
        result: *mut Key,
    ) -> i32;
    fn RegQueryValueExW(
        key: Key,
        name: *const u16,
        reserved: *mut u32,
        kind: *mut u32,
        data: *mut u8,
        size: *mut u32,
    ) -> i32;
    fn RegCloseKey(key: Key) -> i32;
}
#[link(name = "kernel32")]
unsafe extern "system" {
    fn ExpandEnvironmentStringsW(source: *const u16, dest: *mut u16, size: u32) -> u32;
}
#[repr(C)]
struct Guid {
    a: u32,
    b: u16,
    c: u16,
    d: [u8; 8],
}
#[link(name = "shell32")]
unsafe extern "system" {
    fn SHGetKnownFolderPath(
        id: *const Guid,
        flags: u32,
        token: *mut c_void,
        path: *mut *mut u16,
    ) -> i32;
}
#[link(name = "ole32")]
unsafe extern "system" {
    fn CoTaskMemFree(value: *mut c_void);
}
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain([0]).collect()
}
struct Handle(Key);
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            RegCloseKey(self.0);
        }
    }
}
fn open(name: &str) -> Result<Option<Handle>> {
    let mut key = ptr::null_mut();
    let status = unsafe { RegOpenKeyExW(HKCU, wide(name).as_ptr(), 0, 0x20119, &mut key) };
    if status == 2 {
        return Ok(None);
    }
    ensure!(status == 0, "RegOpenKeyExW {name}: {status}");
    Ok(Some(Handle(key)))
}
fn read(key: &str, name: &str) -> Result<Option<RegistryValue>> {
    let Some(key) = open(key)? else {
        return Ok(None);
    };
    let name = wide(name);
    for _ in 0..3 {
        let (mut kind, mut size) = (0, 0);
        let status = unsafe {
            RegQueryValueExW(
                key.0,
                name.as_ptr(),
                ptr::null_mut(),
                &mut kind,
                ptr::null_mut(),
                &mut size,
            )
        };
        if status == 2 {
            return Ok(None);
        }
        ensure!(
            status == 0 && size <= 1024 * 1024,
            "RegQueryValueExW size/status: {size}/{status}"
        );
        let mut bytes = vec![0; size as usize];
        let status = unsafe {
            RegQueryValueExW(
                key.0,
                name.as_ptr(),
                ptr::null_mut(),
                &mut kind,
                bytes.as_mut_ptr(),
                &mut size,
            )
        };
        if status == 234 {
            continue;
        }
        ensure!(status == 0, "RegQueryValueExW data: {status}");
        bytes.truncate(size as usize);
        return Ok(Some(RegistryValue { kind, bytes }));
    }
    anyhow::bail!("registry value changed repeatedly during read")
}
pub fn path() -> Result<Option<RegistryValue>> {
    read("Environment", "Path")
}
pub fn uninstall_exists() -> Result<bool> {
    Ok(open(UNINSTALL)?.is_some())
}
pub fn ownership_exists() -> Result<bool> {
    Ok(open(OWNERSHIP)?.is_some())
}
pub fn uninstall_command() -> Result<Option<RegistryValue>> {
    read(UNINSTALL, "UninstallString")
}
pub fn local_app_data() -> Result<PathBuf> {
    let id = Guid {
        a: 0xf1b32785,
        b: 0x6fba,
        c: 0x4fcf,
        d: [0x9d, 0x55, 0x7b, 0x8e, 0x7f, 0x15, 0x70, 0x91],
    };
    let mut value = ptr::null_mut();
    let result = unsafe { SHGetKnownFolderPath(&id, 0, ptr::null_mut(), &mut value) };
    ensure!(
        result >= 0 && !value.is_null(),
        "SHGetKnownFolderPath failed: {result}"
    );
    let mut length = 0;
    unsafe {
        while *value.add(length) != 0 {
            length += 1;
        }
    }
    let text = unsafe { String::from_utf16(std::slice::from_raw_parts(value, length)) };
    unsafe { CoTaskMemFree(value.cast()) };
    Ok(PathBuf::from(text?))
}
pub fn entry_count(value: Option<&RegistryValue>, app: &str) -> Result<usize> {
    let Some(value) = value else { return Ok(0) };
    let mut count = 0;
    for part in value.text()?.split(';') {
        let part = part.trim().trim_matches('"');
        let expanded = if value.kind == 2 {
            let input = wide(part);
            let size = unsafe { ExpandEnvironmentStringsW(input.as_ptr(), ptr::null_mut(), 0) };
            ensure!(size > 0 && size <= 1024 * 1024, "cannot expand PATH entry");
            let mut output = vec![0; size as usize];
            let written =
                unsafe { ExpandEnvironmentStringsW(input.as_ptr(), output.as_mut_ptr(), size) };
            ensure!(
                written > 0 && written <= size,
                "PATH expansion changed size"
            );
            String::from_utf16(&output[..written as usize - 1]).context("invalid expanded PATH")?
        } else {
            part.into()
        };
        count +=
            usize::from(super::evidence::comparable(&expanded) == super::evidence::comparable(app));
    }
    Ok(count)
}
