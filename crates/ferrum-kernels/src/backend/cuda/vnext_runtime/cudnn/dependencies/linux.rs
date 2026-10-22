//! Linux GNU loader verification. The public `link_map` prefix and dlinfo ABI
//! follow glibc's Dynamic Linker Introspection documentation. We inspect only
//! the retained handle, never traverse the namespace's mutable linked list.

use std::ffi::{c_char, c_void, CStr, OsStr, OsString};
use std::mem::ManuallyDrop;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use libloading::os::unix::{Library as UnixLibrary, RTLD_LOCAL, RTLD_NOW};
use libloading::Library;

#[repr(C)]
struct LinkMapPrefix {
    address: usize,
    name: *const c_char,
    dynamic: *const ElfDynamic,
}

#[repr(C)]
struct ElfDynamic {
    tag: isize,
    value: usize,
}

pub(super) unsafe fn load_checked(path: &Path) -> Result<Library, String> {
    let library = unsafe { UnixLibrary::open(Some(path), RTLD_NOW | RTLD_LOCAL) }
        .map_err(|error| format!("cuDNN dependency load {}: {error}", path.display()))?;
    let handle = library.into_raw();
    // Arm before any introspection that can return Err or unwind. The caller
    // retains its opaque/context grant on failure, matching this native owner.
    let mut library = ManuallyDrop::new(Library::from(unsafe { UnixLibrary::from_raw(handle) }));
    let map = unsafe { link_map(handle) }?;
    let actual = unsafe { mapped_path(map) }?;
    require_selected_path(path, &actual)?;
    let soname = unsafe { soname(map) }?;
    // A previous conflicting SONAME may win a later native dlopen even when an
    // absolute-path load mapped our file separately. NOLOAD adds no new object;
    // reject that ambiguity instead of fingerprinting a library cuDNN won't use.
    let alias =
        unsafe { UnixLibrary::open(Some(&soname), RTLD_NOW | RTLD_LOCAL | libc::RTLD_NOLOAD) }
            .map_err(|error| {
                format!(
            "cuDNN dependency {} is not discoverable by SONAME {} after explicit load: {error}",
            path.display(),
            soname.to_string_lossy()
        )
            })?;
    let alias_handle = alias.into_raw();
    let alias = unsafe { UnixLibrary::from_raw(alias_handle) };
    let alias_path = unsafe { mapped_path(link_map(alias_handle)?) }?;
    require_selected_path(path, &alias_path)?;
    drop(alias);
    Ok(unsafe { ManuallyDrop::take(&mut library) })
}

fn require_selected_path(expected: &Path, actual: &Path) -> Result<(), String> {
    if expected == actual {
        return Ok(());
    }
    Err(format!(
        "cuDNN dependency loader selected {} instead of {}; conflicting native deployment",
        actual.display(),
        expected.display()
    ))
}

unsafe fn link_map(handle: *mut c_void) -> Result<*const LinkMapPrefix, String> {
    let mut map: *const LinkMapPrefix = std::ptr::null();
    let status = unsafe {
        libc::dlinfo(
            handle,
            libc::RTLD_DI_LINKMAP,
            (&mut map as *mut *const LinkMapPrefix).cast(),
        )
    };
    if status != 0 || map.is_null() {
        return Err("cuDNN dependency dlinfo(RTLD_DI_LINKMAP) failed".into());
    }
    Ok(map)
}

unsafe fn mapped_path(map: *const LinkMapPrefix) -> Result<PathBuf, String> {
    let name = unsafe { (*map).name };
    if name.is_null() {
        return Err("cuDNN dependency loader returned no object name".into());
    }
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    if bytes.is_empty() {
        return Err("cuDNN dependency resolved to the main executable".into());
    }
    Path::new(OsStr::from_bytes(bytes))
        .canonicalize()
        .map_err(|error| format!("cuDNN loaded dependency identity: {error}"))
}

unsafe fn soname(map: *const LinkMapPrefix) -> Result<OsString, String> {
    let dynamic = unsafe { (*map).dynamic };
    if dynamic.is_null() {
        return Err("cuDNN dependency has no ELF dynamic segment".into());
    }
    let (mut strings, mut size, mut offset) = (None, None, None);
    // GNU ld.so usually relocates DT_STRTAB in the object's dynamic section;
    // read-only dynamic sections retain the relative value. Resolve either
    // representation against this same loaded object before dereferencing.
    // The trusted loaded ELF guarantees DT_NULL termination. This bound rejects
    // unusually large tables; it is not a parser for untrusted files.
    for index in 0..4096 {
        let entry = unsafe { &*dynamic.add(index) };
        match entry.tag {
            0 => break,
            5 => strings = Some(entry.value), // DT_STRTAB
            10 => size = Some(entry.value),   // DT_STRSZ
            14 => offset = Some(entry.value), // DT_SONAME
            _ => {}
        }
    }
    let (strings, size, offset) = match (strings, size, offset) {
        (Some(strings), Some(size), Some(offset))
            if strings != 0 && offset < size && size <= isize::MAX as usize =>
        {
            (strings, size, offset)
        }
        _ => return Err("cuDNN dependency requires a valid ELF SONAME".into()),
    };
    let strings = unsafe { string_table_address(map, strings) }?;
    let name = unsafe {
        std::slice::from_raw_parts(
            (strings as *const u8).add(offset),
            (size - offset).min(4096),
        )
    };
    let end = name
        .iter()
        .position(|byte| *byte == 0)
        .ok_or("cuDNN dependency ELF SONAME is unterminated or too long")?;
    if end == 0 || name[..end].contains(&b'/') {
        return Err("cuDNN dependency ELF SONAME must be a nonempty basename".into());
    }
    Ok(OsString::from_vec(name[..end].to_vec()))
}

unsafe fn string_table_address(map: *const LinkMapPrefix, value: usize) -> Result<usize, String> {
    let base = unsafe { (*map).address };
    for candidate in [Some(value), base.checked_add(value)].into_iter().flatten() {
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
        // dladdr inspects the loader's mapping metadata; it does not dereference
        // the candidate. Do not guess from whether a pointer is numerically low.
        if unsafe { libc::dladdr(candidate as *const c_void, info.as_mut_ptr()) } != 0 {
            let info = unsafe { info.assume_init() };
            if info.dli_fbase as usize == base {
                return Ok(candidate);
            }
        }
    }
    Err("cuDNN dependency DT_STRTAB does not resolve within its loaded object".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_rejects_a_different_mapped_dependency_identity() {
        assert!(require_selected_path(
            Path::new("/selected/lib.so.1"),
            Path::new("/other/lib.so.1")
        )
        .is_err());
        assert!(require_selected_path(
            Path::new("/selected/lib.so.1"),
            Path::new("/selected/lib.so.1")
        )
        .is_ok());
    }

    #[test]
    fn explicit_dependency_retains_the_actual_soname_object_without_a_gpu() {
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
        assert_ne!(
            unsafe {
                libc::dladdr(
                    libc::getpid as *const () as *const c_void,
                    info.as_mut_ptr(),
                )
            },
            0
        );
        let info = unsafe { info.assume_init() };
        let path = Path::new(OsStr::from_bytes(
            unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes(),
        ))
        .canonicalize()
        .unwrap();
        let library = unsafe { load_checked(&path) }.unwrap();
        let getpid =
            unsafe { library.get::<unsafe extern "C" fn() -> libc::pid_t>(b"getpid\0") }.unwrap();
        assert_eq!(unsafe { getpid() }, unsafe { libc::getpid() });
    }
}
