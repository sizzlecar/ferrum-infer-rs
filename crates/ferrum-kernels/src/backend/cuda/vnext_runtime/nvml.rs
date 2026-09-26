//! Optional NVML queries. Match the CUDA context UUID, never an assumed ordinal.
//! ABI declarations follow NVIDIA's nvml.h; no NVML dependency is required at
//! program startup and unavailable process accounting is not a zero reading.

use std::ffi::{c_char, c_uint, c_void, CString};

use ferrum_types::device_memory::{CudaMemoryReading, CudaMemoryUnavailable as Unavailable};
use libloading::Library;

type Status = c_uint;
type Device = *mut c_void;
type Init = unsafe extern "C" fn() -> Status;
type Shutdown = unsafe extern "C" fn() -> Status;
type DriverVersion = unsafe extern "C" fn(*mut c_char, c_uint) -> Status;
type DeviceByUuid = unsafe extern "C" fn(*const c_char, *mut Device) -> Status;
type DeviceUuid = unsafe extern "C" fn(Device, *mut c_char, c_uint) -> Status;
type Processes = unsafe extern "C" fn(Device, *mut c_uint, *mut ProcessInfo) -> Status;
type MemoryInfo = unsafe extern "C" fn(Device, *mut DeviceMemoryInfo) -> Status;

const SUCCESS: Status = 0;
const NOT_SUPPORTED: Status = 3;
const INSUFFICIENT_SIZE: Status = 7;
const MAX_PROCESSES: usize = 1024;

/// nvmlProcessInfo_t used by nvmlDeviceGetComputeRunningProcesses_v3. The newer
/// protected-memory detail type has a different ABI and is deliberately unused.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: c_uint,
    used_gpu_memory: u64,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

#[repr(C)]
#[derive(Default)]
pub(super) struct DeviceMemoryInfo {
    version: c_uint,
    pub total: u64,
    pub reserved: u64,
    pub free: u64,
    pub used: u64,
}

pub(super) struct Nvml {
    // Kept alive until after shutdown; function pointers never outlive it.
    _library: Library,
    shutdown: Shutdown,
    driver_version: DriverVersion,
    device_by_uuid: Result<DeviceByUuid, Unavailable>,
    device_uuid: Result<DeviceUuid, Unavailable>,
    processes: Result<Processes, Unavailable>,
    memory_info: Result<MemoryInfo, Unavailable>,
}

fn failure(api: &str, code: Status) -> Unavailable {
    if code == NOT_SUPPORTED {
        Unavailable::Unsupported
    } else {
        Unavailable::QueryFailed {
            api: api.to_owned(),
            code: i64::from(code),
        }
    }
}

impl Nvml {
    pub(super) fn load() -> Result<Self, Unavailable> {
        #[cfg(target_os = "windows")]
        const LIBRARY: &str = "nvml.dll";
        #[cfg(not(target_os = "windows"))]
        const LIBRARY: &str = "libnvidia-ml.so.1";
        // SAFETY: this is NVIDIA's platform library, with only documented C ABI
        // symbols resolved. The Library remains owned while any symbol is used.
        unsafe {
            let library = Library::new(LIBRARY).map_err(|_| Unavailable::LibraryUnavailable)?;
            let init = *library
                .get::<Init>(b"nvmlInit_v2\0")
                .map_err(|_| Unavailable::SymbolUnavailable)?;
            let shutdown = *library
                .get::<Shutdown>(b"nvmlShutdown\0")
                .map_err(|_| Unavailable::SymbolUnavailable)?;
            let driver_version = *library
                .get::<DriverVersion>(b"nvmlSystemGetDriverVersion\0")
                .map_err(|_| Unavailable::SymbolUnavailable)?;
            let device_by_uuid = library
                .get::<DeviceByUuid>(b"nvmlDeviceGetHandleByUUID\0")
                .map(|symbol| *symbol)
                .map_err(|_| Unavailable::SymbolUnavailable);
            let device_uuid = library
                .get::<DeviceUuid>(b"nvmlDeviceGetUUID\0")
                .map(|symbol| *symbol)
                .map_err(|_| Unavailable::SymbolUnavailable);
            let processes = library
                .get::<Processes>(b"nvmlDeviceGetComputeRunningProcesses_v3\0")
                .map(|symbol| *symbol)
                .map_err(|_| Unavailable::SymbolUnavailable);
            let status = init();
            if status != SUCCESS {
                return Err(failure("nvmlInit_v2", status));
            }
            let memory_info = library
                .get::<MemoryInfo>(b"nvmlDeviceGetMemoryInfo_v2\0")
                .map(|symbol| *symbol)
                .map_err(|_| Unavailable::SymbolUnavailable);
            Ok(Self {
                _library: library,
                shutdown,
                driver_version,
                device_by_uuid,
                device_uuid,
                processes,
                memory_info,
            })
        }
    }

    pub(super) fn driver_build(&self) -> Result<String, Unavailable> {
        let mut bytes = [0_i8; 128];
        // SAFETY: the function has the header's ABI and the full writable array
        // size is passed. A truncated/nonterminated string is rejected below.
        let status = unsafe { (self.driver_version)(bytes.as_mut_ptr(), bytes.len() as c_uint) };
        if status != SUCCESS {
            return Err(failure("nvmlSystemGetDriverVersion", status));
        }
        bounded_c_text(&bytes).ok_or(Unavailable::QueryChanged)
    }

    pub(super) fn process_memory(&self, uuid: [u8; 16], pid: u32) -> CudaMemoryReading {
        match self.query_process_memory(uuid, pid) {
            Ok(bytes) => CudaMemoryReading::known(bytes),
            Err(reason) => CudaMemoryReading::unavailable(reason),
        }
    }

    fn device_handle(&self, uuid: [u8; 16]) -> Result<Device, Unavailable> {
        let by_uuid = *self.device_by_uuid.as_ref().map_err(Clone::clone)?;
        let get_uuid = *self.device_uuid.as_ref().map_err(Clone::clone)?;
        let expected = format!("GPU-{}", uuid_text(uuid));
        let c_uuid =
            CString::new(expected.as_str()).map_err(|_| Unavailable::DeviceIdentityMismatch)?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: output points to one writable handle, input is a complete C
        // string, and all subsequent calls occur during this initialized lifetime.
        let status = unsafe { by_uuid(c_uuid.as_ptr(), &mut handle) };
        if status != SUCCESS {
            return Err(failure("nvmlDeviceGetHandleByUUID", status));
        }
        if handle.is_null() {
            return Err(Unavailable::DeviceIdentityMismatch);
        }
        let mut actual = [0_i8; 96];
        let status = unsafe { get_uuid(handle, actual.as_mut_ptr(), actual.len() as c_uint) };
        if status != SUCCESS {
            return Err(failure("nvmlDeviceGetUUID", status));
        }
        if !bounded_c_text(&actual).is_some_and(|value| value.eq_ignore_ascii_case(&expected)) {
            return Err(Unavailable::DeviceIdentityMismatch);
        }
        Ok(handle)
    }

    pub(super) fn device_memory(&self, uuid: [u8; 16]) -> Result<DeviceMemoryInfo, Unavailable> {
        let function = *self.memory_info.as_ref().map_err(Clone::clone)?;
        let handle = self.device_handle(uuid)?;
        let mut memory = DeviceMemoryInfo {
            version: (std::mem::size_of::<DeviceMemoryInfo>() as c_uint) | (2 << 24),
            ..Default::default()
        };
        // SAFETY: the documented v2 version/size encoding and full writable C
        // layout are supplied, with the UUID-verified initialized device handle.
        let status = unsafe { function(handle, &mut memory) };
        if status != SUCCESS {
            return Err(failure("nvmlDeviceGetMemoryInfo_v2", status));
        }
        if memory.total == 0
            || [memory.free, memory.used, memory.reserved]
                .into_iter()
                .any(|bytes| bytes > memory.total)
        {
            return Err(Unavailable::QueryChanged);
        }
        Ok(memory)
    }

    fn query_process_memory(&self, uuid: [u8; 16], pid: u32) -> Result<u64, Unavailable> {
        let processes = *self.processes.as_ref().map_err(Clone::clone)?;
        let handle = self.device_handle(uuid)?;
        let mut entries = vec![ProcessInfo::default(); 16];
        for attempt in 0..2 {
            let mut count = entries.len() as c_uint;
            let status = unsafe { processes(handle, &mut count, entries.as_mut_ptr()) };
            if status == SUCCESS {
                if count as usize > entries.len() {
                    return Err(Unavailable::QueryChanged);
                }
                return select_process(&entries[..count as usize], pid);
            }
            if status != INSUFFICIENT_SIZE {
                return Err(failure("nvmlDeviceGetComputeRunningProcesses_v3", status));
            }
            if count as usize > MAX_PROCESSES {
                return Err(Unavailable::CapacityExceeded);
            }
            if attempt == 1 || count as usize <= entries.len() {
                return Err(Unavailable::QueryChanged);
            }
            entries.resize(count as usize, ProcessInfo::default());
        }
        Err(Unavailable::QueryChanged)
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: exactly one successful init belongs to this object. NVML's
        // reference count permits multiple independent users in the process.
        let _ = unsafe { (self.shutdown)() };
    }
}

fn select_process(entries: &[ProcessInfo], pid: u32) -> Result<u64, Unavailable> {
    let mut matches = entries.iter().filter(|entry| entry.pid == pid);
    let first = matches.next().ok_or(Unavailable::ProcessNotListed)?;
    if matches.next().is_some() {
        return Err(Unavailable::AmbiguousProcess);
    }
    if first.used_gpu_memory == u64::MAX {
        return Err(Unavailable::ProcessAccountingUnavailable);
    }
    Ok(first.used_gpu_memory)
}

fn bounded_c_text(bytes: &[i8]) -> Option<String> {
    let end = bytes.iter().position(|byte| *byte == 0)?;
    let bytes = bytes[..end]
        .iter()
        .map(|byte| *byte as u8)
        .collect::<Vec<_>>();
    let text = String::from_utf8(bytes).ok()?;
    (!text.is_empty() && !text.chars().any(char::is_control)).then_some(text)
}

pub(super) fn uuid_text(uuid: [u8; 16]) -> String {
    let mut text = String::with_capacity(36);
    for (index, value) in uuid.into_iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            text.push('-');
        }
        use std::fmt::Write;
        let _ = write!(text, "{value:02x}");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvml_process_selection_preserves_unknown_and_zero_distinction() {
        let entry = |pid, used_gpu_memory| ProcessInfo {
            pid,
            used_gpu_memory,
            ..Default::default()
        };
        assert_eq!(
            select_process(&[entry(2, 500)], 1),
            Err(Unavailable::ProcessNotListed)
        );
        assert_eq!(
            select_process(&[entry(1, u64::MAX)], 1),
            Err(Unavailable::ProcessAccountingUnavailable)
        );
        assert_eq!(
            select_process(&[entry(1, 4), entry(1, 8)], 1),
            Err(Unavailable::AmbiguousProcess)
        );
        assert_eq!(select_process(&[entry(1, 0), entry(2, 100)], 1), Ok(0));
        assert_eq!(select_process(&[entry(1, 4096)], 1), Ok(4096));
    }

    #[test]
    fn nvml_fixed_strings_and_uuid_do_not_accept_truncation() {
        assert!(bounded_c_text(&[65, 66]).is_none());
        assert!(bounded_c_text(&[0]).is_none());
        assert!(bounded_c_text(&[65, 10, 0]).is_none());
        assert_eq!(bounded_c_text(&[65, 66, 0]), Some("AB".to_owned()));
        assert_eq!(
            uuid_text([0xab; 16]),
            "abababab-abab-abab-abab-abababababab"
        );
    }
}
