use std::{
    ffi::{c_void, OsStr, OsString},
    fs, io,
    mem::size_of,
    os::windows::{
        ffi::OsStrExt,
        io::{AsRawHandle, FromRawHandle, OwnedHandle},
    },
    path::Path,
    ptr::{null, null_mut},
};
use windows_sys::Win32::{
    Foundation::{
        DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
    },
    System::{
        Console::{
            GetStdHandle, SetConsoleCtrlHandler, CTRL_BREAK_EVENT, CTRL_C_EVENT, STD_ERROR_HANDLE,
            STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
        },
        JobObjects::{
            CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        },
        Threading::{
            CreateProcessW, DeleteProcThreadAttributeList, GetCurrentProcess, GetExitCodeProcess,
            InitializeProcThreadAttributeList, UpdateProcThreadAttribute, WaitForSingleObject,
            EXTENDED_STARTUPINFO_PRESENT, INFINITE, LPPROC_THREAD_ATTRIBUTE_LIST,
            PROCESS_INFORMATION, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROC_THREAD_ATTRIBUTE_JOB_LIST,
            STARTF_USESTDHANDLES, STARTUPINFOEXW,
        },
    },
};

fn error(operation: &str) -> String {
    format!("{operation}: {}", io::Error::last_os_error())
}

// An explicit handler only affects this process. Using NULL/TRUE instead would
// make children inherit "ignore Ctrl-C", breaking interactive run/serve.
unsafe extern "system" fn console_control(event: u32) -> i32 {
    i32::from(event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT)
}

struct ConsoleHandler;
impl ConsoleHandler {
    fn install() -> Result<Self, String> {
        if unsafe { SetConsoleCtrlHandler(Some(console_control), 1) } == 0 {
            return Err(error("install console handler"));
        }
        Ok(Self)
    }
}
impl Drop for ConsoleHandler {
    fn drop(&mut self) {
        unsafe {
            SetConsoleCtrlHandler(Some(console_control), 0);
        }
    }
}

struct Attributes {
    // Pointer-sized storage keeps the opaque Windows structure aligned.
    _storage: Vec<usize>,
    pointer: LPPROC_THREAD_ATTRIBUTE_LIST,
}
impl Attributes {
    fn new() -> Result<Self, String> {
        let mut bytes = 0;
        unsafe {
            InitializeProcThreadAttributeList(null_mut(), 2, 0, &mut bytes);
        }
        if bytes == 0 || bytes > 65536 {
            return Err(error("size process attributes"));
        }
        let mut storage = vec![0usize; bytes.div_ceil(size_of::<usize>())];
        let pointer = storage.as_mut_ptr().cast();
        if unsafe { InitializeProcThreadAttributeList(pointer, 2, 0, &mut bytes) } == 0 {
            return Err(error("initialize process attributes"));
        }
        Ok(Self {
            _storage: storage,
            pointer,
        })
    }

    // Windows borrows the arrays until CreateProcessW completes. The caller
    // retains both arrays and every underlying handle for that entire interval.
    fn handles(&mut self, attribute: u32, handles: &[HANDLE]) -> Result<(), String> {
        if unsafe {
            UpdateProcThreadAttribute(
                self.pointer,
                0,
                attribute as usize,
                handles.as_ptr().cast(),
                std::mem::size_of_val(handles),
                null_mut(),
                null(),
            )
        } == 0
        {
            return Err(error("set process handle attribute"));
        }
        Ok(())
    }
}
impl Drop for Attributes {
    fn drop(&mut self) {
        unsafe {
            DeleteProcThreadAttributeList(self.pointer);
        }
    }
}

fn job() -> Result<OwnedHandle, String> {
    let handle = unsafe { CreateJobObjectW(null(), null()) };
    if handle.is_null() {
        return Err(error("create child job"));
    }
    let owned = unsafe { OwnedHandle::from_raw_handle(handle) };
    let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if unsafe {
        SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast::<c_void>(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    } == 0
    {
        return Err(error("configure child job"));
    }
    Ok(owned)
}

fn standard_handle(which: u32) -> Result<OwnedHandle, String> {
    let mut handle = unsafe { GetStdHandle(which) };
    if handle == INVALID_HANDLE_VALUE {
        return Err(error("read standard handle"));
    }
    // A detached caller can have no standard handle. Match the normal EOF/
    // discarded-output behavior without adding a console or changing its cwd.
    let null_file = if handle.is_null() {
        Some(
            fs::OpenOptions::new()
                .read(which == STD_INPUT_HANDLE)
                .write(which != STD_INPUT_HANDLE)
                .open("NUL")
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    if let Some(file) = &null_file {
        handle = file.as_raw_handle();
    }
    let mut duplicate = null_mut();
    let process = unsafe { GetCurrentProcess() };
    if unsafe {
        DuplicateHandle(
            process,
            handle,
            process,
            &mut duplicate,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        )
    } == 0
    {
        return Err(error("duplicate standard handle"));
    }
    Ok(unsafe { OwnedHandle::from_raw_handle(duplicate) })
}

fn wide(value: &OsStr) -> Result<Vec<u16>, String> {
    let mut units: Vec<_> = value.encode_wide().collect();
    if units.contains(&0) {
        return Err("process argument contains NUL".into());
    }
    units.push(0);
    Ok(units)
}

fn command_line(
    program: &Path,
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<Vec<u16>, String> {
    let mut output = Vec::new();
    for argument in std::iter::once(program.as_os_str().to_owned()).chain(arguments) {
        if !output.is_empty() {
            output.push(b' ' as u16);
        }
        output.push(b'"' as u16);
        let mut slashes = 0;
        for unit in argument.encode_wide() {
            match unit {
                0 => return Err("process argument contains NUL".into()),
                92 => slashes += 1,
                34 => {
                    output.extend(std::iter::repeat_n(92, slashes * 2 + 1));
                    output.push(34);
                    slashes = 0;
                }
                _ => {
                    output.extend(std::iter::repeat_n(92, slashes));
                    output.push(unit);
                    slashes = 0;
                }
            }
        }
        output.extend(std::iter::repeat_n(92, slashes * 2));
        output.push(b'"' as u16);
        if output.len() >= 32767 {
            return Err("process arguments exceed the Windows command-line limit".into());
        }
    }
    output.push(0);
    Ok(output)
}

pub(super) fn run(
    program: &Path,
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<u32, String> {
    let application = wide(program.as_os_str())?;
    let mut command = command_line(program, arguments)?;
    let owned_job = job()?;
    let input = standard_handle(STD_INPUT_HANDLE)?;
    let output = standard_handle(STD_OUTPUT_HANDLE)?;
    let error_output = standard_handle(STD_ERROR_HANDLE)?;
    let handles = [
        input.as_raw_handle(),
        output.as_raw_handle(),
        error_output.as_raw_handle(),
    ];
    let jobs = [owned_job.as_raw_handle()];
    let mut attributes = Attributes::new()?;
    attributes.handles(PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &handles)?;
    // Windows 10 assigns the job during process creation, without the orphan
    // window of spawning first and then calling AssignProcessToJobObject.
    // https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute
    attributes.handles(PROC_THREAD_ATTRIBUTE_JOB_LIST, &jobs)?;
    let mut startup = STARTUPINFOEXW::default();
    startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = handles[0];
    startup.StartupInfo.hStdOutput = handles[1];
    startup.StartupInfo.hStdError = handles[2];
    startup.lpAttributeList = attributes.pointer;
    let mut information = PROCESS_INFORMATION::default();
    let _console = ConsoleHandler::install()?;
    // No new console/process group; no environment or cwd override. Only the
    // standard handles are inherited, never our kill-on-close job handle.
    if unsafe {
        CreateProcessW(
            application.as_ptr(),
            command.as_mut_ptr(),
            null(),
            null(),
            1,
            EXTENDED_STARTUPINFO_PRESENT,
            null(),
            null(),
            &startup.StartupInfo,
            &mut information,
        )
    } == 0
    {
        return Err(error("start selected Ferrum version"));
    }
    let process = unsafe { OwnedHandle::from_raw_handle(information.hProcess) };
    drop(unsafe { OwnedHandle::from_raw_handle(information.hThread) });
    if unsafe { WaitForSingleObject(process.as_raw_handle(), INFINITE) } != WAIT_OBJECT_0 {
        return Err(error("wait for Ferrum"));
    }
    let mut code = 0;
    if unsafe { GetExitCodeProcess(process.as_raw_handle(), &mut code) } == 0 {
        return Err(error("read Ferrum exit code"));
    }
    // Closing the last job handle also cleans up any leftover descendants.
    Ok(code)
}
