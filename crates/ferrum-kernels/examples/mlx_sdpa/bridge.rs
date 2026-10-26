use super::{
    fixture::{Inputs, Shape, SCALE},
    Result,
};
use libloading::Library;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    ffi::{c_char, c_void},
    fs::File,
    io::Read,
    path::Path,
    ptr::NonNull,
    sync::Arc,
};

#[repr(C)]
struct Config {
    abi: u32,
    batch: u32,
    queries: u32,
    keys: u32,
    heads: u32,
    dim: u32,
    scale: f32,
    reserved: u32,
}

#[repr(C)]
struct Info {
    version: [c_char; 128],
    library: [c_char; 1024],
    device: [c_char; 128],
    architecture: [c_char; 128],
    source_sha256: [c_char; 65],
}

type Create = unsafe extern "C" fn(
    *const Config,
    *const u16,
    u64,
    *const u16,
    *const u16,
    u64,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> i32;
type Evaluate = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32;
type CopyOutput = unsafe extern "C" fn(*mut c_void, *mut u16, u64, *mut c_char, usize) -> i32;
type Close = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32;

struct Api {
    create: Create,
    evaluate: Evaluate,
    copy: CopyOutput,
    close: Close,
    identity: Value,
    // Function pointers and every outstanding handle require this live library.
    _library: Library,
}

pub struct Bridge(Arc<Api>);

impl Bridge {
    pub fn load(path: &Path, expected_version: &str) -> Result<Self> {
        let path = path.canonicalize()?;
        let bridge_sha256 = file_sha256(&path)?;
        // SAFETY: This explicit CLI path is a native-code trust boundary, like
        // loading any external shared library. Symbols below have ABI v1.
        let library = unsafe { Library::new(&path)? };
        let (create, evaluate, copy, close, info) = unsafe {
            let abi = *library.get::<unsafe extern "C" fn() -> u32>(b"ferrum_mlx_sdpa_abi_v1\0")?;
            if abi() != 1 {
                return Err("MLX bridge ABI mismatch".into());
            }
            let get_info = *library
                .get::<unsafe extern "C" fn(*mut Info, *mut c_char, usize) -> i32>(
                    b"ferrum_mlx_sdpa_info_v1\0",
                )?;
            let mut info = Info {
                version: [0; 128],
                library: [0; 1024],
                device: [0; 128],
                architecture: [0; 128],
                source_sha256: [0; 65],
            };
            let mut error = [0; 1024];
            check(get_info(&mut info, error.as_mut_ptr(), error.len()), &error)?;
            (
                *library.get::<Create>(b"ferrum_mlx_sdpa_create_v1\0")?,
                *library.get::<Evaluate>(b"ferrum_mlx_sdpa_evaluate_v1\0")?,
                *library.get::<CopyOutput>(b"ferrum_mlx_sdpa_copy_v1\0")?,
                *library.get::<Close>(b"ferrum_mlx_sdpa_close_v1\0")?,
                info,
            )
        };
        let version = string(&info.version);
        if version != expected_version {
            return Err(format!(
                "MLX runtime version {version:?} differs from expected {expected_version:?}"
            )
            .into());
        }
        let source_sha = format!("{:x}", Sha256::digest(include_bytes!("bridge.cpp")));
        if string(&info.source_sha256) != source_sha {
            return Err("native bridge was not built from this bridge.cpp source".into());
        }
        let mlx_library = Path::new(&string(&info.library)).canonicalize()?;
        let identity = json!({
            "abi":1, "bridge_path":path, "bridge_sha256":bridge_sha256,
            "bridge_source_sha256":source_sha, "mlx_version":version,
            "loaded_mlx_library":mlx_library, "loaded_mlx_library_sha256":file_sha256(&mlx_library)?,
            "device_name":string(&info.device), "architecture":string(&info.architecture),
            "library_identity_scope":"library containing mx::version, resolved from the live symbol; not a hash of all transitive libraries/metallib/JIT kernels",
            "dispatch_policy":"official MLX automatic GPU dispatch, force_fused=false; no architecture override set by this tool"
        });
        Ok(Self(Arc::new(Api {
            create,
            evaluate,
            copy,
            close,
            identity,
            _library: library,
        })))
    }

    pub fn identity(&self) -> &Value {
        &self.0.identity
    }

    pub fn create(&self, shape: Shape, inputs: &Inputs) -> Result<Session> {
        shape.validate()?;
        let q = shape.query_elements()?;
        let kv = shape.key_elements()?;
        if inputs.q.len() != q || inputs.k.len() != kv || inputs.v.len() != kv {
            return Err("bridge input extent differs from typed shape".into());
        }
        let config = Config {
            abi: 1,
            batch: shape.batch as u32,
            queries: shape.queries as u32,
            keys: shape.keys as u32,
            heads: shape.heads as u32,
            dim: shape.dim as u32,
            scale: SCALE,
            reserved: 0,
        };
        let mut raw = std::ptr::null_mut();
        let mut error = [0; 1024];
        // SAFETY: Exact lengths are checked, buffers live across the call; C++
        // copies their BF16 bits into MLX-owned storage before returning.
        let code = unsafe {
            (self.0.create)(
                &config,
                inputs.q.as_ptr(),
                q as u64,
                inputs.k.as_ptr(),
                inputs.v.as_ptr(),
                kv as u64,
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let session = Session {
            raw: NonNull::new(raw),
            api: self.0.clone(),
        };
        check(code, &error)?; // On error, session Drop still retains/retires owner.
        if session.raw.is_none() {
            return Err("MLX bridge returned a null session".into());
        }
        Ok(session)
    }
}

pub struct Session {
    raw: Option<NonNull<c_void>>,
    api: Arc<Api>,
}

impl Session {
    pub fn evaluate(&mut self) -> Result<()> {
        let raw = self.raw.ok_or("MLX session already closed")?;
        let mut error = [0; 1024];
        // SAFETY: This exclusively borrowed session owns the handle.
        check(
            unsafe { (self.api.evaluate)(raw.as_ptr(), error.as_mut_ptr(), error.len()) },
            &error,
        )
    }

    pub fn copy_output(&mut self, output: &mut [u16]) -> Result<()> {
        let raw = self.raw.ok_or("MLX session already closed")?;
        let mut error = [0; 1024];
        // SAFETY: Native validates output extent and availability before copying.
        check(
            unsafe {
                (self.api.copy)(
                    raw.as_ptr(),
                    output.as_mut_ptr(),
                    output.len() as u64,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            &error,
        )
    }

    pub fn close(&mut self) -> Result<()> {
        let Some(raw) = self.raw else {
            return Ok(());
        };
        let mut error = [0; 1024];
        // SAFETY: Native synchronizes before deletion; an error leaves the handle
        // valid and owned so it can be retained instead of unloading its code.
        check(
            unsafe { (self.api.close)(raw.as_ptr(), error.as_mut_ptr(), error.len()) },
            &error,
        )?;
        self.raw = None;
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("MLX session retirement failed; native handle and library retained until process exit: {error}");
            std::mem::forget(self.api.clone());
        }
    }
}

fn string(bytes: &[c_char]) -> String {
    String::from_utf8_lossy(
        &bytes
            .iter()
            .take_while(|x| **x != 0)
            .map(|x| *x as u8)
            .collect::<Vec<_>>(),
    )
    .into_owned()
}

fn check(code: i32, error: &[c_char]) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("MLX bridge status {code}: {}", string(error)).into())
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
