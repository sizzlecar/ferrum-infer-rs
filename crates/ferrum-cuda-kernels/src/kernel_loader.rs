//! Kernel loader: compiles CUDA source to PTX via nvrtc and caches modules.

use candle_core::cuda_backend::CudaDevice;
use std::collections::HashMap;
use std::sync::RwLock;
use tracing::info;

/// Holds compiled CUDA modules, one per kernel source file.
pub struct KernelStore {
    device: CudaDevice,
    loaded: RwLock<HashMap<&'static str, ()>>,
}

impl KernelStore {
    /// Create a store bound to a candle CudaDevice.
    pub fn new(device: CudaDevice) -> Self {
        Self {
            device,
            loaded: RwLock::new(HashMap::new()),
        }
    }

    /// Ensure a kernel module is loaded. `module_name` is used as the cache key.
    /// `cuda_src` is the CUDA C++ source, `func_names` are the entry points.
    pub(crate) fn ensure_loaded(
        &self,
        module_name: &'static str,
        cuda_src: &str,
        func_names: &[&str],
    ) -> candle_core::Result<()> {
        // Fast path: already loaded
        {
            let loaded = self.loaded.read().unwrap();
            if loaded.contains_key(module_name) {
                return Ok(());
            }
        }

        // Slow path: compile and load
        info!("Compiling CUDA kernel module: {}", module_name);

        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_src, opts)
            .map_err(|e| candle_core::Error::Msg(format!("nvrtc compile '{module_name}': {e}")))?;

        // Load all functions from this module
        for &func_name in func_names {
            let _ = self.device.get_or_load_custom_func(func_name, module_name, &ptx)?;
        }

        let mut loaded = self.loaded.write().unwrap();
        loaded.insert(module_name, ());
        info!("CUDA kernel module '{}' ready", module_name);
        Ok(())
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}
