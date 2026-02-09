//! Metal execution context for Apple GPU operations

use crate::metal::error::MetalError;
use ferrum_types::FerrumError;
use metal::{CommandQueue, Device as MTLDevice, Library};
use tracing::debug;

// Include the compiled Metal library
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
include!(concat!(env!("OUT_DIR"), "/metal_lib.rs"));

/// Metal execution context that manages device, command queue, and shader library
pub struct MetalContext {
    pub device: MTLDevice,
    pub command_queue: CommandQueue,
    library: Option<Library>,
}

impl MetalContext {
    /// Create a new Metal context with the default system device
    pub fn new() -> Result<Self, FerrumError> {
        let device =
            MTLDevice::system_default().ok_or_else(|| MetalError::device_not_available())?;

        debug!("Creating Metal context with device: {}", device.name());

        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
            library: None,
        })
    }

    /// Load Metal shader library from embedded data  
    /// This will be called when Metal kernels are needed
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub fn load_shader_library(&mut self) -> Result<(), FerrumError> {
        if self.library.is_some() {
            debug!("Metal shader library already loaded");
            return Ok(());
        }

        if METAL_LIBRARY_DATA.is_empty() {
            debug!("No Metal shader library available, using CPU fallback");
            return Ok(());
        }

        let library = self
            .device
            .new_library_with_data(METAL_LIBRARY_DATA)
            .map_err(|e| {
                MetalError::compilation_failed(format!("Failed to load shader library: {}", e))
            })?;

        self.library = Some(library);
        debug!(
            "Metal shader library loaded successfully ({} bytes)",
            METAL_LIBRARY_DATA.len()
        );
        Ok(())
    }

    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    pub fn load_shader_library(&mut self) -> Result<(), FerrumError> {
        Err(MetalError::generic("Metal not available on this platform"))
    }

    /// Get the shader library (must be loaded first)
    pub fn library(&self) -> Option<&Library> {
        self.library.as_ref()
    }

    /// Get device memory information
    pub fn memory_info(&self) -> (u64, u64) {
        // On Apple Silicon, this is unified memory
        let recommended = self.device.recommended_max_working_set_size();
        let current = self.device.current_allocated_size();
        (current, recommended)
    }
}
