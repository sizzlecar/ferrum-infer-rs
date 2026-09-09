use ferrum_types::{Device, FerrumError, Result};

/// Probe only device initialization, before resolving or allocating a model.
/// Model, capacity and operator failures keep their original error paths.
pub(crate) fn select_device(backend: &str) -> Result<Device> {
    let accelerators = [
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal,
        #[cfg(feature = "cuda")]
        Device::CUDA(0),
    ];
    select_with_probe(backend, &accelerators, |device| match device {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Device::Metal => candle_core::Device::new_metal(0)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        #[cfg(feature = "cuda")]
        Device::CUDA(index) => {
            ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime::probe_device(*index)
                .map_err(|error| error.to_string())
        }
        _ => Err("accelerator support is absent from this binary".to_owned()),
    })
}

fn select_with_probe(
    backend: &str,
    accelerators: &[Device],
    mut probe: impl FnMut(&Device) -> std::result::Result<(), String>,
) -> Result<Device> {
    let requested = backend.trim().to_lowercase();
    let device = match requested.as_str() {
        "cpu" => return Ok(Device::CPU),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        "metal" => Device::Metal,
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        "metal" => return Err(unsupported_backend(&requested)),
        "cuda" => Device::CUDA(0),
        "auto" => {
            for device in accelerators {
                match probe(device) {
                    Ok(()) => return Ok(device.clone()),
                    Err(reason) => {
                        tracing::info!(?device, %reason, "Accelerator initialization unavailable; trying the next backend")
                    }
                }
            }
            return Ok(Device::CPU);
        }
        other => {
            return Err(FerrumError::config(format!(
                "unknown backend {other:?}; expected one of: auto, cpu, metal, cuda"
            )))
        }
    };
    if !accelerators.contains(&device) {
        return Err(unsupported_backend(&requested));
    }
    probe(&device).map_err(|reason| FerrumError::config(format!(
        "requested backend '{requested}' could not initialize: {reason}; use --backend auto or --backend cpu to allow CPU execution"
    )))?;
    Ok(device)
}

fn unsupported_backend(requested: &str) -> FerrumError {
    FerrumError::config(format!(
        "requested backend '{requested}' but this ferrum binary was not built with that backend; use --backend auto/cpu or install the matching package"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an actual CUDA device"]
    fn native_cuda_device_selection_initializes_the_product_driver() {
        assert_eq!(select_device("cuda").unwrap(), Device::CUDA(0));
        #[cfg(not(feature = "metal"))]
        assert_eq!(select_device("auto").unwrap(), Device::CUDA(0));
        assert_eq!(select_device("cpu").unwrap(), Device::CPU);
    }

    #[test]
    fn automatic_selection_requires_a_working_device_and_can_use_cpu() {
        let devices = [Device::CUDA(0), Device::CUDA(1)];
        let mut probed = Vec::new();
        let actual = select_with_probe("auto", &devices, |device| {
            probed.push(device.clone());
            if device == &Device::CUDA(0) {
                Err("device unavailable".into())
            } else {
                Ok(())
            }
        })
        .unwrap();
        assert_eq!(actual, Device::CUDA(1));
        assert_eq!(probed, devices);
        assert_eq!(
            select_with_probe("auto", &devices, |_| Err("driver unavailable".into())).unwrap(),
            Device::CPU
        );
        assert_eq!(
            select_with_probe("auto", &[], |_| panic!("CPU package probed a GPU")).unwrap(),
            Device::CPU
        );
    }

    #[test]
    fn explicit_cpu_skips_probe_and_explicit_gpu_preserves_initialization_error() {
        assert_eq!(
            select_with_probe(" CPU ", &[Device::CUDA(0)], |_| panic!(
                "explicit CPU probed a GPU"
            ))
            .unwrap(),
            Device::CPU
        );
        let error = select_with_probe("cuda", &[Device::CUDA(0)], |_| {
            Err("driver unavailable".into())
        })
        .unwrap_err();
        assert!(error.to_string().contains("driver unavailable"));
        assert!(select_with_probe("cuda", &[], |_| panic!("uncompiled GPU probed")).is_err());
        assert!(select_with_probe("unknown", &[Device::CUDA(0)], |_| panic!(
            "unknown backend probed"
        ))
        .is_err());
    }
}
