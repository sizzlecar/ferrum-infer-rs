//! One fixed native module, retained until all providers and commands retire.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use cudarc::driver::CudaFunction;
use cudarc::nvrtc::Ptx;

use super::*;
use crate::backend::cuda::vnext_runtime::CudaTensorProviderRetention;

pub(super) struct RetainedFunctions {
    // Every CudaFunction retains the same native module/context. Field order
    // releases these owners before returning the one module capacity permit.
    functions: [CudaFunction; 5],
    _provider: CudaTensorProviderRetention,
}

impl RetainedFunctions {
    pub(super) fn load(runtime: &CudaDeviceRuntime) -> Result<Arc<Self>, CudaDeviceRuntimeError> {
        let provider = runtime.reserve_tensor_provider()?;
        let loaded = catch_unwind(AssertUnwindSafe(|| {
            let module = runtime
                .context()
                .load_module(Ptx::from_src(crate::ptx::VNEXT_BF16_SILU.to_owned()))
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 pointwise module load", error)
                })?;
            let load = |kind: Kind| {
                module.load_function(kind.function()).map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 pointwise function load", error)
                })
            };
            Ok::<_, CudaDeviceRuntimeError>([
                load(Kind::Silu)?,
                load(Kind::SiluMul)?,
                load(Kind::Tanh)?,
                load(Kind::Modulation)?,
                load(Kind::GatedResidual)?,
            ])
        }));
        let functions = match loaded {
            Ok(Ok(functions)) => functions,
            Ok(Err(error)) => {
                // A partially failed load has not certified native cleanup.
                std::mem::forget(provider);
                return Err(error);
            }
            Err(payload) => {
                std::mem::forget(provider);
                resume_unwind(payload);
            }
        };
        Ok(Arc::new(Self {
            functions,
            _provider: provider,
        }))
    }

    pub(super) fn function(&self, kind: Kind) -> &CudaFunction {
        &self.functions[kind.index()]
    }
}
