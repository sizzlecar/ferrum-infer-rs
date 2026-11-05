//! BLAS is a compute dependency, not a prerequisite for typed tensor transfers.
use super::*;

pub(super) fn submission_requires_blas<'a>(
    commands: impl IntoIterator<Item = &'a CudaDeviceCommand>,
) -> bool {
    commands
        .into_iter()
        .any(|command| command.tensor_transfer.is_none())
}

impl CudaDeviceRuntime {
    pub(super) fn ensure_submission_blas(
        &self,
        stream: &mut CudaDeviceStream,
        required: bool,
    ) -> Result<(), CudaDeviceRuntimeError> {
        self.validate_stream(stream)?;
        check_initialization_state(&stream.state)?;
        if !required || stream.blas.is_some() {
            return Ok(());
        }
        // Legacy streams are still eagerly initialized. A missing handle there
        // is a broken contract, not permission for an unaccounted lazy create.
        let lane = stream.tensor_lane.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "lazy cuBLAS creation requires the original tensor lane",
            )
        })?;
        let (blas, retained_lane) = create_dependency(&stream.state, Arc::clone(lane), || {
            let trace_enabled = tracing::enabled!(tracing::Level::DEBUG);
            let started = trace_enabled.then(Instant::now);
            let created = CudaBlas::new(Arc::clone(&stream.stream))
                .map(Arc::new)
                .map_err(|error| {
                    CudaDeviceRuntimeError::blas("lazy cuBLAS handle creation", error)
                });
            if trace_enabled {
                tracing::debug!(
                    stream_id = stream.id,
                    runtime_instance = self.runtime_instance,
                    blas_create_us = started.map_or(0, |started| {
                        u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
                    }),
                    succeeded = created.is_ok(),
                    "CUDA stream lazy BLAS creation host timing"
                );
            }
            created
        })?;
        stream.blas = Some(blas);
        // The original stream still owns this exact lane; guard/fence/command
        // retentions will accompany its use. No second reservation was made.
        drop(retained_lane);
        Ok(())
    }
}

impl CudaDeviceStream {
    pub(super) fn enqueue_command(
        &self,
        command: &CudaDeviceCommand,
    ) -> Result<CudaEnqueuedWork, CudaDeviceRuntimeError> {
        if let Some(transfer) = &command.tensor_transfer {
            transfer.enqueue(&self.stream)?;
            return Ok(CudaEnqueuedWork::Static);
        }
        let blas = self.blas.as_ref().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "non-transfer CUDA command lacks its cuBLAS dependency",
            )
        })?;
        command.enqueue(&self.stream, blas)
    }
}

fn check_initialization_state(state: &CudaStreamState) -> Result<(), CudaDeviceRuntimeError> {
    if state.failed.load(Ordering::Acquire) || state.recording.load(Ordering::Acquire) {
        return Err(CudaDeviceRuntimeError::contract(
            "failed or recording CUDA stream cannot initialize submission dependencies",
        ));
    }
    Ok(())
}

fn create_dependency<T, O>(
    state: &CudaStreamState,
    owner: Arc<O>,
    create: impl FnOnce() -> Result<T, CudaDeviceRuntimeError>,
) -> Result<(T, Arc<O>), CudaDeviceRuntimeError> {
    check_initialization_state(state)?;
    // create_native retains the original quota on an uncertain constructor
    // failure. Unlike create_stream, this caller already has a stream facade:
    // poison it on both Err and panic so it cannot repeatedly create unknown
    // native resources against that same retained allowance.
    match catch_unwind(AssertUnwindSafe(|| {
        tensor_eager::create_native(owner, create)
    })) {
        Ok(Ok(created)) => Ok(created),
        Ok(Err(error)) => {
            state.fail();
            Err(error)
        }
        Err(payload) => {
            state.fail();
            resume_unwind(payload)
        }
    }
}

#[cfg(test)]
mod tests;
