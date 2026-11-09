//! Reuse only the retired native stream/BLAS pair, never a logical execution lane.
use super::*;
use crate::backend::cuda_tensor_memory_policy::RetainedNative;
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CudaNativeLaneReuse {
    #[default]
    Off,
    Bounded {
        maximum_idle_lanes: NonZeroUsize,
    },
}

impl CudaNativeLaneReuse {
    pub(super) fn validate(self, maximum_streams: usize) -> Result<(), CudaDeviceRuntimeError> {
        if let Self::Bounded { maximum_idle_lanes } = self {
            if maximum_idle_lanes.get() > maximum_streams {
                return Err(CudaDeviceRuntimeError::contract(
                    "idle native lane limit exceeds the original stream quota",
                ));
            }
        }
        Ok(())
    }
}

pub(super) struct NativeLane {
    // Destroy BLAS (which itself retains the stream) before the stream owner.
    pub(super) blas: Arc<CudaBlasOwner>,
    pub(super) stream: Arc<CudaStream>,
    pub(super) previous_stream_id: u64,
}

type RetiredLane = RetainedNative<NativeLane, Arc<tensor_eager::TensorLaneRetention>>;

/// Owned directly by CudaDeviceRuntime. A lane has only a Weak return address:
/// native -> TensorLaneRetention -> TensorEagerState never points to this cache.
pub(super) struct IdleNativeLanes {
    maximum_idle: usize,
    entries: Mutex<Vec<RetiredLane>>,
}

impl IdleNativeLanes {
    pub(super) fn new(policy: CudaNativeLaneReuse) -> Option<Arc<Self>> {
        match policy {
            CudaNativeLaneReuse::Off => None,
            CudaNativeLaneReuse::Bounded { maximum_idle_lanes } => Some(Arc::new(Self {
                maximum_idle: maximum_idle_lanes.get(),
                entries: Mutex::new(Vec::new()),
            })),
        }
    }

    pub(super) fn take(&self) -> Result<Option<RetiredLane>, CudaDeviceRuntimeError> {
        self.entries
            .lock()
            .map(|mut entries| entries.pop())
            .map_err(|_| CudaDeviceRuntimeError::contract("idle native lane cache poisoned"))
    }

    fn has_room(&self) -> bool {
        self.entries
            .lock()
            .is_ok_and(|entries| entries.len() < self.maximum_idle)
    }

    fn put(&self, lane: RetiredLane) -> bool {
        let Ok(mut entries) = self.entries.lock() else {
            return false;
        };
        if entries.len() >= self.maximum_idle {
            return false;
        }
        entries.push(lane);
        true
    }
}

fn exclusively_owned<T>(owner: &mut Arc<T>) -> bool {
    Arc::get_mut(owner).is_some()
}

// Unlike the single-owner gates above, this is a balance check within a closed
// native ownership graph, not a general atomic Arc uniqueness predicate. The
// facade, exclusively owned CudaBlas, and optional owned workspace retain Arcs.
// Production never exports or downgrades this execution stream Arc: callbacks
// borrow only &CudaStream; fences/guards/graphs retain the paired owners checked
// above. If a future API exposes a native Weak, this contract must be redesigned.
fn only_private_native_pair(stream: &Arc<CudaStream>, blas: &CudaBlasOwner) -> bool {
    Arc::strong_count(stream) == 1 + blas.native_stream_owners() && Arc::weak_count(stream) == 0
}

impl CudaDeviceStream {
    pub(super) fn return_retired_native_lane(&mut self) {
        let Some(cache) = self.idle_native_lanes.upgrade() else {
            return;
        };
        let reason = if !exclusively_owned(&mut self.state) || !self.state.is_quiescent() {
            Some("state_not_exclusive_or_quiescent")
        } else if self.blas.is_none() || self.tensor_lane.is_none() {
            Some("no_initialized_blas")
        } else if !cache.has_room() {
            Some("idle_limit_or_poison")
        } else if !exclusively_owned(&mut self.terminal_readback_pool)
            || !self.terminal_readback_pool.retired()
        {
            Some("terminal_readback_not_retired")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.log_native_return(false, reason);
            return;
        }

        // Destruction must finish before testing exclusivity. A graph whose
        // destruction is indeterminate deliberately leaks the original native
        // owners; the reference counts below then prohibit their reuse.
        let retirement = catch_unwind(AssertUnwindSafe(|| {
            drop(std::mem::replace(
                &mut self.executable_cache,
                CudaExecutableCache::new(),
            ));
            drop(std::mem::replace(
                &mut self.terminal_readback_pool,
                Arc::new(terminal_readback::Pool::new()),
            ));
        }));
        if let Err(payload) = retirement {
            self.retain_uncertain_native_lane();
            resume_unwind(payload);
        }
        if !exclusively_owned(self.blas.as_mut().expect("eligibility checked BLAS"))
            || !exclusively_owned(self.tensor_lane.as_mut().expect("eligibility checked lane"))
            || !only_private_native_pair(
                &self.stream,
                self.blas.as_ref().expect("eligibility checked BLAS"),
            )
        {
            self.log_native_return(false, "external_native_owner_or_uncertain_graph");
            return;
        }
        let blas = self.blas.as_ref().expect("eligibility checked BLAS");
        let lane = self.tensor_lane.as_ref().expect("eligibility checked lane");
        match catch_unwind(AssertUnwindSafe(|| reset_blas(&self.stream, blas))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                self.retain_uncertain_native_lane();
                tracing::warn!(stream_id = self.id, runtime_instance = self.runtime_instance, error = %error, "CUDA retired native lane reset failed; original owners retained");
                return;
            }
            Err(payload) => {
                self.retain_uncertain_native_lane();
                resume_unwind(payload);
            }
        }
        let returned = cache.put(RetainedNative::new(
            NativeLane {
                blas: Arc::clone(blas),
                stream: Arc::clone(&self.stream),
                previous_stream_id: self.id,
            },
            Arc::clone(lane),
        ));
        self.log_native_return(
            returned,
            if returned {
                "retired"
            } else {
                "idle_limit_or_poison"
            },
        );
    }

    fn log_native_return(&self, returned: bool, reason: &'static str) {
        tracing::debug!(
            stream_id = self.id,
            runtime_instance = self.runtime_instance,
            native_lane_returned = returned,
            reason,
            "CUDA native lane retirement"
        );
    }

    fn retain_uncertain_native_lane(&self) {
        self.state.fail();
        std::mem::forget(Arc::clone(&self.stream));
        std::mem::forget(self.blas.clone());
        std::mem::forget(self.tensor_lane.clone());
    }
}

fn reset_blas(stream: &CudaStream, blas: &CudaBlas) -> Result<(), CudaDeviceRuntimeError> {
    use cudarc::cublas::sys;
    stream
        .context()
        .bind_to_thread()
        .map_err(|error| CudaDeviceRuntimeError::driver("retired BLAS context", error))?;
    let handle = *blas.handle();
    // SetStream unconditionally detaches a prior external workspace and restores
    // the vendor default pool, as in CudaBlas::new. SetWorkspace(null, 0) would
    // instead DISABLE that pool and does not restore fresh-handle semantics.
    unsafe {
        sys::cublasSetStream_v2(handle, stream.cu_stream() as _)
            .result()
            .map_err(|error| CudaDeviceRuntimeError::blas("retired BLAS stream reset", error))?;
        sys::cublasSetPointerMode_v2(handle, sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
            .result()
            .map_err(|error| CudaDeviceRuntimeError::blas("retired BLAS pointer reset", error))?;
        sys::cublasSetMathMode(handle, sys::cublasMath_t::CUBLAS_DEFAULT_MATH)
            .result()
            .map_err(|error| CudaDeviceRuntimeError::blas("retired BLAS math reset", error))?;
        sys::cublasSetAtomicsMode(handle, sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED)
            .result()
            .map_err(|error| CudaDeviceRuntimeError::blas("retired BLAS atomics reset", error))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
