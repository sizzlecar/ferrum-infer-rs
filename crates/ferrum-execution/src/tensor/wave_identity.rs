//! Immutable identity metadata for the last admitted tensor execution lane.
use super::*;
use std::sync::Weak;

struct CachedIdentity<R: DeviceRuntime> {
    // Metadata must not extend stream/native-handle ownership after an actor
    // fails or an ordinary one-shot execution retires.
    lane: Weak<ExecutionLane<R>>,
    topology: CompiledSubmissionWaveIdentity,
}

pub(super) struct TensorWaveIdentityCache<R: DeviceRuntime> {
    cached: Option<CachedIdentity<R>>,
    #[cfg(test)]
    compilations: usize,
}

impl<R: DeviceRuntime> Default for TensorWaveIdentityCache<R> {
    fn default() -> Self {
        Self {
            cached: None,
            #[cfg(test)]
            compilations: 0,
        }
    }
}

impl<R: DeviceRuntime> TensorWaveIdentityCache<R> {
    /// Called only after ordinary step/wave admission has checked the runtime
    /// owner. A new lane or plan rebuilds metadata; no request authority is cached.
    pub(super) fn get_or_compile(
        &mut self,
        executable: &dyn ExecutablePlanView,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<&CompiledSubmissionWaveIdentity, VNextError> {
        let plan = executable.execution_plan();
        let matches = self.cached.as_ref().is_some_and(|cached| {
            cached
                .lane
                .upgrade()
                .is_some_and(|previous| Arc::ptr_eq(&previous, lane))
                && cached.topology.lane_id() == lane.id()
                && cached.topology.plan_id() == plan.payload().plan_id()
                && cached.topology.plan_hash() == plan.plan_hash()
                && cached.topology.device_id() == plan.payload().device_id()
                && cached.topology.runtime_implementation_fingerprint()
                    == plan.payload().device_runtime_implementation_fingerprint()
        });
        if !matches {
            let topology = OperationDispatch::compile_submission_wave_identity(executable, lane)?;
            self.cached = Some(CachedIdentity {
                lane: Arc::downgrade(lane),
                topology,
            });
            #[cfg(test)]
            {
                self.compilations += 1;
            }
        }
        Ok(&self
            .cached
            .as_ref()
            .expect("compiled tensor wave identity")
            .topology)
    }
}

#[cfg(test)]
mod tests;
