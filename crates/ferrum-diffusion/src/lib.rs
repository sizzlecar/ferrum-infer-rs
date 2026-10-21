//! Diffusion-domain schedules and numerical contracts.
//!
//! The CPU solver operates on caller-owned FP32 slices. It is a mathematical
//! reference, not a device executor or a resource-allocation authority. Future
//! device implementations must use Ferrum's managed runtime resources.
//!
//! This crate does not implement a model pipeline, latent initialization, a
//! random-number generator, or BF16/CUDA numerical parity.

mod flow_match_euler;

pub use flow_match_euler::{
    EulerStep, EulerStepError, FlowMatchEuler, FlowMatchEulerSchedule, ScheduleError,
    SchedulerTimestep,
};
