//! Diffusion-domain schedules, spatial layouts, and numerical contracts.
//!
//! The CPU solver operates on caller-owned FP32 slices. It is a mathematical
//! reference, not a device executor or a resource-allocation authority. Future
//! device implementations must use Ferrum's managed runtime resources.
//! Spatial layout transforms also borrow caller-owned storage; their checked
//! dimensions do not authorize allocation or execution in the device runtime.
//!
//! This crate does not implement a model pipeline, latent initialization, a
//! random-number generator, or BF16/CUDA numerical parity.

mod flow_match_euler;
mod latent_normalization;
mod spatial_latent;

pub use flow_match_euler::{
    EulerStep, EulerStepError, FlowMatchEuler, FlowMatchEulerSchedule, ScheduleError,
    SchedulerTimestep,
};
pub use latent_normalization::{
    denormalize_latent_f32_into, normalize_latent_f32_into, LatentNormalizationError,
};
pub use spatial_latent::{
    pack_spatial_into, unpack_spatial_into, LatentLayoutError, SingleFrameLatentShape,
    SpatialLatentAxis,
};
