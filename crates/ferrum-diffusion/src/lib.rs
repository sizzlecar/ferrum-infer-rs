//! Diffusion-domain schedules, spatial layouts, and numerical contracts.
//!
//! The FP32 reference borrows caller-owned slices. The BF16 Euler path owns a
//! shared runtime Host account and returns a charged input allocation; it does
//! not create a parallel resource account or allocate device memory.
//! Spatial layout transforms also borrow caller-owned storage; their checked
//! dimensions do not authorize allocation or execution in the device runtime.
//!
//! This crate does not implement a model pipeline, latent initialization, a
//! random-number generator, or experimentally established Torch/CUDA parity.

mod flow_match_euler;
mod latent_normalization;
mod spatial_latent;

pub use flow_match_euler::{
    Bf16EulerError, Bf16EulerOperand, Bf16EulerProfile, EulerStep, EulerStepError, FlowMatchEuler,
    FlowMatchEulerSchedule, ManagedBf16Euler, ScheduleError, SchedulerTimestep,
};
pub use latent_normalization::{
    denormalize_latent_f32_into, normalize_latent_f32_into, LatentNormalizationError,
};
pub use spatial_latent::{
    pack_spatial_into, unpack_spatial_into, LatentLayoutError, SingleFrameLatentShape,
    SpatialLatentAxis,
};
