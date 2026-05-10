//! Candle-tied `TensorFactory` impl used by the registry's stub-executor
//! factory.
//!
//! This module lived at `engine/src/backends/` until the `ComputeBackend`
//! trait was deleted; the directory was renamed to `tensor_factory/` so
//! the layout no longer suggests engine-level backend dispatch happens
//! here. Real GPU dispatch goes through
//! `ferrum_kernels::backend::Backend<B>` and its supertraits / capability
//! bundles. The only remaining job in this module is to mint dummy
//! tensors for the stub model executor (used by tests + the no-model
//! happy path).

pub mod candle;

pub use self::candle::{CandleTensor, CandleTensorFactory, CandleTensorOps};
