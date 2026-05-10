//! Candle-tied `ComputeBackend` implementations.
//!
//! Moved from the (deleted) `ferrum-runtime` crate in Phase 5b/2. Only
//! the candle path stays here — the legacy CPU `ComputeBackend` impl
//! had no consumers and was dropped with the move. New backends should
//! implement `ferrum_kernels::backend::Backend<B>` (the production
//! abstraction), not `ComputeBackend`.

pub mod candle;
pub mod candle_kernel_ops;

pub use self::candle::{CandleBackend, CandleTensor, CandleTensorFactory, CandleTensorOps};
pub use self::candle_kernel_ops::CandleKernelOps;
