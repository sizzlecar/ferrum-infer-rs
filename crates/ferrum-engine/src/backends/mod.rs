//! Candle-tied tensor factory used by the registry's stub-executor
//! factory. Phase 3e+ deleted the legacy `ComputeBackend` trait, so the
//! once-bigger `CandleBackend` shrank to just the `CandleTensorFactory`
//! that mints dummy logits. Real GPU dispatch goes through
//! `ferrum_kernels::backend::Backend<B>`.

pub mod candle;

pub use self::candle::{CandleTensor, CandleTensorFactory, CandleTensorOps};
