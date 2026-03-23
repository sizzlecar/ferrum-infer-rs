//! Backend implementations for different compute devices

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub mod candle_kernel_ops;

#[cfg(feature = "cpu")]
pub mod cpu;

// Re-export backend implementations
#[cfg(feature = "candle")]
pub use self::candle::*;

#[cfg(feature = "candle")]
pub use self::candle_kernel_ops::CandleKernelOps;

#[cfg(feature = "cpu")]
pub use self::cpu::*;
