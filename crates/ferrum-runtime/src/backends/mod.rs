//! Backend implementations for different compute devices

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "cpu")]
pub mod cpu;

// Re-export backend implementations
#[cfg(feature = "candle")]
pub use self::candle::*;

#[cfg(feature = "cpu")]
pub use self::cpu::*;
