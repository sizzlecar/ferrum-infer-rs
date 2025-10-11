//! Model executor implementations

// pub mod candle_executor;  // TODO: Complete Candle integration
pub mod stub_executor;

// pub use candle_executor::CandleModelExecutor;
pub use stub_executor::StubModelExecutor;

