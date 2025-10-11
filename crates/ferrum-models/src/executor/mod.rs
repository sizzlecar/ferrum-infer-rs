//! Model executor implementations

pub mod candle_executor;
pub mod stub_executor;

pub use candle_executor::{CandleModelExecutor, CandleModelExecutorV2, extract_logits_safe};
pub use stub_executor::StubModelExecutor;

