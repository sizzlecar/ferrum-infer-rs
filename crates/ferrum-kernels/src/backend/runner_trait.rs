//! Type-erased runner interface for GenericModelExecutor.
//!
//! ModelRunner<B> is generic over Backend, but the executor needs a concrete type.
//! This trait provides the type-erased interface.

use super::TransformerConfig;

/// Type-erased model runner. GenericModelExecutor holds a `Box<dyn RunnerInterface>`.
pub trait RunnerInterface: Send + Sync {
    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32>;
    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32>;
    fn release(&mut self, cache_id: &str);
    fn reset(&mut self);
    fn config(&self) -> &TransformerConfig;
}

// Blanket impl for any ModelRunner<B>
impl<B: super::Backend> RunnerInterface for super::runner::ModelRunner<B> {
    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        self.prefill(cache_id, tokens)
    }
    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        self.decode(cache_id, token, pos)
    }
    fn release(&mut self, cache_id: &str) {
        self.release(cache_id)
    }
    fn reset(&mut self) {
        self.reset()
    }
    fn config(&self) -> &TransformerConfig {
        self.config()
    }
}
