//! Test utilities and helpers
//!
//! This module provides common utilities for tests to avoid duplication
//! and prevent issues like multiple tracing initializations.

use once_cell::sync::Lazy;
use std::sync::Once;

/// Global test initialization
static TEST_INIT: Once = Once::new();

/// Initialize test environment once for all tests
pub fn init_test_env() {
    TEST_INIT.call_once(|| {
        // Set minimal log level to reduce noise in tests
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "error");
        }
        
        // Try to initialize tracing, ignore if already initialized
        let _ = tracing_subscriber::fmt()
            .with_env_filter("error")
            .with_test_writer()
            .try_init();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_test_env_is_safe_to_call_multiple_times() {
        init_test_env();
        init_test_env();
        init_test_env();
        // Should not panic or cause issues
    }
}

