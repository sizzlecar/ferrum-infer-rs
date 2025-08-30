//! # Ferrum CLI Library
//! 
//! Command-line interface library for the Ferrum LLM inference framework.
//! 
//! ## Features
//! 
//! - **Server Management**: Start and manage inference servers
//! - **Model Testing**: Interactive model testing and validation
//! - **Benchmarking**: Performance testing and profiling
//! - **Configuration**: Config file management and validation
//! - **Health Monitoring**: Server and component health checks
//! - **Cache Operations**: Cache management and debugging
//! - **Development Tools**: Debugging and development utilities

pub mod commands;
pub mod config;
pub mod utils;
pub mod client;
pub mod output;

// Re-exports
pub use config::CliConfig;
pub use output::{OutputFormatter, OutputFormat};
pub use client::FerrumClient;
