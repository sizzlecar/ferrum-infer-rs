//! # Ferrum CLI Library
//!
//! Ollama-style command-line interface for the Ferrum LLM inference framework.
//!
//! ## Commands
//!
//! - `run`: Run a model and start interactive chat
//! - `serve`: Start the HTTP inference server
//! - `stop`: Stop the running server
//! - `pull`: Download a model from HuggingFace
//! - `list`: List downloaded models

pub mod commands;
pub mod config;
pub mod utils;

// Re-exports
pub use config::CliConfig;
