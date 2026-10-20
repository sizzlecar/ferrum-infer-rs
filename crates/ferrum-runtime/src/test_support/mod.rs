//! Shared contract fixtures for downstream tests.
//!
//! These fixtures use public runtime admission, allocation and lifetime APIs.
//! They do not implement device tensor numerics or grant private authority.
//! Available only with the nondefault `test-support` feature (or unit tests).

#[path = "checkpoint.rs"]
pub mod checkpoint_fixture;
#[path = "resource.rs"]
pub mod resource_support;
#[path = "core.rs"]
pub mod vnext_core_contract;
