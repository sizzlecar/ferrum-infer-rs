// Shared implementation stays in this package and is also available to downstream contract tests.
#[path = "../../src/test_support/core.rs"]
mod shared;
pub(crate) use shared::*;
