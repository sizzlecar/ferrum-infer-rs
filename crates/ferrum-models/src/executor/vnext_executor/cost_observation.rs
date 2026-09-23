//! Passive observation of actual physical waves. Caller-supplied lifecycle and
//! host inputs are correlated with actual prepared rows and device attribution.
//! Cold execution identity capture is introduced separately.
pub(super) mod dispatch;

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod product_tests;
