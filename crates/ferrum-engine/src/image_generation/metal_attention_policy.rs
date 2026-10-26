//! Wire identity is available without opening a Metal device.

/// DiT attention implementation selection, independent of weight precision.
/// FusedDense declares the guarded dense operation; unsupported shapes, masks
/// and values use its strict backend fallback. It is not an all-dense promise.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageMetalAttentionPolicy {
    /// Preserve the existing Metal attention profiles.
    #[default]
    Reference,
    /// Explicitly select guarded dense attention and its strict fallback.
    FusedDense,
}
