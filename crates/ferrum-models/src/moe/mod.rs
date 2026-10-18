//! Mixture-of-Experts runtime primitives.
//!
//! Shared routing and dispatch primitives:
//!   - [`router`] — softmax + top-K + optional re-norm to pick which experts
//!     handle each token, and what weight to combine their outputs with.
//!   - [`dispatch`] — load the per-layer expert weight stack from a GGUF
//!     file and run the per-token expert MLPs (gate / up / SiLU·mul /
//!     down) into a weighted sum.
//!
//! CPU dispatch provides a numerical reference for the backend implementations.
//! Production model-family execution is assembled by the typed family registry.

pub mod dispatch;
pub mod router;

pub use dispatch::{
    moe_forward, moe_forward_bucketed, moe_forward_cpu, ExpertStack, MoeBucketPlan,
    MoeForwardBucketedParams, MoeForwardParams, MoeRouteScratch,
};
pub use router::{route, route_into, RouterOutput};
