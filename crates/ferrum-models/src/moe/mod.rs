//! Mixture-of-Experts runtime primitives.
//!
//! Three pieces, each in its own file:
//!   - [`router`] — softmax + top-K + optional re-norm to pick which experts
//!     handle each token, and what weight to combine their outputs with.
//!   - [`dispatch`] — load the per-layer expert weight stack from a GGUF
//!     file and run the per-token expert MLPs (gate / up / SiLU·mul /
//!     down) into a weighted sum.
//!
//! Phase 2 scope: CPU implementation only. Generic `Backend<B>` support is
//! a follow-up — the per-token, per-expert dispatch loop wants buffer
//! slicing and scaled accumulate, which the trait surface doesn't yet
//! expose cleanly. CPU first lets us validate the algorithm + numerics
//! against reference impls before committing to a backend-specific path.

pub mod dispatch;
pub mod layer;
pub mod router;

pub use dispatch::{moe_forward_cpu, ExpertStack};
pub use layer::Qwen3MoeLayer;
pub use router::{route, RouterOutput};
