//! Cold obligations from the installed causal selector. These are checked
//! numerical boundaries, never page ownership or permission to execute.
use super::*;
use ferrum_interfaces::vnext::{
    DecodeContextBoundary, DecodeContextBoundaryKind, DecodeContextCoverage,
};
use std::num::NonZeroU64;

pub(super) fn coverage(
    shape: CausalAttentionShape,
    caps: cost_route::Capabilities,
) -> DecodeContextCoverage {
    let Some(maximum) = NonZeroU64::new(shape.maximum_context_tokens) else {
        return Default::default();
    };
    let unknown = |known_boundaries| DecodeContextCoverage::Unknown {
        maximum_sequence_tokens: Some(maximum),
        known_boundaries,
    };
    if caps.maximum_attention_simdgroups == 0
        || u64::from(caps.maximum_attention_simdgroups) > MAXIMUM_ATTENTION_SIMDGROUPS
        || shape.validate_page_count(caps.kv_type).is_err()
        || params(shape, caps, maximum.get()).is_none()
    {
        return unknown(Vec::new());
    }

    // At most the installed SIMDgroup ramp and one grouped-kernel threshold
    // can change the selected algorithm class for a one-token decode.
    // Use its constants as candidate cuts, then call the original selector
    // on BOTH sides. In particular INT8 never inherits the F16 grouped path.
    let mut candidates = Vec::with_capacity((MAXIMUM_ATTENTION_SIMDGROUPS + 1) as usize);
    candidates.extend(2..=u64::from(caps.maximum_attention_simdgroups));
    candidates.push(GROUPED_DECODE_MINIMUM_CONTEXT);
    candidates.sort_unstable();
    candidates.dedup();
    let mut boundaries = Vec::with_capacity(candidates.len());
    for sequence in candidates {
        if sequence < 2 || sequence > maximum.get() {
            continue;
        }
        let (Some(before), Some(after)) = (
            params(shape, caps, sequence - 1),
            params(shape, caps, sequence),
        ) else {
            return unknown(boundaries);
        };
        let (a, b) = (caps.dispatch_plan(&before), caps.dispatch_plan(&after));
        // selected::kernel binds the entry, specialization, threads and
        // threadgroup memory into the algorithm class. Grid dimensions are
        // numeric work and do not by themselves create another class.
        if (
            a.kind,
            a.threads_per_threadgroup,
            a.threadgroup_memory_bytes,
        ) != (
            b.kind,
            b.threads_per_threadgroup,
            b.threadgroup_memory_bytes,
        ) {
            boundaries.push(DecodeContextBoundary {
                first_sequence_tokens: NonZeroU64::new(sequence).unwrap(),
                kind: DecodeContextBoundaryKind::KernelFamily,
            });
        }
    }
    // These are the kernel-family obligations only. A grouped reduction's
    // changing grid is numeric work in the SAME algorithm class. Its replay
    // partition, exact-span topology and cross-row physical alias decisions
    // remain Unknown; do not turn those unproved replay cuts into independently
    // qualified kernel families or claim that an empty list proves completeness.
    unknown(boundaries)
}

fn params(
    shape: CausalAttentionShape,
    caps: cost_route::Capabilities,
    sequence: u64,
) -> Option<CausalAttentionParams> {
    cost_route::row_params(shape, 1, sequence.checked_sub(1)?, sequence, caps).ok()
}

#[cfg(test)]
mod tests;
