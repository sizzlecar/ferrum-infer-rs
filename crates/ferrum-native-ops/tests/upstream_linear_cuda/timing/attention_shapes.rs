//! Existing MMQ/MMVQ arithmetic on attention projection dimensions, not a new route.
use super::*;

#[test]
#[ignore = "exclusive CUDA MarkerV2 paired attention diagnostic; run correctness first"]
fn marker_v2_m8_attention_mmq_mmvq_paired_working_sets() {
    let l2 = paired_device_l2_bytes();
    // Retained 64-layer GGUF inventory (dimensions are [N, K]): causal Q/K/V,
    // GDN QKV/gate, and GDN/causal output. Exclude the unexecuted blk.64 tensors.
    // These are synthetic finite weights of the observed formats and shapes;
    // no model payload is loaded and graph launch geometry does not supply K.
    for (k, n, formats) in [
        (
            5120,
            1024,
            &[UpstreamLinearFormat::Q4K, UpstreamLinearFormat::Q5K][..],
        ),
        (
            5120,
            6144,
            &[
                UpstreamLinearFormat::Q4K,
                UpstreamLinearFormat::Q5K,
                UpstreamLinearFormat::Iq4Xs,
            ][..],
        ),
        (
            5120,
            10240,
            &[
                UpstreamLinearFormat::Q4K,
                UpstreamLinearFormat::Q5K,
                UpstreamLinearFormat::Iq4Xs,
            ][..],
        ),
        (
            5120,
            12288,
            &[
                UpstreamLinearFormat::Q4K,
                UpstreamLinearFormat::Q5K,
                UpstreamLinearFormat::Iq4Xs,
            ][..],
        ),
        (
            6144,
            5120,
            &[
                UpstreamLinearFormat::Q4K,
                UpstreamLinearFormat::Q5K,
                UpstreamLinearFormat::Iq4Xs,
            ][..],
        ),
    ] {
        for &format in formats {
            // Each route has its own actual-pack oracle and repeat check.
            // MMQ and MMVQ quantization/reduction differ: do not require their
            // packed bytes or outputs to be bitwise equal to one another.
            paired_m8_working_sets(
                format,
                k,
                n,
                l2,
                &["resident", "arena_offset_ring"],
                "marker_v2_attention_m8_direct_paired",
            );
        }
    }
}
