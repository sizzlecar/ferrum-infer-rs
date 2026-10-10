//! Probe the narrower wide-attention region after the 4096 boundary regressed.
use super::*;

#[test]
#[ignore = "exclusive CUDA MarkerV2 paired wide-attention boundary diagnostic; run correctness first"]
fn marker_v2_m8_attention_mmq_mmvq_wide_geometry_boundaries() {
    let l2 = paired_device_l2_bytes();
    // Predeclared second hypothesis: K >= 5120 and physical leaf N >= 5120.
    // Four points probe the corner and nearby aligned dimensions; two smaller
    // dimensions are outside controls. This does not assert a universal optimum.
    for format in [UpstreamLinearFormat::Q4K, UpstreamLinearFormat::Q5K] {
        for (k, n) in [
            (4864, 5120),
            (5120, 4864),
            (5120, 5120),
            (5120, 5376),
            (5376, 5120),
            (5376, 5376),
        ] {
            paired_m8_working_sets(
                format,
                k,
                n,
                l2,
                &["resident", "arena_offset_ring"],
                "marker_v2_attention_m8_wide_boundary_paired",
            );
        }
    }
}
