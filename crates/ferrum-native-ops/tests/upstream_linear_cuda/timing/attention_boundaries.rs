//! Probe a proposed attention geometry boundary with existing MMQ/MMVQ arithmetic.
use super::*;

#[test]
#[ignore = "exclusive CUDA MarkerV2 paired attention boundary diagnostic; run correctness first"]
fn marker_v2_m8_attention_mmq_mmvq_geometry_boundaries() {
    let l2 = paired_device_l2_bytes();
    // Synthetic boundary probes for the proposed K >= 4096 and N >= 4096
    // region. The smaller K or N cases are outside controls, not model shapes.
    // No dispatch policy or performance assertion is encoded in this fixture.
    for format in [UpstreamLinearFormat::Q4K, UpstreamLinearFormat::Q5K] {
        for (k, n) in [
            (3072, 4096),
            (4096, 3072),
            (4096, 4096),
            (4096, 5120),
            (5120, 3072),
            (5120, 4096),
        ] {
            // Preserve each route's oracle and all pairs. MMQ and MMVQ have
            // different quantization/reduction, so cross-route bitwise output
            // equality is not the numerical contract.
            paired_m8_working_sets(
                format,
                k,
                n,
                l2,
                &["resident", "arena_offset_ring"],
                "marker_v2_attention_m8_boundary_paired",
            );
        }
    }
}
