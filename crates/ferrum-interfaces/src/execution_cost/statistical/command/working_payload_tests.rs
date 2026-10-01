use super::*;

#[test]
fn selected_working_payload_covers_live_independent_rows_and_finished_table() {
    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(9);
    let mut digest_capacity = 0;
    builder
        .independent_attention_rows_v2(0u8..9, |builder, row| {
            if let super::super::independent_rows::IndependentRowsDigest::Rows {
                completed, ..
            } = &builder.independent
            {
                digest_capacity = completed.capacity();
            }
            // Distinct real algorithm keys cross the sparse table's first two
            // capacity boundaries while the independent-row digest is still live.
            for lane in [0u8, 1] {
                let class = SelectedAlgorithmClassV1::new(
                    "working.payload",
                    1,
                    [row + 1; 32],
                    [lane + 1; 32],
                )?;
                builder.kernel(
                    class,
                    KernelNumericWorkV1 {
                        logical_units: 1,
                        padded_units: 1,
                        inner_units_per_logical_unit: 3,
                        grid: [1, 1, 1],
                        scratch_bytes: 0,
                        staged_weight_bytes: 0,
                    },
                )?;
            }
            Ok(())
        })
        .unwrap();
    let evidence = builder.finish().unwrap();
    evidence
        .algorithm_work()
        .unwrap()
        .unwrap()
        .validate_command(&evidence)
        .unwrap();
    assert!(evidence.independent_attention_family_v2().is_some());
    let output = evidence.retained_payload_bytes().unwrap();
    let working = SelectedCommandCostEvidenceV1::maximum_working_payload_bytes(18).unwrap();
    assert!(output <= SelectedCommandCostEvidenceV1::maximum_payload_bytes(18).unwrap());
    assert!(
        working
            >= 2 * output
                + std::mem::size_of::<SelectedCommandCostBuilderV1>()
                + digest_capacity * std::mem::size_of::<[u8; 32]>()
    );
}
