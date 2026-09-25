//! A sealed algorithm template contains no old numeric work or algorithm table.
//! Matching it is necessary for passive replay evidence, never sufficient for
//! execution: the runtime must also validate its resident program and ordinal.
use super::{SelectedCommandCostEvidenceV1, StatisticalEvidenceUnknown};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SelectedReplayAlgorithmTemplateV1 {
    schema_version: u32,
    family_signature: [u8; 32],
    fixed_launch_signature: [u8; 32],
    token_count: u64,
    compute_dispatches: u64,
    transfer_commands: u64,
}

impl SelectedReplayAlgorithmTemplateV1 {
    /// Only checked evidence from the captured actual selected command is a
    /// valid template input. Do not construct templates from legacy hashes or
    /// deserialize source JSON into live templates.
    pub fn from_selected(
        evidence: &SelectedCommandCostEvidenceV1,
        tokens: u64,
        compute: u64,
        transfers: u64,
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        evidence.validate_command(tokens, compute, transfers)?;
        evidence
            .algorithm_work()
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)??
            .validate_command(evidence)?;
        if compute == 0 && transfers == 0 {
            return Err(StatisticalEvidenceUnknown::InvalidWork);
        }
        Ok(Self {
            schema_version: 1,
            family_signature: *evidence.family_signature(),
            fixed_launch_signature: evidence.replay_fixed_launch_signature()?,
            token_count: tokens,
            compute_dispatches: compute,
            transfer_commands: transfers,
        })
    }

    /// Validate freshly produced numeric work for the same selected algorithm
    /// sequence. Work may differ because current context/input bindings differ;
    /// callers must derive it from that same invocation, never first capture.
    pub fn validate_binding(
        &self,
        evidence: &SelectedCommandCostEvidenceV1,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.schema_version != 1 || &self.family_signature != evidence.family_signature() {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        if self.fixed_launch_signature != evidence.replay_fixed_launch_signature()? {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        evidence.validate_command(
            self.token_count,
            self.compute_dispatches,
            self.transfer_commands,
        )?;
        evidence
            .algorithm_work()
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)??
            .validate_command(evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_cost::{
        KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
        SelectedCommandCostBuilderV1,
    };

    fn command(
        tokens: u64,
        context: u64,
        class: &str,
        algorithm_table: bool,
    ) -> SelectedCommandCostEvidenceV1 {
        let mut builder = if algorithm_table {
            SelectedCommandCostBuilderV1::new_with_algorithm_work(tokens)
        } else {
            SelectedCommandCostBuilderV1::new(tokens)
        };
        builder
            .kernel_with_replay_geometry(
                SelectedAlgorithmClassV1::new(class, 1, [1; 32], [2; 32]).unwrap(),
                KernelNumericWorkV1 {
                    logical_units: tokens,
                    padded_units: tokens,
                    inner_units_per_logical_unit: context,
                    grid: [u32::try_from(tokens).unwrap(), 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                },
                KernelReplayGeometryV1 {
                    block: [128, 1, 1],
                    dynamic_shared_bytes: 0,
                    fixed_parameters: &[64],
                },
            )
            .unwrap();
        builder.finish().unwrap()
    }

    #[test]
    fn replay_template_accepts_fresh_numeric_context_without_retaining_capture_work() {
        let captured = command(4, 32, "paged.attention", true);
        let template =
            SelectedReplayAlgorithmTemplateV1::from_selected(&captured, 4, 1, 0).unwrap();
        let next = command(4, 257, "paged.attention", true);
        assert_ne!(captured.work(), next.work());
        template.validate_binding(&next).unwrap();
        assert_eq!(
            template,
            SelectedReplayAlgorithmTemplateV1::from_selected(&next, 4, 1, 0).unwrap()
        );
    }

    #[test]
    fn replay_template_rejects_changed_algorithm_tokens_counts_or_missing_assignment() {
        let captured = command(4, 32, "paged.attention", true);
        let template =
            SelectedReplayAlgorithmTemplateV1::from_selected(&captured, 4, 1, 0).unwrap();
        for wrong in [
            command(4, 32, "different.attention", true),
            command(3, 32, "paged.attention", true),
            command(4, 32, "paged.attention", false),
        ] {
            assert!(template.validate_binding(&wrong).is_err());
        }
        assert!(SelectedReplayAlgorithmTemplateV1::from_selected(&captured, 4, 2, 0).is_err());
        assert!(SelectedReplayAlgorithmTemplateV1::from_selected(&captured, 4, 1, 1).is_err());
        assert!(SelectedReplayAlgorithmTemplateV1::from_selected(
            &command(4, 32, "paged.attention", false),
            4,
            1,
            0
        )
        .is_err());
    }
    #[test]
    fn replay_template_rejects_changed_fixed_grid_even_with_equal_grid_product_and_family() {
        let emit = |grid, logical, padded, scratch, staged, block, shared, scalar| {
            let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(4);
            builder
                .kernel_with_replay_geometry(
                    SelectedAlgorithmClassV1::new("same.entry", 1, [1; 32], [2; 32]).unwrap(),
                    KernelNumericWorkV1 {
                        logical_units: logical,
                        padded_units: padded,
                        inner_units_per_logical_unit: 32,
                        grid,
                        scratch_bytes: scratch,
                        staged_weight_bytes: staged,
                    },
                    KernelReplayGeometryV1 {
                        block,
                        dynamic_shared_bytes: shared,
                        fixed_parameters: &[scalar],
                    },
                )
                .unwrap();
            builder.finish().unwrap()
        };
        let captured = emit([4, 2, 1], 4, 4, 16, 0, [128, 1, 1], 0, 64);
        let template =
            SelectedReplayAlgorithmTemplateV1::from_selected(&captured, 4, 1, 0).unwrap();
        for wrong in [
            emit([2, 4, 1], 4, 4, 16, 0, [128, 1, 1], 0, 64),
            emit([8, 1, 1], 4, 4, 16, 0, [128, 1, 1], 0, 64),
            emit([4, 2, 1], 3, 4, 16, 0, [128, 1, 1], 0, 64),
            emit([4, 2, 1], 4, 8, 16, 0, [128, 1, 1], 0, 64),
            emit([4, 2, 1], 4, 4, 32, 0, [128, 1, 1], 0, 64),
            emit([4, 2, 1], 4, 4, 16, 32, [128, 1, 1], 0, 64),
            emit([4, 2, 1], 4, 4, 16, 0, [64, 2, 1], 0, 64),
            emit([4, 2, 1], 4, 4, 16, 0, [128, 1, 1], 32, 64),
            emit([4, 2, 1], 4, 4, 16, 0, [128, 1, 1], 0, 65),
        ] {
            assert_eq!(captured.family_signature(), wrong.family_signature());
            assert_eq!(captured.work().grid_blocks, wrong.work().grid_blocks);
            assert!(template.validate_binding(&wrong).is_err());
        }
    }
    #[test]
    fn replay_template_requires_every_kernel_to_supply_fixed_geometry() {
        let algorithm = SelectedAlgorithmClassV1::new("same.entry", 1, [1; 32], [2; 32]).unwrap();
        let work = KernelNumericWorkV1 {
            logical_units: 4,
            padded_units: 4,
            inner_units_per_logical_unit: 32,
            grid: [4, 1, 1],
            scratch_bytes: 0,
            staged_weight_bytes: 0,
        };
        let geometry = KernelReplayGeometryV1 {
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            fixed_parameters: &[64],
        };
        for missing_first in [true, false] {
            let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(4);
            if missing_first {
                builder.kernel(algorithm, work).unwrap();
            }
            builder
                .kernel_with_replay_geometry(algorithm, work, geometry)
                .unwrap();
            if !missing_first {
                builder.kernel(algorithm, work).unwrap();
            }
            let evidence = builder.finish().unwrap();
            evidence
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(&evidence)
                .unwrap();
            assert_eq!(
                SelectedReplayAlgorithmTemplateV1::from_selected(&evidence, 4, 2, 0),
                Err(StatisticalEvidenceUnknown::MissingProducer)
            );
        }
        let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(4);
        assert_eq!(
            builder.kernel_with_replay_geometry(
                algorithm,
                work,
                KernelReplayGeometryV1 {
                    block: [0, 1, 1],
                    ..geometry
                }
            ),
            Err(StatisticalEvidenceUnknown::InvalidWork)
        );
        assert_eq!(
            builder.kernel_with_replay_geometry(algorithm, work, geometry),
            Err(StatisticalEvidenceUnknown::InvalidWork)
        );
        assert!(builder.finish().is_err());
    }
}
