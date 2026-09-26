//! Real materialization and execution for the three explicit RN-F16 aliases.
//! This is backend conformance, not full-vocab or serving qualification.
use super::*;
use ferrum_kernels::gguf_f16_projection_materializer::{
    gguf_f16_projection_inventory, GGUF_F16_PROJECTION_FORMAT_ID,
    GGUF_F16_PROJECTION_MATERIALIZER_ID,
};
use std::num::NonZeroU64;

#[path = "q8_ffn_cost_family.rs"]
mod ffn_family;

#[path = "gguf_f16_aliases/attribution.rs"]
mod attribution;

const PROFILE: &str = "fixture.explicit.gguf-rn-f16-projections";

fn replacement(operation: &OperationId) -> Option<OperationId> {
    Some(id(match operation.as_str() {
        DENSE_SWIGLU_OPERATION_ID => DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID => {
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID
        }
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID => {
            CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID
        }
        _ => return None,
    }))
}

struct Rounded<T>(T);
impl<T: ModelFamilyProvider<Config = AttentionKind>> ModelFamilyProvider for Rounded<T> {
    type Config = AttentionKind;
    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &AttentionKind,
    ) -> Result<(), VNextError> {
        self.0.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &AttentionKind,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<AttentionKind, VNextError> {
        self.0.parse_config(raw)
    }
    fn semantic_metadata(
        &self,
        config: &AttentionKind,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(config)
    }
    fn weight_schema(&self, config: &AttentionKind) -> Result<WeightSchema, VNextError> {
        self.0.weight_schema(config)
    }
    fn numerical_profiles(
        &self,
        config: &AttentionKind,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let original = self.0.numerical_profiles(config)?;
        let mut profile = original.profiles()[0].clone();
        profile.id = id(PROFILE);
        for operation in &mut profile.operations {
            if let Some(rounded) = replacement(&operation.operation_id) {
                operation.operation_id = rounded;
                operation.version = ContractVersion::new(1, 0);
                operation.multiplication_type = Some(ElementType::F16);
                operation.accumulation_type = Some(ElementType::F32);
            }
        }
        FamilyNumericalProfiles::new(
            self.family_id(),
            original.version(),
            vec![profile],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        config: &AttentionKind,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let original = self.0.semantic_program(config, profile)?;
        let mut blocks = original.blocks().to_vec();
        for block in &mut blocks {
            for node in &mut block.nodes {
                if let Some(operation) = replacement(&node.operation_id) {
                    node.operation_id = operation;
                    node.required_version = ContractVersion::new(1, 0);
                }
            }
        }
        ModelProgram::new(
            self.family_id().clone(),
            original.inputs().to_vec(),
            blocks,
            original.states().to_vec(),
            original.weights().to_vec(),
            original.outputs().to_vec(),
        )?
        .with_checkpoint_inputs(ProgramCheckpointInputs::new(
            id("value.tokens"),
            BTreeSet::new(),
        )?)
    }
}

fn fixture<T: ModelFamilyProvider<Config = AttentionKind> + 'static>(
    definition: T,
    kind: AttentionKind,
    output_node: &str,
) -> Fixture {
    fixture_with_mode(
        definition,
        kind,
        output_node,
        FixtureExecutionMode::Eager,
        1,
    )
}

fn fixture_with_mode<T: ModelFamilyProvider<Config = AttentionKind> + 'static>(
    definition: T,
    kind: AttentionKind,
    output_node: &str,
    mode: FixtureExecutionMode,
    maximum_tokens: u64,
) -> Fixture {
    let states = Family::new(kind).states();
    let registration = TypedFamilyRegistration::new(Rounded(definition));
    let family = registration
        .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(PROFILE))
        .unwrap();
    let inventory = gguf_f16_projection_inventory(&family).unwrap();
    assert!(inventory.converted_f16_bytes > 0 && inventory.retained_consumed_bytes > 0);
    // This is the same selector and composition used by run/serve/teacher.
    let composition = composition(kind, &family);
    assert_eq!(
        composition.3.materializer_id().as_str(),
        GGUF_F16_PROJECTION_MATERIALIZER_ID
    );
    let fixture = Fixture::from_prepared_family_with_composition(
        kind,
        family.clone(),
        states,
        mode,
        None,
        maximum_tokens,
        BTreeMap::new(),
        composition,
    );
    attribution::assert_product_witness(&registration, &family, &fixture, &inventory);
    let executable = fixture.compilation.executable();
    let index = executable
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .position(|n| n.id().as_str() == output_node)
        .unwrap();
    let node = &executable.execution_plan().payload().nodes()[index];
    assert!(gguf_f16_projection_role_v1(node.operation_id(), 2).is_some());
    for value in node.values() {
        if value.role() == ResolvedValueRole::Input
            && gguf_f16_projection_role_v1(node.operation_id(), value.ordinal()).is_some()
        {
            let weight = value.weight().unwrap();
            assert!(matches!(
                weight.physical_layout(),
                PhysicalWeightLayout::Dense { .. }
            ));
            assert_eq!(
                weight.components()[0].physical_element_type(),
                ElementType::F16
            );
        }
    }
    let provider = &fixture.providers.providers()[index];
    assert!(provider
        .descriptor()
        .accepted_weight_formats()
        .contains(&id(GGUF_F16_PROJECTION_FORMAT_ID)));
    for counts in [&[1][..], &[3][..], &[2, 3][..], &[4, 4][..], &[9][..]] {
        let rows = counts
            .iter()
            .map(|&count| OperationCostWorkRow {
                offset: 0,
                count: NonZeroU64::new(count as u64).unwrap(),
                full_input_tokens: NonZeroU64::new(count as u64).unwrap(),
            })
            .collect::<Vec<_>>();
        assert!(
            provider
                .eager_cost_route(executable, &rows)
                .unwrap()
                .is_none(),
            "opaque GEMM cannot borrow native cost evidence"
        );
    }
    fixture
}

fn check_outputs(fixture: &Fixture, node: &str, dtype: ElementType) {
    for counts in [&[1][..], &[3][..], &[2, 3][..], &[4, 4][..], &[9][..]] {
        let first = full_cost_route::run_with_output(fixture, counts, false, node).0;
        let repeated = full_cost_route::run_with_output(fixture, counts, false, node).0;
        assert_eq!(
            first, repeated,
            "same rounded policy must preserve repeated execution"
        );
        assert_eq!(first.len(), counts.len());
        for (row, &count) in first.into_iter().zip(counts) {
            assert_eq!(row.len() as u64, count as u64 * HIDDEN * dtype.size_bytes());
            let values = match dtype {
                ElementType::F16 => row
                    .chunks_exact(2)
                    .map(|x| f16::from_le_bytes(x.try_into().unwrap()).to_f32())
                    .collect::<Vec<_>>(),
                ElementType::F32 => row
                    .chunks_exact(4)
                    .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                    .collect::<Vec<_>>(),
                _ => unreachable!(),
            };
            assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
            assert!(
                values.iter().any(|x| *x != 0.0),
                "RN-F16 output cannot be degenerate"
            );
        }
    }
}

#[test]
fn gguf_rn_f16_cuda_gdn_and_causal_aliases_materialize_and_execute_real_waves() {
    for kind in [AttentionKind::GatedDelta, AttentionKind::Causal] {
        let fixture = fixture(Family::new(kind), kind, "node.attention");
        check_outputs(&fixture, "node.attention", ElementType::F32);
    }
}

#[test]
fn gguf_rn_f16_cuda_ffn_alias_materializes_q4_q6_and_executes_real_waves() {
    // The unchanged F16 trunk supplies the original FFN input ABI. Its GDN
    // remains strict; the independent test above exercises the new GDN alias.
    let kind = AttentionKind::GatedDeltaHadamardF16;
    let definition = ffn_family::Q8FfnFamily {
        base: Family::new(kind),
        policy: ffn_family::FfnPolicy::Strict,
    };
    let fixture = fixture(definition, kind, "node.ffn");
    check_outputs(&fixture, "node.ffn", ElementType::F16);
}

#[path = "gguf_f16_replay.rs"]
mod graph;

#[path = "gguf_f16_library_cost.rs"]
mod library_cost;

#[path = "gguf_f16_gdn_library_cost.rs"]
mod gdn_library_cost;

#[path = "gguf_f16_causal_library_cost.rs"]
mod causal_library_cost;
