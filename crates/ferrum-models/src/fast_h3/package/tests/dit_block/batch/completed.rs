//! Ordinary Core BF16 transport creates completed Host outputs for adapter
//! protocol tests. These explicit diagnostic values are not a DiT oracle.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Config {
    rows: u64,
    columns: u64,
    output: String,
}
struct Family {
    id: ModelFamilyId,
}
const PROFILE: &str = "batch.completed.bf16.v1";
const INPUT: &str = "value.batch.completed.input";
fn ident<T: TryFrom<String>>(s: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.to_owned()).unwrap()
}
fn spec(c: &Config) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![c.rows, c.columns],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
impl Family {
    fn profile(&self, c: &Config) -> NumericalExecutionProfile {
        NumericalExecutionProfile {
            id: NumericalProfileId::new(PROFILE).unwrap(),
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(&c.output),
            boundaries: BTreeMap::from([(id(&c.output), ElementType::Bf16)]),
            states: vec![],
            kv_storage: vec![],
            operations: vec![NumericalOperationContract {
                operation_id: ident(ROW_GATHER_BF16_OPERATION_ID),
                version: ContractVersion::new(1, 0),
                multiplication_type: None,
                accumulation_type: None,
                staged_arithmetic: None,
                composite_arithmetic: None,
            }],
        }
    }
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([ident("metadata.batch.completed")])
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(ident("metadata.batch.completed"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c: Config = serde_json::from_value(raw.clone()).unwrap();
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        assert_eq!(raw, &serde_json::to_value(c).unwrap());
        assert!(c.rows > 0 && c.columns > 0);
        Ok(())
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![self.profile(c)],
            vec![NumericalProfileId::new(PROFILE).unwrap()],
        )
    }
    fn semantic_program(
        &self,
        c: &Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        assert_eq!(p, &self.profile(c));
        ModelProgram::new(
            self.id.clone(),
            vec![id(INPUT)],
            vec![ProgramBlock {
                id: "block.completed".into(),
                nodes: vec![ProgramNode {
                    id: ident("node.completed"),
                    operation_id: ident(ROW_GATHER_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT)],
                    outputs: vec![id(&c.output)],
                    attributes: BTreeMap::from([
                        (ident("batch"), SemanticValue::Unsigned(1)),
                        (ident("column_start"), SemanticValue::Unsigned(0)),
                        (ident("column_count"), SemanticValue::Unsigned(c.columns)),
                        (
                            ident("run_lengths"),
                            SemanticValue::Integers(vec![c.rows as i64]),
                        ),
                        (ident("source_starts"), SemanticValue::Integers(vec![0])),
                        (
                            ident("source_batch_strides"),
                            SemanticValue::Integers(vec![0]),
                        ),
                        (
                            ident("source_row_strides"),
                            SemanticValue::Integers(vec![1]),
                        ),
                    ]),
                }],
            }],
            vec![],
            vec![],
            vec![id(&c.output)],
        )?
        .with_input_tensor_specs(BTreeMap::from([(id(INPUT), spec(c))]))
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
pub(super) fn output(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    name: &str,
    rows: u64,
    columns: u64,
    bytes: &[u8],
) -> TensorExecutionOutput {
    let c = Config {
        rows,
        columns,
        output: name.into(),
    };
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([
        (id(INPUT), spec(&c)),
        (id(name), spec(&c)),
    ]))
    .unwrap();
    run(
        r,
        h,
        Family {
            id: ident("family.batch.completed"),
        },
        &c,
        PROFILE,
        options,
        ComponentWeightSource::None,
        vec![io::managed(
            h,
            INPUT,
            rows,
            columns,
            ElementType::Bf16,
            bytes,
        )],
    )
    .pop()
    .unwrap()
}
