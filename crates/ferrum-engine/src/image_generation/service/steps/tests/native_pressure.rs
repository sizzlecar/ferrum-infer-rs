//! A separately provisioned, real static native buffer. It uses the same
//! allocator/admission driver as model sessions and never changes device limits.
use super::*;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod staging;
const X: &str = "value.service.pressure.input";
const W: &str = "value.service.pressure.table";
const G: &str = "value.service.pressure.gathered";
const Y: &str = "value.service.pressure.output";
const PROFILE: &str = "fixture.service.native_pressure.bf16_transport";
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn spec(rows: u64) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![rows, 1],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
}
struct Family {
    id: ModelFamilyId,
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.service.native_pressure")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(format!("{e}")))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if c.rows == 0 || serde_json::to_value(c).unwrap() != *raw {
            return Err(invalid("noncanonical pressure fixture"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fixture.service.native_pressure"))
    }
    fn weights(&self, c: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fixture.service.native_pressure"),
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: id("component.service.pressure.table"),
                role: WeightComponentRole::Values,
                external_names: vec!["table".into()],
                dimensions: vec![c.rows, 1],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: id("weight.service.pressure.table"),
                dimensions: vec![c.rows, 1],
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: id("component.service.pressure.table"),
                },
                required: true,
            }],
        }))
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Y),
                boundaries: BTreeMap::from([
                    (id(G), ElementType::Bf16),
                    (id(Y), ElementType::Bf16),
                ]),
                states: vec![],
                kv_storage: vec![],
                operations: [
                    ROW_GATHER_BF16_OPERATION_ID,
                    BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                ]
                .into_iter()
                .map(|operation| NumericalOperationContract {
                    operation_id: id(operation),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                })
                .collect(),
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        c: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let mut attributes = BTreeMap::from([
            (id("batch"), SemanticValue::Unsigned(1)),
            (id("column_start"), SemanticValue::Unsigned(0)),
            (id("column_count"), SemanticValue::Unsigned(1)),
        ]);
        for (name, value) in [
            ("run_lengths", 1),
            ("source_starts", 0),
            ("source_batch_strides", 0),
            ("source_row_strides", 1),
        ] {
            attributes.insert(id(name), SemanticValue::Integers(vec![value]));
        }
        ModelProgram::new(
            self.id.clone(),
            vec![id(X)],
            vec![ProgramBlock {
                id: "block.service.pressure".into(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.service.pressure.gather"),
                        operation_id: id(ROW_GATHER_BF16_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::Fixed,
                        inputs: vec![id(W)],
                        outputs: vec![id(G)],
                        attributes,
                    },
                    ProgramNode {
                        id: id("node.service.pressure.concat"),
                        operation_id: id(BATCHED_ROW_CONCAT_BF16_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::Fixed,
                        inputs: vec![id(X), id(G)],
                        outputs: vec![id(Y)],
                        attributes: BTreeMap::from([(id("batch"), SemanticValue::Unsigned(1))]),
                    },
                ],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.service.pressure.table"),
                value_id: id(W),
                tensor: spec(c.rows),
            }],
            vec![id(Y)],
        )?
        .with_input_tensor_specs(BTreeMap::from([(id(X), spec(1))]))
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(super) struct NativePressure(Option<Arc<PlanRuntimeResources<Runtime>>>);
impl NativePressure {
    pub(super) fn close(mut self) {
        let root = self.0.take().unwrap();
        match PlanRuntimeResources::close(root) {
            Ok(PlanRuntimeCloseOutcome::Closed(_)) => (),
            Ok(PlanRuntimeCloseOutcome::Referenced { .. }) => {
                panic!("external root unexpectedly referenced")
            }
            Err(failure) => panic!("external native close: {:?}", failure.failure()),
        }
    }
    pub(super) fn new(pipeline: &QwenImage21Pipeline<Runtime>, bytes: u64) -> Self {
        assert!(bytes > 0 && bytes % 2 == 0);
        let config = Config { rows: bytes / 2 };
        let family = TypedFamilyRegistration::new(Family {
            id: id("family.fixture.service.native_pressure"),
        })
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(PROFILE))
        .unwrap();
        let compiled = ferrum_execution::component::CompiledTensorComponent::compile(
            family,
            &pipeline.execution.registry,
            &pipeline.execution.catalog,
            &pipeline.execution.policy,
            &pipeline.execution.materializers,
            &ProgramPlanCompileOptions::new(BTreeMap::from([
                (id(X), spec(1)),
                (id(G), spec(1)),
                (id(Y), spec(2)),
            ]))
            .unwrap(),
        )
        .unwrap();
        let plan = compiled.executable().execution_plan();
        assert_eq!(plan.payload().memory().static_bytes(), bytes);
        let provisioning = plan
            .provision_static(
                pipeline.execution.runtime.clone(),
                id("request.service.external.native"),
            )
            .unwrap()
            .into_provisioning();
        let StaticProvisioning::Required(permit) = provisioning else {
            panic!("real static allocation required")
        };
        let identity = ResourceTransactionIdentity::for_admission(
            permit.binding(),
            id("run.service.external.native"),
            id("transaction.service.external.native"),
        );
        let committed = ResourceTransaction::begin(
            RuntimeResourceDriver::new(pipeline.execution.runtime.clone()).unwrap(),
            identity,
            permit,
        )
        .unwrap()
        .reserve()
        .unwrap()
        .commit()
        .unwrap_or_else(|_| panic!("native external buffer commit"));
        let root = committed
            .into_plan_runtime()
            .unwrap_or_else(|_| panic!("native external root handoff"));
        Self(Some(root))
    }
}
impl Drop for NativePressure {
    fn drop(&mut self) {
        // This fixture has no submitted work, executor, dynamic backing or
        // other root owners. Close uses the production native release driver.
        if let Some(root) = self.0.take() {
            let _ = PlanRuntimeResources::close(root);
        }
    }
}
