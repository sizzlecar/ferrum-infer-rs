//! Real Reference producer -> D2D request state -> repeated fixed steps.
use super::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::OnceLock;

const X: &str = "value.request.x";
const CACHE: &str = "value.request.cache";
const Y: &str = "value.request.y";
const PROFILE: &str = "profile.request.d2d.bf16";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Config {
    producer: bool,
}
struct Family;
fn spec() -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![2, 2],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn states(config: &Config) -> Vec<StateSpec> {
    if config.producer {
        vec![]
    } else {
        vec![StateSpec {
            id: id("state.request.cache"),
            value_id: id(CACHE),
            tensor: spec(),
            lifetime: StateLifetime::Request,
            capacity_demand: StateCapacityDemand::FixedPerScope,
            initialization: StateInitialization::None,
            checkpoint: StateCheckpointCapability::Unsupported,
        }]
    }
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.request.d2d"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.request.d2d")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if *raw != serde_json::to_value(config).unwrap() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "noncanonical fixture config".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.request.d2d"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidExecutionPlan {
            reason: e.to_string(),
        })
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, config: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(Y),
                boundaries: BTreeMap::from([
                    (id(Y), ElementType::Bf16),
                    (id(CACHE), ElementType::Bf16),
                ]),
                states: states(config),
                operations: vec![NumericalOperationContract {
                    operation_id: id(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let add = |name: &str, left: &str, right: &str, output: &str| ProgramNode {
            id: id(name),
            operation_id: id(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![id(left), id(right)],
            outputs: vec![id(output)],
            attributes: BTreeMap::new(),
        };
        let mut nodes = Vec::new();
        if config.producer {
            nodes.push(add("node.request.cache", X, X, CACHE));
        }
        nodes.push(add("node.request.output", X, CACHE, Y));
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(X)],
            vec![ProgramBlock {
                id: "request".into(),
                nodes,
            }],
            states(config),
            vec![],
            if config.producer {
                vec![id(Y), id(CACHE)]
            } else {
                vec![id(Y)]
            },
        )?
        .with_input_tensor_specs(BTreeMap::from([(id(X), spec())]))
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            32,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16,
                output_bytes: 16,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn input(host: &HostMemoryAccount, values: [f32; 4]) -> ManagedHostTensor {
    let mut owner = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap();
    owner.as_mut().copy_from_slice(&bytes(&values));
    ManagedHostTensor::new(
        id(X),
        ResolvedTensorSpec::new(
            vec![2, 2],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        owner.freeze(),
    )
    .unwrap()
}
fn component(
    c: &ReferenceVNextComposition,
    host: &HostMemoryAccount,
    producer: bool,
) -> TensorComponentSession<ReferenceDeviceRuntime> {
    let family = TypedFamilyRegistration::new(Family)
        .prepare_with_profile(
            &serde_json::to_value(Config { producer }).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let policy = TensorComponentPolicy::new(
        4096,
        c.runtime()
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([(id(X), spec())])).unwrap();
    let compiled = CompiledTensorComponent::compile(
        family,
        c.registry(),
        c.catalog(),
        &policy,
        c.weight_materializers(),
        &options,
    )
    .unwrap();
    compiled
        .initialize(
            c.runtime().clone(),
            host.clone(),
            ComponentWeightSource::None,
            Fixture::init_config(if producer { "producer" } else { "consumer" }),
        )
        .unwrap()
}
fn close(session: TensorComponentSession<ReferenceDeviceRuntime>) {
    match session.close() {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => (),
        Ok(PlanRuntimeCloseOutcome::Referenced { .. }) => panic!("unexpected live resource owner"),
        Err(e) => panic!("close failed: {:?}", e.failure()),
    }
}
fn mappings() -> Vec<TensorRequestStateMapping> {
    vec![TensorRequestStateMapping {
        source_output: id(CACHE),
        destination_state: id(CACHE),
    }]
}

#[test]
fn request_d2d_initialization_rebinds_inputs_without_host_cache_and_retires() {
    let c =
        ReferenceVNextComposition::create_with_capacity(id("device.request.d2d"), 4096).unwrap();
    let host = host();
    let mut producer = component(&c, &host, true);
    let mut consumer = component(&c, &host, false);
    let mut request = consumer
        .begin_request(
            id("run.request"),
            id("request.request"),
            vec![input(&host, [0.; 4])],
        )
        .unwrap();
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        0,
        "admission must not pin initial Host inputs"
    );
    assert!(
        request.execute(vec![input(&host, [1.; 4])]).is_err(),
        "uninitialized state cannot execute"
    );
    assert!(matches!(
        consumer.try_release_idle_dynamic_backing().unwrap(),
        IdleDynamicReleaseOutcome::Busy(_)
    ));
    let before = c.runtime().snapshot();
    let initial = producer
        .execute_initializing_request(
            id("run.prefill"),
            id("request.prefill"),
            vec![input(&host, [1., -2., 3., 4.])],
            &mut request,
            &mappings(),
        )
        .unwrap();
    assert_eq!(initial.len(), 1);
    assert_eq!(initial[0].bytes().bytes(), bytes(&[3., -6., 9., 12.]));
    assert_eq!(
        c.runtime().snapshot().readback_bytes - before.readback_bytes,
        8,
        "cache is not read back to Host"
    );
    close(producer);
    let one = request
        .execute(vec![input(&host, [10., 20., 30., 40.])])
        .unwrap();
    assert_eq!(one[0].bytes().bytes(), bytes(&[12., 16., 36., 48.]));
    drop(initial);
    let two = request
        .execute(vec![input(&host, [-2., 4., -6., -8.])])
        .unwrap();
    assert_eq!(two[0].bytes().bytes(), bytes(&[0.; 4]));
    request.close().unwrap();
    close(consumer);
    assert_eq!(c.runtime().snapshot().live_allocations, 0);
    assert_eq!(one[0].bytes().bytes(), bytes(&[12., 16., 36., 48.]));
    drop((one, two));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn request_wrong_mapping_and_foreign_runtime_reject_before_submission() {
    for foreign in [false, true] {
        let c = ReferenceVNextComposition::create_with_capacity(
            id(format!("device.request.reject.{foreign}")),
            4096,
        )
        .unwrap();
        let other = ReferenceVNextComposition::create_with_capacity(
            id(format!("device.request.other.{foreign}")),
            4096,
        )
        .unwrap();
        let host = host();
        let mut producer = component(&c, &host, true);
        let consumer = component(if foreign { &other } else { &c }, &host, false);
        let mut request = consumer
            .begin_request(
                id("run.rejected"),
                id("request.rejected"),
                vec![input(&host, [0.; 4])],
            )
            .unwrap();
        let mut map = mappings();
        if !foreign {
            map[0].source_output = id("value.absent");
        }
        let count = c.runtime().snapshot().submissions;
        assert!(producer
            .execute_initializing_request(
                id("run.bad"),
                id("request.bad"),
                vec![input(&host, [1.; 4])],
                &mut request,
                &map
            )
            .is_err());
        assert_eq!(c.runtime().snapshot().submissions, count);
        assert!(request.execute(vec![input(&host, [1.; 4])]).is_err());
        request.close().unwrap();
        close(producer);
        close(consumer);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(c.runtime().snapshot().live_allocations, 0);
        assert_eq!(other.runtime().snapshot().live_allocations, 0);
    }
}
