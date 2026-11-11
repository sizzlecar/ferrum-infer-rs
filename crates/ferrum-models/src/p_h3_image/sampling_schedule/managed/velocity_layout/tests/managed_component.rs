//! A real component readback exercises the public output-view boundary. The
//! fixture computes BF16 casts, not a stand-in for an unqualified full DiT.
use super::*;
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
const IMAGE: &str = "value.velocity_fixture.image";
const AUDIO: &str = "value.velocity_fixture.audio";
const PROFILE: &str = "test.h3_velocity_readback.cast_bf16.v1";
fn ident<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {}
fn inputs() -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
    [(IMAGE, vec![4, 8]), (AUDIO, vec![8, 2])]
        .into_iter()
        .map(|(id, dimensions)| {
            (
                ident(id),
                ProgramTensorSpec {
                    dimensions,
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            )
        })
        .collect()
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| ident("family.test.h3_velocity_readback"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([ident("metadata.test.h3_velocity_readback")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> std::result::Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "noncanonical velocity readback fixture".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> std::result::Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(ident("metadata.test.h3_velocity_readback"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> std::result::Result<Config, VNextError> {
        let c =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidExecutionPlan {
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, _: &Config) -> std::result::Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(
        &self,
        _: &Config,
    ) -> std::result::Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: ident(PROFILE),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ident(PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID),
                boundaries: [
                    PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID,
                    PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID,
                ]
                .into_iter()
                .map(|id| (ident(id), ElementType::Bf16))
                .collect(),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: ident(CAST_F32_TO_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![ident(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> std::result::Result<ModelProgram, VNextError> {
        let nodes = [
            ("image", IMAGE, PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID),
            ("audio", AUDIO, PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID),
        ]
        .into_iter()
        .map(|(name, input, output)| ProgramNode {
            id: ident(format!("node.velocity_fixture.{name}")),
            operation_id: ident(CAST_F32_TO_BF16_OPERATION_ID),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![ident(input)],
            outputs: vec![ident(output)],
            attributes: BTreeMap::new(),
        })
        .collect();
        ModelProgram::new(
            self.family_id().clone(),
            inputs().keys().cloned().collect(),
            vec![ProgramBlock {
                id: "block.velocity_fixture".into(),
                nodes,
            }],
            vec![],
            vec![],
            vec![
                ident(PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID),
                ident(PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID),
            ],
        )?
        .with_input_tensor_specs(inputs())
    }
    fn semantic_metadata(
        &self,
        _: &Config,
    ) -> std::result::Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
#[test]
fn actual_reference_readbacks_feed_public_patch_velocity_views_after_component_close() {
    let composition =
        ReferenceVNextComposition::create(ident("device.reference.h3_velocity_views")).unwrap();
    let prepared = TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&serde_json::json!({}), &ident(PROFILE))
        .unwrap();
    let descriptor = composition.runtime().descriptor();
    let policy = TensorComponentPolicy::new(
        descriptor.total_memory_bytes,
        descriptor
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let component = CompiledTensorComponent::compile(
        prepared,
        composition.registry(),
        composition.catalog(),
        &policy,
        composition.weight_materializers(),
        &ProgramPlanCompileOptions::new(inputs()).unwrap(),
    )
    .unwrap();
    let host = account(1 << 20);
    let mut states = [
        state(
            &host,
            "request.a",
            2,
            4,
            PH3ImageVelocityPrecision::Bf16,
            true,
        ),
        state(
            &host,
            "request.b",
            4,
            2,
            PH3ImageVelocityPrecision::Bf16,
            false,
        ),
    ];
    let requests = states.iter().map(|s| s.request.clone()).collect::<Vec<_>>();
    let t = crate::p_h3_image::test_fixture::tiny_transformer(16, 2);
    let layout = PH3ImageVelocityBatchLayout::new(&t, &requests, 4096).unwrap();
    let (image, audio) = physical(&host, &requests, PH3ImageVelocityPrecision::F32, false);
    let image_values = image
        .bytes
        .bytes()
        .chunks_exact(4)
        .map(decode_f32)
        .collect::<Vec<_>>();
    let audio_values = audio
        .bytes
        .bytes()
        .chunks_exact(4)
        .map(decode_f32)
        .collect::<Vec<_>>();
    drop((image, audio));
    let mut session = component
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: ident("run.velocity.init"),
                request_id: ident("request.velocity.init"),
                transaction_id: ident("transaction.velocity.init"),
                static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let outputs = session
        .execute(
            ident("run.velocity.readback"),
            ident("request.velocity.readback"),
            vec![
                tensor(&host, IMAGE, vec![4, 8], &image_values),
                tensor(&host, AUDIO, vec![8, 2], &audio_values),
            ],
        )
        .unwrap();
    session
        .close()
        .unwrap_or_else(|failure| panic!("velocity fixture close: {:?}", failure.failure()));
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    let ports = component
        .executable()
        .execution_plan()
        .payload()
        .tensor_ports()
        .unwrap();
    let image = outputs
        .iter()
        .find(|o| o.value_id().as_str() == PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID)
        .unwrap();
    let audio = outputs
        .iter()
        .find(|o| o.value_id().as_str() == PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID)
        .unwrap();
    let ip = ports
        .outputs()
        .iter()
        .find(|p| p.value_id() == image.value_id())
        .unwrap();
    let ap = ports
        .outputs()
        .iter()
        .find(|p| p.value_id() == audio.value_id())
        .unwrap();
    let before = host.usage().unwrap().committed_bytes();
    assert!(layout
        .image_output(&requests[0].request, ap, image)
        .is_err());
    assert!(layout
        .audio_output(&requests[0].request, ip, audio)
        .is_err());
    let expected = states
        .iter()
        .map(|s| oracle(s, PH3ImageVelocityPrecision::Bf16))
        .collect::<Vec<_>>();
    let pending = states
        .iter_mut()
        .map(|s| {
            let iv = layout.image_output(&s.request.request, ip, image).unwrap();
            let av = layout.audio_output(&s.request.request, ap, audio).unwrap();
            assert!(std::ptr::eq(iv.backing(), image.bytes()));
            assert!(std::ptr::eq(av.backing(), audio.bytes()));
            s.solver
                .prepare_advance(&s.image, iv, &s.audio, av)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(host.usage().unwrap().committed_bytes() > before);
    let successors = pending
        .into_iter()
        .map(PreparedPH3ImageSamplingAdvance::commit)
        .collect::<Vec<_>>();
    for ((s, (_, image, audio)), e) in states.iter_mut().zip(successors).zip(expected) {
        s.image = image;
        s.audio = audio;
        assert_eq!(values(&s.image), e.0);
        assert_eq!(values(&s.audio), e.1);
    }
    drop((outputs, states));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(composition.runtime().snapshot().submissions > 0);
}
