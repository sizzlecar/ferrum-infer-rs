use super::*;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::SafetensorsArchive;
use half::bf16;
use serde_json::json;
#[path = "../../../examples/support/tensor_component.rs"]
mod shared;

fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).unwrap()
}
fn raw(base: u64) -> Value {
    json!({"_class_name":"AutoencoderKLQwenImage21","z_dim":64,"decoder_base_dim":base,
    "dim_mult":[1,2,4,8,8],"num_res_blocks":2,"temperal_downsample":[false,true,true,true],
    "out_channels":4,"is_residual":true,"patch_size":null,"dropout":0.0,"attn_scales":[]})
}
fn real() -> QwenImage21VaeDecoderConfig {
    QwenImage21VaeDecoderConfig::from_hf_json(&serde_json::to_vec(&raw(144)).unwrap(), 1, 4, 6)
        .unwrap()
}

const UP3_SHORTCUT: &str =
    "value.qwen_image21.vae.decoder.decoder.up_blocks.3.first_frame_shortcut";
const UP4_SHORTCUT: &str =
    "value.qwen_image21.vae.decoder.decoder.up_blocks.4.resnets.0.conv_shortcut";

/// The production family intentionally exports only RGBA. Retaining an
/// intermediate allocation does not turn it into an executor completion port;
/// this test adapter explicitly declares the two inspected boundaries.
struct DiagnosticDecoder(QwenImage21VaeDecoderFamily);
impl ModelFamilyProvider for DiagnosticDecoder {
    type Config = QwenImage21VaeDecoderConfig;
    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        self.0.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        self.0.parse_config(raw)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.0.weights(config)
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        self.0.numerical_profiles(config)
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let program = self.0.semantic_program(config, profile)?;
        let mut outputs = program.outputs().to_vec();
        outputs.extend([
            ProgramValueId::new(UP3_SHORTCUT)?,
            ProgramValueId::new(UP4_SHORTCUT)?,
        ]);
        ModelProgram::new(
            program.family_id().clone(),
            program.inputs().to_vec(),
            program.blocks().to_vec(),
            program.states().to_vec(),
            program.weights().to_vec(),
            outputs,
        )?
        .with_input_tensor_specs(program.input_tensor_specs().clone())
    }
    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(config)
    }
}

#[test]
fn full_decoder_matches_real_single_frame_schema_without_temporal_or_encoder_weights() {
    let c = real();
    let family = QwenImage21VaeDecoderFamily::new().unwrap();
    let expected = c.expected_weights().unwrap();
    assert_eq!(expected.len(), 128);
    assert_eq!(expected.iter().map(|w| w.bytes()).sum::<u64>(), 506_140_968);
    assert_eq!(
        c.decoder_channels().unwrap(),
        [1152, 1152, 1152, 576, 288, 144]
    );
    assert_eq!(c.temporal_upsample(), [true, true, true, false]);
    assert_eq!(c.output_tensor().unwrap().dimensions(), [1, 4, 64, 96]);
    let by_name = expected
        .iter()
        .map(|w| (w.external_name(), w.dimensions()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        by_name["decoder.mid_block.attentions.0.norm.gamma"],
        [1152, 1, 1]
    );
    assert_eq!(
        by_name["decoder.mid_block.attentions.0.to_qkv.weight"],
        [3456, 1152, 1, 1]
    );
    assert_eq!(
        by_name["decoder.up_blocks.2.resnets.0.conv_shortcut.weight"],
        [576, 1152, 1, 1]
    );
    assert_eq!(
        by_name["decoder.up_blocks.3.upsampler.resample.1.weight"],
        [288, 288, 3, 3]
    );
    assert_eq!(by_name["decoder.conv_out.weight"], [4, 144, 3, 3]);
    assert!(expected
        .iter()
        .all(|w| !w.external_name().contains("time_conv")
            && !w.external_name().starts_with("encoder.")
            && !w.external_name().starts_with("quant_conv.")));
    let stored = family.weights(&c).unwrap();
    let schema = stored.stored_schema().unwrap();
    assert!(schema.components.iter().all(|w| w.required));
    assert!(schema.tensors.iter().all(|w| w.required));
    assert_eq!(schema.tensors.len(), 128);
    let profiles = family.numerical_profiles(&c).unwrap();
    let program = family
        .semantic_program(&c, &profiles.profiles()[0])
        .unwrap();
    assert_eq!(program.blocks()[0].nodes.len(), 148);
    assert_eq!(program.weights().len(), 128);
    assert_eq!(
        program.outputs(),
        [id(QwenImage21VaeDecoderFamily::OUTPUT_ID)]
    );
    assert_eq!(
        program.input_tensor_specs(),
        &c.input_tensor_specs().unwrap()
    );
    assert!(program.blocks()[0]
        .nodes
        .iter()
        .all(|n| n.work == ProgramNodeWorkSpec::Fixed));
}

#[test]
fn padding_gamma_rank_and_first_frame_shortcuts_are_not_silently_rewritten() {
    let g = graph::build(&real()).unwrap();
    let node = |name: &str| {
        g.nodes
            .iter()
            .find(|n| n.id.as_str() == format!("node.qwen_image21.vae.decoder.{name}"))
            .unwrap()
    };
    for index in 0..4 {
        let n = node(&format!("decoder.up_blocks.{index}.upsampler.resample.1"));
        assert_eq!(
            n.attributes[&id::<AttributeId>("padding_mode")],
            SemanticValue::Text("implicit_zero".into())
        );
        assert_eq!(
            n.attributes[&id::<AttributeId>("padding")],
            SemanticValue::Integers(vec![1; 4])
        );
    }
    assert_eq!(
        node("decoder.conv_in").attributes[&id::<AttributeId>("padding_mode")],
        SemanticValue::Text("explicit_zero".into())
    );
    assert_eq!(
        node("decoder.mid_block.attentions.0.norm")
            .operation_id
            .as_str(),
        CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID
    );
    for (index, stride, offsets) in [
        (0, 1, vec![0, 0, 0, 0]),
        (1, 1, vec![0, 0, 0, 0]),
        (2, 2, vec![1, 1, 1, 1]),
        (3, 2, vec![0, 0, 1, 1]),
    ] {
        let n = node(&format!("decoder.up_blocks.{index}.first_frame_shortcut"));
        assert_eq!(
            n.attributes[&id::<AttributeId>("channel_stride")],
            SemanticValue::Unsigned(stride)
        );
        assert_eq!(
            n.attributes[&id::<AttributeId>("channel_offsets")],
            SemanticValue::Integers(offsets)
        );
        // The outer shortcut reads the block's original value, not the output
        // of its residual stack or its nearest-upsample branch.
        let first = node(&format!("decoder.up_blocks.{index}.resnets.0.norm1"));
        assert_eq!(n.inputs[0], first.inputs[0]);
    }
    assert!(g
        .nodes
        .iter()
        .all(|n| !n.id.as_str().contains("up_blocks.4.upsampler")));
}

#[test]
fn config_roundtrip_rejects_unsupported_execution_and_overflow() {
    let c = real();
    let wire = serde_json::to_value(&c).unwrap();
    assert_eq!(
        serde_json::from_value::<QwenImage21VaeDecoderConfig>(wire.clone()).unwrap(),
        c
    );
    for field in ["frames", "normalized_input"] {
        let mut altered = wire.clone();
        altered[field] = json!(1);
        assert!(serde_json::from_value::<QwenImage21VaeDecoderConfig>(altered).is_err());
    }
    for (field, value) in [
        ("is_residual", json!(false)),
        ("patch_size", json!(2)),
        ("dropout", json!(0.1)),
        ("out_channels", json!(3)),
        ("temperal_downsample", json!([true])),
        ("dim_mult", json!([1, 2, 3, 8, 8])),
    ] {
        let mut changed = raw(144);
        changed[field] = value;
        assert!(
            QwenImage21VaeDecoderConfig::from_hf_json(
                &serde_json::to_vec(&changed).unwrap(),
                1,
                1,
                1
            )
            .is_err(),
            "{field}"
        );
    }
    assert!(QwenImage21VaeDecoderConfig::new(
        1,
        1,
        vec![1, 2, 4, 8, 8],
        2,
        vec![true, true, true, false],
        1,
        u64::MAX,
        1
    )
    .is_err());
}

#[test]
fn complete_tiny_decoder_executes_with_real_weights_and_preserves_up_block_mapping() {
    let c = QwenImage21VaeDecoderConfig::new(
        1,
        1,
        vec![1, 2, 4, 8, 8],
        2,
        vec![true, true, true, false],
        2,
        1,
        1,
    )
    .unwrap();
    let family = TypedFamilyRegistration::new(DiagnosticDecoder(
        QwenImage21VaeDecoderFamily::new().unwrap(),
    ))
    .prepare_with_profile(
        &serde_json::to_value(&c).unwrap(),
        &NumericalProfileId::new(QwenImage21VaeDecoderFamily::REFERENCE_PROFILE_ID).unwrap(),
    )
    .unwrap();
    let mut payloads = BTreeMap::new();
    for w in c.expected_weights().unwrap() {
        let name = w.external_name();
        let shape = w.dimensions();
        let mut data = vec![0_u16; (w.bytes() / 2) as usize];
        if name.ends_with(".gamma") {
            data.fill(0x3f80);
        }
        if name == "decoder.conv_in.bias" {
            for (i, value) in data.iter_mut().enumerate() {
                *value = bf16::from_f32((i + 1) as f32).to_bits();
            }
        }
        if name.ends_with("conv_shortcut.weight") {
            for o in 0..shape[0] as usize {
                data[o * shape[1] as usize + o] = 0x3f80;
            }
        }
        if name == "decoder.conv_out.weight" {
            for o in 0..4 {
                data[o * 9 + 4] = 0x3f80;
            }
        }
        if name == "decoder.conv_out.bias" {
            data.copy_from_slice(&[0xc000, 0xbf00, 0x3f00, 0x4000]);
        }
        payloads.insert(
            name.to_string(),
            (
                shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                data.into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        );
    }
    let directory = tempfile::tempdir().unwrap();
    let views = payloads
        .iter()
        .map(|(name, (shape, data))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.clone(), data)
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let capacity = 64 << 20;
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.vae.decoder.tiny"),
        capacity,
    )
    .unwrap();
    let mut options = c.compile_options().unwrap();
    let shortcut = UP3_SHORTCUT;
    let last_shortcut = UP4_SHORTCUT;
    options.retain_completion_value(id(shortcut));
    options.retain_completion_value(id(last_shortcut));
    let input = [0x00, 0x3f, 0x80, 0xbf];
    let (_,cleanup)=shared::execute_stored_component_with_options(&family,&archive,shared::StoredComponentRuntime {
        runtime:composition.runtime().clone(),registry:composition.registry(),catalog:composition.catalog(),materializers:composition.weight_materializers(),
        capacity_bytes:capacity,staging_bytes:4096,provenance:json!({"scope":"complete five-up-block tiny decoder; analytic channel/bias fixture"}),
    },vec![shared::ComponentInput {value_id:id(QwenImage21VaeDecoderFamily::INPUT_ID),tensor:c.input_tensor().unwrap(),bytes:&input}],options,|outputs,_|{
        assert_eq!(outputs.len(),3);
        for output in outputs {
            let actual=output.bytes().bytes().chunks_exact(2).map(|b|u16::from_le_bytes([b[0],b[1]])).collect::<Vec<_>>();
            match output.value_id().as_str(){
                QwenImage21VaeDecoderFamily::OUTPUT_ID=>{
                    assert_eq!(output.tensor().dimensions(),[2,4,16,16]);
                    // All final hidden values are positive. C=1 normalization
                    // gives 1; SiLU(1)=BF16 3f3b. Identity center weights and
                    // biases [-2,-1/2,+1/2,+2] give these hand-rounded bits.
                    // Values outside [-1,1] prove clamp remains caller-owned.
                    let expected=[0xbfa2,0x3e6c,0x3f9e,0x402f];
                    for (index,value) in actual.into_iter().enumerate(){assert_eq!(value,expected[(index/256)%4]);}
                }
                name if name==shortcut=>{
                    assert_eq!(output.tensor().dimensions(),[2,2,16,16]);
                    for (index,value) in actual.into_iter().enumerate(){
                        let channel=(index/256)%2;let row=(index/16)%16;
                        let expected=if channel==0 {if row%2==0 {0x4000}else{0x4080}}else if row%2==0 {0x40c0}else{0x4100};
                        assert_eq!(value,expected,"up3 shortcut at {index}");
                    }
                }
                name if name==last_shortcut=>{
                    assert_eq!(output.tensor().dimensions(),[2,1,16,16]);
                    for (index,value) in actual.into_iter().enumerate(){assert_eq!(value,if (index/16)%2==0 {0x4000}else{0x4080});}
                }
                other=>panic!("unexpected completion {other}"),
            }
        }
        Ok(json!({"matched":true}))
    }).unwrap();
    assert_eq!(cleanup["managed_host_committed_bytes"], 0);
    assert_eq!(cleanup["managed_host_allocated_bytes"], 0);
    assert_eq!(cleanup["released_static_resources"], 1);
}
