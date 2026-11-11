//! A real selected BF16 checkpoint and original CPU/Core compilation.
use super::super::mtp::{self as model, Qwen35MtpRole as Role};
use super::*;
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_runtime::*;
use serde_json::json;

mod execution;
mod product_wave;

fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
const CAPACITY: u64 = 32 << 20;
const ROWS: u64 = 3;

struct Fixture {
    _directory: tempfile::TempDir,
    sources: Arc<ProductionModelSourceBundle>,
    source: SafetensorsArchive,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path();
        let mut c = test_config();
        c.hf_config["tie_word_embeddings"] = json!(false);
        c.hf_config["text_config"]["tie_word_embeddings"] = json!(false);
        c.hf_config["text_config"]["attn_output_gate"] = json!(true);
        // The base test family defaults to ungated attention. This selected
        // archive explicitly uses the actual MTP/H3 query-plus-gate ABI.
        for weight in &mut c.weights {
            if weight.role == "self_attn_q" {
                weight.dimensions[0] *= 2;
            }
        }
        c.weights.push(FamilyWeight {
            layer_index: None,
            expert_index: None,
            role: "lm_head".into(),
            external_name: "lm_head.weight".into(),
            dimensions: vec![32, 16],
            source_encoding: FamilyWeightSourceEncoding::Dense {
                element_type: ElementType::Bf16,
            },
        });
        std::fs::write(
            path.join("config.json"),
            serde_json::to_vec(&c.hf_config).unwrap(),
        )
        .unwrap();
        std::fs::write(path.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(path.join("tokenizer_config.json"),br#"{"chat_template":"{{ messages }}","bos_token_id":1,"eos_token_id":2,"pad_token_id":0}"#).unwrap();
        let mut tensors = c
            .weights
            .iter()
            .map(|w| {
                (
                    w.external_name.clone(),
                    w.dimensions.iter().map(|n| *n as usize).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        for (name, shape) in [
            ("fc", vec![16, 32]),
            ("pre_fc_norm_embedding", vec![16]),
            ("pre_fc_norm_hidden", vec![16]),
            ("norm", vec![16]),
            ("layers.0.input_layernorm", vec![16]),
            ("layers.0.post_attention_layernorm", vec![16]),
            ("layers.0.self_attn.q_proj", vec![16, 16]),
            ("layers.0.self_attn.k_proj", vec![4, 16]),
            ("layers.0.self_attn.v_proj", vec![4, 16]),
            ("layers.0.self_attn.o_proj", vec![16, 8]),
            ("layers.0.self_attn.q_norm", vec![4]),
            ("layers.0.self_attn.k_norm", vec![4]),
            ("layers.0.mlp.gate_proj", vec![32, 16]),
            ("layers.0.mlp.up_proj", vec![32, 16]),
            ("layers.0.mlp.down_proj", vec![16, 32]),
        ] {
            tensors.push((format!("mtp.{name}.weight"), shape));
        }
        let tensors = tensors
            .into_iter()
            .map(|(name, shape)| {
                let count: usize = shape.iter().product();
                let bytes = (0..count)
                    .flat_map(|i| {
                        let v = if name.contains("norm") {
                            0.125
                        } else if name == "mtp.fc.weight" {
                            let row = i / 32;
                            let col = i % 32;
                            if col == row {
                                1.
                            } else if col == row + 16 {
                                2.
                            } else {
                                0.
                            }
                        } else if name == "mtp.layers.0.self_attn.o_proj.weight"
                            || name == "mtp.layers.0.mlp.down_proj.weight"
                        {
                            0.
                        } else if name == "model.embed_tokens.weight" {
                            if i % 16 == (i / 16) % 16 {
                                1.
                            } else {
                                0.
                            }
                        } else if name == "lm_head.weight" {
                            if i % 16 == (i / 16) % 16 {
                                if i / 16 < 16 {
                                    1.
                                } else {
                                    -1.
                                }
                            } else {
                                0.
                            }
                        } else {
                            ((i % 13) as f32 - 6.) / 128.
                        };
                        bf16::from_f32(v).to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                (name, shape, bytes)
            })
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let checkpoint = path.join("model.safetensors");
        serialize_to_file(views, &None, &checkpoint).unwrap();
        let original = |kind, location: &Path| OriginalModelSource {
            kind,
            location: location.display().to_string(),
            requested_revision: None,
        };
        let sources = Arc::new(
            crate::vnext::open_registered_product_sources(
                path,
                path,
                ProductionWeightArtifact::safetensors_file(&checkpoint),
                OriginalModelSources {
                    semantic: original(ModelSourceKind::LocalDirectory, path),
                    tokenizer: original(ModelSourceKind::LocalDirectory, path),
                    weights: original(ModelSourceKind::LocalFile, &checkpoint),
                },
            )
            .unwrap(),
        );
        let source = sources.weights().open_safetensors_archive().unwrap();
        Self {
            _directory: directory,
            sources,
            source,
        }
    }
    fn prepare(&self, role: Role) -> Arc<PreparedProductionModel> {
        Arc::new(
            model::define_from_sources(self.sources.clone(), role)
                .unwrap()
                .prepare(&id(role.numerical_profile()))
                .unwrap(),
        )
    }
}

fn compile(
    composition: &CpuVNextComposition,
    prepared: &PreparedProductionModel,
) -> ProgramPlanCompilation {
    let family = prepared.family();
    let descriptor = composition.runtime().descriptor();
    let policy = ResolvedRuntimePolicy::new(
        "runtime-policy.mtp.cpu",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            checkpoint_capacity: None,
            capacity_bytes: CAPACITY,
            reserve_bytes: 0,
            maximum_active_sequences: 1,
            dynamic_storage_profile_order: descriptor
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        },
        AdmissionPolicy {
            maximum_queue_depth: 1,
            maximum_scheduled_tokens: ROWS,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            sequence_backing_growth: Default::default(),
            allow_defer: true,
            cancellation_check_interval_steps: 1,
        },
        ferrum_types::AttentionExecutionPolicy::Portable,
        ExecutionDeterminismRequirement::BitwiseSameRuntime,
        None,
    )
    .unwrap();
    let inputs = family
        .program()
        .inputs()
        .iter()
        .map(|input| {
            let (dimensions, dtype) = match input.as_str() {
                "value.input.token_ids" | model::TOKEN_IDS => (vec![ROWS], ElementType::U32),
                "value.input.visual_rows" | model::ACCEPTED_HIDDEN => {
                    (vec![ROWS, 16], ElementType::F16)
                }
                "value.input.visual_replace_mask" => (vec![ROWS], ElementType::U8),
                "value.input.rope_cos"
                | "value.input.rope_sin"
                | model::ROPE_COS
                | model::ROPE_SIN => (vec![ROWS, 2], ElementType::F16),
                "value.input.greedy_token_mask" => (vec![32], ElementType::U8),
                "value.input.greedy_repetition_token_ids" => (vec![1], ElementType::U32),
                "value.input.greedy_repetition_offsets" => (vec![2], ElementType::U32),
                "value.input.greedy_repetition_penalty" => (vec![1], ElementType::F32),
                other => panic!("unexpected input {other}"),
            };
            (input.clone(), tensor_spec(dimensions, dtype))
        })
        .collect();
    let mut options = ProgramPlanCompileOptions::new(inputs).unwrap();
    model::configure_compile_options(family, &mut options).unwrap();
    for output in family.program().outputs() {
        options.retain_completion_value(output.clone());
    }
    ProgramPlanCompiler::compile_with_weight_materializers(
        family,
        composition.catalog(),
        &policy,
        &composition.registry().planning(),
        &WeightMaterializerRegistry::identity_only().unwrap(),
        &options,
    )
    .unwrap()
}

#[test]
fn mtp_source_keeps_exact_union_and_canonical_norm_conversion() {
    let f = Fixture::new();
    let main = f.prepare(Role::MainWithDraftWeights);
    let draft = f.prepare(Role::OneDraft);
    let union = f.prepare(Role::InstallUnion);
    assert_eq!(main.family().weights(), draft.family().weights());
    assert_eq!(main.family().weights(), union.family().weights());
    assert_eq!(
        main.family().program().weights(),
        draft.family().program().weights()
    );
    let schema = draft.family().stored_weight_schema().unwrap();
    let selected = schema
        .components
        .iter()
        .filter(|c| c.id.as_str().starts_with("component.mtp."))
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 14); // gate/up are one physical component, 15 source tensors.
    assert_eq!(
        selected
            .iter()
            .map(|c| c.external_names.len())
            .sum::<usize>(),
        15
    );
    for component in selected {
        if component.external_names[0].contains("norm") {
            assert_eq!(
                component.encoding,
                dense_weight_encoding("final_norm").unwrap()
            );
        }
    }
    assert_eq!(draft.family().program().states().len(), 1);
    assert_eq!(
        draft.family().program().states()[0].id.as_str(),
        "state.mtp.layer.0.kv"
    );
    assert!(draft.family().program().checkpoint_inputs().is_none());
    let attention = draft.family().program().blocks()[0]
        .nodes
        .iter()
        .find(|n| n.id.as_str() == "node.mtp.attention")
        .unwrap();
    assert_eq!(
        &attention.inputs[9..],
        &[id(model::ROPE_COS), id(model::ROPE_SIN)]
    );
    assert_eq!(
        draft.family().program().outputs(),
        &[id(model::GREEDY_TOKENS), id(model::HIDDEN_TAIL)]
    );
    let base = super::super::define_from_sources(f.sources.clone())
        .unwrap()
        .prepare(&id(F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID))
        .unwrap();
    assert_eq!(
        base.family().program().blocks(),
        main.family().program().blocks()
    );
    assert_eq!(
        base.family().program().inputs(),
        main.family().program().inputs()
    );
    assert_eq!(
        base.family().program().outputs(),
        main.family().program().outputs()
    );
}

#[test]
fn mtp_union_compiles_actual_cpu_layout_for_both_consumers() {
    let f = Fixture::new();
    let composition = CpuVNextComposition::create(id("device.mtp.union"), CAPACITY).unwrap();
    let main = f.prepare(Role::MainWithDraftWeights);
    let draft = f.prepare(Role::OneDraft);
    let verify = f.prepare(Role::MainVerify);
    let union = f.prepare(Role::InstallUnion);
    let installed = compile(&composition, &union);
    let main_plan = compile(&composition, &main);
    let draft_plan = compile(&composition, &draft);
    let verify_plan = compile(&composition, &verify);
    assert!(verify_plan
        .executable()
        .execution_plan()
        .requires_selected_prefix_commit());
    for consumer in [&main_plan, &draft_plan, &verify_plan] {
        assert_eq!(
            installed
                .executable()
                .execution_plan()
                .payload()
                .execution_weights(),
            consumer
                .executable()
                .execution_plan()
                .payload()
                .execution_weights()
        );
    }
    execution::assert_shared_installation(
        &f,
        &composition,
        &union,
        installed,
        main_plan,
        draft_plan,
        verify_plan,
    );
}
