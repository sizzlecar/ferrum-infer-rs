//! Synthetic feature rows, real managed inputs and nonzero native attention.
//! This is a causal executor fixture, not a vision encoder or image-quality test.
use super::*;
use ferrum_execution::{CausalInputSegment, CausalRequestInputs, CausalTokenInput};
use ferrum_models::{
    qwen3_vl_vision::Qwen3VLVisionGrid,
    vnext::qwen35::{
        ROPE_COS_INPUT_ID, ROPE_SIN_INPUT_ID, VISUAL_REPLACE_MASK_INPUT_ID, VISUAL_ROWS_INPUT_ID,
    },
};
use ferrum_runtime::{HostMemoryClass, ManagedHostBytes, ProgramValueId};
use safetensors::tensor::{Dtype, TensorView};

pub(super) const POSITIVE_TEXT: &str =
    "Edit <Picture 1>. Keep the subject and change the sky to red.";
pub(super) const NEGATIVE_TEXT: &str =
    "Edit <Picture 1>. Keep the subject and change the sky to blue.";

#[derive(Clone, Copy)]
pub(super) struct VisualCase {
    pub signs: &'static [f32],
    pub quarter_turn: f32,
    pub chunk: usize,
}

pub(super) const MIXED: VisualCase = VisualCase {
    signs: &[1., -1.],
    quarter_turn: 1.,
    chunk: 2,
};
pub(super) const OTHER: VisualCase = VisualCase {
    signs: &[1., 1., 1., 1.],
    quarter_turn: -1.,
    chunk: 3,
};

pub(super) struct VisualFixture {
    pub assets: Arc<NativeTextExpansion>,
    replay: bool,
    // Retain the temporary directory until the actual source owner drops.
    _files: Fixture,
}

impl VisualFixture {
    pub fn new() -> Self {
        let mut files = Fixture::new_two_answers();
        drop(files.take_expansion());
        let config_path = files.directory.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["vision_start_token_id"] = 4.into();
        config["image_token_id"] = 5.into();
        config["vision_end_token_id"] = 6.into();
        config["text_config"]["rope_parameters"] = serde_json::json!({
            "rope_type": "default", "rope_theta": 10000.,
            "partial_rotary_factor": 1., "mrope_interleaved": true,
            "mrope_section": [1, 0, 0]
        });
        std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();

        let weight_path = files.directory.join("model.safetensors");
        let bytes = std::fs::read(&weight_path).unwrap();
        let original = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let owned = original
            .tensors()
            .into_iter()
            .map(|(name, tensor)| {
                let mut values = vec![0f32; tensor.shape().iter().product()];
                match name.as_str() {
                    "model.embed_tokens.weight" => {
                        for (id, row) in values.chunks_exact_mut(2).enumerate() {
                            // The two answer embeddings force a real EOS on
                            // the next decode, even with nonzero cached values.
                            row[0] = if [ANSWER_ID, SECOND_ANSWER_ID].contains(&(id as u32)) {
                                -100.
                            } else {
                                1.
                            };
                        }
                    }
                    "model.lm_head.weight" => {
                        values[ANSWER_ID as usize * 2 + 1] = 1.;
                        values[SECOND_ANSWER_ID as usize * 2 + 1] = -1.;
                        values[EOS_ID as usize * 2] = -2.;
                    }
                    "model.layers.1.self_attn.q_proj.weight"
                    | "model.layers.1.self_attn.k_proj.weight" => {
                        values.copy_from_slice(&[1., 0., 0., 1.]);
                    }
                    "model.layers.1.self_attn.v_proj.weight" => {
                        values.copy_from_slice(&[0., 0., 0., 1.]);
                    }
                    "model.layers.1.self_attn.o_proj.weight" => {
                        values.copy_from_slice(&[0., 0., 0., 32.]);
                    }
                    "model.layers.1.self_attn.q_norm.weight"
                    | "model.layers.1.self_attn.k_norm.weight" => values.fill(1.),
                    _ => {}
                }
                // Qwen's RMS scales are checkpoint deltas (zero means one).
                // The GDN/MLP branches stay zero; full attention does not.
                let bytes = values
                    .into_iter()
                    .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                    .collect::<Vec<_>>();
                (name, tensor.shape().to_vec(), bytes)
            })
            .collect::<Vec<_>>();
        let views = owned.iter().map(|(name, shape, data)| {
            (
                name.as_str(),
                TensorView::new(Dtype::BF16, shape.clone(), data).unwrap(),
            )
        });
        safetensors::tensor::serialize_to_file(views, &None, &weight_path).unwrap();

        let tokenizer_path = files.directory.join("tokenizer.json");
        let mut tokenizer: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&tokenizer_path).unwrap()).unwrap();
        let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
        vocab.retain(|_, id| {
            ![ANSWER_ID as u64, SECOND_ANSWER_ID as u64].contains(&id.as_u64().unwrap())
        });
        vocab.insert(format!("</think> {POSITIVE_TEXT}"), ANSWER_ID.into());
        vocab.insert(format!("</think> {NEGATIVE_TEXT}"), SECOND_ANSWER_ID.into());
        std::fs::write(tokenizer_path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let sources = Arc::new(
            ProductionModelSourceBundle::open_colocated_safetensors(&files.directory).unwrap(),
        );
        let assets = Arc::new(
            NativeTextExpansion::from_sources(sources, protocol_limits())
                .unwrap()
                .with_visual_prompt()
                .unwrap(),
        );
        Self {
            assets,
            replay: false,
            _files: files,
        }
    }

    pub fn with_decode_replay(mut self) -> Self {
        self.replay = true;
        self
    }

    pub fn prepare(&self, case: VisualCase) -> PreparedTextExpansion {
        self.prepare_on(case, ferrum_types::Device::CPU)
    }

    pub fn prepare_on(
        &self,
        case: VisualCase,
        device: ferrum_types::Device,
    ) -> PreparedTextExpansion {
        self.prepare_with_limit(case, device, 2)
    }

    pub fn prepare_with_limit(
        &self,
        case: VisualCase,
        device: ferrum_types::Device,
        max_new_tokens: u32,
    ) -> PreparedTextExpansion {
        let mut engine = EngineConfig::default();
        engine.backend.device = device;
        if self.replay {
            engine.backend.reusable_execution_capture.scope =
                ferrum_types::ReusableExecutionCaptureScope::DecodeOnly;
            engine.backend.reusable_execution_capture.preparation =
                ferrum_types::ReusableExecutionPreparationMode::OnDemand;
        }
        self.assets
            .prepare_edit(
                "change the sky",
                (8, 8),
                &Qwen3VLVisionGrid::new(1, 2, case.signs.len() as u64 * 2).unwrap(),
                2,
                &PH3ImagePromptExpansionOptions {
                    engine,
                    max_new_tokens,
                    prefill_chunk_tokens: case.chunk,
                },
            )
            .unwrap()
    }
}

fn input(host: &HostMemoryAccount, bytes: &[u8]) -> ManagedHostBytes {
    let mut owner = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    owner.as_mut().copy_from_slice(bytes);
    owner.freeze()
}

fn half_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

pub(super) fn bind(
    prepared: &mut PreparedTextExpansion,
    host: &HostMemoryAccount,
    case: VisualCase,
) {
    bind_positions(prepared, host, case, false);
}

pub(super) fn bind_varying_positions(
    prepared: &mut PreparedTextExpansion,
    host: &HostMemoryAccount,
    case: VisualCase,
) {
    bind_positions(prepared, host, case, true);
}

fn bind_positions(
    prepared: &mut PreparedTextExpansion,
    host: &HostMemoryAccount,
    case: VisualCase,
    varying_decode: bool,
) {
    let plan = prepared.visual.as_ref().unwrap();
    let run = &plan.tokens().image_runs()[0];
    assert_eq!(run.rows(), case.signs.len());
    let start = run.token_start();
    let n = plan.tokens().maximum_tokens();
    let features = input(host, &half_bytes(case.signs.iter().flat_map(|&s| [0., s])));
    let mask = input(host, &vec![1; run.rows()]);
    // Exact tables cover every physical prompt/decode position. Only visual
    // keys rotate; the final ordinary query is [1,0]. For the mixed +/- rows,
    // a positive quarter turn favors the negative value, and vice versa.
    let cos = input(
        host,
        &half_bytes((0..n).map(|i| {
            if (start..start + run.rows()).contains(&i) {
                0.
            } else if varying_decode && i >= prepared.tokens.len() {
                [1., 0., -1., 0.][(i - prepared.tokens.len()) % 4]
            } else {
                1.
            }
        })),
    );
    let sin = input(
        host,
        &half_bytes((0..n).map(|i| {
            if (start..start + run.rows()).contains(&i) {
                case.quarter_turn
            } else if varying_decode && i >= prepared.tokens.len() {
                [0., 1., 0., -1.][(i - prepared.tokens.len()) % 4]
            } else {
                0.
            }
        })),
    );
    let values = [
        (VISUAL_ROWS_INPUT_ID, start, run.rows(), features),
        (VISUAL_REPLACE_MASK_INPUT_ID, start, run.rows(), mask),
        (ROPE_COS_INPUT_ID, 0, n, cos),
        (ROPE_SIN_INPUT_ID, 0, n, sin),
    ]
    .into_iter()
    .map(|(id, start, rows, bytes)| {
        CausalTokenInput::new(
            ProgramValueId::new(id).unwrap(),
            vec![CausalInputSegment::new(start, rows, bytes).unwrap()],
        )
    })
    .collect();
    prepared.causal_inputs = Some(Arc::new(
        CausalRequestInputs::new(host, plan.schema(), n, values).unwrap(),
    ));
}

pub(super) fn cohort(
    fixture: &VisualFixture,
    requests: Vec<PreparedTextExpansion>,
) -> PreparedExpansionCohort {
    fixture.assets.prepare_cohort(requests).unwrap()
}
