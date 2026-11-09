use super::*;
use crate::image_generation::p_h3::expansion::{
    mtp_assets::MtpExpansionAssets,
    tests::fixture::{self as original, Fixture as OriginalFixture},
};
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
use ferrum_models::qwen35_config::Qwen35TextConfig;
use ferrum_runtime::DeviceRuntime;
use safetensors::tensor::{Dtype, TensorView};

pub(super) struct Fixture {
    pub(super) assets: Arc<NativeTextExpansion>,
    _original: OriginalFixture,
}

impl Fixture {
    pub(super) fn new() -> Self {
        Self::build(false)
    }

    pub(super) fn nonzero_continuations() -> Self {
        Self::build(true)
    }

    fn build(continuations: bool) -> Self {
        let mut original = OriginalFixture::new_two_answers();
        // No active mmap is edited. Reopen the finished source only after all
        // original fixture mappings and model owners have been dropped.
        drop(original.take_expansion());
        let directory = &original.directory;
        let config_path = directory.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["text_config"]["attn_output_gate"] = true.into();
        config["text_config"]["rope_parameters"] = serde_json::json!({
            "rope_type":"default", "rope_theta":10000., "partial_rotary_factor":1.,
            "mrope_interleaved":true, "mrope_section":[1,0,0]
        });
        let text = Qwen35TextConfig::from_hf_config_value(&config).unwrap();
        let h = text.hidden_size;
        let i = text.dense_intermediate_size.unwrap();
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let weight_path = directory.join("model.safetensors");
        let old = std::fs::read(&weight_path).unwrap();
        let archive = safetensors::SafeTensors::deserialize(&old).unwrap();
        let mut tensors = archive
            .tensors()
            .into_iter()
            .map(|(name, tensor)| {
                assert_eq!(tensor.dtype(), Dtype::BF16);
                if name.ends_with(".self_attn.q_proj.weight") {
                    // The real gated query ABI includes the separate output gate.
                    let shape = vec![text.full_attention_q_proj_total_dim(), h];
                    let bytes = vec![0; shape.iter().product::<usize>() * 2];
                    (name, shape, bytes)
                } else {
                    (name, tensor.shape().to_vec(), tensor.data().to_vec())
                }
            })
            .collect::<Vec<_>>();
        for (role, shape) in [
            ("fc", vec![h, 2 * h]),
            ("pre_fc_norm_embedding", vec![h]),
            ("pre_fc_norm_hidden", vec![h]),
            ("norm", vec![h]),
            ("layers.0.input_layernorm", vec![h]),
            ("layers.0.post_attention_layernorm", vec![h]),
            (
                "layers.0.self_attn.q_proj",
                vec![text.full_attention_q_proj_total_dim(), h],
            ),
            (
                "layers.0.self_attn.k_proj",
                vec![text.full_attention_kv_total_dim(), h],
            ),
            (
                "layers.0.self_attn.v_proj",
                vec![text.full_attention_kv_total_dim(), h],
            ),
            (
                "layers.0.self_attn.o_proj",
                vec![h, text.full_attention_query_total_dim()],
            ),
            ("layers.0.self_attn.q_norm", vec![text.head_dim]),
            ("layers.0.self_attn.k_norm", vec![text.head_dim]),
            ("layers.0.mlp.gate_proj", vec![i, h]),
            ("layers.0.mlp.up_proj", vec![i, h]),
            ("layers.0.mlp.down_proj", vec![h, i]),
        ] {
            let mut values = vec![0.0f32; shape.iter().product()];
            if role == "fc" {
                // The original concatenation is [embedding, accepted hidden].
                // This checkpoint makes the real draft predict from embedding;
                // recurrent/attention arithmetic still executes on real states.
                for row in 0..h {
                    values[row * 2 * h + row] = 1.;
                }
            }
            let bytes = values
                .into_iter()
                .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                .collect();
            tensors.push((format!("mtp.{role}.weight"), shape, bytes));
        }
        if continuations {
            for (name, _, bytes) in &mut tensors {
                if name == "model.embed_tokens.weight" {
                    for (token, vector) in [
                        (ANSWER_ID as usize, [0.0_f32, 1.0]),
                        (SECOND_ANSWER_ID as usize, [-1.0_f32, -1.0]),
                        (0, [1.0_f32, 0.0]),
                    ] {
                        for (column, value) in vector.into_iter().enumerate() {
                            let offset = (token * h + column) * 2;
                            bytes[offset..offset + 2]
                                .copy_from_slice(&half::bf16::from_f32(value).to_le_bytes());
                        }
                    }
                } else if name == "model.lm_head.weight" {
                    // Three valid source tokens form a cycle. EOS remains the
                    // protocol's original token; this fixture is cancelled.
                    bytes[0..4].copy_from_slice(
                        &[
                            half::bf16::from_f32(-1.0).to_le_bytes(),
                            half::bf16::from_f32(-1.0).to_le_bytes(),
                        ]
                        .concat(),
                    );
                    bytes[EOS_ID as usize * h * 2..(EOS_ID as usize + 1) * h * 2].fill(0);
                } else if name == "mtp.fc.weight" {
                    // This draft intentionally misses one transition in each
                    // cycle, exercising rejection after catalogs are populated.
                    bytes.fill(0);
                    for column in 0..h {
                        bytes[column * 2..column * 2 + 2]
                            .copy_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
                    }
                } else if name.contains("norm") {
                    // Gated-delta output normalization uses an ordinary
                    // multiplicative scale: zero would erase its changing state.
                    for value in bytes.chunks_exact_mut(2) {
                        value.copy_from_slice(&half::bf16::from_f32(0.5).to_le_bytes());
                    }
                } else if name != "model.lm_head.weight" {
                    // Real nonzero attention, recurrent and FFN state. The
                    // existing short-EOS fixture stays unchanged above.
                    for (index, value) in bytes.chunks_exact_mut(2).enumerate() {
                        value.copy_from_slice(
                            &half::bf16::from_f32((index % 7 + 1) as f32 / 64.0).to_le_bytes(),
                        );
                    }
                }
            }
        }
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::tensor::serialize_to_file(views, &None, &weight_path).unwrap();
        let sources =
            Arc::new(ProductionModelSourceBundle::open_colocated_safetensors(directory).unwrap());
        let assets = Arc::new(
            NativeTextExpansion::from_sources(sources, original::protocol_limits()).unwrap(),
        );
        Self {
            assets,
            _original: original,
        }
    }

    pub(super) fn requests(&self) -> Vec<PreparedTextExpansion> {
        [(3, 3, 2), (0, 5, 4)]
            .into_iter()
            .map(|(token, count, limit)| {
                let mut request = original::prepared(&self.assets, limit);
                // As in the original tiny engine fixture, these explicit source
                // token rows select distinct real answers without a mock executor.
                request.tokens = vec![TokenId::new(token); count];
                request.engine.runtime.max_model_len = Some(count + limit as usize);
                request
            })
            .collect()
    }

    pub(super) fn machine(
        &self,
        composition: &RuntimeComposition<CpuDeviceRuntime>,
        host: &HostMemoryAccount,
    ) -> MtpExpansionCohort<CpuDeviceRuntime> {
        assert_eq!(composition.runtime().resident_bytes(), 0);
        self.machine_with_requests(composition, host, self.requests())
    }

    pub(super) fn machine_with_requests<R: DeviceRuntime>(
        &self,
        composition: &RuntimeComposition<R>,
        host: &HostMemoryAccount,
        requests: Vec<PreparedTextExpansion>,
    ) -> MtpExpansionCohort<R> {
        let mut prepared = self.assets.prepare_text_cohort(requests).unwrap();
        let capacity = prepared.engine.scheduler.max_running_requests;
        initialization::prepare_engine(&mut prepared).unwrap();
        assert_eq!(prepared.engine.scheduler.max_running_requests, capacity);
        let assets = Arc::new(MtpExpansionAssets::from_native(&self.assets).unwrap());
        let requirements = initialization::host_requirements(
            &self.assets,
            &assets,
            composition,
            &prepared.requests,
            &prepared.engine,
            host,
        )
        .unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert!(requirements.startup_device_bytes() > 0);
        let phase = host.reserve_scope(&requirements.claims()).unwrap();
        let mut initialization = initialization::MtpInitialization::new(
            Arc::clone(&self.assets),
            assets,
            prepared,
            phase,
        );
        let machine = loop {
            match initialization.poll(composition).unwrap() {
                initialization::InitializationPoll::Pending { made_progress } => {
                    // There are no admitted requests during synchronous static
                    // initialization. An unwakeable reference is an error.
                    assert!(made_progress);
                }
                initialization::InitializationPoll::Ready(machine) => break machine,
            }
        };
        assert!(initialization.failure().is_none());
        assert_eq!(
            machine
                .programs
                .main()
                .unwrap()
                .shared_weights_receipt()
                .unwrap()
                .evidence(),
            machine
                .programs
                .draft()
                .unwrap()
                .shared_weights_receipt()
                .unwrap()
                .evidence(),
        );
        machine
    }
}

pub(super) use original::{composition, host, ANSWER_ID, EOS_ID, SECOND_ANSWER_ID};
