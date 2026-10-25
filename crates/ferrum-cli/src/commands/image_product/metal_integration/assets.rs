//! Small real archives from declared family schemas, not trained image weights.
use ferrum_models::{qwen3_vl_text::*, qwen3_vl_vision::*, qwen_image21::*};
use ferrum_runtime::*;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};

fn json_file(root: &Path, name: &str, value: &Value) {
    let path = root.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn archive(root: &Path, name: &str, schemas: &[&WeightSchema], extra: &[(&str, Vec<u64>)]) {
    let mut values = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    for schema in schemas {
        for weight in &schema.components {
            assert_eq!(weight.external_names.len(), 1);
            let shape = weight.dimensions.iter().map(|&v| v as usize).collect();
            let value = (shape, vec![0; weight.physical_bytes().unwrap() as usize]);
            if let Some(previous) = values.insert(weight.external_names[0].clone(), value.clone()) {
                assert_eq!(previous, value);
            }
        }
    }
    for (name, dimensions) in extra {
        let shape = dimensions.iter().map(|&n| n as usize).collect::<Vec<_>>();
        let value = (shape.clone(), vec![0; shape.iter().product::<usize>() * 2]);
        assert!(values.insert(name.to_string(), value).is_none());
    }
    let views = values.iter().map(|(name, (shape, bytes))| {
        (
            name.as_str(),
            TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
        )
    });
    let path = root.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    serialize_to_file(views, &None, &path).unwrap();
}

pub(super) fn package() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let text = json!({"model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"],
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","hidden_act":"silu",
        "attention_bias":false,"attention_dropout":0.0,"num_hidden_layers":3,"hidden_size":8,
        "intermediate_size":16,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,
        "vocab_size":32,"max_position_embeddings":8192,"rms_norm_eps":0.000001,"rope_theta":100,
        "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[2,1,1]}},
        "vision_config":{"model_type":"qwen3_vl","dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh",
        "depth":3,"hidden_size":8,"intermediate_size":16,"num_heads":2,"num_position_embeddings":16,
        "out_hidden_size":8,"deepstack_visual_indexes":[0,1,2],"in_channels":3,"patch_size":16,
        "temporal_patch_size":2,"spatial_merge_size":2}});
    let dit = json!({"_class_name":"QwenImage21Transformer2DModel","num_layers":1,
        "num_attention_heads":1,"attention_head_dim":8,"context_in_dim":8,"in_channels":64,
        "out_channels":64,"patch_size":1,"mlp_ratio":2,"axes_dims_rope":[2,2,4],
        "causal_condition":true,"eps":0.000001});
    let vae = json!({"_class_name":"AutoencoderKLQwenImage21","is_residual":true,"patch_size":null,
        "dropout":0.0,"in_channels":4,"out_channels":4,"attn_scales":[],"z_dim":64,"base_dim":1,
        "dim_mult":[1,2,4,8,8],"num_res_blocks":2,"temperal_downsample":[false,true,true,true],
        "latents_mean":vec![0.;64],"latents_std":vec![1.;64]});
    let scheduler = json!({"_class_name":"FlowMatchEulerDiscreteScheduler","_diffusers_version":"fixture",
        "base_image_seq_len":256,"base_shift":0.5,"invert_sigmas":false,"max_image_seq_len":8192,
        "max_shift":0.9,"num_train_timesteps":1000,"shift":1.0,"shift_terminal":null,
        "stochastic_sampling":false,"time_shift_type":"exponential","use_beta_sigmas":false,
        "use_dynamic_shifting":false,"use_exponential_sigmas":false,"use_karras_sigmas":false});
    let processor = json!({"processor_class":"Qwen3VLProcessor","image_processor":{
        "image_processor_type":"Qwen2VLImageProcessorFast","data_format":"channels_first",
        "do_convert_rgb":true,"do_normalize":true,"do_rescale":true,"do_resize":true,
        "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":1.0/255.0,
        "patch_size":16,"merge_size":2,"temporal_patch_size":2,"resample":3,
        "size":{"shortest_edge":1024,"longest_edge":1048576}}});
    json_file(
        root,
        "model_index.json",
        &json!({"_class_name":"QwenImage21Pipeline",
        "processor":["transformers","Qwen3VLProcessor"],"text_encoder":["transformers","Qwen3VLForConditionalGeneration"],
        "transformer":["diffusers","QwenImage21Transformer2DModel"],"vae":["diffusers","AutoencoderKLQwenImage21"],
        "scheduler":["diffusers","FlowMatchEulerDiscreteScheduler"],"sample_sigmas":[1.0,0.5]}),
    );
    for (name, value) in [
        ("text_encoder/config.json", &text),
        ("transformer/config.json", &dit),
        ("vae/config.json", &vae),
        ("scheduler/scheduler_config.json", &scheduler),
        ("processor/processor_config.json", &processor),
    ] {
        json_file(root, name, value);
    }
    let text_config = Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&text).unwrap(),
        Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]]).unwrap(),
    )
    .unwrap();
    let text_weights = Qwen3VLTextEncoderFamily::new()
        .unwrap()
        .weights(&text_config)
        .unwrap();
    let vision = Qwen3VLVisionArchitecture::from_hf_value(&text).unwrap();
    let vision_config =
        Qwen3VLVisionConfig::new(vision, vec![Qwen3VLVisionGrid::new(1, 2, 2).unwrap()]).unwrap();
    let vision_weights = Qwen3VLVisionFamily::new()
        .unwrap()
        .weights(&vision_config)
        .unwrap();
    archive(
        root,
        "text_encoder/model.safetensors",
        &[
            text_weights.stored_schema().unwrap(),
            vision_weights.stored_schema().unwrap(),
        ],
        &[
            (
                Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME,
                vec![32, 8],
            ),
            (Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME, vec![16, 8]),
        ],
    );
    let dit_config = QwenImage21Config::from_hf_json(
        &serde_json::to_vec(&dit).unwrap(),
        StructureLimits::default(),
    )
    .unwrap();
    archive(
        root,
        "transformer/diffusion_pytorch_model.safetensors",
        &[
            &ExpectedWeights::from_config(&dit_config, StructureLimits::default())
                .unwrap()
                .weight_schema()
                .unwrap(),
        ],
        &[],
    );
    let vae_json = serde_json::to_vec(&vae).unwrap();
    let decoder = QwenImage21VaeDecoderConfig::from_hf_json(&vae_json, 1, 2, 2).unwrap();
    let encoder = QwenImage21VaeEncoderConfig::from_hf_json(&vae_json, 1, 32, 32).unwrap();
    let dw = QwenImage21VaeDecoderFamily::new()
        .unwrap()
        .weights(&decoder)
        .unwrap();
    let ew = QwenImage21VaeEncoderFamily::new()
        .unwrap()
        .weights(&encoder)
        .unwrap();
    archive(
        root,
        "vae/diffusion_pytorch_model.safetensors",
        &[dw.stored_schema().unwrap(), ew.stored_schema().unwrap()],
        &[],
    );
    tokenizer(root);
    directory
}
fn tokenizer(root: &Path) {
    let vocab = [
        "[UNK]",
        "system",
        "Comprehend",
        "and",
        "analyze",
        "the",
        "provided",
        "prompt.",
        "user",
        "assistant",
        "cat",
        "blue",
    ]
    .into_iter()
    .enumerate()
    .map(|(i, s)| (s.to_owned(), i as u32))
    .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.with_post_processor(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
    ));
    tokenizer.add_special_tokens(
        &[
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
        .map(|s| tokenizers::AddedToken::from(s, true)),
    );
    fs::write(
        root.join("processor/tokenizer.json"),
        tokenizer.to_string(false).unwrap(),
    )
    .unwrap();
    json_file(
        root,
        "processor/tokenizer_config.json",
        &json!({"tokenizer_class":"Qwen2Tokenizer","processor_class":"Qwen3VLProcessor",
        "bos_token":null,"pad_token":"<|endoftext|>","eos_token":"<|im_end|>","add_prefix_space":false,"split_special_tokens":false}),
    );
    fs::write(root.join("processor/chat_template.jinja"),"{%- for m in messages -%}{{ '<|im_start|>' + m.role + '\\n' }}{%- for c in m.content -%}{{ c.text }}{%- endfor -%}{{ '<|im_end|>\\n' }}{%- endfor -%}").unwrap();
}
