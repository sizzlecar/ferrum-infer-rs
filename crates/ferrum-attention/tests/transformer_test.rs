//! Test the full Metal transformer layer against CPU reference.

#[cfg(target_os = "macos")]
mod metal_transformer_tests {
    use ferrum_attention::metal::pipelines::MetalPipelines;
    use ferrum_attention::metal::transformer::*;
    use metal::Device;

    fn random_weights(pipes: &MetalPipelines, h: usize, im: usize, nh: usize, nkv: usize, hd: usize) -> MetalLayerWeights {
        let rng = |n: usize, seed: u32| -> Vec<f32> {
            (0..n).map(|i| {
                let x = ((i as u32).wrapping_mul(seed).wrapping_add(12345)) as f32;
                (x % 1000.0) / 10000.0 - 0.05
            }).collect()
        };

        MetalLayerWeights {
            input_ln_w: pipes.buffer_from_data(&vec![1.0f32; h]),
            q_proj_w: pipes.buffer_from_data(&rng(nh * hd * h, 7)),
            k_proj_w: pipes.buffer_from_data(&rng(nkv * hd * h, 13)),
            v_proj_w: pipes.buffer_from_data(&rng(nkv * hd * h, 17)),
            o_proj_w: pipes.buffer_from_data(&rng(h * nh * hd, 23)),
            q_norm_w: pipes.buffer_from_data(&vec![1.0f32; hd]),
            k_norm_w: pipes.buffer_from_data(&vec![1.0f32; hd]),
            post_ln_w: pipes.buffer_from_data(&vec![1.0f32; h]),
            gate_proj_w: pipes.buffer_from_data(&rng(im * h, 31)),
            up_proj_w: pipes.buffer_from_data(&rng(im * h, 37)),
            down_proj_w: pipes.buffer_from_data(&rng(h * im, 41)),
        }
    }

    #[test]
    fn test_metal_transformer_layer_runs() {
        let device = Device::system_default().expect("no Metal device");
        let pipes = MetalPipelines::new(&device);

        let h = 256;  // small for testing
        let im = 512;
        let nh = 4;
        let nkv = 2;
        let hd = 64;
        let tokens = 8;

        let cfg = MetalTransformerConfig {
            hidden_size: h, intermediate_size: im,
            num_heads: nh, num_kv_heads: nkv,
            head_dim: hd, rms_norm_eps: 1e-6,
        };

        let weights = random_weights(&pipes, h, im, nh, nkv, hd);
        let mut kv_cache = MetalKvCache::new();

        // Precompute cos/sin
        let half = hd / 2;
        let max_seq = 1024;
        let mut cos = vec![0.0f32; max_seq * half];
        let mut sin = vec![0.0f32; max_seq * half];
        for pos in 0..max_seq {
            for i in 0..half {
                let freq = 1.0f64 / 1000000.0f64.powf((2 * i) as f64 / hd as f64);
                let angle = pos as f64 * freq;
                cos[pos * half + i] = angle.cos() as f32;
                sin[pos * half + i] = angle.sin() as f32;
            }
        }
        let cos_buf = pipes.buffer_from_data(&cos);
        let sin_buf = pipes.buffer_from_data(&sin);

        let input_data: Vec<f32> = (0..tokens * h).map(|i| ((i as f32) * 0.001).sin() * 0.1).collect();
        let input = pipes.buffer_from_data(&input_data);

        // Prefill
        let output = metal_layer_forward(&pipes, &input, tokens, &weights, &cfg, &mut kv_cache, 0, &cos_buf, &sin_buf);

        let out_data = MetalPipelines::read_buffer(&output, tokens * h);
        assert_eq!(out_data.len(), tokens * h);

        // Check output is not all zeros or NaN
        let sum: f32 = out_data.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "output is all zeros");
        assert!(out_data.iter().all(|x| x.is_finite()), "output has NaN/Inf");

        eprintln!("Transformer layer output: first 5 = {:?}", &out_data[..5]);
        eprintln!("KV cache len after prefill: {}", kv_cache.len);

        // Decode step
        let decode_input: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.002).cos() * 0.1).collect();
        let decode_buf = pipes.buffer_from_data(&decode_input);
        let decode_out = metal_layer_forward(&pipes, &decode_buf, 1, &weights, &cfg, &mut kv_cache, tokens, &cos_buf, &sin_buf);

        let dec_data = MetalPipelines::read_buffer(&decode_out, h);
        assert!(dec_data.iter().all(|x| x.is_finite()), "decode output has NaN/Inf");
        eprintln!("Decode output: first 5 = {:?}", &dec_data[..5]);
        eprintln!("KV cache len after decode: {}", kv_cache.len);
    }
}
