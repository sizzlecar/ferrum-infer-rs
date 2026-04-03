//! Tests for tensor parallel weight sharding logic.
//! Runs on CPU without GPU — verifies sharding math and tensor operations.

use candle_core::{DType, Device, Tensor};

/// Verify column-parallel sharding: Q/K/V split along output (row) dim.
#[test]
fn column_parallel_qkv_sharding() {
    let tp_size = 2;
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = 64;
    let hidden = 2048;

    let q_dim = num_heads * head_dim; // 2048
    let kv_dim = num_kv_heads * head_dim; // 512

    // Simulate Q weight [q_dim, hidden]
    let q_full = Tensor::arange(0f32, (q_dim * hidden) as f32, &Device::Cpu)
        .unwrap()
        .reshape((q_dim, hidden))
        .unwrap();

    // Shard for rank 0
    let q_shard_0 = q_full.narrow(0, 0, q_dim / tp_size).unwrap();
    assert_eq!(q_shard_0.dims(), &[q_dim / tp_size, hidden]);
    assert_eq!(q_shard_0.dims(), &[1024, 2048]);

    // Shard for rank 1
    let q_shard_1 = q_full.narrow(0, q_dim / tp_size, q_dim / tp_size).unwrap();
    assert_eq!(q_shard_1.dims(), &[1024, 2048]);

    // Verify no overlap: first element of shard 1 starts after shard 0
    let v0 = q_shard_0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v1 = q_shard_1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(v0[0], 0.0);
    assert_eq!(v1[0], (q_dim / tp_size * hidden) as f32);

    // Fused QKV shard shape
    let k_full = Tensor::zeros((kv_dim, hidden), DType::F32, &Device::Cpu).unwrap();
    let v_full = Tensor::zeros((kv_dim, hidden), DType::F32, &Device::Cpu).unwrap();
    let k_shard = k_full.narrow(0, 0, kv_dim / tp_size).unwrap();
    let v_shard = v_full.narrow(0, 0, kv_dim / tp_size).unwrap();
    let qkv_shard = Tensor::cat(&[&q_shard_0, &k_shard, &v_shard], 0).unwrap();

    let expected_qkv_dim = q_dim / tp_size + 2 * (kv_dim / tp_size);
    assert_eq!(qkv_shard.dims(), &[expected_qkv_dim, hidden]);
    // 1024 + 2*256 = 1536
    assert_eq!(expected_qkv_dim, 1536);
}

/// Verify row-parallel sharding: O/down split along input (column) dim.
#[test]
fn row_parallel_o_proj_sharding() {
    let tp_size = 2;
    let hidden = 8;
    let q_dim = 8;

    // O weight: [hidden, q_dim] — split columns (small for exact fp32 math)
    let o_full = Tensor::arange(0f32, (hidden * q_dim) as f32, &Device::Cpu)
        .unwrap()
        .reshape((hidden, q_dim))
        .unwrap();

    let shard_size = q_dim / tp_size;
    let o_shard_0 = o_full
        .narrow(1, 0, shard_size)
        .unwrap()
        .contiguous()
        .unwrap();
    let o_shard_1 = o_full
        .narrow(1, shard_size, shard_size)
        .unwrap()
        .contiguous()
        .unwrap();

    // Each shard: [hidden, q_dim/tp]
    assert_eq!(o_shard_0.dims(), &[hidden, shard_size]);
    assert_eq!(o_shard_1.dims(), &[hidden, shard_size]);

    // Row-parallel: input is sharded [1, q_dim/tp] per rank.
    // Each rank does: partial = input_shard @ O_shard.T → [1, hidden]
    // All-reduce(sum) of partials = full result.
    let input_full = Tensor::arange(0f32, q_dim as f32, &Device::Cpu)
        .unwrap()
        .reshape((1, q_dim))
        .unwrap();
    let input_0 = input_full.narrow(1, 0, shard_size).unwrap();
    let input_1 = input_full.narrow(1, shard_size, shard_size).unwrap();

    let partial_0 = input_0.matmul(&o_shard_0.t().unwrap()).unwrap();
    let partial_1 = input_1.matmul(&o_shard_1.t().unwrap()).unwrap();
    assert_eq!(partial_0.dims(), &[1, hidden]);
    assert_eq!(partial_1.dims(), &[1, hidden]);

    // All-reduce (sum) = full O projection result
    let full_result = (partial_0 + partial_1).unwrap();

    // Compare with non-sharded
    let expected = input_full.matmul(&o_full.t().unwrap()).unwrap();
    let diff = (&full_result - &expected)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-1, "Row-parallel result differs: {diff}");
}

/// Verify gate+up fusion with column-parallel sharding.
#[test]
fn column_parallel_gate_up_fusion() {
    let tp_size = 4;
    let hidden = 2048;
    let inter = 5632;
    let inter_shard = inter / tp_size; // 1408

    let gate_full = Tensor::ones((inter, hidden), DType::F32, &Device::Cpu).unwrap();
    let up_full = Tensor::ones((inter, hidden), DType::F32, &Device::Cpu).unwrap();

    for rank in 0..tp_size {
        let gate_shard = gate_full
            .narrow(0, rank * inter_shard, inter_shard)
            .unwrap();
        let up_shard = up_full.narrow(0, rank * inter_shard, inter_shard).unwrap();
        let fused = Tensor::cat(&[&gate_shard, &up_shard], 0).unwrap();

        // Fused: [2*inter_shard, hidden]
        assert_eq!(fused.dims(), &[2 * inter_shard, hidden]);
        assert_eq!(fused.dims(), &[2816, 2048]);
    }
}

/// Verify row-parallel down projection: sharded input → all-reduce → correct result.
#[test]
fn row_parallel_down_proj_sharding() {
    let tp_size = 2;
    let hidden = 8;
    let inter = 16;
    let inter_shard = inter / tp_size;

    let down_full = Tensor::arange(0f32, (hidden * inter) as f32, &Device::Cpu)
        .unwrap()
        .reshape((hidden, inter))
        .unwrap();

    let down_shard_0 = down_full
        .narrow(1, 0, inter_shard)
        .unwrap()
        .contiguous()
        .unwrap();
    let down_shard_1 = down_full
        .narrow(1, inter_shard, inter_shard)
        .unwrap()
        .contiguous()
        .unwrap();

    // Each rank's MLP activation output: [1, inter/tp]
    let act_full = Tensor::arange(1f32, (inter + 1) as f32, &Device::Cpu)
        .unwrap()
        .reshape((1, inter))
        .unwrap();
    let act_0 = act_full.narrow(1, 0, inter_shard).unwrap();
    let act_1 = act_full.narrow(1, inter_shard, inter_shard).unwrap();

    let partial_0 = act_0.matmul(&down_shard_0.t().unwrap()).unwrap();
    let partial_1 = act_1.matmul(&down_shard_1.t().unwrap()).unwrap();

    let full_result = (partial_0 + partial_1).unwrap();
    let expected = act_full.matmul(&down_full.t().unwrap()).unwrap();

    let diff = (&full_result - &expected)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-1, "Down proj row-parallel differs: {diff}");
}

/// Verify column-parallel gate+up → SiLU × mul → row-parallel down end-to-end.
#[test]
fn mlp_block_tp_e2e() {
    let tp_size = 2;
    let hidden = 4;
    let inter = 8;
    let inter_shard = inter / tp_size;

    // Weights
    let gate_full = Tensor::arange(0f32, (inter * hidden) as f32, &Device::Cpu)
        .unwrap()
        .reshape((inter, hidden))
        .unwrap();
    let up_full = Tensor::arange(0f32, (inter * hidden) as f32, &Device::Cpu)
        .unwrap()
        .reshape((inter, hidden))
        .unwrap();
    let down_full = Tensor::arange(0f32, (hidden * inter) as f32, &Device::Cpu)
        .unwrap()
        .reshape((hidden, inter))
        .unwrap();

    let input = Tensor::arange(1f32, (hidden + 1) as f32, &Device::Cpu)
        .unwrap()
        .reshape((1, hidden))
        .unwrap();

    // Full (non-sharded) path
    let gate_out = input.matmul(&gate_full.t().unwrap()).unwrap();
    let up_out = input.matmul(&up_full.t().unwrap()).unwrap();
    let silu_gate = (&gate_out / &(gate_out.neg().unwrap().exp().unwrap() + 1.0).unwrap()).unwrap();
    let act_full = (&silu_gate * &up_out).unwrap();
    let expected = act_full.matmul(&down_full.t().unwrap()).unwrap();

    // TP path: column-parallel gate+up, row-parallel down
    let mut tp_result = Tensor::zeros((1, hidden), DType::F32, &Device::Cpu).unwrap();
    for rank in 0..tp_size {
        let gate_shard = gate_full
            .narrow(0, rank * inter_shard, inter_shard)
            .unwrap();
        let up_shard = up_full.narrow(0, rank * inter_shard, inter_shard).unwrap();
        let down_shard = down_full
            .narrow(1, rank * inter_shard, inter_shard)
            .unwrap()
            .contiguous()
            .unwrap();

        let g = input.matmul(&gate_shard.t().unwrap()).unwrap();
        let u = input.matmul(&up_shard.t().unwrap()).unwrap();
        let silu = (&g / &(g.neg().unwrap().exp().unwrap() + 1.0).unwrap()).unwrap();
        let act = (&silu * &u).unwrap();
        let partial = act.matmul(&down_shard.t().unwrap()).unwrap();
        tp_result = (tp_result + partial).unwrap();
    }

    let diff = (&tp_result - &expected)
        .unwrap()
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-2, "MLP TP E2E differs: {diff}");
}

/// Verify attention head sharding: each rank gets num_heads/tp heads.
#[test]
fn attention_head_sharding() {
    let tp_size = 4;
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = 128;
    let seq_len = 10;

    // Simulate Q: [1, num_heads, seq_len, head_dim]
    let q_full =
        Tensor::zeros((1, num_heads, seq_len, head_dim), DType::F32, &Device::Cpu).unwrap();

    // Each rank gets heads_per_rank = 32/4 = 8 Q heads
    let heads_per_rank = num_heads / tp_size;
    for rank in 0..tp_size {
        let q_shard = q_full
            .narrow(1, rank * heads_per_rank, heads_per_rank)
            .unwrap();
        assert_eq!(q_shard.dims(), &[1, heads_per_rank, seq_len, head_dim]);
    }

    // KV heads per rank = 8/4 = 2
    let kv_heads_per_rank = num_kv_heads / tp_size;
    let kv_full =
        Tensor::zeros((seq_len, num_kv_heads, head_dim), DType::F32, &Device::Cpu).unwrap();
    for rank in 0..tp_size {
        let kv_shard = kv_full
            .narrow(1, rank * kv_heads_per_rank, kv_heads_per_rank)
            .unwrap();
        assert_eq!(kv_shard.dims(), &[seq_len, kv_heads_per_rank, head_dim]);
    }
}

/// Verify embedding is replicated (not sharded).
#[test]
fn embedding_replicated() {
    let vocab = 32000;
    let hidden = 4096;
    let tp_size = 4;

    // Each rank gets the FULL embedding table
    let embed = Tensor::zeros((vocab, hidden), DType::F32, &Device::Cpu).unwrap();
    for _rank in 0..tp_size {
        // No sharding — each rank uses the same full embedding
        assert_eq!(embed.dims(), &[vocab, hidden]);
    }
}

/// Verify norm weights are replicated (not sharded).
#[test]
fn norms_replicated() {
    let hidden = 4096;
    let tp_size = 4;

    let norm_w = Tensor::ones(hidden, DType::F32, &Device::Cpu).unwrap();
    for _rank in 0..tp_size {
        assert_eq!(norm_w.dims(), &[hidden]);
    }
}

/// Verify sharded GEMM dimensions match for TP=4.
#[test]
fn tp4_dimension_consistency() {
    let tp_size = 4;
    let hidden = 4096;
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = 128;
    let inter = 11008;

    // Per-rank dimensions
    let q_dim_shard = (num_heads / tp_size) * head_dim; // 8 * 128 = 1024
    let kv_dim_shard = (num_kv_heads / tp_size) * head_dim; // 2 * 128 = 256
    let inter_shard = inter / tp_size; // 2752

    // QKV fused shard
    let qkv_shard_dim = q_dim_shard + 2 * kv_dim_shard; // 1024 + 512 = 1536
    assert_eq!(qkv_shard_dim, 1536);

    // O proj: [hidden, q_dim_shard] — row-parallel, output is hidden (full)
    assert_eq!(q_dim_shard, 1024);

    // gate_up fused shard
    assert_eq!(2 * inter_shard, 5504);

    // down proj: [hidden, inter_shard] — row-parallel, output is hidden (full)
    assert_eq!(inter_shard, 2752);

    // Verify hidden stays full (not sharded) — needed for replicated norms
    assert_eq!(hidden, 4096);
}
