#![cfg(feature = "cuda-sage-vsa-fp4")]
//! Managed sparse routing qualification at Q=K=0. Uniform probabilities and
//! dyadic V permit an independent scalar average oracle. This does not qualify
//! general softmax/two-level-P rounding, QKV producers, or complete FastH3 VSA.
#[path = "support/sparse_fp4_dynamic_cases.rs"]
mod dynamic;
#[path = "support/sparse_fp4_attention_component.rs"]
mod fixture;
use ferrum_execution::{
    component::{ComponentInitializationConfig, ComponentWeightSource},
    TensorExecutorConfig,
};
use ferrum_kernels::{
    backend::cuda::{
        vnext_ops::{
            cuda_tensor_runtime_config, cuda_tensor_runtime_enable_sage_vsa_fp4,
            cuda_tensor_runtime_enable_sage_vsa_fp4_regions,
            cuda_vnext_sparse_fp4_attention_checked_host_provider,
            cuda_vnext_sparse_fp4_attention_region_topk_provider,
        },
        vnext_runtime::{
            CudaDeviceRuntime, CudaPinnedWeightUploadConfig, CudaTensorRuntimeConfig,
            CudaTensorRuntimeOptions,
        },
    },
    native_ops::sage_vsa_fp4::{
        CheckedHostVsaPattern, CheckedVsaRegions, ListShape, SageVsaFp4Config,
    },
};
use ferrum_runtime::*;
use fixture::{id, Config, Producer, INPUTS, SCORES};
use std::{num::NonZeroUsize, sync::Arc};

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            32 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                transfer_staging_bytes: 8 << 20,
                // Q=256 tiles produces 8 MiB of BF16 output plus physical
                // alignment; the cross-packet geometry case needs that backing.
                output_bytes: 16 << 20,
                weight_backing_bytes: 8 << 20,
            },
        )
        .unwrap(),
        CudaDeviceRuntime::pinned_weight_upload_alignment().unwrap(),
    )
    .unwrap()
}
fn lists(host: &HostMemoryAccount, values: &[i32]) -> ManagedHostBytes {
    let mut owner = host
        .allocate_zeroed(HostMemoryClass::TransferStaging, values.len() * 4)
        .unwrap();
    for (dst, value) in owner.as_mut().chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    owner.freeze()
}
fn active(row: usize, keys: usize) -> Vec<usize> {
    if keys == 1 || row % 3 == 0 {
        vec![0]
    } else if keys == 4 && row % 3 == 2 {
        (0..keys).rev().collect()
    } else {
        vec![keys - 1, 0]
    }
}
fn v_code(c: &Config, batch_head: usize, key: usize, channel: usize, half_only: bool) -> u8 {
    if half_only && key % 128 >= 64 {
        return 7;
    } // masked rows deliberately differ
    match c.producer {
        Producer::CheckedHost => [2, 4, 6, 0][(batch_head + key / 128 + channel / 16) % 4],
        // Channel groups encode different bits of the block index, so routing
        // to key+4 cannot accidentally pass the original periodic V oracle.
        Producer::RegionTopK => [0, 2, 4, 6][(batch_head + ((key / 128) >> (channel / 16))) % 4],
    }
}
fn v_scale(batch_head: usize, channel: usize) -> (u8, f64) {
    [(0x30, 0.5), (0x38, 1.0), (0x40, 2.0)][(batch_head + channel / 16) % 3]
}
fn operands(c: &Config, half_only: bool) -> [Vec<u8>; 6] {
    let bh = (c.batches * c.heads) as usize;
    let q = c.query_blocks as usize * 128;
    let k = c.key_blocks as usize * 128;
    let mut v = vec![0; bh * 128 * k / 2];
    let mut sv = vec![0; bh * 128 * k / 16];
    for head in 0..bh {
        for channel in 0..128 {
            for key in 0..k {
                let offset = (head * 128 + channel) * k + key;
                v[offset / 2] |= v_code(c, head, key, channel, half_only) << ((offset % 2) * 4);
                let row = channel % 64;
                let group = key / 16;
                // Independent transcription of upstream 4d transpose-quantizer scale
                // storage: 64-row atoms, 4 scale columns, 16 interleaved row lanes.
                let sf = head * 128 * (k / 16)
                    + (channel / 64) * 64 * (k / 16)
                    + (group / 4) * 256
                    + group % 4
                    + (row / 16) * 4
                    + (row % 16) * 16;
                sv[sf] = v_scale(head, channel).0;
            }
        }
    }
    [
        vec![0; bh * q * 64],
        vec![0; bh * k * 64],
        v,
        vec![0x38; bh * q * 8],
        vec![0x38; bh * k * 8],
        sv,
    ]
}
fn expected(c: &Config, half_only: bool) -> Vec<u16> {
    expected_selected(c, half_only, |row| active(row, c.key_blocks as usize))
}
fn expected_selected(
    c: &Config,
    half_only: bool,
    select: impl Fn(usize) -> Vec<usize>,
) -> Vec<u16> {
    let mut result = Vec::new();
    let blocks = c.query_blocks as usize;
    let magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for bh in 0..(c.batches * c.heads) as usize {
        for q in 0..blocks * 128 {
            let selected = select(bh * blocks + q / 128);
            let valid = if half_only { 64 } else { 128 };
            for channel in 0..128 {
                let sum = selected
                    .iter()
                    .flat_map(|&block| (0..valid).map(move |r| block * 128 + r))
                    .map(|key| {
                        magnitudes[v_code(c, bh, key, channel, half_only) as usize]
                            * v_scale(bh, channel).1
                    })
                    .sum::<f64>();
                result.push(
                    half::bf16::from_f32((sum / (selected.len() * valid) as f64) as f32).to_bits(),
                );
            }
        }
    }
    result
}

#[test]
#[ignore = "requires SM120 and locked cuda-sage-vsa-fp4 artifact; routing/ownership diagnostic only"]
fn sparse_fp4_checked_host_uses_core_and_matches_uniform_attention_average() {
    let runtime = runtime();
    for (b, h, q, k, half_only) in [(1, 1, 1, 1, false), (2, 3, 2, 4, false), (1, 2, 2, 3, true)] {
        checked_host_case(&runtime, b, h, q, k, half_only);
    }
}
fn runtime() -> Arc<CudaDeviceRuntime> {
    let mut device = cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap();
    // Both tests share this physical device and its live capacity account;
    // declare both profiles under one consistent runtime fingerprint.
    cuda_tensor_runtime_enable_sage_vsa_fp4(&mut device).unwrap();
    cuda_tensor_runtime_enable_sage_vsa_fp4_regions(&mut device).unwrap();
    Arc::new(
        CudaDeviceRuntime::new_tensor_eager_with_options(
            CudaTensorRuntimeConfig {
                device,
                usable_memory_bytes: 1 << 30,
                bootstrap_memory_bytes: 256 << 20,
                stream_memory_bytes: 16 << 20,
                maximum_streams: 2,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 64,
                opaque_memory_profile: "vsa-routing-diagnostic-unqualified-capacity".into(),
            },
            CudaTensorRuntimeOptions {
                pinned_weight_upload: Some(CudaPinnedWeightUploadConfig {
                    minimum_allocation_bytes: 0,
                    maximum_allocation_bytes: NonZeroUsize::new(1 << 20).unwrap(),
                    maximum_registered_bytes: NonZeroUsize::new(8 << 20).unwrap(),
                }),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}
fn checked_host_case(
    runtime: &Arc<CudaDeviceRuntime>,
    b: u32,
    h: u32,
    q: u32,
    k: u32,
    half_only: bool,
) {
    let host = host();
    let shape = ListShape::new(b, h, q, k).unwrap();
    let mut indices = vec![i32::MIN; (b * h * q * k) as usize];
    let mut counts = vec![0; (b * h * q) as usize];
    for (row, count) in counts.iter_mut().enumerate() {
        let chosen = active(row, k as usize);
        *count = chosen.len() as i32;
        for (column, key) in chosen.into_iter().enumerate() {
            indices[row * k as usize + column] = key as i32;
        }
    }
    let half = [64, if half_only { 0 } else { 64 }].repeat(k as usize);
    let pattern = Arc::new(
        CheckedHostVsaPattern::new(
            shape,
            lists(&host, &indices),
            lists(&host, &counts),
            lists(&host, &half),
        )
        .unwrap(),
    );
    let pattern_bytes = host.usage().unwrap().committed_bytes();
    let config = Config {
        batches: b,
        heads: h,
        query_blocks: q,
        key_blocks: k,
        pattern_sha256: pattern.digest().into(),
        producer: Producer::CheckedHost,
    };
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(
            sparse_fp4_attention_checked_host_contract().unwrap(),
        )],
        vec![cuda_vnext_sparse_fp4_attention_checked_host_provider(
            &runtime,
            &SageVsaFp4Config {
                opaque_memory_bytes: 64 << 20,
                opaque_memory_profile: "vsa-routing-module-unqualified-capacity".into(),
            },
            Arc::clone(&pattern),
        )
        .unwrap()],
    )
    .unwrap();
    execute_case(
        runtime,
        &host,
        &config,
        &registry,
        pattern_bytes,
        half_only,
        |_| (None, expected(&config, half_only)),
    );
    drop(registry);
    drop(pattern);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        0,
        "last pattern owner retires"
    );
}

fn execute_case(
    runtime: &Arc<CudaDeviceRuntime>,
    host: &HostMemoryAccount,
    config: &Config,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    pattern_bytes: usize,
    half_only: bool,
    trial_data: impl Fn(usize) -> (Option<Vec<u8>>, Vec<u16>),
) {
    let [b, h, q, k] = [
        config.batches,
        config.heads,
        config.query_blocks,
        config.key_blocks,
    ];
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.vsa.test"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let mut forged = config.clone();
    forged.pattern_sha256 = "00".repeat(32);
    assert!(
        forged
            .try_compile(runtime.as_ref(), &registry, &catalog, &materializers)
            .is_err(),
        "digest alone cannot supply a payload"
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), pattern_bytes);
    let component = config
        .try_compile(runtime.as_ref(), &registry, &catalog, &materializers)
        .unwrap();
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.vsa.init"),
                request_id: id("request.vsa.init"),
                transaction_id: id("transaction.vsa.init"),
                static_policy: StaticInitializationPolicy::new(1024, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let operands = operands(&config, half_only);
    for trial in 0..2 {
        let (scores, expected) = trial_data(trial);
        let mut inputs: Vec<_> = INPUTS
            .into_iter()
            .zip(&operands)
            .map(|(name, bytes)| config.managed(&host, name, bytes))
            .collect();
        if let Some(scores) = scores {
            inputs.push(config.managed(host, SCORES, &scores));
        }
        let output = session
            .execute(
                id(format!("run.vsa.{trial}")),
                id(format!("request.vsa.{trial}")),
                inputs,
            )
            .unwrap();
        assert_eq!(
            output.len(),
            1,
            "the upstream noncausal path does not initialize LSE"
        );
        let actual = output[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        if let Some((index, (&actual, &expected))) = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .find(|(_, (a, e))| a != e)
        {
            panic!("shape B{b} H{h} Q{q} K{k}, half={half_only}, element {index}: actual={actual:#06x} expected={expected:#06x}");
        }
        assert_eq!(actual.len(), expected.len());
        assert!(host.owns(output[0].bytes()));
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|e| panic!("VSA close: {:?}", e.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        pattern_bytes,
        "only explicit pattern remains after retirement"
    );
    drop(component);
}
