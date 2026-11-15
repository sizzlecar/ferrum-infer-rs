//! Included by the managed GPU fixture to reuse its actual Core executor path.
use super::*;

#[test]
#[ignore = "requires SM120 and rebuilt Sage region-producer artifact; dynamic routing/owner qualification"]
fn sparse_fp4_gpu_region_selection_uses_same_core_command_and_changes_each_invocation() {
    let runtime = runtime();
    // Power-of-two selected counts keep this Q=K=0 dyadic oracle independent
    // of general online-softmax/two-level-P approximation qualification.
    for (b, h, q, k, region_values, half_only) in [
        (2, 3, 4, 5, vec![1, 5, 3], false),
        (1, 2, 4, 6, vec![2, 4, 1, 4, 6, 1], true),
        (1, 1, 129, 130, vec![1, 130, 1], false),
        // 514 half-valid words cross a kernel-argument packet boundary.
        (1, 1, 256, 257, vec![1, 257, 1], false),
    ] {
        let host = host();
        let shape = ListShape::new(b, h, q, k).unwrap();
        let halves = [64, if half_only { 0 } else { 64 }].repeat(k as usize);
        let pattern = Arc::new(
            CheckedVsaRegions::new(shape, lists(&host, &region_values), lists(&host, &halves))
                .unwrap(),
        );
        let charged = host.usage().unwrap().committed_bytes();
        let config = Config {
            batches: b,
            heads: h,
            query_blocks: q,
            key_blocks: k,
            pattern_sha256: pattern.digest().into(),
            producer: Producer::RegionTopK,
        };
        let registry = OperationRuntimeRegistry::new(
            vec![Box::new(
                sparse_fp4_attention_region_topk_contract().unwrap(),
            )],
            vec![cuda_vnext_sparse_fp4_attention_region_topk_provider(
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
            &runtime,
            &host,
            &config,
            &registry,
            charged,
            half_only,
            |trial| {
                let keys = k as usize;
                let prefix = (k - q) as usize;
                let scores: Vec<f32> = (0..b as usize * h as usize * keys * keys)
                    .map(|i| {
                        let column = i % keys;
                        let row = i / keys;
                        if trial == 0 {
                            ((column * 47 + row * 13) % 257) as f32
                        } else {
                            match (column + row) % 7 {
                                0 => f32::NEG_INFINITY,
                                1 => f32::NAN,
                                2 => -0.,
                                3 => 0.,
                                _ => ((column + row) % 3) as f32,
                            }
                        }
                    })
                    .collect();
                let checked = pattern.checked().unwrap();
                let mut selections = Vec::new();
                let mut cutoff_tie_regions = 0;
                for row in 0..b as usize * h as usize * q as usize {
                    let fullrow = (row / q as usize) * keys + prefix + row % q as usize;
                    let values = &scores[fullrow * keys..(fullrow + 1) * keys];
                    let mut output = vec![0; keys];
                    let count = checked.select_row(values, &mut output).unwrap() as usize;
                    for region in region_values.chunks_exact(3) {
                        if region[2] == region[1] - region[0] {
                            continue;
                        }
                        if (region[0]..region[1]).any(|a| {
                            output[..count].contains(&a)
                                && (region[0]..region[1]).any(|z| {
                                    !output[..count].contains(&z)
                                        && (values[a as usize].to_bits()
                                            == values[z as usize].to_bits()
                                            || (values[a as usize].is_nan()
                                                && values[z as usize].is_nan()))
                                })
                        }) {
                            cutoff_tie_regions += 1;
                        }
                    }
                    selections.push(
                        output[..count]
                            .iter()
                            .map(|&n| n as usize)
                            .collect::<Vec<_>>(),
                    );
                }
                eprintln!("VSA dynamic B{b} H{h} Q{q} K{k} trial{trial}: cutoff-tie regions={cutoff_tie_regions}; nonfinite={}",scores.iter().filter(|x|!x.is_finite()).count());
                let expected = expected_selected(&config, half_only, |row| selections[row].clone());
                (
                    Some(scores.into_iter().flat_map(f32::to_le_bytes).collect()),
                    expected,
                )
            },
        );
        drop(registry);
        drop(pattern);
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            0,
            "dynamic producer retains no hidden Host owner"
        );
    }
}
