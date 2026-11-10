//! Independent explicit-position decode rows through original CUDA launchers.
use super::*;

fn execute(runtime: &CudaDeviceRuntime, packed: bool) -> (Vec<u16>, Vec<u16>, Vec<Vec<u16>>) {
    let stream = runtime.context().default_stream();
    let provider =
        CudaCausalPagedAttentionProvider::new(runtime, AttentionExecutionPolicy::Portable).unwrap();
    let functions = &provider.functions;
    let mut shape = shape(256, false);
    shape.query_heads = 16;
    shape.key_value_heads = 4;
    shape.query_features = shape.query_heads * shape.head_dim;
    shape.query_projection_features = shape.query_features * 2;
    shape.kv_features = shape.key_value_heads * shape.head_dim;
    shape.output_gate = true;
    shape.rope_dim = 64;
    shape.maximum_context_tokens = 2048;
    let cuda = shape.cuda_shape().unwrap();
    // Different live lengths cross the serial/parallel scan and page boundaries.
    let prefixes = [254_usize, 255, 1023];
    let pages = prefixes
        .iter()
        .enumerate()
        .map(|(row, &prefix)| {
            (0..shape.physical_state_bytes((prefix + 1) as u64).unwrap() / VNEXT_KV_PAGE_BYTES)
                .map(|page| {
                    Guarded::new(
                        &stream,
                        &(0..VNEXT_KV_PAGE_BYTES as usize / 2)
                            .map(|i| sample(i + page as usize * 31 + row * 19, 13))
                            .collect::<Vec<_>>(),
                        f16::from_f32(79.0),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let slot_bytes = shape.binding_slot_bytes().unwrap();
    let binding_layout = BindingLayout {
        required_bytes: slot_bytes * prefixes.len() as u64,
        slot_bytes,
    };
    let mut bytes = vec![0_u8; binding_layout.required_bytes as usize];
    let mut launches = Vec::new();
    for (row, &prefix) in prefixes.iter().enumerate() {
        let entries = shape.table_entries((prefix + 1) as u64).unwrap() as i32;
        let slot = &mut bytes[row * slot_bytes as usize..(row + 1) * slot_bytes as usize];
        for (word, value) in [entries, prefix as i32, 1, prefix as i32 + 1, row as i32, 0]
            .into_iter()
            .enumerate()
        {
            slot[word * 4..word * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let pointers = pages[row]
            .iter()
            .map(|page| page.pointer(&stream))
            .collect::<Vec<_>>();
        let table = binding_addresses(shape.kv_layout().unwrap(), entries, &pointers).unwrap();
        for (index, address) in table.iter().enumerate() {
            let start = BINDING_CONTROL_BYTES as usize + index * 8;
            slot[start..start + 8].copy_from_slice(&address.to_le_bytes());
        }
        let mut current = launch(shape, prefix, 1);
        current.path = CausalAttentionKernelPath::VllmAddressedFallback;
        current.table_entries_i32 = entries;
        current.binding_offset = row as u64 * slot_bytes;
        current.packed_token_start = row as u64;
        current.rotary = Some([row * 2, row * 2 + 1]);
        launches.push(current);
    }
    assert!(
        explicit_decode::PreparedExplicitDecode::for_launches(&launches, binding_layout, true)
            .unwrap()
            .is_some()
    );
    let binding = Guarded::new(&stream, &bytes, 0xBA_u8);
    let n = prefixes.len();
    let qwidth = shape.query_projection_features as usize;
    let kvwidth = shape.kv_features as usize;
    let width = shape.query_features as usize;
    let rope = shape.rope_dim as usize / 2;
    let q = Guarded::new(
        &stream,
        &(0..n * qwidth).map(|i| sample(i, 3)).collect::<Vec<_>>(),
        f16::from_f32(80.0),
    );
    let k = Guarded::new(
        &stream,
        &(0..n * kvwidth).map(|i| sample(i, 5)).collect::<Vec<_>>(),
        f16::from_f32(81.0),
    );
    let v = Guarded::new(
        &stream,
        &(0..n * kvwidth).map(|i| sample(i, 7)).collect::<Vec<_>>(),
        f16::from_f32(82.0),
    );
    let norm = Guarded::new(
        &stream,
        &vec![f16::ONE; shape.head_dim as usize],
        f16::from_f32(83.0),
    );
    let angles = (0..n * rope)
        .map(|i| (i % rope) as f64 * 0.03 + (i / rope) as f64 * 0.71)
        .collect::<Vec<_>>();
    let cos = Guarded::new(
        &stream,
        &angles
            .iter()
            .map(|x| f16::from_f64(x.cos()))
            .collect::<Vec<_>>(),
        f16::from_f32(84.0),
    );
    let sin = Guarded::new(
        &stream,
        &angles
            .iter()
            .map(|x| f16::from_f64(x.sin()))
            .collect::<Vec<_>>(),
        f16::from_f32(85.0),
    );
    let query = Guarded::new(&stream, &vec![f16::NAN; n * width], f16::from_f32(86.0));
    let output = Guarded::new(&stream, &vec![f16::NAN; n * width], f16::from_f32(87.0));
    let prepare = |row: usize| {
        let control = binding.pointer(&stream) + row as u64 * slot_bytes;
        launch_prepare_with_rope(
            &stream,
            &functions.prepare,
            q.pointer(&stream) + (row * qwidth * 2) as u64,
            k.pointer(&stream) + (row * kvwidth * 2) as u64,
            v.pointer(&stream) + (row * kvwidth * 2) as u64,
            norm.pointer(&stream),
            norm.pointer(&stream),
            query.pointer(&stream) + (row * width * 2) as u64,
            control,
            control + BINDING_CONTROL_BYTES,
            launches[row],
            cuda,
            1,
            None,
            Some([
                cos.pointer(&stream) + (row * rope * 2) as u64,
                sin.pointer(&stream) + (row * rope * 2) as u64,
            ]),
        )
        .unwrap();
    };
    let gate = |row: usize| {
        launch_attention_gate(
            &stream,
            &functions.attention_gate,
            output.pointer(&stream) + (row * width * 2) as u64,
            q.pointer(&stream) + (row * qwidth * 2) as u64,
            1,
            cuda,
        )
        .unwrap()
    };
    if packed {
        for row in 0..n {
            prepare(row);
        }
        launch_fallback_attention(
            &stream,
            functions,
            query.pointer(&stream),
            q.pointer(&stream),
            binding.pointer(&stream),
            binding.pointer(&stream) + BINDING_CONTROL_BYTES,
            output.pointer(&stream),
            launches[0],
            cuda,
            1,
            packed_fallback_geometry(&launches, binding_layout).unwrap(),
        )
        .unwrap();
        for row in 0..n {
            gate(row);
        }
    } else {
        for row in 0..n {
            prepare(row);
            let control = binding.pointer(&stream) + row as u64 * slot_bytes;
            launch_fallback_attention(
                &stream,
                functions,
                query.pointer(&stream) + (row * width * 2) as u64,
                q.pointer(&stream) + (row * qwidth * 2) as u64,
                control,
                control + BINDING_CONTROL_BYTES,
                output.pointer(&stream) + (row * width * 2) as u64,
                launches[row],
                cuda,
                1,
                None,
            )
            .unwrap();
            gate(row);
        }
    }
    let query_values = query.read(&stream);
    let output_values = output.read(&stream);
    assert!(query_values
        .iter()
        .chain(&output_values)
        .all(|x| x.is_finite()));
    assert_ne!(&output_values[..width], &output_values[width..2 * width]);
    for immutable in [&q, &k, &v, &norm, &cos, &sin] {
        immutable.assert_unchanged(&stream);
    }
    binding.assert_unchanged(&stream);
    (
        query_values.into_iter().map(f16::to_bits).collect(),
        output_values.into_iter().map(f16::to_bits).collect(),
        pages
            .iter()
            .map(|row| {
                row.iter()
                    .flat_map(|page| page.read(&stream))
                    .map(f16::to_bits)
                    .collect()
            })
            .collect(),
    )
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn explicit_decode_batched_ctas_preserve_independent_positions_kv_and_gates_on_cuda() {
    let runtime = runtime();
    assert_eq!(execute(&runtime, false), execute(&runtime, true));
}
