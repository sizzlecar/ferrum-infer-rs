//! Same-partition continuation of the real prepare/attention kernels. Inputs
//! are projected Q/K/V, so this does not claim full-provider projection or
//! residual conformance, or authorize the public checkpoint API.

use super::*;
use std::{mem::size_of, ops::Range};

#[path = "checkpoint_copy_test_support.rs"]
mod native;

const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIM * 2;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const KV_BYTES_PER_TOKEN: usize = 2 * KV_WIDTH * size_of::<f16>();
const PREFIX: usize = 17;
const TOTAL: usize = 35;

#[test]
fn private_native_checkpoint_preserves_causal_state_and_continuation_bits() {
    metal::objc::rc::autoreleasepool(|| {
        let device = Device::system_default().expect("checkpoint continuation requires Metal");
        let queue = device.new_command_queue();
        let pipelines = MetalCausalAttentionPipelines::new(&device).unwrap();
        let inputs = Inputs {
            query: half_values(TOTAL * QUERY_WIDTH, 0.013, 0.31),
            key: half_values(TOTAL * KV_WIDTH, 0.017, 0.27),
            value: half_values(TOTAL * KV_WIDTH, 0.019, 0.22),
            query_norm: half_values(HEAD_DIM, 0.019, 0.94),
            key_norm: half_values(HEAD_DIM, 0.023, 0.89),
        };
        // Each pair has one participant, the same token spans, page ABI,
        // packing (none), and production attention-dispatch selection.
        let prefix = [0..15, 15..PREFIX];
        let suffix = [PREFIX..25, 25..34, 34..TOTAL];
        eprintln!("causal native continuation: prefix={prefix:?}, suffix={suffix:?}, participants=1, projected-input ABI, Hq=16 Hkv=4 D=256 RoPE=64/interleaved, page_bytes={VNEXT_KV_PAGE_BYTES}");
        let source = private_pages(&device, &queue);
        let cold = private_pages(&device, &queue);
        let restored = private_pages(&device, &queue);
        let prefix_output = inputs.run(&device, &queue, &pipelines, &source, &prefix);
        let cold_prefix_output = inputs.run(&device, &queue, &pipelines, &cold, &prefix);
        native::assert_output_bits("cold prefix", &cold_prefix_output, &prefix_output);
        let prefix_bytes = (PREFIX * KV_BYTES_PER_TOKEN) as u64;
        assert!(prefix_bytes > VNEXT_KV_PAGE_BYTES);
        assert!(!prefix_bytes.is_multiple_of(VNEXT_KV_PAGE_BYTES));
        let saved = read_pages_bytes(&device, &queue, &source);
        native::assert_bits(
            "cold prefix state",
            &read_pages_bytes(&device, &queue, &cold),
            &saved,
        );
        let checkpoint = native::private_filled(&device, &queue, prefix_bytes, 0xAB);
        copy_prefix(&queue, &source, &checkpoint, PrefixCopy::Capture);
        native::assert_bits(
            "captured valid KV",
            &native::read_bytes(&device, &queue, &checkpoint),
            &saved[..prefix_bytes as usize],
        );

        let source_output = inputs.run(&device, &queue, &pipelines, &source, &suffix);
        // Restore only after the source has continued into the captured page's
        // previous slack and into a third page. The compact snapshot is owned.
        native::assert_bits(
            "checkpoint after source continuation",
            &native::read_bytes(&device, &queue, &checkpoint),
            &saved[..prefix_bytes as usize],
        );
        copy_prefix(&queue, &restored, &checkpoint, PrefixCopy::Restore);
        native::assert_bits(
            "restored prefix and untouched capacity",
            &read_pages_bytes(&device, &queue, &restored),
            &saved,
        );
        let cold_output = inputs.run(&device, &queue, &pipelines, &cold, &suffix);
        let restored_output = inputs.run(&device, &queue, &pipelines, &restored, &suffix);
        native::assert_output_bits("cold suffix", &cold_output, &source_output);
        native::assert_output_bits("restored suffix", &restored_output, &source_output);
        let final_source = read_pages_bytes(&device, &queue, &source);
        native::assert_bits(
            "cold final KV",
            &read_pages_bytes(&device, &queue, &cold),
            &final_source,
        );
        native::assert_bits(
            "restored final KV",
            &read_pages_bytes(&device, &queue, &restored),
            &final_source,
        );
        assert!(
            final_source[TOTAL * KV_BYTES_PER_TOKEN..]
                .iter()
                .all(|&byte| byte == 0xFF),
            "KV writes changed capacity after the completed frontier"
        );
        native::assert_bits(
            "checkpoint after all branches",
            &native::read_bytes(&device, &queue, &checkpoint),
            &saved[..prefix_bytes as usize],
        );
    });
}

struct Inputs {
    query: Vec<f16>,
    key: Vec<f16>,
    value: Vec<f16>,
    query_norm: Vec<f16>,
    key_norm: Vec<f16>,
}

impl Inputs {
    fn run(
        &self,
        device: &Device,
        queue: &CommandQueueRef,
        pipelines: &MetalCausalAttentionPipelines,
        pages: &[Buffer],
        spans: &[Range<usize>],
    ) -> Vec<u16> {
        spans
            .iter()
            .flat_map(|span| {
                let params = CausalAttentionParams {
                    page_elements: VNEXT_KV_PAGE_BYTES as u32 / size_of::<f16>() as u32,
                    page_count: pages.len() as u32,
                    position_start: span.start as u32,
                    tokens: span.len() as u32,
                    query_heads: QUERY_HEADS as u32,
                    key_value_heads: KV_HEADS as u32,
                    head_dim: HEAD_DIM as u32,
                    rope_dim: 64,
                    query_projection_stride: QUERY_WIDTH as u32,
                    query_head_stride: (2 * HEAD_DIM) as u32,
                    kv_projection_stride: KV_WIDTH as u32,
                    output_gate: 1,
                    rope_interleaved: 1,
                    attention_simdgroups: pipelines
                        .attention_simdgroups_for_context(span.end as u64),
                    epsilon: 1.0e-6,
                    rope_theta: 10_000_000.0,
                };
                run_segment_bits(
                    device,
                    queue,
                    pipelines,
                    SegmentInputs {
                        query_raw: &self.query[span.start * QUERY_WIDTH..span.end * QUERY_WIDTH],
                        key_raw: &self.key[span.start * KV_WIDTH..span.end * KV_WIDTH],
                        value_raw: &self.value[span.start * KV_WIDTH..span.end * KV_WIDTH],
                    },
                    &self.query_norm,
                    &self.key_norm,
                    pages,
                    &params,
                    attention_dispatch_plan(&params),
                )
            })
            .collect()
    }
}

fn private_pages(device: &Device, queue: &CommandQueueRef) -> Vec<Buffer> {
    (0..(TOTAL * KV_BYTES_PER_TOKEN).div_ceil(VNEXT_KV_PAGE_BYTES as usize))
        .map(|_| native::private_filled(device, queue, VNEXT_KV_PAGE_BYTES, 0xFF))
        .collect()
}

fn read_pages_bytes(device: &Device, queue: &CommandQueueRef, pages: &[Buffer]) -> Vec<u8> {
    pages
        .iter()
        .flat_map(|page| native::read_bytes(device, queue, page))
        .collect()
}

enum PrefixCopy {
    Capture,
    Restore,
}

fn copy_prefix(
    queue: &CommandQueueRef,
    pages: &[Buffer],
    compact: &BufferRef,
    direction: PrefixCopy,
) {
    let mut copied = 0;
    for page in pages {
        if copied == compact.length() {
            break;
        }
        let bytes = page.length().min(compact.length() - copied);
        match direction {
            PrefixCopy::Capture => native::copy(queue, page, 0, compact, copied, bytes),
            PrefixCopy::Restore => native::copy(queue, compact, copied, page, 0, bytes),
        }
        copied += bytes;
    }
    assert_eq!(copied, compact.length());
}
