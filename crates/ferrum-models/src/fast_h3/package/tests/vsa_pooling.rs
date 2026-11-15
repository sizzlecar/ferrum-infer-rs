use super::input_projection::{close, execute, host, prepare, session, Counting};
use super::*;
use crate::fast_h3::{vsa::*, *};
use ferrum_execution::component::ComponentWeightSource;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use half::bf16;
use std::sync::atomic::{AtomicUsize, Ordering};
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn pool_limits() -> FastH3VsaPoolLimits {
    FastH3VsaPoolLimits {
        maximum_batches: 2,
        maximum_tiles: 128,
        maximum_columns: 1024,
        maximum_count_bytes: 512,
        maximum_metadata_bytes: 4096,
    }
}
fn layout(text: usize) -> FastH3VsaLayout {
    let c = FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        include_bytes!("../../fixtures/transformer.json"),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    FastH3VsaLayout::ref2va(
        &c,
        text,
        &[FastH3Reference::Video {
            latent_frames: 5,
            latent_height: 10,
            latent_width: 18,
            audio_latents_per_channel: Some(2),
        }],
        &FastH3VideoGeometry::new(64, 64, 5).unwrap(),
        FastH3VsaLayoutLimits {
            maximum_packed_rows: 10000,
            maximum_tiles: 128,
            maximum_metadata_bytes: 1 << 20,
        },
    )
    .unwrap()
}
fn managed(
    host: &HostMemoryAccount,
    name: &str,
    shape: Vec<u64>,
    bytes: &[u8],
) -> ManagedHostTensor {
    let mut b = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
        .unwrap();
    b.as_mut().copy_from_slice(bytes);
    ManagedHostTensor::new(
        id(name),
        ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous)
            .unwrap(),
        b.freeze(),
    )
    .unwrap()
}
#[test]
fn checked_layout_counts_are_stored_once_and_pool_tiled_rows_with_head_order_and_masked_nan_padding(
) {
    let l = layout(3);
    let t = config();
    let c = FastH3VsaPoolConfig::from_layout(&l, 2, &t, pool_limits()).unwrap();
    let account = host(4 << 20);
    let source = FastH3VsaPoolSource::new(&c, &l, &account, pool_limits()).unwrap();
    let counted = Counting {
        source: &source,
        reads: AtomicUsize::new(0),
    };
    let composition =
        ReferenceVNextComposition::create_with_capacity(id("device.reference.vsa.pool"), 4 << 20)
            .unwrap();
    let prepared = prepare(
        FastH3VsaPoolFamily::new(pool_limits()).unwrap(),
        &c,
        FastH3VsaPoolFamily::PROFILE_ID,
    );
    let mut s = session(
        &composition,
        prepared,
        c.compile_options(pool_limits()).unwrap(),
        ComponentWeightSource::Stored(&counted),
        &account,
    );
    assert_eq!(counted.reads.load(Ordering::SeqCst), 1);
    let shape = c.input_shape().unwrap();
    let (rows, columns) = (shape[1] as usize, shape[2] as usize);
    let h = c.heads() as usize;
    let d = c.head_dim() as usize;
    let tiles = c.tiles() as usize;
    let transport_limits = FastH3VsaRowTransportLimits {
        maximum_rows: 10000,
        maximum_columns: 1024,
        maximum_batches: 2,
        maximum_map_bytes: 1 << 20,
        maximum_metadata_bytes: 4096,
    };
    let (transport, transport_source) = FastH3VsaRowTransportSource::prepare(
        &l,
        FastH3VsaRowTransport::Tile,
        2,
        columns as u64,
        &account,
        transport_limits,
    )
    .unwrap();
    let prepared = prepare(
        FastH3VsaRowTransportFamily::new(transport_limits).unwrap(),
        &transport,
        FastH3VsaRowTransportFamily::PROFILE_ID,
    );
    let mut tiling = session(
        &composition,
        prepared,
        transport.compile_options(transport_limits).unwrap(),
        ComponentWeightSource::Stored(&transport_source),
        &account,
    );
    let mut retained = None;
    for trial in 0..2 {
        let mut input = Vec::new();
        let mut expected = vec![0f32; 2 * h * tiles * d];
        for batch in 0..2 {
            for tile in 0..tiles {
                let valid = usize::from(l.valid_rows_per_tile()[tile]);
                for r in 0..FAST_H3_VSA_TILE_ROWS {
                    for col in 0..columns {
                        let value = if r < valid {
                            bf16::from_f32(
                                (batch as f32 * 2.)
                                    + (tile % 5) as f32 / 8.
                                    + r as f32 / 128.
                                    + (col % 11) as f32 / 32.,
                            )
                        } else if trial == 0 {
                            bf16::from_f32(0.)
                        } else {
                            bf16::from_bits(0x7fc1)
                        };
                        input.extend_from_slice(&value.to_bits().to_le_bytes());
                    }
                }
                for head in 0..h {
                    for ch in 0..d {
                        let col = head * d + ch;
                        let sum = (0..valid)
                            .map(|r| {
                                f64::from(
                                    bf16::from_f32(
                                        (batch as f32 * 2.)
                                            + (tile % 5) as f32 / 8.
                                            + r as f32 / 128.
                                            + (col % 11) as f32 / 32.,
                                    )
                                    .to_f32(),
                                )
                            })
                            .sum::<f64>();
                        expected[((batch * h + head) * tiles + tile) * d + ch] =
                            (sum / valid as f64) as f32;
                    }
                }
            }
        }
        assert_eq!(input.len(), 2 * rows * columns * 2);
        // Execute the actual checked map producer before pooling. The source
        // is in original packed order, including spatial border permutations.
        let packed_rows = l.packed().sequence_rows();
        let mut packed = vec![0u8; 2 * packed_rows * columns * 2];
        for batch in 0..2 {
            for (original, &tiled) in l.packed_to_tile_rows().iter().enumerate() {
                let src = (batch * rows + tiled as usize) * columns * 2;
                let dst = (batch * packed_rows + original) * columns * 2;
                packed[dst..dst + columns * 2].copy_from_slice(&input[src..src + columns * 2]);
            }
        }
        let tile_outputs = execute(
            &mut tiling,
            vec![managed(
                &account,
                FastH3VsaRowTransportFamily::INPUT_ID,
                vec![2, packed_rows as u64, columns as u64],
                &packed,
            )],
        );
        let mut tiled = tile_outputs[0].bytes().bytes().to_vec();
        if trial == 1 {
            for batch in 0..2 {
                for (tile, &valid) in l.valid_rows_per_tile().iter().enumerate() {
                    for row in usize::from(valid)..FAST_H3_VSA_TILE_ROWS {
                        let offset =
                            (batch * rows + tile * FAST_H3_VSA_TILE_ROWS + row) * columns * 2;
                        for chunk in tiled[offset..offset + columns * 2].chunks_exact_mut(2) {
                            chunk.copy_from_slice(&0x7fc1u16.to_le_bytes());
                        }
                    }
                }
            }
        }
        assert_eq!(
            tiled, input,
            "real gather output, with explicit poisoned padding in the second trial"
        );
        let out = execute(
            &mut s,
            vec![managed(
                &account,
                FastH3VsaPoolFamily::INPUT_ID,
                shape.clone(),
                &tiled,
            )],
        );
        assert_eq!(out[0].tensor().dimensions(), c.output_shape());
        for (actual, expected) in out[0].bytes().bytes().chunks_exact(4).zip(&expected) {
            assert_eq!(
                u32::from_le_bytes(actual.try_into().unwrap()),
                expected.to_bits(),
                "FP64 exact dyadic sum/division oracle"
            );
        }
        retained = Some(out);
    }
    assert_eq!(counted.reads.load(Ordering::SeqCst), 1);
    close(s);
    close(tiling);
    drop(transport_source);
    drop(counted);
    drop(source);
    let out = retained.unwrap();
    assert!(account.owns(out[0].bytes()));
    drop(out);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}
#[test]
fn pooling_rejects_foreign_layout_count_identity_and_insufficient_host_before_source_publication() {
    let l = layout(3);
    let c = FastH3VsaPoolConfig::from_layout(&l, 1, &config(), pool_limits()).unwrap();
    let account = host(4);
    assert!(FastH3VsaPoolSource::new(&c, &l, &account, pool_limits()).is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let account = host(4096);
    assert!(FastH3VsaPoolSource::new(&c, &layout(4), &account, pool_limits()).is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let mut wire = serde_json::to_value(&c).unwrap();
    wire["tiles"] = 0.into();
    assert!(FastH3VsaPoolFamily::new(pool_limits())
        .unwrap()
        .parse_config(&wire)
        .is_err());
    let mut bound = pool_limits();
    bound.maximum_count_bytes = 1;
    assert!(FastH3VsaPoolConfig::from_layout(&l, 1, &config(), bound).is_err());
}
