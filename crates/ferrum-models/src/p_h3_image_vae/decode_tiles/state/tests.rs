//! Actual managed Host stages with explicit synthetic patch outputs. These do
//! not pretend to execute the raw-W8 decoder on a missing Reference provider.
use super::*;
use crate::p_h3_image_vae::{
    pipeline_adapter::{
        PH3ImageVaeAdapter, PH3ImageVaeDecoderPreset, PreparedPH3ImageVaeDecoderTiles,
    },
    test_support::TinyPH3ImageVaeFixture,
    tiling::PH3ImageVaeSpatialPolicy,
};

fn id<T: TryFrom<String>>(s: &str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.to_owned()).unwrap()
}
fn host(alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            64 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16 << 20,
                output_bytes: 32 << 20,
                transfer_staging_bytes: 8 << 20,
                weight_backing_bytes: 8 << 20,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
fn policy() -> PH3ImageVaeSpatialPolicy {
    PH3ImageVaeSpatialPolicy::Tiled {
        tile_pixels: 32,
        minimum_overlap_pixels: 16,
        max_tiles: 32,
    }
}
fn prepare(
    f: &TinyPH3ImageVaeFixture,
    batch: u64,
    tiles: usize,
) -> PreparedPH3ImageVaeDecoderTiles {
    PH3ImageVaeAdapter::new(f.source(), f.metadata(), f.limits())
        .unwrap()
        .prepare_decoder_with_tile_batch(
            batch,
            96,
            64,
            policy(),
            PH3ImageVaeDecoderPreset::F32A8FullAttention,
            PH3ImageVaeTileBatch::new(tiles).unwrap(),
        )
        .unwrap()
}
fn sample(p: &PreparedPH3ImageVaeDecoderTiles, host: &HostMemoryAccount) -> ManagedHostTensor {
    let tensor = p.normalized_latent_spec().unwrap();
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            tensor.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    for (i, b) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
        b.copy_from_slice(&((i % 31) as f32 / 32.).to_le_bytes());
    }
    ManagedHostTensor::new(id("value.grouped.sample"), tensor, bytes.freeze()).unwrap()
}
fn state<'a>(
    p: &PreparedPH3ImageVaeDecoderTiles,
    source: &'a ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource,
    normalized: &'a ManagedHostTensor,
    input: HostMemoryAccount,
    output: HostMemoryAccount,
) -> PH3ImageVaeDecodeState<'a> {
    let configs = p
        .shapes()
        .iter()
        .map(|p| {
            serde_json::from_value::<PH3ImageVaeDecoderExecutionConfig>(
                p.family().canonical_config().clone(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let shapes = p
        .shapes()
        .iter()
        .zip(&configs)
        .map(|(p, c)| {
            DecodeShape::new(
                c,
                p.tile_count(),
                p.family().family_id().clone(),
                p.family().program().fingerprint().unwrap(),
            )
            .unwrap()
        })
        .collect();
    PH3ImageVaeDecodeState::new(
        PH3ImageVaeTilePlan::new(96, 64, 16, policy()).unwrap(),
        p.shapes()[0].request_batch(),
        p.tile_batch(),
        shapes,
        PH3ImageVaeLatentStatistics::new(configs[0].architecture(), source).unwrap(),
        normalized,
        id("request.grouped.host"),
        input,
        output,
    )
    .unwrap()
}
fn values(group: PH3ImageVaeTileGroup, batch: usize, tile_samples: usize) -> Vec<f32> {
    (0..group.tile_count() * batch * tile_samples)
        .map(|i| {
            let tile = group.first_tile() + i / (batch * tile_samples);
            let request = (i / tile_samples) % batch;
            (tile as f32 * 0.03125)
                + request as f32 * 0.125
                + (i % tile_samples % 13) as f32 * 0.0078125
        })
        .collect()
}
fn install_pixels(s: &mut PH3ImageVaeDecodeState<'_>, group: PH3ImageVaeTileGroup) -> Result<()> {
    let tensor = s.shape(group.tile_count())?.layout.pixel_spec()?;
    let mut bytes = s.output_host.allocate_zeroed(
        HostMemoryClass::Output,
        tensor.minimum_storage_bytes()? as usize,
    )?;
    for (dst, x) in
        bytes
            .as_mut()
            .chunks_exact_mut(4)
            .zip(values(group, s.request_batch as usize, 3 * 32 * 32))
    {
        dst.copy_from_slice(&x.to_le_bytes());
    }
    s.pending = Some(Pending::Pixels(tensor, bytes.freeze()));
    s.pending_group = Some(group);
    s.next_commit = 0;
    Ok(())
}

#[test]
fn grouped_partial_commit_retains_original_owner_and_cursor_then_matches_single_tiles() {
    let f = TinyPH3ImageVaeFixture::new().unwrap();
    let source = f.source();
    let p = prepare(&f, 2, 4);
    let parent = host(64);
    let sample = sample(&p, &parent);
    let r = p.host_requirements(&parent).unwrap();
    let scope = parent.reserve_scope(&r.scope_claims()).unwrap();
    let h = scope.account();
    let mut s = state(&p, &source, &sample, h.clone(), h.clone());
    let group = s.next_group().unwrap().unwrap();
    install_pixels(&mut s, group).unwrap();
    let physical = |n| parent.backing_bytes(n).unwrap();
    let first_peak = r.canvas_output_bytes
        + physical(4 * 2 * 3 * 32 * 32 * 4)
        + physical(2 * 3 * 32 * 32 * 4)
        + physical(2 * 3 * 32 * 16 * 4)
        + physical(2 * 3 * 16 * 96 * 4);
    assert!(r.output_peak_bytes > first_peak);
    let available_blocker = r.output_peak_bytes - first_peak;
    let blocker_logical = (1..=available_blocker)
        .rev()
        .find(|&n| physical(n) <= available_blocker)
        .unwrap();
    assert!(available_blocker - physical(blocker_logical) < physical(2 * 3 * 32 * 16 * 4));
    let blocker = h
        .allocate_zeroed(HostMemoryClass::Output, blocker_logical)
        .unwrap();
    let pointer = match &s.pending {
        Some(Pending::Pixels(_, p)) => p.bytes().as_ptr(),
        _ => unreachable!(),
    };
    assert!(s.commit_pending().is_err());
    assert_eq!(s.completed_tiles(), 1);
    assert_eq!(s.next_commit, 1);
    assert_eq!(s.next_group().unwrap(), Some(group));
    assert!(s.commit_pending().is_err());
    assert_eq!(s.completed_tiles(), 1);
    assert_eq!(
        match &s.pending {
            Some(Pending::Pixels(_, p)) => p.bytes().as_ptr(),
            _ => unreachable!(),
        },
        pointer
    );
    drop(blocker);
    assert!(!s.commit_pending().unwrap());
    assert_eq!(s.completed_tiles(), 4);
    while let Some(group) = s.next_group().unwrap() {
        install_pixels(&mut s, group).unwrap();
        s.commit_pending().unwrap();
    }
    let (_, actual) = s.finish().unwrap();
    scope.close().unwrap();
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        sample.bytes().backing_bytes() + actual.backing_bytes()
    );
    let single = prepare(&f, 2, 1);
    let scope = parent
        .reserve_scope(&single.host_requirements(&parent).unwrap().scope_claims())
        .unwrap();
    let h = scope.account();
    let mut s = state(&single, &source, &sample, h.clone(), h);
    while let Some(group) = s.next_group().unwrap() {
        install_pixels(&mut s, group).unwrap();
        s.commit_pending().unwrap();
    }
    let (_, expected) = s.finish().unwrap();
    assert_eq!(
        actual.bytes(),
        expected.bytes(),
        "ordered seam arithmetic must be unchanged"
    );
    scope.close().unwrap();
    drop((sample, actual, expected));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

fn host_stages(f: &TinyPH3ImageVaeFixture, alignment: usize, short: bool) -> Result<()> {
    let source = f.source();
    let p = prepare(f, 2, 4);
    let parent = host(alignment);
    let sample = sample(&p, &parent);
    let r = p.host_requirements(&parent)?;
    let result = (|| -> Result<_> {
        let scope = parent.reserve_scope(&[
            (HostMemoryClass::InputPreprocess, r.input_peak_bytes),
            (
                HostMemoryClass::Output,
                r.output_peak_bytes - usize::from(short),
            ),
        ])?;
        let h = scope.account();
        let mut s = state(&p, &source, &sample, h.clone(), h.clone());
        let cache = (
            s.cosine.bytes().bytes().as_ptr(),
            s.sine.bytes().bytes().as_ptr(),
        );
        while let Some(group) = s.next_group()? {
            let inputs = s.inputs(group)?;
            assert_eq!(
                inputs[0].tensor(),
                &s.shape(group.tile_count())?.layout.row_spec()?
            );
            assert_eq!(inputs[1].bytes().bytes().as_ptr(), cache.0);
            assert_eq!(inputs[2].bytes().bytes().as_ptr(), cache.1);
            assert_eq!(
                inputs[1].bytes().backing_bytes() + inputs[2].bytes().backing_bytes(),
                r.cached_rotary_input_bytes
            );
            let layout = &s.shape(group.tile_count())?.layout;
            let patch_spec = layout.patch_output_spec()?;
            let patch = h
                .allocate_zeroed(
                    HostMemoryClass::Output,
                    patch_spec.minimum_storage_bytes()? as usize,
                )?
                .freeze();
            drop(inputs);
            let (tensor, pixels) = layout.unpatchify(&h, &patch_spec, &patch)?;
            drop(patch);
            s.pending = Some(Pending::Pixels(tensor, pixels));
            s.pending_group = Some(group);
            s.next_commit = 0;
            s.commit_pending()?;
        }
        let (_, output) = s.finish()?;
        scope.close()?;
        assert_eq!(
            parent.usage()?.committed_bytes(),
            sample.bytes().backing_bytes() + output.backing_bytes()
        );
        drop(output);
        Ok(())
    })();
    drop(sample);
    assert_eq!(parent.usage()?.committed_bytes(), 0);
    result
}
#[test]
fn grouped_exact_host_grants_cover_tail_and_each_cursor_one_byte_less_rejects() {
    let f = TinyPH3ImageVaeFixture::new().unwrap();
    for alignment in [1, 64, 4096] {
        host_stages(&f, alignment, false).unwrap();
        let e = host_stages(&f, alignment, true).unwrap_err();
        assert!(
            e.chain().any(|e| matches!(
                e.downcast_ref::<HostMemoryError>(),
                Some(HostMemoryError::ScopeCapacity {
                    class: HostMemoryClass::Output,
                    ..
                })
            )),
            "{e:#}"
        );
    }
}
#[test]
fn grouped_failed_execution_seals_shape_selection_and_cannot_publish_canvas() {
    let f = TinyPH3ImageVaeFixture::new().unwrap();
    let source = f.source();
    let p = prepare(&f, 1, 4);
    let parent = host(64);
    let sample = sample(&p, &parent);
    let mut s = state(&p, &source, &sample, parent.clone(), parent.clone());
    s.execution_failed = true;
    assert!(s.next_group().is_err());
    assert!(s.finish().is_err());
    drop(sample);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn abandoning_pending_group_keeps_original_scope_charge_until_last_pixel_alias_drops() {
    let f = TinyPH3ImageVaeFixture::new().unwrap();
    let source = f.source();
    let p = prepare(&f, 2, 4);
    let parent = host(4096);
    let sample = sample(&p, &parent);
    let scope = parent
        .reserve_scope(&p.host_requirements(&parent).unwrap().scope_claims())
        .unwrap();
    let h = scope.account();
    let mut state = state(&p, &source, &sample, h.clone(), h);
    let group = state.next_group().unwrap().unwrap();
    install_pixels(&mut state, group).unwrap();
    let alias = match &state.pending {
        Some(Pending::Pixels(_, owner)) => owner.slice(0..4).unwrap(),
        _ => unreachable!(),
    };
    assert!(alias.backing_bytes() > alias.len());
    scope.close().unwrap();
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        parent.usage().unwrap().allocated_bytes()
    );
    drop(state);
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        sample.bytes().backing_bytes() + alias.backing_bytes()
    );
    drop(alias);
    assert_eq!(
        parent.usage().unwrap().committed_bytes(),
        sample.bytes().backing_bytes()
    );
    drop(sample);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}
