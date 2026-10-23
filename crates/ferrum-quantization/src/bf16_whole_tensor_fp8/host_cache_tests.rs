use super::*;

fn cached(host: &HostMemoryAccount, cap: u64) -> Bf16WholeTensorFp8Materializer {
    Bf16WholeTensorFp8Materializer::new(host.clone(), vec![vec![id("weight.cache")]])
        .unwrap()
        .with_host_cache(NonZeroU64::new(cap).unwrap())
        .unwrap()
}

#[test]
fn same_immutable_source_hits_exact_payload_and_preserves_one_original_charge() {
    let (_directory, archive, specs) = source(&[[0x43e0, 0x3f80, 0xbf80, 0]], Dtype::BF16);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let host = account(8); // The retained group leaves no capacity for another conversion.
    let conversion = cached(&host, 8);
    let first = conversion
        .materialize_components(&archive, &refs, &outputs)
        .unwrap();
    assert_eq!(first[0].bytes(), &[126, 56, 184, 0]);
    let second = conversion
        .materialize_components(&archive, &refs, &outputs)
        .unwrap();
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(a.bytes(), b.bytes());
        assert!(std::ptr::eq(a.bytes(), b.bytes()));
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    let upload_alias = second[0].retained_host_memory().unwrap().clone();
    drop(first);
    drop(second);
    drop(conversion);
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    assert_eq!(upload_alias.bytes(), &[126, 56, 184, 0]);
    drop(upload_alias);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn same_ids_in_different_archives_miss_and_eviction_does_not_release_upload_alias() {
    let (_dir_a, a, specs) = source(&[[0x43e0, 0x3f80, 0xbf80, 0]], Dtype::BF16);
    let (_dir_b, b, _) = source(&[[0x43e0, 0x4360, 0, 0]], Dtype::BF16);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let host = account(8);
    let conversion = cached(&host, 8);
    let old = conversion
        .materialize_components(&a, &refs, &outputs)
        .unwrap();
    // Eviction drops the cache's owner, but cannot steal the upload's charge.
    assert!(conversion
        .materialize_components(&b, &refs, &outputs)
        .is_err());
    assert_eq!(old[0].bytes(), &[126, 56, 184, 0]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    drop(old);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let new = conversion
        .materialize_components(&b, &refs, &outputs)
        .unwrap();
    assert_eq!(new[0].bytes(), &[126, 118, 0, 0]);
    drop(new);
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    drop(conversion);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn cache_caps_are_explicit_and_oversized_groups_use_normal_uncached_conversion() {
    let (_directory, archive, specs) = source(&[[0x43e0, 0, 0, 0]], Dtype::BF16);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let host = account(8);
    let disabled =
        Bf16WholeTensorFp8Materializer::new(host.clone(), vec![vec![id("weight.cache")]]).unwrap();
    let enabled = cached(&host, 7);
    assert_eq!(disabled.descriptor(), enabled.descriptor());
    for conversion in [&disabled, &enabled] {
        drop(
            conversion
                .materialize_components(&archive, &refs, &outputs)
                .unwrap(),
        );
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    assert!(disabled
        .with_host_cache(NonZeroU64::new(9).unwrap())
        .is_err());
}

#[test]
fn cached_conversion_preserves_ordered_groups_and_exact_output_subsets() {
    let (_directory, archive, specs) = source(
        &[[0x43e0, 0x3f80, 0, 0], [0x4360, 0xbf80, 0, 0]],
        Dtype::BF16,
    );
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let host = account(64);
    let conversion = cached(&host, 64);
    let outputs = derived.iter().collect::<Vec<_>>();
    let complete = conversion
        .materialize_components(&archive, &refs, &outputs)
        .unwrap();
    let selected = [&derived[3], &derived[1]];
    let subset = conversion
        .materialize_components(&archive, &refs, &selected)
        .unwrap();
    assert_eq!(subset[0].bytes(), complete[3].bytes());
    assert_eq!(subset[1].bytes(), complete[1].bytes());
    let again = conversion
        .materialize_components(&archive, &refs, &selected)
        .unwrap();
    assert!(std::ptr::eq(subset[0].bytes(), again[0].bytes()));
    assert!(conversion
        .materialize_components(&archive, &[refs[1], refs[0]], &outputs)
        .is_err());
    let reversed = [refs[1], refs[0]];
    let reversed_derived = plan::derive_group(&reversed).unwrap();
    let reordered = conversion
        .materialize_components(
            &archive,
            &reversed,
            &reversed_derived.iter().collect::<Vec<_>>(),
        )
        .unwrap();
    assert_eq!(reordered[0].bytes(), complete[1].bytes());
    assert_eq!(reordered[1].bytes(), complete[0].bytes());
}

#[test]
fn failed_conversion_never_publishes_or_evicts_a_valid_group() {
    let (_dir_a, good, specs) = source(&[[0x43e0, 0, 0, 0]], Dtype::BF16);
    let (_dir_b, bad, _) = source(&[[0x7fc0, 0, 0, 0]], Dtype::BF16);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let host = account(8);
    let conversion = cached(&host, 8);
    drop(
        conversion
            .materialize_components(&good, &refs, &outputs)
            .unwrap(),
    );
    for _ in 0..2 {
        assert!(conversion
            .materialize_components(&bad, &refs, &outputs)
            .is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    }
    assert_eq!(
        conversion
            .materialize_components(&good, &refs, &outputs)
            .unwrap()[0]
            .bytes(),
        &[126, 0, 0, 0]
    );
    drop(conversion);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn cache_rejects_unretained_source_bytes_before_lookup() {
    struct Unretained;
    impl WeightComponentSource for Unretained {
        fn component<'a>(
            &'a self,
            spec: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'a>, VNextError> {
            WeightComponentPayload::new(
                spec,
                spec.external_names[0].clone(),
                "fixture",
                spec.dimensions.clone(),
                ElementType::Bf16,
                std::borrow::Cow::Borrowed(&[0xe0_u8, 0x43, 0, 0, 0, 0, 0, 0][..]),
            )
        }
    }
    let spec = source_spec("part0");
    let derived = plan::derive_group(&[&spec]).unwrap();
    let host = account(8);
    assert!(cached(&host, 8)
        .materialize_components(&Unretained, &[&spec], &derived.iter().collect::<Vec<_>>())
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn cache_eviction_keeps_recently_reused_group() {
    let (_dir_a, a, specs) = source(&[[0x43e0, 0x3f80, 0, 0]], Dtype::BF16);
    let (_dir_b, b, _) = source(&[[0x43e0, 0x4360, 0, 0]], Dtype::BF16);
    let (_dir_c, c, _) = source(&[[0x43e0, 0x42e0, 0, 0]], Dtype::BF16);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let host = account(32);
    let conversion = cached(&host, 16);
    let first_a = conversion
        .materialize_components(&a, &refs, &outputs)
        .unwrap();
    let first_b = conversion
        .materialize_components(&b, &refs, &outputs)
        .unwrap();
    drop(
        conversion
            .materialize_components(&a, &refs, &outputs)
            .unwrap(),
    );
    drop(
        conversion
            .materialize_components(&c, &refs, &outputs)
            .unwrap(),
    );
    let repeated_a = conversion
        .materialize_components(&a, &refs, &outputs)
        .unwrap();
    assert!(std::ptr::eq(first_a[0].bytes(), repeated_a[0].bytes()));
    // B is evicted, but this upload still owns its 8-byte backing in addition
    // to the two cached groups. Cache capacity never replaces Host admission.
    assert_eq!(host.usage().unwrap().committed_bytes(), 24);
    assert_eq!(first_b[0].bytes(), &[126, 118, 0, 0]);
    drop(first_b);
    assert_eq!(host.usage().unwrap().committed_bytes(), 16);
    drop(first_a);
    drop(repeated_a);
    drop(conversion);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
