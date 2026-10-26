use super::*;

fn mmap_source(
    values: &[Vec<u16>],
) -> (
    tempfile::TempDir,
    SafetensorsArchive,
    Vec<WeightComponentSpec>,
) {
    let directory = tempfile::tempdir().unwrap();
    let specs = values
        .iter()
        .enumerate()
        .map(|(index, values)| {
            let mut spec = source_spec(&format!("lookup{index}"));
            // Independent row counts with the same input width form one group.
            spec.dimensions = vec![values.len() as u64, 1];
            spec
        })
        .collect::<Vec<_>>();
    let bytes = values
        .iter()
        .map(|values| {
            values
                .iter()
                .flat_map(|bits| bits.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let tensors = specs
        .iter()
        .zip(&bytes)
        .map(|(spec, bytes)| {
            (
                spec.external_names[0].as_str(),
                TensorView::new(Dtype::BF16, vec![bytes.len() / 2, 1], bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(tensors, &None, &directory.path().join("model.safetensors")).unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    (directory, archive, specs)
}

fn finite_values(elements: usize) -> Vec<u16> {
    // amax=448 makes scale=1; include both signs, ties, FP8 subnormals and -0.
    let pattern = [0x43e0, 0xc3e0, 0x4360, 0x3f88, 0xbf98, 0, 0x8000, 0x3b00];
    (0..elements)
        .map(|index| pattern[index % pattern.len()])
        .collect()
}

fn aligned_account(bytes: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(account(bytes).limits(), alignment).unwrap()
}

#[test]
fn lookup_matches_original_scalar_for_every_admitted_signed_bf16_at_varied_scales() {
    let first_valid = (1_u16..0x7f80)
        .find(|&bits| conversion::scale(bf16::from_bits(bits).to_f32()).is_ok())
        .unwrap();
    assert!(conversion::scale(bf16::from_bits(first_valid - 1).to_f32()).is_err());
    let host = account(lookup::MAGNITUDE_ENTRIES);
    for maximum_bits in [0, first_valid, 0x3f81, 0x40e0, 0x43e0, 0x7f7f] {
        let maximum = bf16::from_bits(maximum_bits).to_f32();
        let (_, inverse) = conversion::scale(maximum).unwrap();
        let table = lookup::LookupTable::new(&host, maximum, inverse).unwrap();
        assert_eq!(
            host.usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::WeightBacking),
            lookup::MAGNITUDE_ENTRIES,
        );
        for magnitude in 0..=maximum_bits {
            for sign in [0_u16, 0x8000] {
                let bits = magnitude | sign;
                let scalar = conversion::encode(bf16::from_bits(bits).to_f32() * inverse);
                assert_eq!(
                    table.encode(bits),
                    scalar,
                    "amax={maximum_bits:04x}, input={bits:04x}"
                );
            }
        }
        assert_eq!(table.encode(0), 0);
        assert_eq!(table.encode(0x8000), 0x80);
        drop(table);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn real_mmap_conversion_preserves_payload_across_lookup_cutoff() {
    let cutoff = lookup::MINIMUM_PACKED_ELEMENTS;
    assert_eq!(lookup::workspace_bytes(cutoff - 1), 0);
    assert_eq!(lookup::workspace_bytes(cutoff), lookup::MAGNITUDE_ENTRIES);
    assert_eq!(
        lookup::workspace_bytes(cutoff + 1),
        lookup::MAGNITUDE_ENTRIES
    );
    for elements in [cutoff - 1, cutoff, cutoff + 1] {
        let values = finite_values(elements as usize);
        let expected = values
            .iter()
            .map(|&bits| conversion::encode(bf16::from_bits(bits).to_f32()))
            .collect::<Vec<_>>();
        let (_directory, archive, specs) = mmap_source(&[values]);
        let refs = specs.iter().collect::<Vec<_>>();
        let derived = plan::derive_group(&refs).unwrap();
        // Output order does not change lookup selection or payload ranges.
        let outputs = [&derived[1], &derived[0]];
        let payload_bytes = elements as usize + 4;
        let host = account(payload_bytes + lookup::workspace_bytes(elements));
        let payloads = conversion::materialize(&host, &archive, &refs, &outputs).unwrap();
        assert_eq!(payloads[0].bytes(), &1_f32.to_le_bytes());
        assert_eq!(payloads[1].bytes(), expected.as_slice());
        assert_eq!(host.usage().unwrap().committed_bytes(), payload_bytes);
        drop(payloads);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn lookup_and_output_have_separate_aligned_claims_and_only_output_is_retained() {
    let elements = lookup::MINIMUM_PACKED_ELEMENTS as usize + 1;
    let (_directory, archive, specs) = mmap_source(&[finite_values(elements)]);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let sizing = aligned_account(1 << 20, 4096);
    let output_claim = sizing.backing_bytes(elements + 4).unwrap();
    let lookup_claim = sizing.backing_bytes(lookup::MAGNITUDE_ENTRIES).unwrap();
    let peak = output_claim + lookup_claim;
    assert!(
        peak > sizing
            .backing_bytes(elements + 4 + lookup::MAGNITUDE_ENTRIES)
            .unwrap()
    );

    let host = aligned_account(peak, 4096);
    let payloads = conversion::materialize(&host, &archive, &refs, &outputs).unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), output_claim);
    let retained = payloads[0].retained_host_memory().unwrap().clone();
    drop(payloads);
    drop(archive);
    assert_eq!(retained.bytes()[6], 0x80);
    assert_eq!(retained.bytes().len(), elements);
    assert_eq!(host.usage().unwrap().committed_bytes(), output_claim);
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);

    let archive = SafetensorsArchive::open(_directory.path()).unwrap();
    let below = aligned_account(peak - 1, 4096);
    assert!(conversion::materialize(&below, &archive, &refs, &outputs).is_err());
    assert_eq!(below.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn large_sources_do_not_force_lookup_for_scales_or_small_requested_subsets() {
    let elements = lookup::MINIMUM_PACKED_ELEMENTS as usize + 1;
    let (_directory, archive, specs) =
        mmap_source(&[finite_values(elements), vec![0x4360, 0x8000]]);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let scale_only = account(4);
    let scale = conversion::materialize(&scale_only, &archive, &refs, &[&derived[3]]).unwrap();
    assert_eq!(scale[0].bytes(), &1_f32.to_le_bytes());
    assert_eq!(scale_only.usage().unwrap().committed_bytes(), 4);
    drop(scale);
    assert_eq!(scale_only.usage().unwrap().committed_bytes(), 0);

    let subset_host = account(6);
    let subset =
        conversion::materialize(&subset_host, &archive, &refs, &[&derived[3], &derived[1]])
            .unwrap();
    // The unrequested large matrix's amax=448 still defines the shared scale.
    assert_eq!(subset[0].bytes(), &1_f32.to_le_bytes());
    assert_eq!(subset[1].bytes(), &[118, 0x80]);
    assert_eq!(subset_host.usage().unwrap().committed_bytes(), 6);
    drop(subset);
    assert_eq!(subset_host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn large_conversion_rejects_nonfinite_group_sources_and_invalid_scale_before_allocation() {
    let elements = lookup::MINIMUM_PACKED_ELEMENTS as usize + 1;
    for poison in [0x7f80, 0xff80, 0x7fc0, 0xff81] {
        let (_directory, archive, specs) = mmap_source(&[finite_values(elements), vec![poison]]);
        let refs = specs.iter().collect::<Vec<_>>();
        let derived = plan::derive_group(&refs).unwrap();
        let host = account(1);
        // The poison lies in a source whose packed output is not requested.
        let error = conversion::materialize(&host, &archive, &refs, &[&derived[0]])
            .err()
            .expect("every source contributes to group qualification");
        assert!(error.to_string().contains("nonfinite BF16 source"));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    let (_directory, archive, specs) = mmap_source(&[vec![0x0001; elements]]);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let host = account(1);
    let error = conversion::materialize(&host, &archive, &refs, &[&derived[0]])
        .err()
        .expect("invalid reciprocal must be rejected before allocating lookup");
    assert!(error.to_string().contains("scale or reciprocal"));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn cached_large_group_reuses_payload_without_reallocating_lookup() {
    let elements = lookup::MINIMUM_PACKED_ELEMENTS as usize + 1;
    let (_directory, archive, specs) = mmap_source(&[finite_values(elements)]);
    let refs = specs.iter().collect::<Vec<_>>();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let payload_bytes = elements + 4;
    let host = account(payload_bytes + lookup::MAGNITUDE_ENTRIES);
    let materializer =
        Bf16WholeTensorFp8Materializer::new(host.clone(), vec![vec![id("weight.lookup")]])
            .unwrap()
            .with_host_cache(NonZeroU64::new(payload_bytes as u64).unwrap())
            .unwrap();
    let first = materializer
        .materialize_components(&archive, &refs, &outputs)
        .unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), payload_bytes);
    let occupied = host
        .reserve(HostMemoryClass::WeightBacking, lookup::MAGNITUDE_ENTRIES)
        .unwrap();
    // No room remains for a second output or a temporary lookup table.
    let second = materializer
        .materialize_components(&archive, &refs, &outputs)
        .unwrap();
    for (left, right) in first.iter().zip(&second) {
        assert!(std::ptr::eq(left.bytes(), right.bytes()));
        assert_eq!(left.bytes(), right.bytes());
    }
    drop(occupied);
    let upload = second[0].retained_host_memory().unwrap().clone();
    drop((first, second, materializer));
    assert_eq!(host.usage().unwrap().committed_bytes(), payload_bytes);
    assert_eq!(upload.bytes()[6], 0x80);
    drop(upload);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
