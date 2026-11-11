use super::*;

fn id(s: &str) -> ProgramValueId {
    ProgramValueId::new(s).unwrap()
}
fn host(alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}
fn bytes(host: &HostMemoryAccount, data: &[u8]) -> ManagedHostBytes {
    let mut buffer = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len())
        .unwrap();
    buffer.as_mut().copy_from_slice(data);
    buffer.freeze()
}
fn halfs(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
        .collect()
}
fn schema() -> CausalInputSchema {
    CausalInputSchema::new(vec![
        CausalTokenInputSpec {
            value_id: id("input.features"),
            element_type: ElementType::F16,
            row_dimensions: vec![2],
            storage: CausalTokenInputStorage::ZeroOutsideSegments,
        },
        CausalTokenInputSpec {
            value_id: id("input.mask"),
            element_type: ElementType::U8,
            row_dimensions: vec![],
            storage: CausalTokenInputStorage::ZeroOutsideSegments,
        },
        CausalTokenInputSpec {
            value_id: id("input.rotary"),
            element_type: ElementType::F16,
            row_dimensions: vec![1],
            storage: CausalTokenInputStorage::Exact,
        },
    ])
    .unwrap()
}
fn request(host: &HostMemoryAccount) -> CausalRequestInputs {
    CausalRequestInputs::new(
        host,
        &schema(),
        6,
        vec![
            CausalTokenInput::new(
                id("input.features"),
                vec![
                    CausalInputSegment::new(1, 2, bytes(host, &halfs(&[0.25, -0.5, 1.25, 2.0])))
                        .unwrap(),
                ],
            ),
            CausalTokenInput::new(
                id("input.mask"),
                vec![CausalInputSegment::new(1, 2, bytes(host, &[1, 1])).unwrap()],
            ),
            CausalTokenInput::new(
                id("input.rotary"),
                vec![
                    CausalInputSegment::new(0, 6, bytes(host, &halfs(&[0., 1., 2., 3., 4., 5.])))
                        .unwrap(),
                ],
            ),
        ],
    )
    .unwrap()
}
fn prepared(
    request: &CausalRequestInputs,
    host: &HostMemoryAccount,
    range: Range<usize>,
) -> Vec<PreparedInput> {
    let demand = request
        .schema
        .zero_demands(range.len())
        .unwrap()
        .into_iter()
        .map(|n| (HostMemoryClass::InputPreprocess, n))
        .collect::<Vec<_>>();
    let mut claims = host.reserve_many(&demand).unwrap().into_iter();
    let mut allocate = || {
        claims
            .next()
            .unwrap()
            .allocate_zeroed()
            .map_err(|e| FerrumError::resource_exhausted(e.to_string()))
    };
    request.prepare(range, &mut allocate).unwrap()
}

#[test]
fn causal_inputs_real_sparse_owners_materialize_only_current_span_and_decode_zeros() {
    let host = host(1);
    let request = request(&host);
    let persistent = host.usage().unwrap().allocated_bytes();
    let wave = prepared(&request, &host, 0..4);
    let feature = wave
        .iter()
        .find(|x| x.value_id == id("input.features"))
        .unwrap();
    assert_eq!(
        feature.pieces[0].bytes.bytes(),
        halfs(&[0., 0., 0.25, -0.5, 1.25, 2., 0., 0.])
    );
    let mask = wave
        .iter()
        .find(|x| x.value_id == id("input.mask"))
        .unwrap();
    assert_eq!(mask.pieces[0].bytes.bytes(), [0, 1, 1, 0]);
    let rotary = wave
        .iter()
        .find(|x| x.value_id == id("input.rotary"))
        .unwrap();
    assert_eq!(rotary.pieces[0].bytes.bytes(), halfs(&[0., 1., 2., 3.]));
    assert!(rotary.pieces[0]
        .bytes
        .same_region(&request.values[2].segments[0].bytes.slice(0..8).unwrap()));
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        persistent + schema().additional_bytes(&host, 4).unwrap()
    );
    drop(wave);
    assert_eq!(host.usage().unwrap().allocated_bytes(), persistent);
    let decode = prepared(&request, &host, 5..6);
    assert_eq!(decode[0].pieces[0].bytes.bytes(), halfs(&[0., 0.]));
    assert_eq!(decode[1].pieces[0].bytes.bytes(), [0]);
    assert_eq!(decode[2].pieces[0].bytes.bytes(), halfs(&[5.]));
    drop(decode);
    drop(request);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn selected_prefix_conditioning_keeps_original_charge_until_core_releases_it() {
    let account = host(1);
    let request = request(&account);
    let before = account.usage().unwrap();
    let conditioning = request.sequence_conditioning(&account).unwrap();
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        before.allocated_bytes()
    );
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        before.committed_bytes()
    );
    assert!(request.sequence_conditioning(&host(1)).is_err());

    // Core may outlive the product-side request during cancellation/fence
    // recovery. It must retain the original charge without another payload.
    drop(request);
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        before.allocated_bytes()
    );
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        before.committed_bytes()
    );
    drop(conditioning);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn causal_inputs_exact_segments_keep_original_owner_and_source_offsets_after_request_drop() {
    let host = host(1);
    let schema = CausalInputSchema::new(vec![CausalTokenInputSpec {
        value_id: id("input.table"),
        element_type: ElementType::F16,
        row_dimensions: vec![1],
        storage: CausalTokenInputStorage::Exact,
    }])
    .unwrap();
    let request = CausalRequestInputs::new(
        &host,
        &schema,
        6,
        vec![CausalTokenInput::new(
            id("input.table"),
            vec![
                CausalInputSegment::new(0, 3, bytes(&host, &halfs(&[10., 11., 12.]))).unwrap(),
                CausalInputSegment::new(3, 3, bytes(&host, &halfs(&[13., 14., 15.]))).unwrap(),
            ],
        )],
    )
    .unwrap();
    let wave = request
        .prepare(2..5, &mut || {
            panic!("Exact inputs never allocate another payload")
        })
        .unwrap();
    let binding = ResolvedInputs {
        schema,
        ports: BTreeMap::from([(
            id("input.table"),
            vec![InputPort {
                node: NodeId::new("node.table").unwrap(),
                ordinal: 3,
            }],
        )]),
    };
    let uploads = binding.uploads(&wave, 2..5, 1).unwrap();
    assert_eq!(uploads.len(), 2);
    assert_eq!(uploads[0].participant_index(), 1);
    assert_eq!(uploads[0].logical_offset_bytes(), 4);
    assert_eq!(uploads[1].logical_offset_bytes(), 6);
    assert_eq!(uploads[0].bytes(), halfs(&[12.]));
    assert_eq!(uploads[1].bytes(), halfs(&[13., 14.]));
    drop(request);
    drop(wave);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 12);
    assert_eq!(uploads[1].bytes(), halfs(&[13., 14.]));
    drop(uploads);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn causal_inputs_reject_incomplete_exact_coverage_overlap_foreign_and_wrong_class() {
    let other = host(1);
    let host = host(1);
    let schema = CausalInputSchema::new(vec![CausalTokenInputSpec {
        value_id: id("input.exact"),
        element_type: ElementType::F16,
        row_dimensions: vec![1],
        storage: CausalTokenInputStorage::Exact,
    }])
    .unwrap();
    let value = |segments| vec![CausalTokenInput::new(id("input.exact"), segments)];
    assert!(CausalRequestInputs::new(
        &host,
        &schema,
        3,
        value(vec![CausalInputSegment::new(
            0,
            2,
            bytes(&host, &halfs(&[1., 2.]))
        )
        .unwrap()])
    )
    .is_err());
    assert!(CausalRequestInputs::new(
        &host,
        &schema,
        3,
        value(vec![
            CausalInputSegment::new(0, 2, bytes(&host, &halfs(&[1., 2.]))).unwrap(),
            CausalInputSegment::new(1, 2, bytes(&host, &halfs(&[2., 3.]))).unwrap(),
        ])
    )
    .is_err());
    assert!(CausalRequestInputs::new(
        &host,
        &schema,
        3,
        value(vec![CausalInputSegment::new(
            0,
            3,
            bytes(&other, &halfs(&[1., 2., 3.]))
        )
        .unwrap()])
    )
    .is_err());
    let output = host
        .allocate_zeroed(HostMemoryClass::Output, 6)
        .unwrap()
        .freeze();
    assert!(CausalRequestInputs::new(
        &host,
        &schema,
        3,
        value(vec![CausalInputSegment::new(0, 3, output).unwrap()])
    )
    .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(other.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn causal_inputs_alignment_query_matches_real_scope_and_one_byte_short_refuses_before_allocation() {
    let host = host(8);
    let request = request(&host);
    let persistent = host.usage().unwrap().allocated_bytes();
    let bytes = request.schema.additional_bytes(&host, 3).unwrap();
    let demand = request
        .schema
        .zero_demands(3)
        .unwrap()
        .into_iter()
        .map(|n| (HostMemoryClass::InputPreprocess, n))
        .collect::<Vec<_>>();
    let short = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes - 1)])
        .unwrap();
    assert!(short.account().reserve_many(&demand).is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), persistent);
    short.close().unwrap();
    let exact = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, bytes)])
        .unwrap();
    let wave = prepared(&request, &exact.account(), 2..5);
    assert_eq!(host.usage().unwrap().allocated_bytes(), persistent + bytes);
    drop(wave);
    exact.close().unwrap();
    drop(request);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn materialize_rows_exact_slice_retains_owner_and_cross_segment_copy_refunds() {
    let account = host(8);
    let schema = CausalInputSchema::new(vec![CausalTokenInputSpec {
        value_id: id("input.table"),
        element_type: ElementType::F16,
        row_dimensions: vec![2],
        storage: CausalTokenInputStorage::Exact,
    }])
    .unwrap();
    let request = CausalRequestInputs::new(
        &account,
        &schema,
        4,
        vec![CausalTokenInput::new(
            id("input.table"),
            vec![
                CausalInputSegment::new(0, 2, bytes(&account, &halfs(&[1., -2., 3., -4.])))
                    .unwrap(),
                CausalInputSegment::new(2, 2, bytes(&account, &halfs(&[5., -6., 7., -8.])))
                    .unwrap(),
            ],
        )],
    )
    .unwrap();
    let persistent = account.usage().unwrap().allocated_bytes();
    let first_backing = account
        .backing_bytes(request.values[0].segments[0].bytes.len())
        .unwrap();
    let (dtype, slice) = request
        .materialize_rows(&account, &id("input.table"), 1..2)
        .unwrap();
    assert_eq!(dtype, ElementType::F16);
    assert_eq!(slice.bytes(), halfs(&[3., -4.]));
    assert!(slice.same_region(&request.values[0].segments[0].bytes.slice(4..8).unwrap()));
    assert_eq!(account.usage().unwrap().allocated_bytes(), persistent);
    let (dtype, joined) = request
        .materialize_rows(&account, &id("input.table"), 1..3)
        .unwrap();
    assert_eq!(dtype, ElementType::F16);
    assert_eq!(joined.bytes(), halfs(&[3., -4., 5., -6.]));
    assert_eq!(joined.class(), HostMemoryClass::InputPreprocess);
    assert!(account.owns(&joined));
    let joined_backing = account.backing_bytes(joined.len()).unwrap();
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        persistent + joined_backing
    );
    drop(request);
    assert_eq!(slice.bytes(), halfs(&[3., -4.]));
    assert_eq!(joined.bytes(), halfs(&[3., -4., 5., -6.]));
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        first_backing + joined_backing
    );
    drop(slice);
    assert_eq!(account.usage().unwrap().allocated_bytes(), joined_backing);
    drop(joined);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn materialize_rows_sparse_holes_are_zero_and_preserve_multicolumn_row_offsets() {
    let account = host(8);
    let mut values = request(&account).values;
    values
        .iter_mut()
        .find(|value| value.value_id == id("input.features"))
        .unwrap()
        .segments
        .push(CausalInputSegment::new(4, 1, bytes(&account, &halfs(&[3., -3.]))).unwrap());
    let request = CausalRequestInputs::new(&account, &schema(), 6, values).unwrap();
    let persistent = account.usage().unwrap().allocated_bytes();
    let (dtype, features) = request
        .materialize_rows(&account, &id("input.features"), 0..6)
        .unwrap();
    assert_eq!(dtype, ElementType::F16);
    assert_eq!(
        features.bytes(),
        halfs(&[0., 0., 0.25, -0.5, 1.25, 2., 0., 0., 3., -3., 0., 0.])
    );
    let (dtype, mask) = request
        .materialize_rows(&account, &id("input.mask"), 2..5)
        .unwrap();
    assert_eq!(dtype, ElementType::U8);
    assert_eq!(mask.bytes(), &[1, 0, 0]);
    for buffer in [&features, &mask] {
        assert!(account.owns(buffer));
        assert_eq!(buffer.class(), HostMemoryClass::InputPreprocess);
    }
    let backing =
        account.backing_bytes(features.len()).unwrap() + account.backing_bytes(mask.len()).unwrap();
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        persistent + backing
    );
    drop((features, mask));
    assert_eq!(account.usage().unwrap().allocated_bytes(), persistent);
    drop(request);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn materialize_rows_rejects_foreign_account_unknown_value_and_invalid_ranges_without_charge() {
    let account = host(8);
    let other = host(8);
    let request = request(&account);
    let before = account.usage().unwrap();
    assert!(request
        .materialize_rows(&other, &id("input.rotary"), 1..2)
        .is_err());
    assert!(request
        .materialize_rows(&other, &id("input.features"), 0..6)
        .is_err());
    assert!(request
        .materialize_rows(&account, &id("input.unknown"), 0..1)
        .is_err());
    for range in [0..0, Range { start: 4, end: 2 }, 5..7, 6..7, 0..usize::MAX] {
        assert!(request
            .materialize_rows(&account, &id("input.features"), range)
            .is_err());
    }
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        before.allocated_bytes()
    );
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        before.committed_bytes()
    );
    assert_eq!(other.usage().unwrap().committed_bytes(), 0);
    drop(request);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn materialize_rows_respects_scope_capacity_and_refunds_failed_and_completed_copies() {
    let account = host(8);
    let request = request(&account);
    let persistent = account.usage().unwrap().allocated_bytes();
    let output_bytes = halfs(&[0., 0., 0.25, -0.5, 1.25, 2., 0., 0.]);
    let backing = account.backing_bytes(output_bytes.len()).unwrap();
    let short = account
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, backing - 1)])
        .unwrap();
    let before = account.usage().unwrap();
    assert!(request
        .materialize_rows(&short.account(), &id("input.features"), 0..4)
        .is_err());
    assert_eq!(account.usage().unwrap().allocated_bytes(), persistent);
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        before.committed_bytes()
    );
    short.close().unwrap();
    let exact = account
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, backing)])
        .unwrap();
    let (_, output) = request
        .materialize_rows(&exact.account(), &id("input.features"), 0..4)
        .unwrap();
    assert_eq!(output.bytes(), output_bytes);
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        persistent + backing
    );
    drop(output);
    exact.close().unwrap();
    assert_eq!(account.usage().unwrap().committed_bytes(), persistent);
    drop(request);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
