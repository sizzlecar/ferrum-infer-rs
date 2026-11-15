use super::*;

#[test]
fn temporal_patches_preserve_channel_time_space_and_batch_order() {
    let shape = SpatiotemporalLatentShape::new([2, 2, 4, 2, 4], [2, 1, 2]).unwrap();
    assert_eq!(shape.sequence_dims(), [2, 8, 8]);
    // Independent enumeration of the first request's eight tokens. Each
    // channel occupies 32 dense elements; time advances by eight elements.
    let first = [
        0, 1, 8, 9, 32, 33, 40, 41, 2, 3, 10, 11, 34, 35, 42, 43, 4, 5, 12, 13, 36, 37, 44, 45, 6,
        7, 14, 15, 38, 39, 46, 47, 16, 17, 24, 25, 48, 49, 56, 57, 18, 19, 26, 27, 50, 51, 58, 59,
        20, 21, 28, 29, 52, 53, 60, 61, 22, 23, 30, 31, 54, 55, 62, 63,
    ];
    let expected: Vec<u16> = first.into_iter().chain(first.map(|v| v + 64)).collect();
    let input: Vec<u16> = (0..128).collect();
    let mut packed = vec![u16::MAX; 128];
    pack_spatiotemporal_into(shape, &input, &mut packed).unwrap();
    assert_eq!(packed, expected);
    let mut dense = vec![u16::MAX; 128];
    unpack_spatiotemporal_into(shape, &expected, &mut dense).unwrap();
    assert_eq!(dense, input);
    let mut visited = Vec::new();
    shape.for_each_packed_span(|packed, dense| {
        assert_eq!(packed.start, visited.len());
        assert_eq!(packed.len(), dense.len());
        assert!(dense.end <= shape.element_count());
        visited.extend(dense.map(|i| input[i]));
    });
    assert_eq!(visited, expected);
}

#[test]
fn contiguous_patch_rows_preserve_odd_widths_and_opaque_elements() {
    let shape = SpatiotemporalLatentShape::new([2, 2, 2, 3, 15], [1, 3, 5]).unwrap();
    let input: Vec<[u8; 3]> = (0..shape.element_count())
        .map(|i| [i as u8, (i >> 8) as u8, 0xa5])
        .collect();
    let mut packed = vec![[0; 3]; input.len()];
    pack_spatiotemporal_into(shape, &input, &mut packed).unwrap();
    // Decode each dense coordinate independently, then locate its token and
    // channel-major position. This also covers spans with no word alignment.
    for (dense, value) in input.iter().enumerate() {
        let x = dense % 15;
        let y = dense / 15 % 3;
        let t = dense / 45 % 2;
        let c = dense / 90 % 2;
        let n = dense / 180;
        let token = n * 6 + t * 3 + x / 5;
        let within = c * 15 + y * 5 + x % 5;
        assert_eq!(packed[token * 30 + within], *value);
    }
    let mut dense = vec![[0; 3]; input.len()];
    unpack_spatiotemporal_into(shape, &packed, &mut dense).unwrap();
    assert_eq!(dense, input);
}

#[test]
fn unit_temporal_and_spatial_patches_match_existing_single_frame_contract() {
    let old = crate::SingleFrameLatentShape::new(2, 3, 2, 5).unwrap();
    let shape = SpatiotemporalLatentShape::new(old.ncfhw_dims(), [1, 1, 1]).unwrap();
    let input: Vec<_> = (0..old.element_count()).collect();
    let mut old_packed = vec![0; input.len()];
    let mut new_packed = vec![0; input.len()];
    crate::pack_spatial_into(old, &input, &mut old_packed).unwrap();
    pack_spatiotemporal_into(shape, &input, &mut new_packed).unwrap();
    assert_eq!(new_packed, old_packed);
    assert_eq!(shape.sequence_dims(), old.sequence_dims());
    assert!(crate::SingleFrameLatentShape::from_ncfhw_dims([1, 3, 2, 2, 5]).is_err());
}

#[test]
fn invalid_storage_is_rejected_before_any_write() {
    let shape = SpatiotemporalLatentShape::new([1, 1, 2, 2, 2], [2, 2, 2]).unwrap();
    let mut output = [77u8; 8];
    assert!(pack_spatiotemporal_into(shape, &[1; 7], &mut output).is_err());
    assert_eq!(output, [77; 8]);
    assert!(unpack_spatiotemporal_into(shape, &[1; 9], &mut output).is_err());
    assert_eq!(output, [77; 8]);
    assert!(pack_spatiotemporal_into(shape, &[1; 8], &mut output[..7]).is_err());
    assert_eq!(output, [77; 8]);
    assert_eq!(
        shape.byte_len::<()>(),
        Err(SpatiotemporalLayoutError::ZeroSizedElement)
    );
}

#[test]
fn axes_patches_and_physical_storage_bounds_are_checked() {
    for axis in 0..5 {
        let mut dims = [1, 2, 4, 6, 8];
        dims[axis] = 0;
        assert!(matches!(
            SpatiotemporalLatentShape::new(dims, [1, 2, 2]),
            Err(SpatiotemporalLayoutError::ZeroDimension { .. })
        ));
    }
    for patch in [[0, 2, 2], [3, 2, 2], [1, 4, 2], [1, 2, 16]] {
        assert!(matches!(
            SpatiotemporalLatentShape::new([1, 2, 4, 6, 8], patch),
            Err(SpatiotemporalLayoutError::IncompletePatch { .. })
        ));
    }
    assert_eq!(
        SpatiotemporalLatentShape::new([2, usize::MAX, 1, 1, 1], [1; 3]),
        Err(SpatiotemporalLayoutError::ElementCountOverflow)
    );
    assert!(matches!(
        SpatiotemporalLatentShape::new([1, isize::MAX as usize + 1, 1, 1, 1], [1; 3]),
        Err(SpatiotemporalLayoutError::AddressSpaceExceeded { .. })
    ));
    let shape = SpatiotemporalLatentShape::new([1, 1, 1, 1, isize::MAX as usize], [1; 3]).unwrap();
    assert!(matches!(
        shape.byte_len::<u16>(),
        Err(SpatiotemporalLayoutError::AddressSpaceExceeded { .. })
    ));
    assert_eq!(
        shape.byte_len::<u32>(),
        Err(SpatiotemporalLayoutError::ByteCountOverflow)
    );
}
