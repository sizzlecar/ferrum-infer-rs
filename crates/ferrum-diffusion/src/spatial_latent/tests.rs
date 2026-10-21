use super::*;

// Independent, hand-arranged fixture with N=2, C=2, H=2, W=3. Each value is
// 100 * batch + 10 * channel + row * width + column. These expected orders are
// written directly, not obtained by calling either transform under test.
const PLANAR: [i32; 24] = [
    0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 100, 101, 102, 103, 104, 105, 110, 111, 112, 113,
    114, 115,
];
const SEQUENCE: [i32; 24] = [
    0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 100, 110, 101, 111, 102, 112, 103, 113, 104, 114,
    105, 115,
];

#[test]
fn pack_matches_hand_arranged_multibatch_rectangular_order() {
    let shape = SingleFrameLatentShape::new(2, 2, 2, 3).unwrap();
    let mut output = [-1; 24];
    pack_spatial_into(shape, &PLANAR, &mut output).unwrap();
    assert_eq!(output, SEQUENCE);
}

#[test]
fn unpack_matches_independently_arranged_channels_first_order() {
    let shape = SingleFrameLatentShape::new(2, 2, 2, 3).unwrap();
    let mut output = [-1; 24];
    unpack_spatial_into(shape, &SEQUENCE, &mut output).unwrap();
    assert_eq!(output, PLANAR);
}

#[test]
fn descriptor_keeps_axes_and_single_frame_view_explicit() {
    let shape = SingleFrameLatentShape::new(2, 3, 2, 4).unwrap();
    assert_eq!(shape.nchw_dims(), [2, 3, 2, 4]);
    assert_eq!(shape.ncfhw_dims(), [2, 3, 1, 2, 4]);
    assert_eq!(shape.sequence_dims(), [2, 8, 3]);
    assert_eq!(shape.element_count(), 48);
    assert_eq!(shape.byte_len::<u16>(), Ok(96));
    assert_eq!(
        SingleFrameLatentShape::from_ncfhw_dims([2, 3, 1, 2, 4]),
        Ok(shape)
    );
    // The same element count cannot authorize folding multiple frames into H.
    for frames in [0, 2, usize::MAX] {
        assert_eq!(
            SingleFrameLatentShape::from_ncfhw_dims([2, 3, frames, 1, 4]),
            Err(LatentLayoutError::UnsupportedFrameCount { frames })
        );
    }
    assert_ne!(shape, SingleFrameLatentShape::new(2, 3, 4, 2).unwrap());
}

#[test]
fn singleton_axes_and_odd_spatial_dimensions_need_no_patch_padding() {
    for shape in [
        SingleFrameLatentShape::new(1, 3, 1, 1).unwrap(),
        SingleFrameLatentShape::new(1, 1, 1, 3).unwrap(),
        SingleFrameLatentShape::new(3, 1, 1, 1).unwrap(),
    ] {
        let input = [2_u16, 7, 19];
        let mut output = [0; 3];
        pack_spatial_into(shape, &input, &mut output).unwrap();
        assert_eq!(output, input);
        unpack_spatial_into(shape, &input, &mut output).unwrap();
        assert_eq!(output, input);
    }
}

#[test]
fn zero_axes_and_overflow_are_rejected_without_allocating() {
    for (axis, dims) in [
        (SpatialLatentAxis::Batch, [0, 1, 1, 1]),
        (SpatialLatentAxis::Channels, [1, 0, 1, 1]),
        (SpatialLatentAxis::Height, [1, 1, 0, 1]),
        (SpatialLatentAxis::Width, [1, 1, 1, 0]),
    ] {
        assert_eq!(
            SingleFrameLatentShape::new(dims[0], dims[1], dims[2], dims[3]),
            Err(LatentLayoutError::ZeroDimension { axis })
        );
    }
    for [batch, channels, height, width] in [
        [1, 1, usize::MAX, 2],
        [usize::MAX, 2, 1, 1],
        [usize::MAX / 2 + 1, 1, 1, 2],
    ] {
        assert_eq!(
            SingleFrameLatentShape::new(batch, channels, height, width),
            Err(LatentLayoutError::ElementCountOverflow)
        );
    }
    let too_many = isize::MAX as usize + 1;
    assert_eq!(
        SingleFrameLatentShape::new(1, 1, 1, too_many),
        Err(LatentLayoutError::AddressSpaceExceeded { bytes: too_many })
    );
}

#[test]
fn byte_sizes_enforce_actual_element_width_and_address_space() {
    // These descriptors do not allocate buffers or fabricate oversized slices.
    let max_bytes = isize::MAX as usize;
    let one_byte_boundary = SingleFrameLatentShape::new(1, 1, 1, max_bytes).unwrap();
    assert_eq!(one_byte_boundary.byte_len::<u8>(), Ok(max_bytes));
    assert_eq!(
        one_byte_boundary.byte_len::<u16>(),
        Err(LatentLayoutError::AddressSpaceExceeded {
            bytes: max_bytes * 2
        })
    );
    assert_eq!(
        one_byte_boundary.byte_len::<u32>(),
        Err(LatentLayoutError::ByteCountOverflow)
    );
    let f32_boundary = SingleFrameLatentShape::new(1, 1, 1, max_bytes / 4).unwrap();
    assert_eq!(f32_boundary.byte_len::<f32>(), Ok(max_bytes / 4 * 4));
    let beyond_f32 = SingleFrameLatentShape::new(1, 1, 1, max_bytes / 4 + 1).unwrap();
    assert!(matches!(
        beyond_f32.byte_len::<f32>(),
        Err(LatentLayoutError::AddressSpaceExceeded { .. })
    ));
    let mut output = [17.0_f32];
    assert!(pack_spatial_into(beyond_f32, &[1.0], &mut output).is_err());
    assert_eq!(output, [17.0]);
    assert!(unpack_spatial_into(beyond_f32, &[1.0], &mut output).is_err());
    assert_eq!(output, [17.0]);
}

#[test]
fn both_transforms_reject_short_and_long_buffers_before_any_write() {
    let shape = SingleFrameLatentShape::new(2, 2, 2, 3).unwrap();
    for transform in [pack_spatial_into::<i32>, unpack_spatial_into::<i32>] {
        let mut output = [-1; 24];
        for input in [&PLANAR[..23], &[7; 25][..]] {
            assert_eq!(
                transform(shape, input, &mut output),
                Err(LatentLayoutError::InputLengthMismatch {
                    expected: 24,
                    actual: input.len()
                })
            );
            assert_eq!(output, [-1; 24]);
        }
        let mut short = [-1; 23];
        assert_eq!(
            transform(shape, &PLANAR, &mut short),
            Err(LatentLayoutError::OutputLengthMismatch {
                expected: 24,
                actual: 23
            })
        );
        assert_eq!(short, [-1; 23]);
        let mut long = [-1; 25];
        assert_eq!(
            transform(shape, &PLANAR, &mut long),
            Err(LatentLayoutError::OutputLengthMismatch {
                expected: 24,
                actual: 25
            })
        );
        assert_eq!(long, [-1; 25]);
    }
}

#[test]
fn zero_sized_elements_are_not_tensor_storage() {
    let shape = SingleFrameLatentShape::new(1, 1, 1, 1).unwrap();
    assert_eq!(
        shape.byte_len::<()>(),
        Err(LatentLayoutError::ZeroSizedElement)
    );
    assert_eq!(
        pack_spatial_into(shape, &[()], &mut [()]),
        Err(LatentLayoutError::ZeroSizedElement)
    );
    assert_eq!(
        unpack_spatial_into(shape, &[()], &mut [()]),
        Err(LatentLayoutError::ZeroSizedElement)
    );
}

#[test]
fn floating_point_payload_bits_are_reordered_without_arithmetic() {
    let shape = SingleFrameLatentShape::new(1, 2, 1, 2).unwrap();
    let planar_bits = [0x7fc0_1234, 0x8000_0000, 0x7f80_0000, 0x3f80_0000];
    let sequence_bits = [0x7fc0_1234, 0x7f80_0000, 0x8000_0000, 0x3f80_0000];
    let mut output = [0.0_f32; 4];
    pack_spatial_into(shape, &planar_bits.map(f32::from_bits), &mut output).unwrap();
    assert_eq!(output.map(f32::to_bits), sequence_bits);
    unpack_spatial_into(shape, &sequence_bits.map(f32::from_bits), &mut output).unwrap();
    assert_eq!(output.map(f32::to_bits), planar_bits);
}
