use super::*;

#[test]
fn checked_rectangles_distinguish_read_bytes_from_physical_extent() {
    let region = StridedCopyRegion::new(7, 11, 6, 3, 32, 6).unwrap();
    assert_eq!(region.length_bytes().unwrap(), 18);
    assert_eq!(region.source_end_bytes().unwrap(), 77);
    assert_eq!(region.destination_end_bytes().unwrap(), 29);
    assert_eq!(region.reversed().reversed(), region);
    assert!(StridedCopyRegion::new(0, 0, 0, 1, 1, 1).is_err());
    assert!(StridedCopyRegion::new(0, 0, 8, 2, 7, 8).is_err());
    assert!(StridedCopyRegion::new(u64::MAX, 0, 1, 1, 1, 1).is_err());
    assert!(StridedCopyRegion::new(0, 0, 1, u64::MAX, 2, 1).is_err());
}

#[test]
fn translated_tail_groups_rows_instead_of_expanding_one_command_per_row() {
    let region = StridedCopyRegion::new(0, 0, 6, 1024, 32, 6).unwrap();
    let fragments = split_strided_copy(region, &[32742], &[6144], 8192, || Ok::<(), ()>(()))
        .unwrap_or_else(|_| panic!("valid rectangle rejected"));
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].region, region);
    let split = split_strided_copy(region, &[32742], &[3072, 3072], 8192, || Ok::<(), ()>(()))
        .unwrap_or_else(|_| panic!("valid rectangle split rejected"));
    assert_eq!(split.len(), 2);
    assert_eq!(split[0].region.height(), 512);
    assert_eq!(split[1].region.height(), 512);
    assert_eq!(split[1].source_segment, 0);
    assert_eq!(split[1].destination_segment, 1);
}

#[test]
fn physical_splits_preserve_exact_bytes_including_rows_split_between_segments() {
    let region = StridedCopyRegion::new(0, 0, 5, 7, 11, 5).unwrap();
    let source_lengths = [13, 17, 41];
    let destination_lengths = [8, 11, 16];
    let fragments = split_strided_copy(region, &source_lengths, &destination_lengths, 8192, || {
        Ok::<(), ()>(())
    })
    .unwrap_or_else(|_| panic!("valid split rejected"));
    let source = (0..71).map(|x| x as u8).collect::<Vec<_>>();
    let mut compact = vec![255; 35];
    for f in &fragments {
        let source_base = source_lengths[..f.source_segment].iter().sum::<u64>();
        let destination_base = destination_lengths[..f.destination_segment]
            .iter()
            .sum::<u64>();
        for row in 0..f.region.height() {
            let a = (source_base
                + f.region.source_offset_bytes()
                + row * f.region.source_pitch_bytes()) as usize;
            let b = (destination_base
                + f.region.destination_offset_bytes()
                + row * f.region.destination_pitch_bytes()) as usize;
            let n = f.region.width_bytes() as usize;
            compact[b..b + n].copy_from_slice(&source[a..a + n]);
        }
    }
    let expected = (0..7)
        .flat_map(|row| source[row * 11..row * 11 + 5].iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(compact, expected);
    assert!(matches!(
        split_strided_copy(region, &source_lengths, &destination_lengths, 0, || Ok::<
            (),
            (),
        >(
            ()
        )),
        Err(StridedCopySplitError::LimitExceeded)
    ));
    assert!(matches!(
        split_strided_copy(
            region,
            &source_lengths,
            &destination_lengths,
            8192,
            || Err::<(), ()>(())
        ),
        Err(StridedCopySplitError::Poll(()))
    ));
}
