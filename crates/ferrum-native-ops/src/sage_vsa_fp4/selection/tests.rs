use super::*;

#[test]
fn region_selection_preserves_independent_budgets_and_dense_prefix() {
    let shape = SageVsaFp4ListShape::new(2, 3, 7, 9).unwrap();
    let half = [64; 18];
    let metadata = CheckedRegionSelection::new(shape, &[2, 5, 1, 5, 9, 2], &half).unwrap();
    let scores = [900., -900., 2., 1., 3., 7., 4., 9., 6.];
    let mut indices = [99; 9];
    assert_eq!(metadata.select_row(&scores, &mut indices).unwrap(), 5);
    assert_eq!(indices, [7, 5, 4, 1, 0, 0, 0, 0, 0]);
    let all_indices = indices.repeat(2 * 3 * 7);
    let counts = vec![5; 2 * 3 * 7];
    CheckedCpuBlockLists::for_shape(shape, &all_indices, &counts, &half).unwrap();
}

#[test]
fn region_selection_total_order_keeps_negative_infinity_and_reports_tie_policy() {
    let shape = SageVsaFp4ListShape::new(1, 1, 7, 8).unwrap();
    let half = [64; 16];
    let metadata = CheckedRegionSelection::new(shape, &[1, 8, 4], &half).unwrap();
    let scores = [
        f32::NAN,
        -0.,
        0.,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        0.,
        f32::NAN,
    ];
    let mut out = [0; 8];
    metadata.select_row(&scores, &mut out).unwrap();
    assert_eq!(out, [7, 5, 4, 2, 0, 0, 0, 0]);
    let mut tied = [0; 8];
    metadata
        .select_row(&[f32::NEG_INFINITY; 8], &mut tied)
        .unwrap();
    assert_eq!(tied, [4, 3, 2, 1, 0, 0, 0, 0]);
    let mut nan = [0; 8];
    metadata.select_row(&[f32::NAN; 8], &mut nan).unwrap();
    assert_eq!(nan, tied);
}

#[test]
fn region_selection_matches_independent_sort_across_warp_and_cta_boundaries() {
    for (keys, prefix, keep) in [(33, 1, 7), (129, 2, 31), (259, 3, 128)] {
        let shape = SageVsaFp4ListShape::new(1, 2, keys - prefix, keys).unwrap();
        let regions = [prefix as i32, keys as i32, keep as i32];
        let half = vec![64; keys as usize * 2];
        let metadata = CheckedRegionSelection::new(shape, &regions, &half).unwrap();
        let scores: Vec<f32> = (0..keys).map(|i| ((i * 47) % 29) as f32 - 14.).collect();
        let mut ranked: Vec<usize> = (prefix as usize..keys as usize).collect();
        ranked.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
        ranked.truncate(keep as usize);
        ranked.extend(0..prefix as usize);
        ranked.sort_unstable_by(|a, b| b.cmp(a));
        let mut actual = vec![-1; keys as usize];
        let count = metadata.select_row(&scores, &mut actual).unwrap() as usize;
        assert_eq!(
            &actual[..count],
            ranked.iter().map(|&i| i as i32).collect::<Vec<_>>()
        );
        assert!(actual[count..].iter().all(|&n| n == 0));
    }
}

#[test]
fn region_selection_rejects_holes_overlap_empty_budget_and_invalid_half_tiles() {
    let shape = SageVsaFp4ListShape::new(1, 1, 5, 6).unwrap();
    for regions in [
        vec![],
        vec![0, 6, 2],
        vec![1, 3, 1, 4, 6, 1],
        vec![1, 4, 1, 3, 6, 1],
        vec![1, 6, 0],
        vec![1, 6, 6],
        vec![1, 5, 1],
        vec![1, 6],
    ] {
        assert!(CheckedRegionSelection::new(shape, &regions, &[64; 12]).is_err());
    }
    let mut half = [64; 12];
    half[4] = 7;
    assert!(CheckedRegionSelection::new(shape, &[1, 6, 1], &half).is_err());
    half[5] = 0;
    assert!(CheckedRegionSelection::new(shape, &[1, 6, 1], &half).is_ok());
    assert!(CheckedRegionSelection::new(
        SageVsaFp4ListShape::new(1, 1, 6, 6).unwrap(),
        &[0, 6, 1],
        &[64; 12]
    )
    .is_err());
}
