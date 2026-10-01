use super::*;

#[test]
fn support_diagnostic_matches_complete_joint_support_without_changing_it() {
    let points = [[2, 8], [8, 2]];
    let support = JointSupport::new(points.iter().map(|point| point.as_slice())).unwrap();
    assert_eq!(
        support.diagnose(&[1, 2]),
        Some(StructuredFitSupportReasonV1::BelowMinimum {
            support_axis: 0,
            query: 1,
            minimum: 2,
            maximum: 8,
        })
    );
    assert_eq!(
        support.diagnose(&[9, 2]),
        Some(StructuredFitSupportReasonV1::AboveAllFitMax {
            support_axis: 0,
            query: 9,
            minimum: 2,
            maximum: 8,
        })
    );
    assert_eq!(
        support.diagnose(&[5, 5]),
        Some(StructuredFitSupportReasonV1::NoJointDominator {
            support_axis: 0,
            query: 5,
            minimum: 2,
            maximum: 8,
            first_fit_point_upper: 2,
        })
    );
    for first in 0..=10 {
        for second in 0..=10 {
            let query = [first, second];
            let before = support.contains(&query);
            assert_eq!(support.diagnose(&query).is_none(), before);
            assert_eq!(support.contains(&query), before);
        }
    }
}
