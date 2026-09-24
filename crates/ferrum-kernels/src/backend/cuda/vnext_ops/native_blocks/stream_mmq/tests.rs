use super::*;
use ferrum_interfaces::vnext::WeightId;

fn part(offset: u32) -> weights::MatrixPart {
    weights::MatrixPart {
        component_id: WeightId::new(format!("fixture.part.{offset}")).unwrap(),
        format: weights::MatrixFormat::Block(GgufBlockFormat::Q4K),
        rows: 768,
        columns: 512,
        output_offset: offset,
        transform: None,
        signs_region: None,
    }
}

#[test]
fn stream_mmq_geometry_requires_complete_physical_b8_pair() {
    let pair = [part(0), part(768)];
    assert!(eligible(&pair, 8, 512, 768));
    for rows in [1, 4, 7, 9, 16, 128] {
        assert!(!eligible(&pair, rows, 512, 768));
    }
    for hidden in [0, 511, 513] {
        assert!(!eligible(&pair, 8, hidden, 768));
    }
    assert!(!eligible(&pair[..1], 8, 512, 768));
    for index in 0..2 {
        let mut wrong = pair.clone();
        wrong[index].format = weights::MatrixFormat::Block(GgufBlockFormat::Q5K);
        assert!(!eligible(&wrong, 8, 512, 768));
        let mut wrong = pair.clone();
        wrong[index].output_offset += 1;
        assert!(!eligible(&wrong, 8, 512, 768));
        let mut wrong = pair.clone();
        wrong[index].signs_region = Some(0);
        assert!(!eligible(&wrong, 8, 512, 768));
    }
}

#[test]
fn stream_mmq_workspace_covers_every_partition_and_metadata_without_alias() {
    for hidden in [256, 512, 4096] {
        for intermediate in [1u32, 65, 129, 12288] {
            for budget in [1, 3, 17, 337] {
                let w = Workspace::new(hidden, intermediate, budget).unwrap();
                let tiles = u64::from(intermediate).div_ceil(128);
                let units = tiles * u64::from(hidden / 256);
                assert_eq!(u64::from(w.ctas), units.min(u64::from(budget)));
                assert_eq!(w.words_bytes, 8 * u64::from(hidden));
                assert_eq!(w.scales_bytes, u64::from(hidden));
                assert_eq!(w.partial_offset, w.words_bytes + w.scales_bytes * 2);
                assert_eq!(
                    w.total_bytes - w.partial_offset,
                    (tiles + u64::from(w.ctas)) * 1024 * 4
                );
                assert_eq!(w.partial_offset % 16, 0);
            }
        }
    }
    assert!(Workspace::new(257, 768, 3).is_err());
    assert!(Workspace::new(256, 768, 0).is_err());
}
