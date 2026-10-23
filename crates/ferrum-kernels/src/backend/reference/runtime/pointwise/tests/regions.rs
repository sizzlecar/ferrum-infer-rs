use super::*;
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

#[test]
fn allocation_partitions_use_exact_nonzero_regions_and_preserve_sentinels() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        // All three-role partitions and representative four-role partitions,
        // including shared output allocations and one through four owners.
        let partitions = match kind {
            Kind::ModulationScale => vec![
                vec![0, 1, 2],
                vec![0, 0, 1],
                vec![0, 1, 0],
                vec![0, 1, 1],
                vec![0, 0, 0],
            ],
            Kind::GatedResidual => vec![
                vec![0, 1, 2, 3],
                vec![0, 1, 2, 0],
                vec![0, 1, 2, 1],
                vec![0, 1, 2, 2],
                vec![0, 0, 1, 1],
                vec![0, 0, 0, 0],
            ],
        };
        for partition in partitions {
            for mode in RowBroadcast::ALL {
                let storage = (0..partition.len())
                    .map(|index| {
                        buffer(
                            &runtime,
                            &format!("resource.pointwise.partition-{index}"),
                            &[0xa5; 40],
                            16,
                            ElementType::Bf16,
                        )
                    })
                    .collect::<Vec<_>>();
                let input = storage[partition[0]].region(2..10).unwrap();
                let parameter = storage[partition[1]]
                    .region(10..if mode == RowBroadcast::Row { 14 } else { 18 })
                    .unwrap();
                let update = if kind == Kind::GatedResidual {
                    storage[partition[2]].region(18..26).unwrap()
                } else {
                    input.clone()
                };
                let output_slot = *partition.last().unwrap();
                let output = storage[output_slot].region(26..34).unwrap();
                input.write(&encoded(&[1., 2., 3., 4.]));
                parameter.write(&encoded(if mode == RowBroadcast::Row {
                    &[0., 1.]
                } else {
                    &[0., 1., 2., -1.]
                }));
                if kind == Kind::GatedResidual {
                    update.write(&encoded(&[2., 4., 6., 8.]));
                }
                let before = storage
                    .iter()
                    .map(|s| s.region(0..40).unwrap().read())
                    .collect::<Vec<_>>();
                let case = Fixture {
                    kind,
                    input,
                    parameter,
                    update,
                    output,
                    shape: ReferencePointwiseShape {
                        rows: 2,
                        width: 2,
                        broadcast: mode,
                    },
                };
                submit(&runtime, vec![case.command().unwrap()]).unwrap();
                let expected = match (kind, mode) {
                    (Kind::ModulationScale, RowBroadcast::Row) => [1., 4., 3., 8.],
                    (Kind::ModulationScale, RowBroadcast::Elementwise) => [1., 4., 9., 0.],
                    (Kind::GatedResidual, RowBroadcast::Row) => [1., 6., 3., 12.],
                    (Kind::GatedResidual, RowBroadcast::Elementwise) => [1., 6., 15., -4.],
                };
                assert_eq!(case.output.read(), encoded(&expected));
                for (slot, allocation) in storage.iter().enumerate() {
                    let after = allocation.region(0..40).unwrap().read();
                    for byte in 0..40 {
                        if slot != output_slot || !(26..34).contains(&byte) {
                            assert_eq!(after[byte], before[slot][byte]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn private_launch_allows_exact_and_partial_read_aliases_with_disjoint_output() {
    // This checks storage safety only. Public binding validation additionally
    // restricts overlapping reads to reuse of the same logical value identity.
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        for mode in RowBroadcast::ALL {
            let storage = buffer(
                &runtime,
                "resource.pointwise.read-alias",
                &[0xa5; 16],
                16,
                ElementType::Bf16,
            );
            let input = storage.region(0..8).unwrap();
            input.write(&encoded(&[1., 2., 3., 4.]));
            let parameter = storage
                .region(0..if mode == RowBroadcast::Row { 4 } else { 8 })
                .unwrap();
            let case = Fixture {
                kind,
                update: input.clone(),
                input,
                parameter,
                output: storage.region(8..16).unwrap(),
                shape: ReferencePointwiseShape {
                    rows: 2,
                    width: 2,
                    broadcast: mode,
                },
            };
            submit(&runtime, vec![case.command().unwrap()]).unwrap();
            let expected = if mode == RowBroadcast::Row {
                [2., 6., 6., 12.]
            } else {
                [2., 6., 12., 20.]
            };
            assert_eq!(case.output.read(), encoded(&expected));
            assert_eq!(case.input.read(), encoded(&[1., 2., 3., 4.]));
        }
        let storage = buffer(
            &runtime,
            "resource.pointwise.partial-read",
            &[0xa5; 20],
            16,
            ElementType::Bf16,
        );
        storage
            .region(0..12)
            .unwrap()
            .write(&encoded(&[1., 2., 3., 4., 5., 6.]));
        let case = Fixture {
            kind,
            input: storage.region(0..8).unwrap(),
            parameter: storage.region(2..10).unwrap(),
            update: storage.region(4..12).unwrap(),
            output: storage.region(12..20).unwrap(),
            shape: ReferencePointwiseShape {
                rows: 2,
                width: 2,
                broadcast: RowBroadcast::Elementwise,
            },
        };
        submit(&runtime, vec![case.command().unwrap()]).unwrap();
        let expected = match kind {
            Kind::ModulationScale => [3., 8., 15., 24.],
            Kind::GatedResidual => [7., 14., 23., 34.],
        };
        assert_eq!(case.output.read(), encoded(&expected));
        assert_eq!(
            storage.region(0..12).unwrap().read(),
            encoded(&[1., 2., 3., 4., 5., 6.])
        );
    }
}

#[test]
fn output_overlap_with_every_read_operand_is_rejected_before_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        for mode in RowBroadcast::ALL {
            let storage = (0..3)
                .map(|index| {
                    buffer(
                        &runtime,
                        &format!("resource.pointwise.alias-{index}"),
                        &[0xa5; 32],
                        16,
                        ElementType::Bf16,
                    )
                })
                .collect::<Vec<_>>();
            let case = Fixture {
                kind,
                input: storage[0].region(4..12).unwrap(),
                parameter: storage[1]
                    .region(4..if mode == RowBroadcast::Row { 8 } else { 12 })
                    .unwrap(),
                update: storage[2].region(4..12).unwrap(),
                output: storage[0].region(20..28).unwrap(),
                shape: ReferencePointwiseShape {
                    rows: 2,
                    width: 2,
                    broadcast: mode,
                },
            };
            // An odd offset entirely inside the allocation must fail for
            // alignment, independently of the physical bounds check.
            let mut misaligned = case.clone();
            misaligned.input = storage[0].region(5..13).unwrap();
            assert!(misaligned.command().is_err());
            for read_slot in 0..if kind == Kind::GatedResidual { 3 } else { 2 } {
                for offset in [2, 4, 6] {
                    let mut overlap = case.clone();
                    overlap.output = storage[read_slot].region(offset..offset + 8).unwrap();
                    assert!(overlap.command().is_err());
                }
            }
            for allocation in storage {
                assert_eq!(allocation.region(0..32).unwrap().read(), [0xa5; 32]);
            }
        }
    }
}

#[test]
fn opposite_roles_across_three_and_four_slot_tables_share_one_lock_order() {
    let runtime = Arc::new(ReferenceDeviceRuntime::new(config()).unwrap());
    let a = buffer(
        &runtime,
        "resource.pointwise.concurrent-a",
        &[0xa5; 16],
        16,
        ElementType::Bf16,
    );
    let b = buffer(
        &runtime,
        "resource.pointwise.concurrent-b",
        &[0xa5; 16],
        16,
        ElementType::Bf16,
    );
    for source in [&a, &b] {
        source
            .region(0..8)
            .unwrap()
            .write(&encoded(&[1., 2., 3., 4.]));
    }
    let modulation = Fixture {
        kind: Kind::ModulationScale,
        input: a.region(0..8).unwrap(),
        parameter: b.region(0..8).unwrap(),
        update: a.region(0..8).unwrap(),
        output: b.region(8..16).unwrap(),
        shape: ReferencePointwiseShape {
            rows: 2,
            width: 2,
            broadcast: RowBroadcast::Elementwise,
        },
    };
    let gated = Fixture {
        kind: Kind::GatedResidual,
        input: b.region(0..8).unwrap(),
        parameter: a.region(0..8).unwrap(),
        update: b.region(0..8).unwrap(),
        output: a.region(8..16).unwrap(),
        shape: modulation.shape,
    };
    let mod_slots =
        AllocationLocks::new([&modulation.input, &modulation.parameter, &modulation.output]).slots;
    let gate_slots =
        AllocationLocks::new([&gated.input, &gated.parameter, &gated.update, &gated.output]).slots;
    assert_eq!(mod_slots, [gate_slots[1], gate_slots[0], gate_slots[0]]);
    assert_eq!(gate_slots[0], gate_slots[2]);
    assert_eq!(gate_slots[1], gate_slots[3]);
    let barrier = Arc::new(Barrier::new(3));
    let (sent, received) = mpsc::channel();
    let handles = [modulation, gated].map(|case| {
        let runtime = runtime.clone();
        let barrier = barrier.clone();
        let sent = sent.clone();
        std::thread::spawn(move || {
            let command = case.command().unwrap();
            barrier.wait();
            submit(&runtime, vec![command]).unwrap();
            sent.send(()).unwrap();
        })
    });
    barrier.wait();
    for _ in 0..2 {
        received
            .recv_timeout(Duration::from_secs(10))
            .expect("opposite operand roles must not deadlock");
    }
    for handle in handles {
        handle.join().unwrap();
    }
    for output in [&a, &b] {
        assert_eq!(
            output.region(8..16).unwrap().read(),
            encoded(&[2., 6., 12., 20.])
        );
        assert_eq!(
            output.region(0..8).unwrap().read(),
            encoded(&[1., 2., 3., 4.])
        );
    }
}
