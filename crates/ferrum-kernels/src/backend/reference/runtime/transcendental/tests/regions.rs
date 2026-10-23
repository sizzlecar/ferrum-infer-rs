use super::*;
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

#[test]
fn allocation_partitions_use_exact_offsets_and_preserve_surrounding_bytes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        let partitions = if matches!(kind, Kind::SiluMul) {
            vec![
                vec![0, 1, 2],
                vec![0, 0, 1],
                vec![0, 1, 0],
                vec![0, 1, 1],
                vec![0, 0, 0],
            ]
        } else {
            vec![vec![0, 1], vec![0, 0]]
        };
        for partition in partitions {
            let storage = (0..partition.len())
                .map(|i| {
                    buffer(
                        &runtime,
                        &format!("resource.activation.partition-{i}"),
                        &[0xa5; 32],
                        16,
                        ElementType::Bf16,
                    )
                })
                .collect::<Vec<_>>();
            let input = storage[partition[0]].region(2..10).unwrap();
            input.write(&bytes(&[0x3f80, 0xc2b4, 0, 1]));
            let up = if matches!(kind, Kind::SiluMul) {
                storage[partition[1]].region(10..18).unwrap()
            } else {
                input.clone()
            };
            if matches!(kind, Kind::SiluMul) {
                up.write(&bytes(&[0x3f81, 0xbf80, 0xbf80, 0x3f80]));
            }
            let output_slot = *partition.last().unwrap();
            let output = storage[output_slot].region(18..26).unwrap();
            let before = storage
                .iter()
                .map(|b| b.region(0..32).unwrap().read())
                .collect::<Vec<_>>();
            let case = Fixture {
                kind,
                input,
                up,
                output,
                shape: ReferenceActivationShape { rows: 2, width: 2 },
            };
            submit(&runtime, vec![case.command().unwrap()]).unwrap();
            check(
                &case.output,
                &match kind {
                    Kind::Silu => [0x3f3b, 0x8000, 0, 0],
                    Kind::SiluMul => [0x3f3c, 0, 0x8000, 0],
                    Kind::Tanh => [0x3f43, 0xbf80, 0, 1],
                },
            );
            for (slot, allocation) in storage.iter().enumerate() {
                let after = allocation.region(0..32).unwrap().read();
                for index in 0..32 {
                    if slot != output_slot || !(18..26).contains(&index) {
                        assert_eq!(after[index], before[slot][index]);
                    }
                }
            }
        }
    }
}

#[test]
fn private_read_aliases_are_safe_but_output_aliases_and_odd_offsets_are_rejected() {
    // These private launch checks do not widen core's same-logical-value rule
    // for overlapping readonly bindings. That identity is checked upstream.
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for partial in [false, true] {
        let storage = buffer(
            &runtime,
            "resource.activation.read-alias",
            &[0xa5; 18],
            16,
            ElementType::Bf16,
        );
        storage
            .region(0..10)
            .unwrap()
            .write(&bytes(&[0x3f80, 0x3f80, 0, 0, 0x3f80]));
        let case = Fixture {
            kind: Kind::SiluMul,
            input: storage.region(0..8).unwrap(),
            up: storage.region(if partial { 2..10 } else { 0..8 }).unwrap(),
            output: storage.region(10..18).unwrap(),
            shape: ReferenceActivationShape { rows: 2, width: 2 },
        };
        submit(&runtime, vec![case.command().unwrap()]).unwrap();
        check(
            &case.output,
            if partial {
                &[0x3f3b, 0, 0, 0]
            } else {
                &[0x3f3b, 0x3f3b, 0, 0]
            },
        );
        assert_eq!(
            storage.region(0..10).unwrap().read(),
            bytes(&[0x3f80, 0x3f80, 0, 0, 0x3f80])
        );
    }
    for kind in KINDS {
        let storage = [
            "resource.activation.alias-input",
            "resource.activation.alias-up",
        ]
        .map(|name| buffer(&runtime, name, &[0xa5; 32], 16, ElementType::Bf16));
        let case = Fixture {
            kind,
            input: storage[0].region(4..12).unwrap(),
            up: storage[1].region(4..12).unwrap(),
            output: storage[0].region(20..28).unwrap(),
            shape: ReferenceActivationShape { rows: 2, width: 2 },
        };
        for read_slot in 0..if matches!(kind, Kind::SiluMul) { 2 } else { 1 } {
            for offset in [2, 4, 6] {
                let mut wrong = case.clone();
                wrong.output = storage[read_slot].region(offset..offset + 8).unwrap();
                assert!(wrong.command().is_err());
            }
        }
        let mut wrong = case.clone();
        wrong.input = storage[0].region(5..13).unwrap();
        assert!(wrong.command().is_err());
        let mut wrong = case.clone();
        wrong.output = storage[0].region(19..27).unwrap();
        assert!(wrong.command().is_err());
        if matches!(kind, Kind::SiluMul) {
            let mut wrong = case.clone();
            wrong.up = storage[1].region(5..13).unwrap();
            assert!(wrong.command().is_err());
        }
        for allocation in storage {
            assert_eq!(allocation.region(0..32).unwrap().read(), [0xa5; 32]);
        }
    }
}

#[test]
fn every_operand_is_checked_against_the_submitting_runtime() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        let local = fixture(&runtime, kind, 2, &[0x3f80; 4], &[0x3f80; 4]);
        let other = fixture(&foreign, kind, 2, &[0x3f80; 4], &[0x3f80; 4]);
        for role in 0..if matches!(kind, Kind::SiluMul) { 3 } else { 2 } {
            let mut mixed = local.clone();
            match role {
                0 => mixed.input = other.input.clone(),
                1 => mixed.output = other.output.clone(),
                2 => mixed.up = other.up.clone(),
                _ => unreachable!(),
            }
            // Geometry alone is valid; actual submit rejects foreign ownership.
            assert!(submit(&runtime, vec![mixed.command().unwrap()]).is_err());
            assert_eq!(local.output.read(), [0xa5; 8]);
            assert_eq!(other.output.read(), [0xa5; 8]);
        }
    }
    assert_eq!(runtime.snapshot().submissions, 0);
    assert_eq!(foreign.snapshot().submissions, 0);
}

#[test]
fn opposite_unary_and_binary_roles_share_one_global_lock_order() {
    let runtime = Arc::new(ReferenceDeviceRuntime::new(config()).unwrap());
    let a = buffer(
        &runtime,
        "resource.activation.concurrent-a",
        &[0xa5; 16],
        16,
        ElementType::Bf16,
    );
    let b = buffer(
        &runtime,
        "resource.activation.concurrent-b",
        &[0xa5; 16],
        16,
        ElementType::Bf16,
    );
    for input in [&a, &b] {
        input.region(0..8).unwrap().write(&bytes(&[0x3f80; 4]));
    }
    let unary = Fixture {
        kind: Kind::Tanh,
        input: a.region(0..8).unwrap(),
        up: a.region(0..8).unwrap(),
        output: b.region(8..16).unwrap(),
        shape: ReferenceActivationShape { rows: 2, width: 2 },
    };
    let binary = Fixture {
        kind: Kind::SiluMul,
        input: b.region(0..8).unwrap(),
        up: a.region(0..8).unwrap(),
        output: a.region(8..16).unwrap(),
        shape: unary.shape,
    };
    let unary_slots = AllocationLocks::new([&unary.input, &unary.output]).slots;
    let binary_slots = AllocationLocks::new([&binary.input, &binary.up, &binary.output]).slots;
    assert_eq!(unary_slots, [binary_slots[1], binary_slots[0]]);
    assert_eq!(binary_slots[1], binary_slots[2]);
    let barrier = Arc::new(Barrier::new(3));
    let (sent, received) = mpsc::channel();
    let handles = [unary, binary].map(|case| {
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
            .expect("opposite activation roles must not deadlock");
    }
    for handle in handles {
        handle.join().unwrap();
    }
    check(&a.region(8..16).unwrap(), &[0x3f3b; 4]);
    check(&b.region(8..16).unwrap(), &[0x3f43; 4]);
    for input in [&a, &b] {
        assert_eq!(input.region(0..8).unwrap().read(), bytes(&[0x3f80; 4]));
    }
}
