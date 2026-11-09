use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

fn geometry() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        16,
    )
    .unwrap()
}
fn budgets(
    h: &HostMemoryAccount,
) -> (
    ExecutionHostBudget,
    usize,
    [(ExecutionHostBudget, usize, usize, usize); 2],
) {
    let p = |n| h.backing_bytes(n).unwrap();
    (
        ExecutionHostBudget {
            input: p(17) + p(33) + p(3),
            output: p(9),
            staging: p(7),
            weights: p(31) + p(2),
        },
        p(2),
        [
            (
                ExecutionHostBudget {
                    input: p(21),
                    output: p(41),
                    staging: p(5),
                    weights: p(15) + p(4),
                },
                p(4),
                p(25),
                p(17),
            ),
            (
                ExecutionHostBudget {
                    input: p(45),
                    output: p(25),
                    staging: p(5),
                    weights: p(47) + p(5),
                },
                p(5),
                p(57),
                p(33),
            ),
        ],
    )
}
fn account(
    b: ExecutionHostBudget,
    input_extra: usize,
    output_extra: usize,
    short: Option<HostMemoryClass>,
) -> HostMemoryAccount {
    let mut caps = HostMemoryClassLimits {
        input_preprocess_bytes: b.input + input_extra,
        output_bytes: b.output + output_extra,
        transfer_staging_bytes: b.staging,
        weight_backing_bytes: b.weights,
    };
    match short {
        Some(HostMemoryClass::InputPreprocess) => caps.input_preprocess_bytes -= 1,
        Some(HostMemoryClass::WeightBacking) => caps.weight_backing_bytes -= 1,
        None => (),
        _ => unreachable!(),
    }
    let total = caps.input_preprocess_bytes
        + caps.output_bytes
        + caps.transfer_staging_bytes
        + caps.weight_backing_bytes;
    HostMemoryAccount::with_allocation_alignment(HostMemoryLimits::new(total, caps).unwrap(), 16)
        .unwrap()
}
#[test]
fn visual_cohort_scope_covers_prior_inputs_and_max_conversion_with_original_credits_once() {
    let g = geometry();
    let p = |n| g.backing_bytes(n).unwrap();
    let (causal, persistent, visual) = budgets(&g);
    let b = sequential_budget(causal, persistent, visual.into_iter()).unwrap();
    assert_eq!(b.input, p(17) + p(57) + p(45));
    assert_eq!(b.weights, p(2) + p(4) + p(5) + p(47));
    assert!(b.weights < causal.weights + visual[0].0.weights + visual[1].0.weights);
    let host = account(b, p(11) + p(13), p(7) + p(7), None);
    let ingress = [
        host.allocate_zeroed(HostMemoryClass::InputPreprocess, 11)
            .unwrap(),
        host.allocate_zeroed(HostMemoryClass::InputPreprocess, 13)
            .unwrap(),
    ];
    let credits = host
        .reserve_scope(&[(HostMemoryClass::Output, p(7) + p(7))])
        .unwrap();
    let phase = host.reserve_scope(&b.claims()).unwrap();
    let work = phase.account();
    let caches = [
        work.allocate_zeroed(HostMemoryClass::WeightBacking, 2)
            .unwrap(),
        work.allocate_zeroed(HostMemoryClass::WeightBacking, 4)
            .unwrap(),
        work.allocate_zeroed(HostMemoryClass::WeightBacking, 5)
            .unwrap(),
    ];
    let d0 = work
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 25)
        .unwrap();
    let v0 = work
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 21)
        .unwrap();
    let out0 = work.allocate_zeroed(HostMemoryClass::Output, 41).unwrap();
    drop((d0, v0));
    let p0 = work
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 17)
        .unwrap();
    drop(out0);
    let d1 = work
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 57)
        .unwrap();
    let v1 = work
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 45)
        .unwrap();
    assert!(work
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .is_err());
    let conversion = work
        .allocate_zeroed(HostMemoryClass::WeightBacking, 47)
        .unwrap();
    assert!(work
        .allocate_zeroed(HostMemoryClass::WeightBacking, 1)
        .is_err());
    drop(conversion);
    let out1 = work.allocate_zeroed(HostMemoryClass::Output, 25).unwrap();
    drop((d1, v1));
    let p1 = work
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 33)
        .unwrap();
    drop(out1);
    let wave = work
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 3)
        .unwrap();
    let output = work.allocate_zeroed(HostMemoryClass::Output, 9).unwrap();
    let staging = work
        .allocate_zeroed(HostMemoryClass::TransferStaging, 7)
        .unwrap();
    let conversion = work
        .allocate_zeroed(HostMemoryClass::WeightBacking, 31)
        .unwrap();
    drop((p0, p1, wave, output, staging, conversion, caches, work));
    phase.close().unwrap();
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        p(11) + p(13) + p(7) + p(7)
    );
    drop((ingress, credits));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn visual_cohort_short_input_or_weight_grant_refuses_atomically_and_overflows_are_errors() {
    let g = geometry();
    let p = |n| g.backing_bytes(n).unwrap();
    let (causal, persistent, visual) = budgets(&g);
    let b = sequential_budget(causal, persistent, visual.into_iter()).unwrap();
    for class in [
        HostMemoryClass::InputPreprocess,
        HostMemoryClass::WeightBacking,
    ] {
        let host = account(b, p(11) + p(13), p(7) + p(7), Some(class));
        let ingress = [
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, 11)
                .unwrap(),
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, 13)
                .unwrap(),
        ];
        let credits = host
            .reserve_scope(&[(HostMemoryClass::Output, p(7) + p(7))])
            .unwrap();
        let before = host.usage().unwrap().committed_bytes();
        assert!(host.reserve_scope(&b.claims()).is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        drop((ingress, credits));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    let mut bad = visual;
    bad[1].2 = usize::MAX;
    assert!(sequential_budget(causal, persistent, bad.into_iter()).is_err());
    assert!(sequential_budget(causal, usize::MAX, visual.into_iter()).is_err());
}
