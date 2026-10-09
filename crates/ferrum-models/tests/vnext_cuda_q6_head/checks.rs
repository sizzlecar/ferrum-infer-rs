use super::*;
use runtime::{Fixture, Path};

fn tokens(rows: usize, count: usize) -> Vec<Arc<[u32]>> {
    (0..rows)
        .map(|p| {
            (0..count)
                .map(|t| 3 + ((p * 7 + t * 11) % 59) as u32)
                .collect::<Vec<_>>()
                .into()
        })
        .collect()
}
fn sessions(f: &Fixture, name: &str, tokens: &[Arc<[u32]>]) -> Vec<Arc<SequenceSession<Runtime>>> {
    tokens
        .iter()
        .enumerate()
        .map(|(p, t)| f.admit(&format!("{name}.{p}"), t.clone()))
        .collect()
}
fn oracle(f: &Fixture, output: &[Vec<u8>], tokens: &[Arc<[u32]>], last: usize) {
    assert_eq!(output.len(), tokens.len());
    for (p, (bytes, t)) in output.iter().zip(tokens).enumerate() {
        assert_eq!(bytes.len(), OUTPUTS as usize * 4);
        for (n, (raw, expected)) in bytes
            .chunks_exact(4)
            .zip(f.source.expected(t[last]))
            .enumerate()
        {
            let got = f32::from_le_bytes(raw.try_into().unwrap());
            // These binary-fraction inputs are exactly Q8 representable; the
            // general actual-pack oracle remains a separate primitive gate.
            assert!(
                got.is_finite() && (f64::from(got) - expected).abs() <= 2e-5,
                "p={p} n={n} token={} got={got} exact={expected}",
                t[last]
            );
        }
    }
}
pub fn verify() {
    // Actual physical dispatch dimensions include J boundaries and M>32 strict
    // fallback. Causal/token-history logic cannot silently pick a first row.
    for rows in [1, 4, 8, 16, 17, 32, 33] {
        let f = Fixture::new(Head::Q6, rows, None, false);
        let t = tokens(rows as usize, 4);
        let s = sessions(&f, "packed", &t);
        let eager = f.execute(&s, &t, 0..1, Path::Eager);
        oracle(&f, &eager, &t, 0);
        if rows <= 32 {
            let reference = Fixture::new(Head::Q6, rows, None, false);
            let reference_sessions = sessions(&reference, "eager-reference", &t);
            assert_eq!(
                eager,
                reference.execute(&reference_sessions, &t, 0..1, Path::Eager)
            );
            for (position, path) in [(1, Path::Warm), (2, Path::Warm), (3, Path::Replay)] {
                let output = f.execute(&s, &t, position..position + 1, path);
                oracle(&f, &output, &t, position);
                assert_ne!(
                    eager, output,
                    "replay must consume the changed token upload"
                );
                assert_eq!(
                    output,
                    reference.execute(&reference_sessions, &t, position..position + 1, Path::Eager),
                    "same legal history: full eager/replay output bits"
                );
            }
        }
    }
    for head in [Head::Q6, Head::Dense] {
        let f = Fixture::new(head, 4, None, false);
        let t = tokens(4, 3);
        let s = sessions(&f, "multiple-tokens", &t);
        let output = f.execute(&s, &t, 0..3, Path::Eager);
        oracle(&f, &output, &t, 2);
    }
}
pub fn markers() {
    let healthy = Fixture::new(Head::Q6, 4, None, false);
    let t: Vec<Arc<[u32]>> = vec![
        vec![3; 4].into(),
        vec![4; 4].into(),
        vec![5; 4].into(),
        vec![6; 4].into(),
    ];
    let s = sessions(&healthy, "healthy", &t);
    let expected = healthy.execute(&s, &t, 0..1, Path::Eager);
    oracle(&healthy, &expected, &t, 0);
    for bad_weight in [false, true] {
        let f = Fixture::new(Head::Q6, 4, (!bad_weight).then_some(4), bad_weight);
        let s = sessions(&f, "poison", &t);
        for (position, path) in [Path::Eager, Path::Warm, Path::Warm, Path::Replay]
            .into_iter()
            .enumerate()
        {
            let output = f.execute(&s, &t, position..position + 1, path);
            for p in 0..4 {
                if bad_weight || p == 1 {
                    assert!(
                        output[p]
                            .chunks_exact(4)
                            .all(|v| u32::from_le_bytes(v.try_into().unwrap()) == 0x7fc00000),
                        "whole-row/leaf marker publication"
                    );
                } else {
                    assert_eq!(
                        output[p], expected[p],
                        "bad neighbor cannot poison healthy participant"
                    );
                }
            }
        }
    }
}
pub fn shared_plan_lanes() {
    let f = Fixture::new(Head::Q6, 8, None, false);
    let lane = f.resources.create_execution_lane().unwrap();
    lane.configure_reusable_executables(DeviceReusableExecutionPlan::on_demand(8).unwrap())
        .unwrap();
    let reaper = CompletionReaper::new();
    let a: Vec<Arc<[u32]>> = tokens(4, 1).iter().map(|t| vec![t[0]; 4].into()).collect();
    let b: Vec<Arc<[u32]>> = a.iter().map(|t| vec![t[0] + 1; 4].into()).collect();
    let sa = sessions(&f, "lane-a", &a);
    let sb = sessions(&f, "lane-b", &b);
    let barrier = std::sync::Barrier::new(3);
    let submitted = runtime::SubmissionRendezvous::default();
    // Real independent lanes share the admitted Plan flags and weights. This
    // proves concurrent submission correctness, not GPU overlap or speedup.
    let (observed_a, observed_b) = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            barrier.wait();
            f.on_lane_observed(
                &f.lane,
                &f.reaper,
                &sa,
                &a,
                0..1,
                Path::Eager,
                Some(&submitted),
            )
        });
        let second = scope.spawn(|| {
            barrier.wait();
            f.on_lane_observed(&lane, &reaper, &sb, &b, 0..1, Path::Eager, Some(&submitted))
        });
        barrier.wait();
        (first.join().unwrap(), second.join().unwrap())
    });
    let mut work: Vec<_> = observed_a
        .dependency_work
        .iter()
        .chain(&observed_b.dependency_work)
        .copied()
        .collect();
    work.sort_unstable();
    assert_eq!(work, [(0, 0, 1), (1, 1, 1)], "two concurrent consumers hold the same live validation: exactly one memset/scan, one wait each");
    let (oa, ob) = (observed_a.outputs, observed_b.outputs);
    oracle(&f, &oa, &a, 0);
    oracle(&f, &ob, &b, 0);
    for (offset, path) in [Path::Warm, Path::Warm, Path::Replay]
        .into_iter()
        .enumerate()
    {
        let position = offset + 1;
        assert_eq!(oa, f.execute(&sa, &a, position..position + 1, path));
        assert_eq!(
            ob,
            f.on_lane(&lane, &reaper, &sb, &b, position..position + 1, path)
        );
    }
}
