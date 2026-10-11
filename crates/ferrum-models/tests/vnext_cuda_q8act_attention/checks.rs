use super::*;
use runtime::{Fixture, Path};

pub fn verify(kind: AttentionKind) {
    for participants in [1_u32, 3, 8, 9] {
        let eager = Fixture::for_attention(kind, false, participants);
        let graph = Fixture::for_attention(kind, true, participants);
        let tokens: Vec<Arc<[u32]>> = (0..participants)
            .map(|participant| {
                (0..44)
                    .map(|position| ((position * 7 + participant * 3 + 1) % 32) as u32)
                    .collect()
            })
            .collect();
        let eager_sessions: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| eager.admit(&format!("eager-{i}"), Arc::clone(t)))
            .collect();
        let graph_sessions: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| graph.admit(&format!("graph-{i}"), Arc::clone(t)))
            .collect();

        // Build actual recurrent state and a complete first KV block. This
        // is a separate multi-token comparison, not an assumption that batch
        // partitioning has the same floating-point association.
        let prefix_path = if kind == AttentionKind::Causal {
            Path::EagerBoundary
        } else {
            Path::Warm
        };
        for (range, path) in [(0..16, Path::Warm), (16..32, prefix_path)] {
            let expected =
                eager.execute_participants(&eager_sessions, &tokens, range.clone(), Path::Eager);
            let actual = graph.execute_participants(&graph_sessions, &tokens, range.clone(), path);
            expected.assert_same(&actual);
            actual.dump(kind, participants, range, path);
        }

        // Two adaptive waves establish the actual catalog topology; two
        // subsequent calls require replay with different source positions and
        // deliberately different scratch fill from the independent eager lane.
        // Crossing position32 also checks physical KV block addressing.
        for position in 32..36 {
            let path = if position < 34 {
                Path::Warm
            } else {
                Path::Replay
            };
            let range = position..position + 1;
            let expected =
                eager.execute_participants(&eager_sessions, &tokens, range.clone(), Path::Eager);
            let actual = graph.execute_participants(&graph_sessions, &tokens, range.clone(), path);
            expected.assert_same(&actual);
            actual.dump(kind, participants, range, path);
        }

        // Revisit the real multi-token boundary after a reusable graph exists:
        // its presence must not qualify an otherwise unsupported FP16 KV wave.
        for (range, path) in [(36..38, Path::Warm), (38..40, prefix_path)] {
            let expected =
                eager.execute_participants(&eager_sessions, &tokens, range.clone(), Path::Eager);
            let actual = graph.execute_participants(&graph_sessions, &tokens, range.clone(), path);
            expected.assert_same(&actual);
            actual.dump(kind, participants, range, path);
        }
        for session in eager_sessions.iter().chain(&graph_sessions) {
            session.try_complete().unwrap();
        }
    }
}
