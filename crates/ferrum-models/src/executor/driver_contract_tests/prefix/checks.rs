//! Backend numerical oracles remain in models; execution owns only the
//! acknowledgement/identity/cancellation scenario over these real sessions.
use super::*;
use ferrum_execution::test_support::prefix::{
    assert_token_ack_failure_preserves_other_incarnation, assert_token_ack_preserves_incarnation,
};

const BOUNDARY: usize = 129;
const INPUT: usize = 138;

fn tokens() -> Arc<[u32]> {
    (0..INPUT)
        .map(|index| ((index * 7 + 3) % 32) as u32)
        .collect()
}

fn execute_prefix(fixture: &Fixture, session: &Arc<SequenceSession<Runtime>>, tokens: &Arc<[u32]>) {
    fixture
        .execute(session, Arc::clone(tokens), 0..128)
        .assert_state_nonzero();
    fixture
        .execute(session, Arc::clone(tokens), 128..BOUNDARY)
        .assert_state_nonzero();
}

#[test]
fn native_prefix_tokens_ack_preserves_incarnation_and_gpu_continuation() {
    let fixture = Fixture::new(AttentionKind::CausalInt8);
    let tokens = tokens();
    let source = fixture.admit("token-ack-source", Arc::clone(&tokens));
    let cold = fixture.admit("token-ack-cold", Arc::clone(&tokens));
    execute_prefix(&fixture, &source, &tokens);
    execute_prefix(&fixture, &cold, &tokens);
    let checkpoint = fixture.capture(&source);
    assert_eq!(checkpoint.completed_tokens(), BOUNDARY);
    source.try_abort_if_quiescent().unwrap();
    drop(source);
    let target = fixture.admit("token-ack-target", Arc::clone(&tokens));
    assert_token_ack_preserves_incarnation(
        target,
        &tokens,
        BOUNDARY,
        |session| fixture.restore_publication(session, &checkpoint, Arc::clone(&tokens)),
        |session| fixture.assert_restore_rejected(session, &checkpoint, Arc::clone(&tokens)),
        |session| {
            for range in [BOUNDARY..137, 137..INPUT] {
                let expected = fixture.execute(&cold, Arc::clone(&tokens), range.clone());
                let actual = fixture.execute(session, Arc::clone(&tokens), range);
                expected.assert_same(&actual, "token-ack restored output/payload/scales");
            }
            cold.try_complete().unwrap();
        },
    );
}

#[test]
fn native_prefix_tokens_ack_failure_cancels_only_the_original_target() {
    let fixture = Fixture::new(AttentionKind::CausalInt8);
    let tokens = tokens();
    let source = fixture.admit("token-ack-failure-source", Arc::clone(&tokens));
    execute_prefix(&fixture, &source, &tokens);
    let checkpoint = fixture.capture(&source);
    let expected = fixture.execute(&source, Arc::clone(&tokens), BOUNDARY..INPUT);
    source.try_complete().unwrap();
    drop(source);
    for replaced in [false, true] {
        let target = fixture.admit(&format!("token-ack-old-{replaced}"), Arc::clone(&tokens));
        assert_token_ack_failure_preserves_other_incarnation(
            target,
            &tokens,
            |session| fixture.restore_publication(session, &checkpoint, Arc::clone(&tokens)),
            || replaced.then(|| fixture.admit("token-ack-replacement", Arc::clone(&tokens))),
            |session| fixture.assert_restore_rejected(session, &checkpoint, Arc::clone(&tokens)),
            |session| {
                execute_prefix(&fixture, session, &tokens);
                expected.assert_same(
                    &fixture.execute(session, Arc::clone(&tokens), BOUNDARY..INPUT),
                    "same-request replacement was not cancelled or given old state",
                );
            },
        );
    }
}
