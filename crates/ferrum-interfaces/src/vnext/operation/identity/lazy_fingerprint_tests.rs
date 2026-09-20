use super::*;
use crate::vnext::{
    BatchParticipantAuthority, ExecutionFrameId, ExecutionIdentityParts, NodeInvocationId,
    RequestAuthorityId, RequestIdentity, RunId, SequenceAuthorityId, SpanId,
    EXECUTION_IDENTITY_VERSION,
};
use sha2::{Digest, Sha256};
use std::sync::Barrier;

struct Fixture {
    node_index: u32,
    node_id: NodeId,
    operation_id: OperationId,
    provider_id: ProviderId,
    provider_fingerprint: String,
    semantics: ProviderExecutionSemantics,
    work_fingerprint: String,
    participants: Vec<BatchOperationParticipantIdentity>,
}

impl Fixture {
    fn new() -> Self {
        let mut fixture = Self {
            node_index: 3,
            node_id: NodeId::new("node.linear").unwrap(),
            operation_id: OperationId::new("operation.linear").unwrap(),
            provider_id: ProviderId::new("provider.cpu").unwrap(),
            provider_fingerprint: "a".repeat(64),
            semantics: ProviderExecutionSemantics::bitwise_eager_and_replay(),
            work_fingerprint: "b".repeat(64),
            participants: Vec::new(),
        };
        for slot in [2_u32, 9] {
            let frame = ExecutionFrameId::try_from(u64::from(slot) + 1).unwrap();
            let parts = ExecutionIdentityParts {
                version: EXECUTION_IDENTITY_VERSION,
                run_id: RunId::new("run.lazy-fingerprint").unwrap(),
                request_id: RequestIdentity::new(format!("request.{slot}")).unwrap(),
                sequence: u64::from(slot) + 10,
                plan_id: Some(PlanId::new("plan.linear").unwrap()),
                plan_hash: Some(serde_json::from_value(serde_json::json!("c".repeat(64))).unwrap()),
                frame_id: Some(frame),
                node_invocation_id: Some(NodeInvocationId::try_from(7_u64).unwrap()),
                node_id: Some(fixture.node_id.clone()),
                operation_id: Some(fixture.operation_id.clone()),
                provider_id: Some(fixture.provider_id.clone()),
                device_id: Some(DeviceId::new("cpu.identity-test").unwrap()),
                resource_pool_id: None,
                resource_pool_identity_fingerprint: None,
                provisioning_run_id: None,
                provisioning_request_id: None,
                transaction_id: None,
                active_sequence_slot: Some(slot),
                admission_generation: Some(2),
                activation_epoch: Some(3),
                runtime_implementation_fingerprint: Some("d".repeat(64)),
                active_sequence_fingerprint: Some("e".repeat(64)),
                completed_sequence_fingerprint: None,
                aborted_sequence_fingerprint: None,
                resource_id: None,
                resource_generation: None,
                resource_batch_fingerprint: None,
                span_id: SpanId::new(format!("span.node/{slot}")).unwrap(),
                parent_span_id: Some(SpanId::new("span.wave").unwrap()),
                async_links: vec![SpanId::new("span.prepare").unwrap()],
            };
            let key = ParticipantNodeKey::new(
                BatchParticipantAuthority::new(
                    SequenceAuthorityId::test_only(slot, 2),
                    RequestAuthorityId::test_only(slot + 20, 4),
                ),
                frame,
                fixture.node_id.clone(),
            );
            fixture
                .participants
                .push(BatchOperationParticipantIdentity::new(
                    4 + fixture.participants.len() as u32,
                    key,
                    ExecutionIdentityEnvelope::new(parts).unwrap(),
                ));
        }
        fixture
    }

    fn node(&self) -> Result<BatchOperationNodeIdentity, VNextError> {
        BatchOperationNodeIdentity::from_validated(
            self.node_index,
            self.node_id.clone(),
            self.operation_id.clone(),
            self.provider_id.clone(),
            self.provider_fingerprint.clone(),
            self.semantics,
            self.work_fingerprint.clone(),
            self.participants.clone(),
        )
    }

    fn change_parts(&mut self, change: impl FnOnce(&mut ExecutionIdentityParts)) {
        let participant = &self.participants[0];
        let mut parts = participant.identity().parts().clone();
        change(&mut parts);
        self.participants[0] = BatchOperationParticipantIdentity::new(
            participant.participant_index(),
            participant.node_key().clone(),
            ExecutionIdentityEnvelope::new(parts).unwrap(),
        );
    }
}

// The old eager format is represented independently of the production node
// serializer and fingerprint helper. In particular, participant Arc wrappers
// are bypassed and the pre-existing flat envelope fields are serialized.
#[derive(Serialize)]
struct LegacyParticipant<'a> {
    participant_index: u32,
    node_key: &'a ParticipantNodeKey,
    identity: &'a ExecutionIdentityParts,
}

#[derive(Serialize)]
struct LegacyFingerprintInput<'a> {
    domain: &'static str,
    node_index: u32,
    node_id: &'a NodeId,
    operation_id: &'a OperationId,
    provider_id: &'a ProviderId,
    provider_implementation_fingerprint: &'a str,
    provider_execution_semantics: ProviderExecutionSemantics,
    work_shape_fingerprint: &'a str,
    participants: &'a [LegacyParticipant<'a>],
}

#[derive(Serialize)]
struct LegacyNodeWire<'a> {
    node_index: u32,
    node_id: &'a NodeId,
    operation_id: &'a OperationId,
    provider_id: &'a ProviderId,
    provider_implementation_fingerprint: &'a str,
    provider_execution_semantics: ProviderExecutionSemantics,
    work_shape_fingerprint: &'a str,
    participants: &'a [LegacyParticipant<'a>],
    fingerprint: &'a str,
}

fn legacy_evidence(fixture: &Fixture) -> (String, Vec<u8>) {
    let participants = fixture
        .participants
        .iter()
        .map(|participant| LegacyParticipant {
            participant_index: participant.participant_index(),
            node_key: participant.node_key(),
            identity: participant.identity().parts(),
        })
        .collect::<Vec<_>>();
    let fingerprint_bytes = serde_json::to_vec(&LegacyFingerprintInput {
        domain: "ferrum.runtime-vnext.batch-operation-node-identity.v2",
        node_index: fixture.node_index,
        node_id: &fixture.node_id,
        operation_id: &fixture.operation_id,
        provider_id: &fixture.provider_id,
        provider_implementation_fingerprint: &fixture.provider_fingerprint,
        provider_execution_semantics: fixture.semantics,
        work_shape_fingerprint: &fixture.work_fingerprint,
        participants: &participants,
    })
    .unwrap();
    let fingerprint = format!("{:x}", Sha256::digest(fingerprint_bytes));
    let wire_bytes = serde_json::to_vec(&LegacyNodeWire {
        node_index: fixture.node_index,
        node_id: &fixture.node_id,
        operation_id: &fixture.operation_id,
        provider_id: &fixture.provider_id,
        provider_implementation_fingerprint: &fixture.provider_fingerprint,
        provider_execution_semantics: fixture.semantics,
        work_shape_fingerprint: &fixture.work_fingerprint,
        participants: &participants,
        fingerprint: &fingerprint,
    })
    .unwrap();
    (fingerprint, wire_bytes)
}

#[test]
fn lazy_node_fingerprint_preserves_cold_and_warm_clone_equality() {
    let fixture = Fixture::new();
    let node = fixture.node().unwrap();
    let cold_clone = node.clone();
    assert!(node.fingerprint.get().is_none());
    assert!(cold_clone.fingerprint.get().is_none());
    assert_eq!(node, cold_clone);
    assert!(node.fingerprint.get().is_none());

    let expected = legacy_evidence(&fixture).0;
    assert_eq!(node.fingerprint(), expected);
    assert!(cold_clone.fingerprint.get().is_none());
    assert_eq!(node, cold_clone);
    let warm_clone = node.clone();
    assert_eq!(warm_clone.fingerprint.get(), Some(&expected));
    assert_eq!(warm_clone, cold_clone);
    assert_eq!(cold_clone.fingerprint(), expected);

    let mut different = Fixture::new();
    different.work_fingerprint = "f".repeat(64);
    assert_ne!(node, different.node().unwrap());
}

#[test]
fn lazy_node_fingerprint_and_wire_match_independent_legacy_bytes() {
    let fixture = Fixture::new();
    let (expected_digest, expected_wire) = legacy_evidence(&fixture);
    let node = fixture.node().unwrap();
    assert!(node.fingerprint.get().is_none());
    assert_eq!(serde_json::to_vec(&node).unwrap(), expected_wire);
    assert_eq!(node.fingerprint(), expected_digest);
    assert_eq!(serde_json::to_vec(&node.clone()).unwrap(), expected_wire);
}

#[test]
fn lazy_node_construction_still_rejects_invalid_participants_and_digests() {
    type Mutation = fn(&mut Fixture);
    let cases: &[(&str, Mutation)] = &[
        ("empty", |f| f.participants.clear()),
        ("index gap", |f| {
            let p = &f.participants[1];
            f.participants[1] = BatchOperationParticipantIdentity::new(
                p.participant_index() + 1,
                p.node_key().clone(),
                p.identity().clone(),
            );
        }),
        ("noncanonical order", |f| {
            f.participants.reverse();
            for (index, p) in f.participants.iter_mut().enumerate() {
                *p = BatchOperationParticipantIdentity::new(
                    4 + index as u32,
                    p.node_key().clone(),
                    p.identity().clone(),
                );
            }
        }),
        ("duplicate key", |f| {
            let p = &f.participants[0];
            f.participants[1] = BatchOperationParticipantIdentity::new(
                p.participant_index() + 1,
                p.node_key().clone(),
                p.identity().clone(),
            );
        }),
        ("frame projection", |f| {
            f.change_parts(|p| p.frame_id = Some(ExecutionFrameId::try_from(99_u64).unwrap()))
        }),
        ("node projection", |f| {
            f.change_parts(|p| p.node_id = Some(NodeId::new("node.other").unwrap()))
        }),
        ("operation projection", |f| {
            f.change_parts(|p| p.operation_id = Some(OperationId::new("operation.other").unwrap()))
        }),
        ("provider projection", |f| {
            f.change_parts(|p| p.provider_id = Some(ProviderId::new("provider.other").unwrap()))
        }),
        ("provider digest case", |f| {
            f.provider_fingerprint = "A".repeat(64)
        }),
        ("provider digest length", |f| {
            f.provider_fingerprint.pop().map(|_| ()).unwrap()
        }),
        ("work digest hex", |f| f.work_fingerprint = "g".repeat(64)),
        ("work digest length", |f| f.work_fingerprint.clear()),
    ];
    for (name, mutate) in cases {
        let mut fixture = Fixture::new();
        mutate(&mut fixture);
        assert!(
            matches!(fixture.node(), Err(VNextError::InvalidExecutionPlan { .. })),
            "invalid {name} must be rejected before fingerprint access",
        );
    }
}

#[test]
fn lazy_node_concurrent_first_serialize_and_fingerprint_are_consistent() {
    let fixture = Fixture::new();
    let expected = legacy_evidence(&fixture);
    let node = fixture.node().unwrap();
    let barrier = Barrier::new(4);
    assert!(node.fingerprint.get().is_none());
    std::thread::scope(|scope| {
        let handles = (0..4)
            .map(|worker| {
                let node = &node;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    if worker % 2 == 0 {
                        let digest = node.fingerprint().to_owned();
                        (digest, serde_json::to_vec(node).unwrap())
                    } else {
                        let bytes = serde_json::to_vec(node).unwrap();
                        (node.fingerprint().to_owned(), bytes)
                    }
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
    assert_eq!(node.fingerprint.get(), Some(&expected.0));
}
