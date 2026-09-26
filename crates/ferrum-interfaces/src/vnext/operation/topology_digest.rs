//! The original v3 program topology digest, shared by actual and future selection.
use super::foundation::invalid_operation;
use super::*;
use crate::vnext::*;
use sha2::{Digest, Sha256};
pub(super) struct ReusableTopologyDigest {
    digest: Sha256,
}
impl ReusableTopologyDigest {
    pub(super) fn new(nodes: usize) -> Result<Self, VNextError> {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.runtime-vnext.reusable-program-topology.v3\0");
        digest.update(
            u64::try_from(nodes)
                .map_err(|_| invalid_operation("topology count overflow"))?
                .to_le_bytes(),
        );
        Ok(Self { digest })
    }
    pub(super) fn append(
        &mut self,
        wave_node_index: usize,
        plan_node_index: usize,
        node_id: &NodeId,
        provider_id: &ProviderId,
        declared: ProviderReplayEquivalence,
        topology: &ReusableExecutionTopology,
    ) -> Result<(), VNextError> {
        let (topology_kind, topology_bytes) = match (
                declared,
                topology,
            ) {
                (
                    ProviderReplayEquivalence::BitwiseEagerEquivalent,
                    ReusableExecutionTopology::Static,
                ) => (0_u8, &[][..]),
                (
                    ProviderReplayEquivalence::BitwiseEagerEquivalent,
                    ReusableExecutionTopology::Dynamic(topology),
                ) => (1_u8, &topology.as_bytes()[..]),
                (_, ReusableExecutionTopology::EagerBoundary) => (2_u8, &[][..]),
                (
                    ProviderReplayEquivalence::Ineligible,
                    ReusableExecutionTopology::Static | ReusableExecutionTopology::Dynamic(_),
                ) => {
                    return Err(invalid_operation(format!(
                        "provider `{}` returned reusable topology without a bitwise eager-equivalence contract",
                        provider_id
                    )))
                }
            };
        let wave_node_index = u64::try_from(wave_node_index)
            .map_err(|_| invalid_operation("reusable topology wave node index exceeds u64"))?;
        let plan_node_index = u64::try_from(plan_node_index)
            .map_err(|_| invalid_operation("reusable topology plan node index exceeds u64"))?;
        let node_id = node_id.as_str().as_bytes();
        let provider_id = provider_id.as_str().as_bytes();
        let node_id_len = u64::try_from(node_id.len())
            .map_err(|_| invalid_operation("reusable topology node id exceeds u64"))?;
        let provider_id_len = u64::try_from(provider_id.len())
            .map_err(|_| invalid_operation("reusable topology provider id exceeds u64"))?;
        let topology_len = u64::try_from(topology_bytes.len())
            .map_err(|_| invalid_operation("reusable topology payload exceeds u64"))?;
        self.digest.update(wave_node_index.to_le_bytes());
        self.digest.update(plan_node_index.to_le_bytes());
        self.digest.update(node_id_len.to_le_bytes());
        self.digest.update(node_id);
        self.digest.update(provider_id_len.to_le_bytes());
        self.digest.update(provider_id);
        self.digest.update([topology_kind]);
        self.digest.update(topology_len.to_le_bytes());
        self.digest.update(topology_bytes);
        Ok(())
    }
    pub(super) fn finish(self) -> DeviceReusableExecutionTopologyFingerprint {
        DeviceReusableExecutionTopologyFingerprint::from_sha256(self.digest.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn digest(second: &ReusableExecutionTopology) -> DeviceReusableExecutionTopologyFingerprint {
        let mut digest = ReusableTopologyDigest::new(2).unwrap();
        for (n, topology) in [ReusableExecutionTopology::Static, second.clone()]
            .iter()
            .enumerate()
        {
            digest
                .append(
                    n,
                    n,
                    &NodeId::new(format!("node.{n}")).unwrap(),
                    &ProviderId::new(format!("provider.{n}")).unwrap(),
                    ProviderReplayEquivalence::BitwiseEagerEquivalent,
                    topology,
                )
                .unwrap();
        }
        digest.finish()
    }
    #[test]
    fn future_topology_refactor_preserves_the_existing_v3_wire_digest() {
        // Fixed legacy v3 byte encoding (two rows, static then dynamic) proves
        // moving the helper does not invalidate resident program identities.
        let actual = digest(&ReusableExecutionTopology::Dynamic(
            DeviceReusableExecutionTopologyFingerprint::from_sha256([7; 32]),
        ));
        let hex = actual
            .as_bytes()
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect::<String>();
        assert_eq!(
            hex,
            "a87bb8382d767305de247801d1241fe7560f5e48f2629a18eea2be39b6d2bd13"
        );
        assert_ne!(actual, digest(&ReusableExecutionTopology::EagerBoundary));
        assert_ne!(actual, digest(&ReusableExecutionTopology::Static));
    }
    #[test]
    fn future_topology_never_grants_replay_to_an_ineligible_provider() {
        let mut digest = ReusableTopologyDigest::new(1).unwrap();
        assert!(digest
            .append(
                0,
                0,
                &NodeId::new("node.test").unwrap(),
                &ProviderId::new("provider.test").unwrap(),
                ProviderReplayEquivalence::Ineligible,
                &ReusableExecutionTopology::Static
            )
            .is_err());
    }
}
