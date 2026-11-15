//! Identity lookups over a validated immutable program. Deserialization goes
//! through ModelProgram::new, so no caller can supply or retain a stale index.
use super::*;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ProgramIndex {
    nodes: BTreeMap<NodeId, (usize, usize)>,
    states: BTreeMap<ProgramValueId, usize>,
    weights: BTreeMap<ProgramValueId, usize>,
}

impl ProgramIndex {
    pub(super) fn new(
        blocks: &[ProgramBlock],
        states: &[StateSpec],
        weights: &[WeightReference],
    ) -> Self {
        Self {
            nodes: blocks
                .iter()
                .enumerate()
                .flat_map(|(block, value)| {
                    value
                        .nodes
                        .iter()
                        .enumerate()
                        .map(move |(node, value)| (value.id.clone(), (block, node)))
                })
                .collect(),
            states: states
                .iter()
                .enumerate()
                .map(|(i, s)| (s.value_id.clone(), i))
                .collect(),
            weights: weights
                .iter()
                .enumerate()
                .map(|(i, w)| (w.value_id.clone(), i))
                .collect(),
        }
    }
}

impl ModelProgram {
    pub fn node(&self, id: &NodeId) -> Option<&ProgramNode> {
        let &(block, node) = self.index.nodes.get(id)?;
        Some(&self.blocks[block].nodes[node])
    }

    pub fn state_for_value(&self, id: &ProgramValueId) -> Option<&StateSpec> {
        self.index.states.get(id).map(|&i| &self.states[i])
    }

    pub fn weight_for_value(&self, id: &ProgramValueId) -> Option<&WeightReference> {
        self.index.weights.get(id).map(|&i| &self.weights[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::vnext_core_contract::{TestConfig, TestFamily};

    fn id<T: TryFrom<String>>(value: &str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value.to_owned()).unwrap()
    }

    #[test]
    fn lookups_preserve_multiblock_identity_after_sort_clone_and_wire_roundtrip() {
        let family = TypedFamilyRegistration::new(TestFamily)
            .prepare_with_profile(
                &serde_json::to_value(TestConfig { width: 4 }).unwrap(),
                &id("fixture.f32"),
            )
            .unwrap();
        let original = family.program();
        let mut states = original.states().to_vec();
        let mut state = states[0].clone();
        state.id = id("state.alpha");
        state.value_id = id("value.z_state");
        states.push(state);
        let mut weights = original.weights().to_vec();
        let mut weight = weights[0].clone();
        weight.weight_id = id("weight.alpha");
        weight.value_id = id("value.z_weight");
        weights.push(weight);
        let mut blocks = original.blocks().to_vec();
        let mut second = blocks[0].nodes[0].clone();
        second.id = id("node.alpha");
        second.inputs = vec![
            id("value.output"),
            id("value.z_state"),
            id("value.z_weight"),
        ];
        second.outputs = vec![id("value.final")];
        blocks.push(ProgramBlock {
            id: "block.alpha".into(),
            nodes: vec![second],
        });
        let program = ModelProgram::new(
            original.family_id().clone(),
            original.inputs().to_vec(),
            blocks,
            states,
            weights,
            vec![id("value.final")],
        )
        .unwrap();
        let wire = serde_json::to_value(&program).unwrap();
        assert!(wire.get("index").is_none());
        let restored: ModelProgram = serde_json::from_value(wire.clone()).unwrap();
        for p in [&program, &program.clone(), &restored] {
            assert_eq!(serde_json::to_value(p).unwrap(), wire);
            assert_eq!(p.fingerprint().unwrap(), program.fingerprint().unwrap());
            for node in p.blocks().iter().flat_map(|block| &block.nodes) {
                assert_eq!(p.node(&node.id), Some(node));
            }
            for value in p
                .inputs()
                .iter()
                .chain(p.outputs())
                .chain(p.weights().iter().map(|w| &w.value_id))
                .chain(p.states().iter().map(|s| &s.value_id))
            {
                assert_eq!(
                    p.weight_for_value(value),
                    p.weights().iter().find(|w| &w.value_id == value)
                );
                assert_eq!(
                    p.state_for_value(value),
                    p.states().iter().find(|s| &s.value_id == value)
                );
            }
            assert!(p.node(&id("node.unknown")).is_none());
            assert!(p.weight_for_value(&id("value.unknown")).is_none());
            assert!(p.state_for_value(&id("value.unknown")).is_none());
        }
        let mut injected = wire;
        injected
            .as_object_mut()
            .unwrap()
            .insert("index".into(), serde_json::json!({"nodes":{}}));
        assert!(serde_json::from_value::<ModelProgram>(injected).is_err());
    }
}
