//! Optional delayed expansion for a composed, fixed stateless graph.
//! Operands, arithmetic, weights and every node survive unchanged. Storage reuse
//! and execution ordering are still checked by the ordinary Core compiler.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorProgramSchedule {
    #[default]
    DeclarationOrder,
    /// Postpone non-shrinking intermediates until their first consumer while
    /// keeping reductions/quantization early. The compiled memory plan, not
    /// this heuristic, determines whether the result fits device capacity.
    DelayedExpansion,
    /// Prefer ready operations that retire larger intermediates, then delay
    /// expansions. Core still proves ordering, alias safety and actual bytes.
    EarlyCompaction,
    /// Drain unshared producer/consumer chains before starting peer branches.
    /// Shared frontiers still use compaction priority; expansions remain late.
    ConsumerChains,
}

impl StoredComposition {
    pub fn with_schedule(mut self, schedule: TensorProgramSchedule) -> Result<Self, VNextError> {
        if schedule == TensorProgramSchedule::DeclarationOrder {
            return Ok(self);
        }
        let p = &self.program;
        let [block] = p.blocks() else {
            return Err(invalid("delayed expansion requires one composed block"));
        };
        if !p.states().is_empty()
            || p.checkpoint_inputs().is_some()
            || !self.profile.states.is_empty()
            || !self.profile.kv_storage.is_empty()
            || block
                .nodes
                .iter()
                .any(|n| n.work != ProgramNodeWorkSpec::Fixed)
        {
            return Err(invalid("delayed expansion requires fixed stateless work"));
        }
        let nodes = match schedule {
            TensorProgramSchedule::DeclarationOrder => unreachable!("handled above"),
            TensorProgramSchedule::DelayedExpansion => {
                delayed_expansion(&block.nodes, self.options.tensor_specs())?
            }
            TensorProgramSchedule::EarlyCompaction => {
                let ordered = early_compaction(
                    &block.nodes,
                    self.options.tensor_specs(),
                    p.outputs(),
                    false,
                )?;
                delayed_expansion(&ordered, self.options.tensor_specs())?
            }
            TensorProgramSchedule::ConsumerChains => {
                let ordered =
                    early_compaction(&block.nodes, self.options.tensor_specs(), p.outputs(), true)?;
                delayed_expansion(&ordered, self.options.tensor_specs())?
            }
        };
        self.program = ModelProgram::new(
            p.family_id().clone(),
            p.inputs().to_vec(),
            vec![ProgramBlock {
                id: block.id.clone(),
                nodes,
            }],
            vec![],
            p.weights().to_vec(),
            p.outputs().to_vec(),
        )?
        .with_input_tensor_specs(p.input_tensor_specs().clone())?;
        self.profile.validate_program(&self.program)?;
        self.schema
            .validate_program_references(self.program.family_id(), &self.program)?;
        Ok(self)
    }
}

fn early_compaction(
    nodes: &[ProgramNode],
    specs: &BTreeMap<ProgramValueId, ProgramTensorSpec>,
    outputs: &[ProgramValueId],
    complete_chains: bool,
) -> Result<Vec<ProgramNode>, VNextError> {
    let producers = nodes
        .iter()
        .enumerate()
        .flat_map(|(i, node)| node.outputs.iter().map(move |v| (v, i)))
        .collect::<BTreeMap<_, _>>();
    let retained = outputs.iter().collect::<BTreeSet<_>>();
    let inputs = nodes
        .iter()
        .map(|node| node.inputs.iter().collect::<BTreeSet<_>>())
        .collect::<Vec<_>>();
    let mut uses = BTreeMap::<&ProgramValueId, usize>::new();
    let mut pending = vec![0; nodes.len()];
    let mut successors = vec![Vec::new(); nodes.len()];
    for (index, input) in inputs.iter().enumerate() {
        let mut dependencies = BTreeSet::new();
        for &value in input {
            *uses.entry(value).or_default() += 1;
            if let Some(&producer) = producers.get(value) {
                dependencies.insert(producer);
            }
        }
        pending[index] = dependencies.len();
        for producer in dependencies {
            successors[producer].push(index);
        }
    }
    let mut ready = pending
        .iter()
        .enumerate()
        .filter_map(|(i, &count)| (count == 0).then_some(i))
        .collect::<BTreeSet<_>>();
    let bytes = |value: &ProgramValueId| -> Result<i128, VNextError> {
        Ok(specs
            .get(value)
            .map(ProgramTensorSpec::byte_len)
            .transpose()?
            .map_or(0, i128::from))
    };
    let mut ordered = Vec::with_capacity(nodes.len());
    let mut continuation = None;
    while !ready.is_empty() {
        let mut choice = None;
        for &index in &ready {
            if continuation.is_some_and(|next| index != next) {
                continue;
            }
            let mut score = 0i128;
            for &input in &inputs[index] {
                // External inputs remain uploaded for the whole wave; only
                // a last use of an internal non-output can retire storage.
                if producers.contains_key(input) && !retained.contains(input) && uses[input] == 1 {
                    score += bytes(input)?;
                }
            }
            for output in &nodes[index].outputs {
                score -= bytes(output)?;
            }
            // Preserve declaration order for ties. This is only a heuristic:
            // aliased values and physical pools are charged by Core afterward.
            if choice.is_none_or(|(_, best)| score > best) {
                choice = Some((index, score));
            }
        }
        let (index, _) = choice.expect("nonempty ready set");
        ready.remove(&index);
        ordered.push(nodes[index].clone());
        for input in &inputs[index] {
            *uses.get_mut(input).expect("counted input") -= 1;
        }
        for &next in &successors[index] {
            pending[next] -= 1;
            if pending[next] == 0 {
                ready.insert(next);
            }
        }
        continuation = None;
        if complete_chains && !nodes[index].outputs.iter().any(|v| retained.contains(v)) {
            if let [next] = successors[index].as_slice() {
                continuation = if ready.contains(next) {
                    Some(*next)
                } else {
                    // A sole consumer can need one other ready prerequisite,
                    // such as the initial destination of a sequence of writes.
                    // Materialize that dependency instead of accumulating all
                    // peer branches. At a shared frontier, resume compaction.
                    inputs[*next]
                        .iter()
                        .filter_map(|value| producers.get(value).copied())
                        .filter(|index| ready.contains(index))
                        .min()
                };
            }
        }
    }
    if ordered.len() != nodes.len() {
        return Err(invalid("compaction schedule contains cyclic dependencies"));
    }
    Ok(ordered)
}

fn delayed_expansion(
    nodes: &[ProgramNode],
    specs: &BTreeMap<ProgramValueId, ProgramTensorSpec>,
) -> Result<Vec<ProgramNode>, VNextError> {
    let mut consumers = BTreeMap::<&ProgramValueId, Vec<usize>>::new();
    for (i, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            consumers.entry(input).or_default().push(i);
        }
    }
    let mut order = (0..nodes.len()).collect::<Vec<_>>();
    let mut positions = order.clone();
    // Work backwards so a producer sees the final placement of its consumers.
    // Moving only toward the first consumer preserves every data dependency.
    // Do not sink a compression: retaining its larger input would defeat the
    // intended saving (for example BF16 -> packed quantization -> wide GEMM).
    for index in (0..nodes.len()).rev() {
        let extent = |values: &[ProgramValueId]| -> Result<u128, VNextError> {
            values
                .iter()
                .collect::<BTreeSet<_>>()
                .into_iter()
                // Stored weight operands are already resident and are absent
                // from activation geometry; moving a node cannot retire them.
                .filter_map(|v| specs.get(v))
                .try_fold(0u128, |total, tensor| {
                    Ok(total + u128::from(tensor.byte_len()?))
                })
        };
        let current = positions[index];
        let output_bytes = extent(&nodes[index].outputs)?;
        let input_bytes = extent(&nodes[index].inputs)?;
        let first_consumer = nodes[index]
            .outputs
            .iter()
            .filter_map(|value| consumers.get(value))
            .flatten()
            .map(|&consumer| positions[consumer])
            .min()
            .unwrap_or(nodes.len());
        if output_bytes < input_bytes {
            continue;
        }
        // The validated ModelProgram is topological; this is a reschedule of
        // all its nodes, not a pruning pass or a bypass of Core alias checks.
        let target = first_consumer - 1;
        if target > current {
            order.remove(current);
            order.insert(target, index);
            for position in current..=target {
                positions[order[position]] = position;
            }
        }
    }
    Ok(order
        .into_iter()
        .map(|index| nodes[index].clone())
        .collect())
}
