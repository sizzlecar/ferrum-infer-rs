use super::*;

fn value(name: &str) -> ProgramValueId {
    ProgramValueId::new(format!("value.{name}")).unwrap()
}
fn projection(name: &str, input: &str, outputs: u64) -> ProgramNode {
    ProgramNode {
        id: NodeId::new(format!("node.{name}")).unwrap(),
        operation_id: OperationId::new(DENSE_LINEAR_FP8_BF16_OPERATION_ID).unwrap(),
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs: vec![value(input), value(&format!("weight.{name}"))],
        outputs: vec![value(name)],
        attributes: BTreeMap::from([
            (
                AttributeId::new("in_features").unwrap(),
                SemanticValue::Unsigned(16),
            ),
            (
                AttributeId::new("out_features").unwrap(),
                SemanticValue::Unsigned(outputs),
            ),
        ]),
    }
}
fn fixture(blocks: Vec<Vec<ProgramNode>>, state_input: bool) -> ModelProgram {
    let nodes = blocks.iter().flatten().collect::<Vec<_>>();
    let weights = nodes
        .iter()
        .filter(|node| node.inputs.len() == 2)
        .map(|node| {
            let SemanticValue::Unsigned(outputs) =
                node.attributes[&AttributeId::new("out_features").unwrap()]
            else {
                unreachable!()
            };
            WeightReference {
                weight_id: WeightId::new(format!("weight.{}", node.id.as_str())).unwrap(),
                value_id: node.inputs[1].clone(),
                tensor: ProgramTensorSpec {
                    dimensions: vec![outputs, 16],
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }
        })
        .collect();
    let states = if state_input {
        vec![StateSpec {
            id: StateId::new("state.input").unwrap(),
            value_id: value("input"),
            tensor: ProgramTensorSpec {
                dimensions: vec![2, 16],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
            lifetime: StateLifetime::Request,
            capacity_demand: StateCapacityDemand::FixedPerScope,
            initialization: StateInitialization::None,
            checkpoint: StateCheckpointCapability::Unsupported,
        }]
    } else {
        Vec::new()
    };
    let outputs = nodes.iter().flat_map(|node| node.outputs.clone()).collect();
    ModelProgram::new(
        ModelFamilyId::new("family.fp8.shared-quant.fixture").unwrap(),
        if state_input {
            vec![value("request")]
        } else {
            vec![value("input")]
        },
        blocks
            .into_iter()
            .enumerate()
            .map(|(index, nodes)| ProgramBlock {
                id: format!("block.{index}"),
                nodes,
            })
            .collect(),
        states,
        weights,
        outputs,
    )
    .unwrap()
}

#[test]
fn nonadjacent_projections_preserve_outputs_and_preceding_input_dependencies() {
    let mut consume_q = projection("consume_q", "q", 16);
    consume_q.operation_id = OperationId::new("operation.fixture.consume").unwrap();
    consume_q.inputs.truncate(1);
    consume_q.attributes.clear();
    let original = fixture(
        vec![vec![
            projection("q", "input", 16),
            consume_q,
            projection("k", "input", 16),
            projection("v", "input", 16),
            projection("gate", "consume_q", 16),
            projection("up", "consume_q", 16),
        ]],
        false,
    );
    let fused = share_input_quantization(original.clone()).unwrap();
    let nodes = &fused.blocks()[0].nodes;
    let qkv = nodes
        .iter()
        .find(|node| node.operation_id.as_str() == DENSE_LINEAR_FP8_TRIPLE_BF16_OPERATION_ID)
        .unwrap();
    assert_eq!(
        qkv.inputs,
        [
            value("input"),
            value("weight.q"),
            value("weight.k"),
            value("weight.v")
        ]
    );
    assert_eq!(qkv.outputs, [value("q"), value("k"), value("v")]);
    let pair_index = nodes
        .iter()
        .position(|node| node.operation_id.as_str() == DENSE_LINEAR_FP8_PAIR_BF16_OPERATION_ID)
        .unwrap();
    let producer_index = nodes
        .iter()
        .position(|node| node.outputs == [value("consume_q")])
        .unwrap();
    assert!(producer_index < pair_index);
    assert_eq!(
        nodes[pair_index].inputs,
        [value("consume_q"), value("weight.gate"), value("weight.up")]
    );
    assert_eq!(nodes[pair_index].outputs, [value("gate"), value("up")]);
    assert_eq!(fused.inputs(), original.inputs());
    assert_eq!(fused.outputs(), original.outputs());
    assert_eq!(fused.weights(), original.weights());
    assert_eq!(fused.states(), original.states());
    assert_eq!(share_input_quantization(fused.clone()).unwrap(), fused);
}

#[test]
fn fusion_never_crosses_blocks_state_backing_or_shape_contracts() {
    let separate = fixture(
        vec![
            vec![projection("q", "input", 16)],
            vec![projection("k", "input", 16)],
        ],
        false,
    );
    assert_eq!(
        share_input_quantization(separate.clone()).unwrap(),
        separate
    );
    let state = fixture(
        vec![vec![
            projection("q", "input", 16),
            projection("k", "input", 16),
        ]],
        true,
    );
    assert_eq!(share_input_quantization(state.clone()).unwrap(), state);
    let different_shape = fixture(
        vec![vec![
            projection("q", "input", 16),
            projection("k", "input", 32),
        ]],
        false,
    );
    assert_eq!(
        share_input_quantization(different_shape.clone()).unwrap(),
        different_shape
    );
    let mut token_work = projection("k", "input", 16);
    token_work.work = ProgramNodeWorkSpec::tokens(value("input"), 0);
    let different_work = fixture(vec![vec![projection("q", "input", 16), token_work]], false);
    assert_eq!(
        share_input_quantization(different_work.clone()).unwrap(),
        different_work
    );
}
