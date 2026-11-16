//! Validate actual Stored interpretations at standard operation ports. This is
//! a core contract test, not a claim that the CUDA providers executed.
use super::*;

#[test]
fn shared_unit_quantization_and_all_packed_projections_bind_standard_contracts() {
    let (_dir, checkpoint) = open(&metadata(), tensors(), 1, limits());
    let checkpoint = checkpoint.unwrap();
    let c = checkpoint.config();
    let family = Qwen3VLTextNvfp4Family::new().unwrap();
    let graph = program::build(c, family.family_id()).unwrap();
    let schema = c.schema().unwrap();
    let options = c.compile_options().unwrap();
    let mut shapes = options.tensor_specs().clone();
    for weight in graph.weights() {
        shapes.insert(weight.value_id.clone(), weight.tensor.clone());
    }
    for p in &c.projections {
        let alpha = config::value(&p.alpha_component().unwrap()).unwrap();
        let gemm = graph.blocks()[0]
            .nodes
            .iter()
            .find(|n| {
                n.operation_id.as_str() == NVFP4_MATMUL_BF16_OPERATION_ID && n.inputs[4] == alpha
            })
            .unwrap();
        let quant = graph.blocks()[0]
            .nodes
            .iter()
            .find(|n| n.outputs.contains(&gemm.inputs[0]))
            .unwrap();
        for (id, columns) in [(&quant.inputs[0], p.columns), (&gemm.outputs[0], p.rows)] {
            shapes.insert(
                id.clone(),
                ProgramTensorSpec {
                    dimensions: vec![c.text().layout().rows(), columns],
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            );
        }
    }
    let quantizers: Vec<_> = graph.blocks()[0]
        .nodes
        .iter()
        .filter(|n| n.operation_id.as_str() == NVFP4_QUANTIZE_BF16_OPERATION_ID)
        .collect();
    let consumers: Vec<_> = quantizers
        .iter()
        .map(|q| {
            graph.blocks()[0]
                .nodes
                .iter()
                .filter(|n| {
                    n.operation_id.as_str() == NVFP4_MATMUL_BF16_OPERATION_ID
                        && n.inputs[0] == q.outputs[0]
                })
                .count()
        })
        .collect();
    assert_eq!(consumers, [3, 1, 2, 1]);
    for node in &graph.blocks()[0].nodes {
        let contract = match node.operation_id.as_str() {
            NVFP4_QUANTIZE_BF16_OPERATION_ID => nvfp4_quantize_bf16_contract(),
            NVFP4_MATMUL_BF16_OPERATION_ID => nvfp4_matmul_bf16_contract(),
            _ => continue,
        }
        .unwrap();
        contract
            .descriptor()
            .validate_attributes(&node.attributes)
            .unwrap();
        let mut bindings = Vec::new();
        for (input, ids) in [(true, &node.inputs), (false, &node.outputs)] {
            for (ordinal, id) in ids.iter().enumerate() {
                let spec = &shapes[id];
                let weight =
                    graph.weights().iter().find(|w| &w.value_id == id).map(|w| {
                        ResolvedWeightBinding::from_schema(&schema, &w.weight_id).unwrap()
                    });
                let resource = ResourceId::new(format!("resource.{id}")).unwrap();
                let storage = if let Some(weight) = &weight {
                    let component = schema
                        .components
                        .iter()
                        .find(|c| &c.id == weight.components()[0].component_id())
                        .unwrap();
                    ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
                        Some(component.id.clone()),
                        resource,
                        0,
                        component.physical_bytes().unwrap(),
                        component.physical_element_type(),
                    )
                    .unwrap()])
                    .unwrap()
                } else {
                    ResolvedValueStorage::single(
                        resource,
                        0,
                        spec.byte_len().unwrap(),
                        spec.element_type,
                    )
                    .unwrap()
                };
                bindings.push(
                    ResolvedValueBinding::new(
                        id.clone(),
                        if input {
                            ResolvedValueRole::Input
                        } else {
                            ResolvedValueRole::Output
                        },
                        ordinal as u32,
                        ResolvedTensorSpec::new(
                            spec.dimensions.clone(),
                            spec.element_type,
                            spec.layout.clone(),
                        )
                        .unwrap(),
                        if input {
                            TensorAccess::Read
                        } else {
                            TensorAccess::Write
                        },
                        AliasPolicy::NoAlias,
                        if weight.is_some() {
                            BufferUsage::Weights
                        } else {
                            BufferUsage::Activations
                        },
                        weight,
                        storage,
                    )
                    .unwrap(),
                );
            }
        }
        contract
            .descriptor()
            .validate_resolved_bindings(&bindings)
            .unwrap();
    }
}
