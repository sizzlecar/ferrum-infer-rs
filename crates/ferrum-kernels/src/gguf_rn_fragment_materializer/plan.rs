use super::*;
use crate::gguf_f16_projection_materializer as legacy;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GgufRnFragmentInventoryV1 {
    pub source_schema_fingerprint: String,
    pub execution_schema_fingerprint: String,
    pub weights: Vec<legacy::GgufF16ProjectionWeightInventoryV1>,
    pub source_consumed_bytes: u64,
    pub converted_f16_bytes: u64,
    pub packed_fragment_bytes: u64,
    pub retained_consumed_bytes: u64,
    pub unique_consumed_execution_bytes: u64,
    /// Placement alignment and allocation peaks belong to the resource plan.
    pub includes_placement_alignment: bool,
}
pub(super) struct Prepared {
    pub schema: WeightSchema,
    pub sources: BTreeMap<WeightId, Vec<WeightId>>,
    pub inventory: GgufRnFragmentInventoryV1,
}

pub(super) fn prepare(family: &PreparedModelFamily) -> Result<Prepared, VNextError> {
    prepare_program(family.weight_schema(), family.program())
}

pub(super) fn prepare_program(
    original: &WeightSchema,
    program: &ModelProgram,
) -> Result<Prepared, VNextError> {
    // Keep the existing source schema, exact storage, composite order, numeric
    // conversion and every-consumer checks, including old RN attention roles.
    let mut prepared =
        legacy::plan::prepare_program_with_roles(original, program, |operation, ordinal| {
            gguf_rn_f16_fragment_role_v1(operation, ordinal)
                .or_else(|| gguf_f16_projection_role_v1(operation, ordinal))
        })?;
    let mut packets = BTreeMap::<WeightId, (WeightComponentSpec, Vec<WeightId>)>::new();
    let mut dual_weights = BTreeSet::new();
    for weight in &prepared.inventory.weights {
        let is_fragment = |consumer: &legacy::GgufF16ProjectionConsumerV1| {
            gguf_rn_f16_fragment_role_v1(&consumer.operation, consumer.input_ordinal).is_some()
        };
        if !weight.consumers.iter().any(is_fragment) {
            continue;
        }
        if !weight.consumers.iter().all(is_fragment) {
            return Err(invalid(
                "RN fragment logical weight also has a consumer requiring another physical ABI",
            ));
        }
        dual_weights.insert(weight.logical_weight.clone());
        let tensor = prepared
            .schema
            .tensors
            .iter_mut()
            .find(|t| t.id == weight.logical_weight)
            .ok_or_else(|| invalid("RN fragment logical tensor absent"))?;
        let PhysicalWeightLayout::Dense {
            component_id: dense_id,
        } = &tensor.physical_layout
        else {
            return Err(invalid(
                "RN fragment source did not produce the original dense RN representation",
            ));
        };
        let dense_id = dense_id.clone();
        let ids = prepared
            .sources
            .get(&dense_id)
            .ok_or_else(|| invalid("RN fragment source order absent"))?
            .clone();
        let sources = ids
            .iter()
            .map(|id| {
                original
                    .components
                    .iter()
                    .find(|s| &s.id == id)
                    .ok_or_else(|| invalid("RN fragment original source component absent"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (packet_plan, component) = conversion::fragment_component(&sources)?;
        if RnF16FragmentPlanV1::from_dimensions(packet_plan.source_format(), &tensor.dimensions)?
            != packet_plan
        {
            return Err(invalid(
                "RN fragment combined source shape differs from logical projection",
            ));
        }
        let packet_id = component.id.clone();
        if original.components.iter().any(|c| c.id == packet_id) {
            return Err(invalid(
                "RN fragment derived identity collides with an original source",
            ));
        }
        if let Some((prior, prior_sources)) = packets.get(&packet_id) {
            if prior != &component || prior_sources != &ids {
                return Err(invalid("RN fragment derived group identity conflict"));
            }
        } else {
            packets.insert(packet_id.clone(), (component, ids));
        }
        tensor.physical_layout = PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
            dense_values: PhysicalWeightComponentBinding {
                component_id: dense_id,
                storage: PhysicalStorageLayout::exact_contiguous(),
            },
            fragment_values: PhysicalWeightComponentBinding {
                component_id: packet_id,
                storage: PhysicalStorageLayout::exact_contiguous(),
            },
            source_format: packet_plan.source_format(),
        };
    }
    if dual_weights.is_empty() {
        return Err(invalid(
            "RN fragment materializer requires the explicit fragment operation",
        ));
    }
    let mut packed_fragment_bytes = 0u64;
    for (id, (component, sources)) in packets {
        packed_fragment_bytes = packed_fragment_bytes
            .checked_add(component.physical_bytes()?)
            .ok_or_else(|| invalid("RN fragment inventory overflow"))?;
        prepared.sources.insert(id, sources);
        prepared.schema.components.push(component);
    }
    prepared.schema.components.sort_by(|a, b| a.id.cmp(&b.id));
    prepared.schema.format_id = WeightFormatId::new(GGUF_RN_FRAGMENT_FORMAT_ID)?;
    prepared.schema.layout_id = WeightLayoutId::new(GGUF_RN_FRAGMENT_LAYOUT_ID)?;
    prepared.schema.version = VERSION;
    prepared.schema.validate(program.family_id())?;
    for weight in &mut prepared.inventory.weights {
        if dual_weights.contains(&weight.logical_weight) {
            weight.execution_components = prepared
                .schema
                .physical_component_refs(&weight.logical_weight)?
                .iter()
                .map(|c| c.id.clone())
                .collect();
            weight.conversion = "authorized_gguf_rn_f16_dense_and_fragment_v1";
        }
    }
    let inventory = GgufRnFragmentInventoryV1 {
        source_schema_fingerprint: prepared.inventory.source_schema_fingerprint,
        execution_schema_fingerprint: prepared.schema.fingerprint()?,
        weights: prepared.inventory.weights,
        source_consumed_bytes: prepared.inventory.source_consumed_bytes,
        converted_f16_bytes: prepared.inventory.converted_f16_bytes,
        packed_fragment_bytes,
        retained_consumed_bytes: prepared.inventory.retained_consumed_bytes,
        unique_consumed_execution_bytes: prepared
            .inventory
            .unique_consumed_execution_bytes
            .checked_add(packed_fragment_bytes)
            .ok_or_else(|| invalid("RN fragment total execution bytes overflow"))?,
        includes_placement_alignment: false,
    };
    Ok(Prepared {
        schema: prepared.schema,
        sources: prepared.sources,
        inventory,
    })
}
