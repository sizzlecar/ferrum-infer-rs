//! Explicit diagnostic scope, generated from the complete pinned GGUF inventory.
//! Header descriptors are not a payload checksum: the caller declares the full
//! model SHA already independently verified, and both captures must bind it.
use super::*;
use ferrum_interfaces::vnext::{ModelArtifactSourceRole, ProductModelSourceIdentity};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Scope {
    MetalPairedQ4GateUpSharedTailV1,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub schema_version: u32,
    pub scope: Scope,
    pub inventory: VNextTeacherRawArtifact,
    pub model_sha256: String,
    pub model_bytes: u64,
}

// Exact fields exported by ferrum-quantization/examples/gguf_inventory.rs.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InventoryReport {
    file: String,
    file_length_source: String,
    local_file_bytes: u64,
    payload_verified: bool,
    tensor_payloads_materialized: bool,
    inventory: Inventory,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inventory {
    schema_version: u32,
    architecture: String,
    quantization_version: Option<u64>,
    hadamard: Option<Value>,
    declared_file_bytes: u64,
    tensor_data_offset: u64,
    tensor_payload_bytes: u64,
    split: Option<Value>,
    tensor_counts_by_dtype: BTreeMap<String, usize>,
    tensors: Vec<Tensor>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Tensor {
    name: String,
    dtype: String,
    ggml_type: u32,
    candle_dtype_available: bool,
    quantization_format: Option<String>,
    dimensions: Vec<u64>,
    block_axis: usize,
    logical_values_per_block: u64,
    bytes_per_block: u64,
    elements: u64,
    absolute_offset: u64,
    bytes: u64,
}
#[derive(Debug, Clone, Serialize)]
pub(super) struct LeafSet {
    pub layer: usize,
    pub hidden: u64,
    pub intermediate: u64,
    pub eligible_weights: bool,
}
impl LeafSet {
    pub fn work(&self, rows: u64) -> Result<(u64, u64)> {
        let h = self.hidden;
        let i = self.intermediate;
        let logical = i.checked_mul(3).and_then(|n| n.checked_add(h));
        let inner = h
            .checked_mul(i)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(i));
        Ok((
            logical
                .and_then(|n| n.checked_mul(rows))
                .context("logical work overflow")?,
            inner
                .and_then(|n| n.checked_mul(rows))
                .context("inner work overflow")?,
        ))
    }
}

pub(super) fn load(
    path: &Path,
    captures: [&capture::CheckedCapture; 2],
) -> Result<(Value, BTreeMap<String, LeafSet>)> {
    let bytes = bounded_file(path)?;
    let config: Config = serde_json::from_slice(&bytes)?;
    ensure!(
        config.schema_version == 1,
        "unsupported native audit configuration"
    );
    ensure!(
        files::canonical_sha(&config.model_sha256) && config.model_bytes > 0,
        "incomplete full GGUF content pin"
    );
    for capture in captures {
        let identity = capture
            .manifest
            .identity
            .as_ref()
            .context("missing capture identity")?;
        let source: ProductModelSourceIdentity =
            serde_json::from_value(identity.model_source.clone())?;
        source.validate().map_err(anyhow::Error::msg)?;
        let weights = &source
            .resolved_sources
            .for_role(ModelArtifactSourceRole::Weights)
            .files;
        ensure!(
            weights.len() == 1
                && weights[0].size_bytes == config.model_bytes
                && weights[0].sha256 == config.model_sha256,
            "native audit GGUF pin differs from actual capture weight source"
        );
    }
    let inventory_bytes =
        bounded_artifact(path.parent().unwrap_or(Path::new(".")), &config.inventory)?;
    let nodes = inventory_nodes(&inventory_bytes, config.model_bytes)?;
    Ok((
        json!({"configuration_sha256":sha256(&bytes),"scope":config.scope,
        "inventory":config.inventory,"model_sha256":config.model_sha256,"model_bytes":config.model_bytes,
        "binding":"declared_independently_verified_full_file_sha_matched_to_both_actual_model_sources",
        "payload_rehashed_by_this_audit":false,"inventory_producer":"ferrum-quantization/examples/gguf_inventory.rs"}),
        nodes,
    ))
}

pub(super) fn inventory_nodes(bytes: &[u8], model_bytes: u64) -> Result<BTreeMap<String, LeafSet>> {
    let report: InventoryReport = serde_json::from_slice(bytes)?;
    ensure!(
        !report.file.is_empty()
            && report.file_length_source == "local_file"
            && report.local_file_bytes == model_bytes
            && !report.payload_verified
            && !report.tensor_payloads_materialized,
        "unsupported GGUF inventory producer envelope"
    );
    let inventory = report.inventory;
    ensure!(
        inventory.schema_version == 1
            && inventory.architecture == "qwen35"
            && inventory.hadamard.is_none()
            && inventory.split.is_none()
            && inventory.declared_file_bytes == model_bytes
            && inventory.tensor_data_offset < model_bytes,
        "unsupported transformed/split GGUF inventory scope"
    );
    let _ = inventory.quantization_version;
    let mut names = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut payload = 0_u64;
    let mut layers = BTreeMap::<usize, BTreeMap<String, Tensor>>::new();
    for tensor in inventory.tensors {
        ensure!(names.insert(tensor.name.clone()), "duplicate GGUF tensor");
        ensure!(
            tensor.bytes > 0
                && tensor.absolute_offset >= inventory.tensor_data_offset
                && tensor
                    .absolute_offset
                    .checked_add(tensor.bytes)
                    .is_some_and(|end| end <= model_bytes),
            "GGUF tensor payload range outside pinned file"
        );
        payload = payload
            .checked_add(tensor.bytes)
            .context("GGUF payload byte overflow")?;
        *counts.entry(tensor.dtype.clone()).or_default() += 1;
        // Do not select an eligible subset first: every gate/up/down tensor is
        // accounted for and every observed FFN node must join this full set.
        if let Some(rest) = tensor.name.strip_prefix("blk.") {
            if let Some((layer, role)) = rest.split_once(".ffn_") {
                if matches!(role, "gate.weight" | "up.weight" | "down.weight") {
                    let index: usize = layer.parse().context("invalid FFN layer index")?;
                    ensure!(layer == index.to_string(), "noncanonical FFN layer index");
                    layers
                        .entry(index)
                        .or_default()
                        .insert(role.to_owned(), tensor);
                }
            }
        }
    }
    ensure!(
        payload == inventory.tensor_payload_bytes && counts == inventory.tensor_counts_by_dtype,
        "GGUF tensor inventory population/count differs"
    );
    ensure!(!layers.is_empty(), "GGUF inventory has no dense FFN layers");
    let mut nodes = BTreeMap::new();
    for (expected, (layer, tensors)) in layers.into_iter().enumerate() {
        ensure!(
            layer == expected && tensors.len() == 3,
            "GGUF FFN inventory omits a layer or leaf"
        );
        let gate = tensors.get("gate.weight").context("missing gate leaf")?;
        let up = tensors.get("up.weight").context("missing up leaf")?;
        let down = tensors.get("down.weight").context("missing down leaf")?;
        for tensor in [gate, up, down] {
            let (type_id, block_bytes, format) = match tensor.dtype.as_str() {
                "Q4K" => (12, 144, "quantization.gguf.q4-k"),
                "Q6K" => (14, 210, "quantization.gguf.q6-k"),
                _ => anyhow::bail!("unsupported FFN weight format in native audit scope"),
            };
            ensure!(
                tensor.dimensions.len() == 2
                    && !tensor.dimensions.contains(&0)
                    && tensor.block_axis == 1
                    && tensor.logical_values_per_block == 256
                    && tensor.bytes_per_block == block_bytes
                    && tensor.ggml_type == type_id
                    && tensor.candle_dtype_available
                    && tensor.quantization_format.as_deref() == Some(format)
                    && tensor.dimensions[0].checked_mul(tensor.dimensions[1])
                        == Some(tensor.elements)
                    && tensor.dimensions[1].is_multiple_of(256)
                    && (tensor.elements / 256).checked_mul(block_bytes) == Some(tensor.bytes),
                "inconsistent FFN tensor physical ABI"
            );
        }
        let i = gate.dimensions[0];
        let h = gate.dimensions[1];
        ensure!(
            gate.dtype == "Q4K"
                && up.dtype == "Q4K"
                && up.dimensions == gate.dimensions
                && down.dimensions == [h, i]
                && h >= 1024
                && i >= 1024
                && i.is_multiple_of(256),
            "FFN leaves outside paired-Q4 diagnostic scope"
        );
        nodes.insert(
            format!("node.layer.{layer}.feed_forward"),
            LeafSet {
                layer,
                hidden: h,
                intermediate: i,
                eligible_weights: down.dtype == "Q6K",
            },
        );
    }
    ensure!(
        nodes.values().any(|leaf| leaf.eligible_weights),
        "no eligible Q6-down FFN in inventory"
    );
    Ok(nodes)
}
