use super::*;
use ferrum_bench_core::images::ImageSample;
use serde_json::{Map, Value};
use std::{collections::BTreeSet, io::Read, path::Path, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComfyWorkflowSpec {
    pub path: PathBuf,
    pub expected_sha256: Option<String>,
    /// One explicit node whose history output must contain exactly one PNG.
    pub output_node: String,
    pub steps: ComfySteps,
    pub bindings: Vec<ComfyBinding>,
    /// Exact original graph IDs required by endpoint execution_audit mode.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_execution_nodes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComfySteps {
    /// Declared workflow setting; never inferred from a model or node name.
    Fixed {
        count: u32,
    },
    Bound {
        default: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyInputValue {
    Prompt,
    Seed,
    Width,
    Height,
    UploadedImage,
    OutputPrefix,
    Steps,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComfyBinding {
    pub node: String,
    pub input: String,
    pub value: ComfyInputValue,
    /// Validate the declared graph role without model-name routing.
    pub expected_class: String,
}

#[derive(Debug)]
pub(super) struct PreparedWorkflow {
    graph: Arc<Map<String, Value>>,
    spec: ComfyWorkflowSpec,
    pub(super) sha256: String,
}

pub(super) fn read_bounded(path: &Path, maximum: usize) -> anyhow::Result<Vec<u8>> {
    ensure!(
        maximum > 0 && maximum < isize::MAX as usize,
        "metadata byte limit must fit the address space"
    );
    let file = std::fs::File::open(path).context("open protocol metadata")?;
    let metadata = file.metadata().context("inspect protocol metadata")?;
    ensure!(
        metadata.is_file() && metadata.len() <= maximum as u64,
        "protocol metadata is not a regular file within its byte limit"
    );
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(metadata.len() as usize)
        .context("reserve protocol metadata")?;
    file.take(maximum as u64 + 1)
        .read_to_end(&mut bytes)
        .context("read protocol metadata")?;
    ensure!(
        bytes.len() <= maximum,
        "protocol metadata exceeds byte limit"
    );
    Ok(bytes)
}

impl PreparedWorkflow {
    pub(super) fn contains_node(&self, node: &str) -> bool {
        self.graph.contains_key(node)
    }

    pub(super) fn required_execution_nodes(&self) -> &[String] {
        &self.spec.required_execution_nodes
    }

    pub(super) fn validate_execution_audit(&self) -> anyhow::Result<()> {
        let mut unique = BTreeSet::new();
        ensure!(
            !self.spec.required_execution_nodes.is_empty()
                && self
                    .spec
                    .required_execution_nodes
                    .iter()
                    .all(|node| !node.is_empty() && node.len() <= 256 && unique.insert(node)),
            "Comfy execution audit requires unique nonempty original node IDs within 256 bytes"
        );
        let mut closure = BTreeSet::new();
        let mut pending = vec![self.spec.output_node.as_str()];
        while let Some(id) = pending.pop() {
            if !closure.insert(id) {
                continue;
            }
            let inputs = self
                .graph
                .get(id)
                .and_then(|node| node.get("inputs"))
                .and_then(Value::as_object)
                .context("Comfy output dependency references an absent node")?;
            for value in inputs.values() {
                if let Some(link) = value
                    .as_array()
                    .filter(|link| link.len() == 2 && link[1].as_u64().is_some())
                {
                    if let Some(parent) = link[0].as_str() {
                        pending.push(parent);
                    }
                }
            }
        }
        ensure!(
            self.spec
                .required_execution_nodes
                .iter()
                .all(|node| closure.contains(node.as_str())),
            "Comfy audited nodes must belong to the declared PNG output dependency closure"
        );
        Ok(())
    }

    pub(super) fn load(
        spec: &ComfyWorkflowSpec,
        edit: bool,
        limits: &ComfyProtocolLimits,
    ) -> anyhow::Result<Self> {
        let bytes = read_bounded(&spec.path, limits.maximum_workflow_bytes)?;
        let digest = sha256(&bytes);
        ensure!(
            spec.expected_sha256
                .as_ref()
                .is_none_or(|expected| expected == &digest),
            "Comfy workflow SHA256 differs from the declaration"
        );
        let value: Value = serde_json::from_slice(&bytes).context("invalid Comfy workflow JSON")?;
        let graph = value
            .get("prompt")
            .unwrap_or(&value)
            .as_object()
            .context("Comfy workflow requires an API prompt object, not a UI graph")?;
        ensure!(
            !graph.is_empty() && graph.contains_key(&spec.output_node),
            "Comfy workflow has no declared output node"
        );
        for (id, node) in graph {
            ensure!(
                !id.is_empty()
                    && id.len() <= 256
                    && node
                        .get("class_type")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
                    && node.get("inputs").is_some_and(Value::is_object),
                "Comfy API node requires an identity, class_type and inputs object"
            );
        }
        let mut targets = BTreeSet::new();
        let mut roles = BTreeSet::new();
        for binding in &spec.bindings {
            ensure!(
                !binding.node.is_empty()
                    && !binding.input.is_empty()
                    && targets.insert((binding.node.clone(), binding.input.clone())),
                "Comfy input bindings require unique non-empty targets"
            );
            let node = graph
                .get(&binding.node)
                .context("Comfy binding references an absent node")?;
            ensure!(
                node.get("class_type").and_then(Value::as_str)
                    == Some(binding.expected_class.as_str()),
                "Comfy binding class differs from the declared graph contract"
            );
            let original = node
                .get("inputs")
                .and_then(|v| v.get(&binding.input))
                .context("Comfy binding references an absent input")?;
            let string = matches!(
                binding.value,
                ComfyInputValue::Prompt
                    | ComfyInputValue::UploadedImage
                    | ComfyInputValue::OutputPrefix
            );
            ensure!(
                if string {
                    original.is_string()
                } else {
                    original.as_u64().is_some()
                },
                "Comfy binding must replace a literal of its declared type, not a graph link"
            );
            roles.insert(binding.value);
        }
        for role in [
            ComfyInputValue::Prompt,
            ComfyInputValue::Seed,
            ComfyInputValue::Width,
            ComfyInputValue::Height,
        ] {
            ensure!(
                roles.contains(&role),
                "Comfy workflow is missing a dataset input binding"
            );
        }
        ensure!(
            roles.contains(&ComfyInputValue::UploadedImage) == edit,
            "Comfy uploaded-image binding must match the request mode"
        );
        match spec.steps {
            ComfySteps::Fixed { count } => ensure!(
                count > 0 && !roles.contains(&ComfyInputValue::Steps),
                "fixed Comfy steps require a positive count and no steps binding"
            ),
            ComfySteps::Bound { default } => ensure!(
                default > 0 && roles.contains(&ComfyInputValue::Steps),
                "bound Comfy steps require a positive default and actual input binding"
            ),
        }
        Ok(Self {
            graph: Arc::new(graph.clone()),
            spec: spec.clone(),
            sha256: digest,
        })
    }

    pub(super) fn validate_sample(&self, sample: &ImageSample) -> anyhow::Result<()> {
        if let (ComfySteps::Fixed { count }, Some(requested)) =
            (&self.spec.steps, sample.num_inference_steps)
        {
            ensure!(
                *count == requested,
                "dataset inference steps differ from the fixed Comfy workflow declaration"
            );
        }
        Ok(())
    }

    pub(super) fn bind(
        &self,
        sample: &ImageSample,
        uploaded: Option<&str>,
        request: &str,
    ) -> Result<Value, RequestFailure> {
        let mut graph = self.graph.as_ref().clone();
        for binding in &self.spec.bindings {
            let value = match binding.value {
                ComfyInputValue::Prompt => Value::String(sample.prompt.clone()),
                ComfyInputValue::Seed => Value::from(sample.seed),
                ComfyInputValue::Width => Value::from(sample.width),
                ComfyInputValue::Height => Value::from(sample.height),
                ComfyInputValue::UploadedImage => Value::String(
                    uploaded
                        .ok_or((
                            ImageFailureKind::ClientTask,
                            "Comfy image binding has no completed upload",
                        ))?
                        .into(),
                ),
                ComfyInputValue::OutputPrefix => Value::String(format!("ferrum-bench/{request}")),
                ComfyInputValue::Steps => {
                    Value::from(sample.num_inference_steps.unwrap_or(match self.spec.steps {
                        ComfySteps::Bound { default } => default,
                        ComfySteps::Fixed { count } => count,
                    }))
                }
            };
            // Shape, target and type were validated once against this immutable graph.
            graph
                .get_mut(&binding.node)
                .and_then(|node| node.get_mut("inputs"))
                .and_then(Value::as_object_mut)
                .ok_or((
                    ImageFailureKind::ClientTask,
                    "prepared Comfy graph contract changed",
                ))?
                .insert(binding.input.clone(), value);
        }
        Ok(Value::Object(graph))
    }

    pub(super) fn output_node(&self) -> &str {
        &self.spec.output_node
    }
}
