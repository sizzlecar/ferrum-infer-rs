//! Explicit visual-input mode. Selection covers expanded image feature rows;
//! vision-start/end tokens remain ordinary embeddings and are not scattered.
use super::*;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen3VLTextComfyInputMode {
    #[default]
    Embeddings,
    VisualEmbeddings,
}
impl Qwen3VLTextComfyInputMode {
    pub(super) fn is_embeddings(&self) -> bool {
        *self == Self::Embeddings
    }
}
pub(super) fn validate(
    text: &Qwen3VLTextConfig,
    mode: Qwen3VLTextComfyInputMode,
) -> Result<(), VNextError> {
    match (mode, text.deepstack()) {
        (Qwen3VLTextComfyInputMode::Embeddings, None) => Ok(()),
        (Qwen3VLTextComfyInputMode::VisualEmbeddings, Some(runs))
            if text.architecture().layers() >= DenseFamily::DEEPSTACK_IDS.len() as u64 => {
            runs.validate(text.layout(), text.architecture().hidden_size())?;
            ResolvedTensorSpec::new(vec![runs.rows()?, text.architecture().hidden_size()], ElementType::F32, ResolvedTensorLayout::Contiguous)?;
            Ok(())
        }
        _ => Err(invalid("visual_inputs", "visual injection must be explicitly selected with checked runs and all three decoder injection layers")),
    }
}
pub(super) fn replacement(
    c: &Qwen3VLTextComfyConfig,
) -> Result<Option<(ProgramBlock, ProgramValueId)>, VNextError> {
    let Some(runs) = c.visual_inputs() else {
        return Ok(None);
    };
    let output = ProgramValueId::new("value.qwen3_vl.text.comfy.visual_replaced")?;
    Ok(Some((
        ProgramBlock {
            id: "block.qwen3_vl.text.comfy.visual_embeddings".into(),
            nodes: vec![ProgramNode {
                id: NodeId::new("node.qwen3_vl.text.comfy.visual_embeddings")?,
                operation_id: OperationId::new(ROW_SCATTER_REPLACE_F32_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![
                    ProgramValueId::new(Qwen3VLTextComfyFamily::EMBEDDINGS_ID)?,
                    ProgramValueId::new(Qwen3VLTextComfyFamily::VISUAL_EMBEDDINGS_ID)?,
                ],
                outputs: vec![output.clone()],
                attributes: runs.attributes()?,
            }],
        },
        output,
    )))
}
