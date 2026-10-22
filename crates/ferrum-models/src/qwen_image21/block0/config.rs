use super::*;
use serde::{Deserialize, Deserializer, Serialize};

/// Explicit conditioning geometry. Elementwise inputs contain the already
/// selected t0/target rows; Row supplies one common row to the whole sequence.
/// This choice does not derive conditioning from the attention visibility mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenImage21Block0Broadcast {
    Elementwise,
    Row,
}
impl QwenImage21Block0Broadcast {
    pub const fn mode(self) -> RowBroadcast {
        match self {
            Self::Elementwise => RowBroadcast::Elementwise,
            Self::Row => RowBroadcast::Row,
        }
    }
    pub const fn as_str(self) -> &'static str {
        self.mode().as_str()
    }
}

/// Checked configuration for a complete block, not a complete diffusion model.
/// Four conditioning inputs are raw scales/gates with the same explicit layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QwenImage21Block0Config {
    attention: QwenImage21AttentionConfig,
    broadcast: QwenImage21Block0Broadcast,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    attention: QwenImage21AttentionConfig,
    broadcast: QwenImage21Block0Broadcast,
}
impl<'de> Deserialize<'de> for QwenImage21Block0Config {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(d)?;
        Self::new(wire.attention, wire.broadcast).map_err(serde::de::Error::custom)
    }
}

impl QwenImage21Block0Config {
    pub fn new(
        attention: QwenImage21AttentionConfig,
        broadcast: QwenImage21Block0Broadcast,
    ) -> Result<Self, VNextError> {
        let config = Self {
            attention,
            broadcast,
        };
        config.validate()?;
        Ok(config)
    }
    pub fn attention(&self) -> &QwenImage21AttentionConfig {
        &self.attention
    }
    pub fn transformer(&self) -> &QwenImage21Config {
        self.attention.transformer()
    }
    pub const fn broadcast(&self) -> QwenImage21Block0Broadcast {
        self.broadcast
    }

    /// Shapes enter the family program identity and are checked by the compiler
    /// against supplied input specs before resource admission or weight loading.
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        let rows = self.attention.layout().rows();
        let hidden = self.transformer().hidden_size();
        let parameter_rows = match self.broadcast {
            QwenImage21Block0Broadcast::Elementwise => rows,
            QwenImage21Block0Broadcast::Row => 1,
        };
        let matrix = |rows| ProgramTensorSpec {
            dimensions: vec![rows, hidden],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        };
        let mut specs = BTreeMap::from([(
            ProgramValueId::new(QwenImage21Block0Family::INPUT_ID)?,
            matrix(rows),
        )]);
        for name in [
            QwenImage21Block0Family::SCALE1_ID,
            QwenImage21Block0Family::GATE1_ID,
            QwenImage21Block0Family::SCALE2_ID,
            QwenImage21Block0Family::GATE2_ID,
        ] {
            specs.insert(ProgramValueId::new(name)?, matrix(parameter_rows));
        }
        specs.insert(
            ProgramValueId::new(QwenImage21Block0Family::FREQUENCIES_ID)?,
            ProgramTensorSpec {
                dimensions: vec![
                    u64::from(self.attention.layout().sequence_length()),
                    self.transformer().head_dim() / 2,
                    2,
                ],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        );
        for (id, spec) in &specs {
            spec.validate(id.as_str())?;
        }
        Ok(specs)
    }

    pub(super) fn validate(&self) -> Result<(), VNextError> {
        // The nested type owns its attention validation; reconstruct through its
        // public checked constructor instead of duplicating mask/scale rules.
        let checked = QwenImage21AttentionConfig::new(
            self.transformer().clone(),
            self.attention.layout().clone(),
        )?;
        if checked != self.attention {
            return Err(invalid(
                "attention",
                "nested attention config is not canonical",
            ));
        }
        let specs = self.input_tensor_specs()?;
        let input = &specs[&ProgramValueId::new(QwenImage21Block0Family::INPUT_ID)?];
        let resolved = ResolvedTensorSpec::new(
            input.dimensions.clone(),
            input.element_type,
            input.layout.clone(),
        )?;
        LayerNormBf16Geometry::new(
            &resolved,
            &resolved,
            self.transformer().hidden_size(),
            self.transformer().epsilon(),
        )?;
        Ok(())
    }
}
