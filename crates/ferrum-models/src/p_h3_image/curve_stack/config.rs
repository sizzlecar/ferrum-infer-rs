use super::*;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "layout",
    content = "projections",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PH3ImageStackProjectionStorage {
    Uniform(PH3ImageBlockProjectionStorage),
    PerLayer(Vec<PH3ImageBlockProjectionStorage>),
}
impl Default for PH3ImageStackProjectionStorage {
    fn default() -> Self {
        Self::Uniform(PH3ImageBlockProjectionStorage::default())
    }
}
impl PH3ImageStackProjectionStorage {
    pub fn all_w4(&self) -> bool {
        match self {
            Self::Uniform(p) => p.all_w4(),
            Self::PerLayer(p) => p.iter().all(|p| p.all_w4()),
        }
    }
    pub(super) fn get(&self, index: u64) -> Result<PH3ImageBlockProjectionStorage, VNextError> {
        match self {
            Self::Uniform(p) => Ok(*p),
            Self::PerLayer(p) => p
                .get(usize::try_from(index).map_err(|_| invalid("layer exceeds address space"))?)
                .copied()
                .ok_or_else(|| invalid("missing per-layer projection storage")),
        }
    }
    pub(super) fn unique(&self) -> BTreeSet<PH3ImageBlockProjectionStorage> {
        match self {
            Self::Uniform(p) => BTreeSet::from([*p]),
            Self::PerLayer(p) => p.iter().copied().collect(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PH3ImageCurveStackConfig {
    pub(super) block: PH3ImageCurveBlockConfig,
    pub(super) layers: u64,
    pub(super) projection_storage: PH3ImageStackProjectionStorage,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    block: PH3ImageCurveBlockConfig,
    layers: u64,
    #[serde(default)]
    projection_storage: PH3ImageStackProjectionStorage,
}
impl<'de> Deserialize<'de> for PH3ImageCurveStackConfig {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        let c = Self {
            block: w.block,
            layers: w.layers,
            projection_storage: w.projection_storage,
        };
        c.validate().map_err(serde::de::Error::custom)?;
        Ok(c)
    }
}
impl PH3ImageCurveStackConfig {
    pub fn new(
        t: &PH3ImageTransformerConfig,
        curve: PH3ImageCurveTimeEmbedding,
        basis_rows: u64,
        requests: Vec<PH3ImageCurveBlockRequest>,
    ) -> Result<Self, VNextError> {
        let c = Self {
            block: PH3ImageCurveBlockConfig::new(t, curve, 0, basis_rows, requests)?,
            layers: t.layers(),
            projection_storage: PH3ImageStackProjectionStorage::default(),
        };
        c.validate()?;
        Ok(c)
    }
    pub fn layers(&self) -> u64 {
        self.layers
    }
    pub fn with_projection_storage(
        mut self,
        storage: PH3ImageStackProjectionStorage,
    ) -> Result<Self, VNextError> {
        self.block = self.block.with_projection_storage(storage.get(0)?)?;
        self.projection_storage = storage;
        self.validate()?;
        Ok(self)
    }
    pub fn storage_for_layer(
        &self,
        layer: u64,
    ) -> Result<PH3ImageBlockProjectionStorage, VNextError> {
        if layer >= self.layers {
            return Err(invalid("projection layer outside stack"));
        }
        self.projection_storage.get(layer)
    }
    pub fn from_checkpoint(
        t: &PH3ImageTransformerConfig,
        curve: PH3ImageCurveTimeEmbedding,
        basis_rows: u64,
        requests: Vec<PH3ImageCurveBlockRequest>,
        source: &ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource,
        limits: PH3ImageCurveStackLimits,
    ) -> anyhow::Result<Self> {
        limits.validate()?;
        let c = Self::new(t, curve, basis_rows, requests)?;
        let count = usize::try_from(c.layers)?;
        anyhow::ensure!(
            count
                .checked_mul(std::mem::size_of::<PH3ImageBlockProjectionStorage>())
                .and_then(|n| n.checked_add(c.metadata_bytes().ok()?))
                .is_some_and(|n| n <= limits.max_metadata_bytes),
            "per-layer projection metadata exceeds declared capacity"
        );
        let mut storage = Vec::new();
        storage.try_reserve_exact(count)?;
        for layer in 0..c.layers {
            storage.push(PH3ImageBlockProjectionStorage::from_checkpoint(
                source, layer,
            )?);
        }
        c.with_projection_storage(PH3ImageStackProjectionStorage::PerLayer(storage))
            .map_err(Into::into)
    }
    pub fn block_template(&self) -> &PH3ImageCurveBlockConfig {
        &self.block
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        self.block
            .input_tensor_specs()?
            .into_iter()
            .map(|(id, tensor)| {
                Ok((
                    if id.as_str() == PH3ImageCurveBlockFamily::INPUT_ID {
                        ProgramValueId::new(PH3ImageCurveStackFamily::INPUT_ID)?
                    } else {
                        id
                    },
                    tensor,
                ))
            })
            .collect()
    }
    pub fn compile_options(
        &self,
        limits: PH3ImageCurveStackLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        PH3ImageCurveStackFamily::new(limits)?.compile_options(self)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if self.layers == 0
            || usize::try_from(self.layers).is_err()
            || self.block.block_index() != 0
        {
            return Err(invalid(
                "full stack requires nonzero addressable layers and a block-zero template",
            ));
        }
        if matches!(&self.projection_storage,PH3ImageStackProjectionStorage::PerLayer(p) if u64::try_from(p.len()).ok()!=Some(self.layers))
            || self.block.projection_storage() != self.projection_storage.get(0)?
        {
            return Err(invalid("per-layer projection storage must exactly cover all layers and match template zero"));
        }
        Ok(())
    }
    pub(super) fn metadata_bytes(&self) -> Result<usize, VNextError> {
        let heap = self.block.requests().iter().try_fold(0usize, |sum, r| {
            r.runs()
                .len()
                .checked_mul(std::mem::size_of::<super::super::PH3ImageCurveBlockRowRun>())
                .and_then(|n| sum.checked_add(n))
                .and_then(|n| n.checked_add(std::mem::size_of::<PH3ImageCurveBlockRequest>()))
                .ok_or_else(|| invalid("routing metadata overflows"))
        })?;
        let storage = match &self.projection_storage {
            PH3ImageStackProjectionStorage::Uniform(_) => 0,
            PH3ImageStackProjectionStorage::PerLayer(p) => p
                .len()
                .checked_mul(std::mem::size_of::<PH3ImageBlockProjectionStorage>())
                .ok_or_else(|| invalid("projection metadata bytes overflow"))?,
        };
        heap.checked_add(std::mem::size_of::<Self>())
            .and_then(|n| n.checked_add(storage))
            .ok_or_else(|| invalid("configuration metadata overflows"))
    }
}
