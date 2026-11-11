use super::*;
use serde::Deserializer;
/// A consecutive target interval uses one explicitly selected curve-basis row.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PH3ImageFinalLayerTimeRun {
    length: u64,
    time_index: u64,
}
impl PH3ImageFinalLayerTimeRun {
    pub fn new(length: u64, time_index: u64) -> Result<Self, VNextError> {
        if length == 0 {
            return Err(invalid("target time run must be positive"));
        }
        Ok(Self { length, time_index })
    }
    pub const fn length(&self) -> u64 {
        self.length
    }
    pub const fn time_index(&self) -> u64 {
        self.time_index
    }
}
/// The original packed document is [conditioning prefix | audio | target image].
/// Prefix rows include text and references. They never enter a velocity head.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PH3ImageFinalLayerRequest {
    prefix_rows: u64,
    audio: Vec<PH3ImageFinalLayerTimeRun>,
    image: Vec<PH3ImageFinalLayerTimeRun>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    prefix_rows: u64,
    audio: Vec<PH3ImageFinalLayerTimeRun>,
    image: Vec<PH3ImageFinalLayerTimeRun>,
}
impl<'de> Deserialize<'de> for PH3ImageFinalLayerRequest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = RequestWire::deserialize(d)?;
        Self::new(w.prefix_rows, w.audio, w.image).map_err(serde::de::Error::custom)
    }
}
impl PH3ImageFinalLayerRequest {
    pub fn new(
        prefix_rows: u64,
        audio: Vec<PH3ImageFinalLayerTimeRun>,
        image: Vec<PH3ImageFinalLayerTimeRun>,
    ) -> Result<Self, VNextError> {
        let r = Self {
            prefix_rows,
            audio,
            image,
        };
        r.validate()?;
        Ok(r)
    }
    pub fn from_layout(
        layout: PH3ImageRequestLayout<'_>,
        time_indices: &[u64],
    ) -> Result<Self, VNextError> {
        if time_indices.len() != layout.rows() {
            return Err(invalid("time-index rows differ from packed request"));
        }
        let runs =
            |range: std::ops::Range<usize>| -> Result<Vec<PH3ImageFinalLayerTimeRun>, VNextError> {
                let mut runs: Vec<PH3ImageFinalLayerTimeRun> = Vec::new();
                for &time_index in &time_indices[range] {
                    if let Some(last) = runs.last_mut().filter(|r| r.time_index == time_index) {
                        last.length = last
                            .length
                            .checked_add(1)
                            .ok_or_else(|| invalid("target run overflows"))?;
                    } else {
                        runs.push(PH3ImageFinalLayerTimeRun::new(1, time_index)?);
                    }
                }
                Ok(runs)
            };
        Self::new(
            layout.audio_rows().start as u64,
            runs(layout.audio_rows())?,
            runs(layout.target_rows())?,
        )
    }
    pub const fn prefix_rows(&self) -> u64 {
        self.prefix_rows
    }
    pub fn audio_runs(&self) -> &[PH3ImageFinalLayerTimeRun] {
        &self.audio
    }
    pub fn image_runs(&self) -> &[PH3ImageFinalLayerTimeRun] {
        &self.image
    }
    pub fn audio_rows(&self) -> Result<u64, VNextError> {
        sum_runs(&self.audio)
    }
    pub fn image_rows(&self) -> Result<u64, VNextError> {
        sum_runs(&self.image)
    }
    pub fn input_rows(&self) -> Result<u64, VNextError> {
        self.prefix_rows
            .checked_add(self.audio_rows()?)
            .and_then(|n| n.checked_add(self.image_rows().ok()?))
            .ok_or_else(|| invalid("packed request rows overflow"))
    }
    fn validate(&self) -> Result<(), VNextError> {
        if self.prefix_rows == 0
            || self.audio_rows()? != PH3ImageRequestLayout::AUDIO_ROWS as u64
            || self.image_rows()? == 0
        {
            return Err(invalid("requires a conditioning prefix, the declared four audio rows and positive target-image rows"));
        }
        for runs in [&self.audio, &self.image] {
            if runs.is_empty()
                || runs.iter().any(|r| r.length == 0)
                || runs.windows(2).any(|p| p[0].time_index == p[1].time_index)
            {
                return Err(invalid("time routing must be positive canonical RLE"));
            }
        }
        self.input_rows()?;
        Ok(())
    }
}
fn sum_runs(runs: &[PH3ImageFinalLayerTimeRun]) -> Result<u64, VNextError> {
    runs.iter().try_fold(0u64, |sum, r| {
        sum.checked_add(r.length)
            .ok_or_else(|| invalid("target row count overflows"))
    })
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PH3ImageFinalLayerConfig {
    pub(super) hidden: u64,
    pub(super) image_width: u64,
    pub(super) audio_width: u64,
    pub(super) basis: u64,
    pub(super) grid: u64,
    pub(super) basis_rows: u64,
    pub(super) epsilon: CanonicalRational,
    pub(super) requests: Vec<PH3ImageFinalLayerRequest>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    hidden: u64,
    image_width: u64,
    audio_width: u64,
    basis: u64,
    grid: u64,
    basis_rows: u64,
    epsilon: CanonicalRational,
    requests: Vec<PH3ImageFinalLayerRequest>,
}
impl<'de> Deserialize<'de> for PH3ImageFinalLayerConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = Wire::deserialize(d)?;
        let c = Self {
            hidden: w.hidden,
            image_width: w.image_width,
            audio_width: w.audio_width,
            basis: w.basis,
            grid: w.grid,
            basis_rows: w.basis_rows,
            epsilon: w.epsilon,
            requests: w.requests,
        };
        c.validate().map_err(serde::de::Error::custom)?;
        Ok(c)
    }
}
impl PH3ImageFinalLayerConfig {
    pub fn new(
        t: &PH3ImageTransformerConfig,
        curve: PH3ImageCurveTimeEmbedding,
        basis_rows: u64,
        requests: Vec<PH3ImageFinalLayerRequest>,
    ) -> Result<Self, VNextError> {
        let c = Self {
            hidden: t.hidden_size(),
            image_width: t.patch_input_dim(),
            audio_width: t.audio_in_channels(),
            basis: curve.basis_dim() as u64,
            grid: curve.grid_points() as u64,
            basis_rows,
            epsilon: CanonicalRational::from_decimal_str(&t.final_norm_eps().to_string())?,
            requests,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn requests(&self) -> &[PH3ImageFinalLayerRequest] {
        &self.requests
    }
    pub const fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub const fn image_velocity_width(&self) -> u64 {
        self.image_width
    }
    pub const fn audio_velocity_width(&self) -> u64 {
        self.audio_width
    }
    pub const fn basis_rows(&self) -> u64 {
        self.basis_rows
    }
    pub fn input_rows(&self) -> Result<u64, VNextError> {
        self.requests.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.input_rows()?)
                .ok_or_else(|| invalid("input rows overflow"))
        })
    }
    pub fn image_rows(&self) -> Result<u64, VNextError> {
        self.requests.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.image_rows()?)
                .ok_or_else(|| invalid("image rows overflow"))
        })
    }
    pub fn audio_rows(&self) -> Result<u64, VNextError> {
        self.requests.iter().try_fold(0u64, |n, r| {
            n.checked_add(r.audio_rows()?)
                .ok_or_else(|| invalid("audio rows overflow"))
        })
    }
    pub fn input_tensor_specs(
        &self,
    ) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        self.validate()?;
        Ok(BTreeMap::from([
            (
                ProgramValueId::new(PH3ImageFinalLayerFamily::INPUT_ID)?,
                tensor(vec![self.input_rows()?, self.hidden], ElementType::Bf16),
            ),
            (
                ProgramValueId::new(PH3ImageFinalLayerFamily::BASIS_ID)?,
                tensor(vec![self.basis_rows, self.basis], ElementType::F32),
            ),
        ]))
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        program::compile_options(self)
    }
    pub(super) fn validate(&self) -> Result<(), VNextError> {
        if [
            self.hidden,
            self.image_width,
            self.audio_width,
            self.basis_rows,
            self.basis,
            self.grid,
        ]
        .contains(&0)
            || self.requests.is_empty()
        {
            return Err(invalid(
                "all final dimensions and request inventory must be positive",
            ));
        }
        self.hidden
            .checked_mul(2)
            .ok_or_else(|| invalid("parameter width overflows"))?;
        for request in &self.requests {
            request.validate()?;
            for run in request.audio_runs().iter().chain(request.image_runs()) {
                if run.time_index >= self.basis_rows
                    || i64::try_from(run.time_index).is_err()
                    || i64::try_from(run.length).is_err()
                {
                    return Err(invalid(
                        "time routing lies outside declared basis rows or attribute extent",
                    ));
                }
            }
        }
        PH3ImageCurveTimeEmbedding::new(
            usize::try_from(self.grid).map_err(|_| invalid("grid exceeds address space"))?,
            usize::try_from(self.basis).map_err(|_| invalid("basis exceeds address space"))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        let x = ResolvedTensorSpec::new(
            vec![1, self.hidden],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let gamma = ResolvedTensorSpec::new(
            vec![self.hidden],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        RowRmsNormSingleRoundBf16Geometry::new(&x, &gamma, &x, self.epsilon)?;
        for shape in [
            vec![self.input_rows()?, self.hidden],
            vec![self.image_rows()?, self.image_width],
            vec![self.audio_rows()?, self.audio_width],
        ] {
            tensor(shape, ElementType::Bf16).validate("final layer tensor")?;
        }
        Ok(())
    }
}
pub(super) fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
