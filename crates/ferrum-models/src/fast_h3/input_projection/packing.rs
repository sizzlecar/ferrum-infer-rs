use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Stream {
    Video,
    Audio,
    RefinedText,
}
impl Stream {
    fn index(self) -> usize {
        match self {
            Self::Video => 0,
            Self::Audio => 1,
            Self::RefinedText => 2,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    stream: Stream,
    rows: u64,
}
/// Exact BF16 row transport. The text operand must already be refined. This
/// does not refine text, create attention isolation, or synthesize padded rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3InputPackingConfig {
    rows: [u64; 3],
    hidden: u64,
    runs: Vec<Run>,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    batches: u64,
}
fn one() -> u64 {
    1
}
fn is_one(value: &u64) -> bool {
    *value == 1
}
impl FastH3InputPackingConfig {
    pub fn from_layout(
        layout: &FastH3PackedLayout,
        hidden: u64,
        limits: FastH3InputLimits,
    ) -> Result<Self, VNextError> {
        Self::from_batched_layout(layout, hidden, 1, limits)
    }
    /// Each input stream contains B equal request segments. The result contains
    /// B complete documents in that same order; transport never mixes requests.
    pub fn from_batched_layout(
        layout: &FastH3PackedLayout,
        hidden: u64,
        batches: u64,
        limits: FastH3InputLimits,
    ) -> Result<Self, VNextError> {
        let mut runs = Vec::<Run>::new();
        for segment in layout.segments() {
            let stream = match super::stream(segment.kind) {
                0 => Stream::Video,
                1 => Stream::Audio,
                _ => Stream::RefinedText,
            };
            let rows = segment.packed_rows.len().try_into().map_err(invalid)?;
            if let Some(last) = runs.last_mut().filter(|r| r.stream == stream) {
                last.rows = last
                    .rows
                    .checked_add(rows)
                    .ok_or_else(|| invalid("packing run overflow"))?;
            } else {
                if runs.len() >= limits.maximum_segments.min(ROW_GATHER_BF16_MAX_RUNS) {
                    return Err(invalid("packing run capacity exceeded"));
                }
                runs.push(Run { stream, rows });
            }
        }
        let c = Self {
            rows: counts(layout)?,
            hidden,
            runs,
            batches,
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn sequence_rows(&self) -> Result<u64, VNextError> {
        self.rows.iter().try_fold(0u64, |n, &r| {
            n.checked_add(r)
                .ok_or_else(|| invalid("packed sequence extent overflow"))
        })
    }
    pub fn hidden_size(&self) -> u64 {
        self.hidden
    }
    pub fn batches(&self) -> u64 {
        self.batches
    }
    pub(in crate::fast_h3) fn same_request_partition(&self, other: &Self) -> bool {
        self.rows == other.rows && self.hidden == other.hidden && self.runs == other.runs
    }
    /// Per-request [video, audio, refined text] extents.
    pub fn request_rows(&self) -> [u64; 3] {
        self.rows
    }
    pub fn total_rows(&self) -> Result<u64, VNextError> {
        self.sequence_rows()?
            .checked_mul(self.batches)
            .ok_or_else(|| invalid("packed batch extent overflow"))
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(self.program()?.1)
    }
    pub(super) fn validate(&self, limits: FastH3InputLimits) -> Result<(), VNextError> {
        limits.validate()?;
        check_rows(&self.rows, limits)?;
        check_rows(&[self.total_rows()?], limits)?;
        self.checked_partition()?;
        if self.runs.len() > limits.maximum_segments
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("packing metadata capacity exceeded"));
        }
        for s in self.program()?.1.values() {
            if s.byte_len()? > isize::MAX as u64 {
                return Err(invalid("packing tensor exceeds address space"));
            }
        }
        Ok(())
    }
    // Public compile_options can also receive a deserialized config directly.
    // Validate its complete partition before constructing specs or run vectors.
    fn checked_partition(&self) -> Result<(u64, u64), VNextError> {
        let total = self.sequence_rows()?;
        let text_video = self.rows[2]
            .checked_add(self.rows[0])
            .ok_or_else(|| invalid("text/video extent overflows"))?;
        if self.batches == 0
            || self.total_rows()? > i32::MAX as u64
            || self.hidden == 0
            || self.hidden > i32::MAX as u64
            || self.rows.iter().any(|&r| r == 0 || r > i32::MAX as u64)
            || self.runs.len() > ROW_GATHER_BF16_MAX_RUNS
            || self.runs.len() < 3
            || self.runs[0].stream != Stream::RefinedText
            || self.runs.last().is_none_or(|r| r.stream != Stream::Video)
            || self.runs[1..]
                .iter()
                .any(|r| r.stream == Stream::RefinedText)
            || self.runs.windows(2).any(|w| w[0].stream == w[1].stream)
        {
            return Err(invalid(
                "packing partition is empty, noncanonical or exceeds capacity",
            ));
        }
        let mut counts = [0u64; 3];
        for r in &self.runs {
            if r.rows == 0 {
                return Err(invalid("empty packing run"));
            }
            let i = r.stream.index();
            counts[i] = counts[i]
                .checked_add(r.rows)
                .ok_or_else(|| invalid("packing row count overflows"))?;
        }
        if counts != self.rows {
            return Err(invalid(
                "packing must cover each original source exactly once",
            ));
        }
        Ok((text_video, total))
    }
    pub(super) fn program(
        &self,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
        type F = FastH3InputPackingFamily;
        let (text_video_rows, total_rows) = self.checked_partition()?;
        let text: ProgramValueId = id(F::REFINED_TEXT_ID)?;
        let video: ProgramValueId = id(F::VIDEO_ID)?;
        let audio: ProgramValueId = id(F::AUDIO_ID)?;
        let first: ProgramValueId = id("value.fast_h3.input_packing.text_video")?;
        let all: ProgramValueId = id("value.fast_h3.input_packing.source_groups")?;
        let output: ProgramValueId = id(F::OUTPUT_ID)?;
        let inputs = BTreeMap::from([
            (
                text.clone(),
                tensor(
                    vec![self.batches * self.rows[2], self.hidden],
                    ElementType::Bf16,
                ),
            ),
            (
                video.clone(),
                tensor(
                    vec![self.batches * self.rows[0], self.hidden],
                    ElementType::Bf16,
                ),
            ),
            (
                audio.clone(),
                tensor(
                    vec![self.batches * self.rows[1], self.hidden],
                    ElementType::Bf16,
                ),
            ),
        ]);
        let mut specs = inputs.clone();
        for (v, rows) in [
            (first.clone(), text_video_rows),
            (all.clone(), total_rows),
            (output.clone(), total_rows),
        ] {
            specs.insert(
                v,
                tensor(vec![self.batches * rows, self.hidden], ElementType::Bf16),
            );
        }
        let concat = || {
            Ok(BTreeMap::from([(
                id("batch")?,
                SemanticValue::Unsigned(self.batches),
            )]))
        };
        let mut offsets = [0u64; 3];
        let base = [self.rows[2], text_video_rows, 0];
        let mut starts = Vec::with_capacity(self.runs.len());
        let mut lengths = Vec::with_capacity(self.runs.len());
        for r in &self.runs {
            let i = r.stream.index();
            starts.push(
                i64::try_from(
                    base[i]
                        .checked_add(offsets[i])
                        .ok_or_else(|| invalid("packing source offset overflows"))?,
                )
                .map_err(invalid)?,
            );
            lengths.push(i64::try_from(r.rows).map_err(invalid)?);
            offsets[i] = offsets[i]
                .checked_add(r.rows)
                .ok_or_else(|| invalid("packing source extent overflows"))?;
        }
        let attrs = BTreeMap::from([
            (id("batch")?, SemanticValue::Unsigned(self.batches)),
            (id("column_start")?, SemanticValue::Unsigned(0)),
            (id("column_count")?, SemanticValue::Unsigned(self.hidden)),
            (id("run_lengths")?, SemanticValue::Integers(lengths)),
            (id("source_starts")?, SemanticValue::Integers(starts)),
            (
                id("source_batch_strides")?,
                SemanticValue::Integers(vec![
                    if self.batches == 1 {
                        0
                    } else {
                        total_rows as i64
                    };
                    self.runs.len()
                ]),
            ),
            (
                id("source_row_strides")?,
                SemanticValue::Integers(vec![1; self.runs.len()]),
            ),
        ]);
        let nodes = vec![
            program::node(
                "input_packing",
                "concat_text_video",
                BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                vec![text.clone(), video.clone()],
                first.clone(),
                concat()?,
            )?,
            program::node(
                "input_packing",
                "concat_audio",
                BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                vec![first, audio.clone()],
                all.clone(),
                concat()?,
            )?,
            program::node(
                "input_packing",
                "reorder",
                ROW_GATHER_BF16_OPERATION_ID,
                vec![all],
                output.clone(),
                attrs,
            )?,
        ];
        Ok((
            ModelProgram::new(
                id(F::FAMILY_ID)?,
                vec![text, video, audio],
                vec![ProgramBlock {
                    id: "block.fast_h3.input_packing".into(),
                    nodes,
                }],
                vec![],
                vec![],
                vec![output],
            )?
            .with_input_tensor_specs(inputs)?,
            specs,
        ))
    }
}
