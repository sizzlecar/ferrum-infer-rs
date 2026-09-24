use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationReferenceCurve {
    pub total_prompt_tokens: NonZeroU32,
    pub input_tokens_sha256: [u8; 32],
    pub partition: Vec<ProfileWaveShapeV2>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationReferencePlan {
    pub reference_revision: NonZeroU64,
    pub protocol: ReferenceProtocolV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub piecewise: Option<PiecewiseReferenceSpec>,
    pub decode_input_tokens: NonZeroU32,
    pub decode_input_tokens_sha256: [u8; 32],
    pub curves: Vec<CalibrationReferenceCurve>,
    pub limits: SloPrefillReferenceLimits,
}
impl CalibrationReferencePlan {
    pub fn protocol_sha256(&self) -> std::result::Result<[u8; 32], ReferenceError> {
        match &self.piecewise {
            Some(spec) => spec.protocol_sha256(&self.protocol),
            None => self.protocol.sha256(),
        }
    }

    pub(super) fn validate_discovery(
        &self,
        session: &Arc<()>,
        discovery: &[CalibrationReferenceDiscoverySample],
    ) -> Result<profile::ProfileFingerprint> {
        self.limits.validate().map_err(invalid)?;
        if self.curves.is_empty()
            || self.curves.len() > self.limits.max_curves.get()
            || discovery.is_empty()
            || discovery.len() > self.limits.max_samples.get()
        {
            return Err(invalid("reference discovery exceeds curve/sample limits"));
        }
        let fingerprint =
            profile::ProfileFingerprint::from(&discovery[0].witness.sample.fingerprint);
        // Reuse protocol validation before retaining any trial.
        ReferenceCalibrationBuilder::new(
            self.reference_revision,
            fingerprint.clone(),
            self.protocol.clone(),
            NonZeroU64::MIN,
            self.limits.clone(),
        )
        .map_err(reference_error)?;
        if let Some(spec) = &self.piecewise {
            spec.validate().map_err(reference_error)?;
            let minimum = self.curves.iter().map(|row| row.total_prompt_tokens).min();
            let maximum = self.curves.iter().map(|row| row.total_prompt_tokens).max();
            if minimum != Some(spec.minimum_prompt_tokens)
                || maximum != Some(spec.maximum_prompt_tokens)
            {
                return Err(invalid(
                    "piecewise domain requires actual boundary input anchors",
                ));
            }
        }
        let mut expected = 1_usize;
        let mut points = 0_usize;
        let mut lengths = std::collections::BTreeSet::new();
        for curve in &self.curves {
            let segments = match &self.piecewise {
                Some(spec) => spec
                    .segment_count(curve.total_prompt_tokens.get())
                    .map_err(reference_error)? as u64,
                None => u64::from(curve.total_prompt_tokens.get())
                    .div_ceil(u64::from(self.protocol.granule_tokens.get())),
            };
            if curve.input_tokens_sha256 == [0; 32]
                || !lengths.insert(curve.total_prompt_tokens)
                || curve.partition.len() as u64 != segments
                || curve
                    .partition
                    .len()
                    .checked_add(1)
                    .is_none_or(|n| n > self.limits.max_points_per_curve.get())
            {
                return Err(invalid(
                    "reference curve must freeze all original-input endpoints",
                ));
            }
            let mut offset = 0_u32;
            for shape in &curve.partition {
                let count = match &self.piecewise {
                    Some(spec) => spec
                        .next_count(curve.total_prompt_tokens.get(), offset)
                        .map_err(reference_error)?
                        .get(),
                    None => self
                        .protocol
                        .granule_tokens
                        .get()
                        .min(curve.total_prompt_tokens.get() - offset),
                };
                if shape.exact.prefill_chunks.len() != 1
                    || !shape.exact.decode_kv_tokens.is_empty()
                    || shape.exact.prefill_chunks[0].offset != offset
                    || shape.exact.prefill_chunks[0].count.get() != count
                    || shape.exact.prefill_chunks[0].total_prompt_tokens
                        != curve.total_prompt_tokens
                {
                    return Err(invalid(
                        "reference partition differs from frozen granule/full input",
                    ));
                }
                offset = offset
                    .checked_add(count)
                    .ok_or_else(|| invalid("reference endpoint overflow"))?;
            }
            expected = expected
                .checked_add(curve.partition.len())
                .ok_or_else(|| invalid("reference sample overflow"))?;
            points = points
                .checked_add(curve.partition.len() + 1)
                .ok_or_else(|| invalid("reference point overflow"))?;
        }
        if points > self.limits.max_points.get()
            || expected != discovery.len()
            || expected
                .checked_mul(self.protocol.repetitions.get() + 1)
                .is_none_or(|n| n > self.limits.max_samples.get())
            || serde_json::to_vec(self).map_err(json_error)?.len()
                > self.limits.max_file_bytes.get()
        {
            return Err(invalid(
                "reference needs exactly one discovery per frozen shape within its budgets",
            ));
        }
        let mut ordinals = std::collections::BTreeSet::new();
        if discovery.iter().any(|row| {
            !Arc::ptr_eq(&row.session, session)
                || profile::ProfileFingerprint::from(&row.witness.sample.fingerprint) != fingerprint
                || !ordinals.insert(row.witness.accepted)
        }) {
            return Err(invalid(
                "reference discovery source/session identity changed",
            ));
        }
        let mut used = std::collections::BTreeSet::new();
        for curve in &self.curves {
            for shape in &curve.partition {
                let matches: Vec<_> = discovery
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| {
                        row.input.original_input_tokens == curve.total_prompt_tokens.get() as usize
                            && row.input.original_input_tokens_sha256 == curve.input_tokens_sha256
                            && row.witness.matches(shape, self.protocol.prefill_host)
                    })
                    .map(|(index, _)| index)
                    .collect();
                if matches.len() != 1 || !used.insert(matches[0]) {
                    return Err(invalid(
                        "prefill partition was not uniquely observed during discovery",
                    ));
                }
            }
        }
        let decode: Vec<_> = discovery
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                row.input.original_input_tokens == self.decode_input_tokens.get() as usize
                    && row.input.original_input_tokens_sha256 == self.decode_input_tokens_sha256
                    && row
                        .witness
                        .matches(&self.protocol.decode_shape, self.protocol.decode_host)
            })
            .map(|(index, _)| index)
            .collect();
        if decode.len() != 1 || !used.insert(decode[0]) || used.len() != discovery.len() {
            return Err(invalid(
                "decode reference was not uniquely observed during discovery",
            ));
        }
        Ok(fingerprint)
    }
}
