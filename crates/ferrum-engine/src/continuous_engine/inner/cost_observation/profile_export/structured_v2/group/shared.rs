//! Source4 coordination. Child numerical states remain the original collector.
//! This writer is the only owner of physical records and durable source bytes.
use super::*;
impl StructuredCalibrationGroupV2 {
    fn drain_metadata(&mut self) -> Vec<Vec<serde_json::Value>> {
        self.children
            .iter_mut()
            .map(|c| c.source.take_staged())
            .collect()
    }
    pub(super) fn flush_common_metadata(&mut self) -> Result<(), ExportError> {
        if self.shared.is_none() {
            return Ok(());
        }
        let mut records = self.drain_metadata();
        if records.iter().all(Vec::is_empty) {
            return Ok(());
        }
        let first = records.remove(0);
        if records.iter().any(|r| r.len() != first.len()) {
            return Err(ExportError::Source(
                "shared child lifecycle cardinality differs",
            ));
        }
        for (i, mut record) in first.into_iter().enumerate() {
            if record["kind"] == "unsubmitted" {
                let mut members = vec![record
                    .as_object_mut()
                    .unwrap()
                    .remove("member")
                    .ok_or(ExportError::Source("missing unsubmitted membership"))?];
                for other in &mut records {
                    members.push(
                        other[i]
                            .as_object_mut()
                            .ok_or(ExportError::Source("invalid shared event"))?
                            .remove("member")
                            .ok_or(ExportError::Source("missing unsubmitted membership"))?,
                    );
                    if other[i] != record {
                        return Err(ExportError::Source("shared unsubmitted call differs"));
                    }
                }
                record["members"] = serde_json::Value::Array(members);
                self.shared.as_mut().unwrap().record(&record)?;
            } else {
                if records.iter().any(|r| r[i] != record) {
                    return Err(ExportError::Source(
                        "shared original lifecycle differs across children",
                    ));
                }
                self.shared
                    .as_mut()
                    .unwrap()
                    .record(&serde_json::json!({"kind":"common","record":record}))?;
            }
        }
        Ok(())
    }
    pub(super) fn flush_shared_completion(
        &mut self,
        members: &[Option<u64>],
    ) -> Result<(), ExportError> {
        let records = self.drain_metadata();
        let mut physical = None;
        let mut terminals = Vec::with_capacity(records.len());
        for (i, records) in records.into_iter().enumerate() {
            let mut completed = Vec::new();
            for mut record in records {
                if record["kind"] == "completed" {
                    if physical.is_some() || record["member"] != serde_json::to_value(members[i])? {
                        return Err(ExportError::Source(
                            "shared physical completion is duplicated or mismatched",
                        ));
                    }
                    record.as_object_mut().unwrap().remove("member");
                    record["members"] = serde_json::to_value(members)?;
                    physical = Some(record);
                } else if record["kind"] == "request_completed" {
                    completed.push(record);
                } else {
                    return Err(ExportError::Source("unexpected shared settlement event"));
                }
            }
            terminals.push(completed);
        }
        if terminals.iter().skip(1).any(|v| v != &terminals[0]) {
            return Err(ExportError::Source(
                "shared original request terminals differ",
            ));
        }
        let source = self.shared.as_mut().unwrap();
        source
            .record(&physical.ok_or(ExportError::Source("shared original completion missing"))?)?;
        for record in &terminals[0] {
            source.record(&serde_json::json!({"kind":"common","record":record}))?;
        }
        for child in &mut self.children {
            child.source.collect_completed(false);
        }
        Ok(())
    }
    pub(super) fn flush_shared_failed_completion(
        &mut self,
        members: &[Option<u64>],
        failures: &[serde_json::Value],
        reason: &str,
    ) -> Result<(), ExportError> {
        let records = self.drain_metadata();
        let mut original = records
            .into_iter()
            .flatten()
            .find(|r| r["kind"] == "completed");
        let source = self.shared.as_mut().unwrap();
        if let Some(record) = &mut original {
            record.as_object_mut().unwrap().remove("member");
            record["members"] = serde_json::to_value(members)?;
            // The original failing payload is retained exactly; absence of a
            // numeric record/sidecar stays explicit and cannot qualify replay.
            source.record(record)?;
        }
        source.record(&serde_json::json!({"kind":"phase_failed","reason":reason,
            "child_failures":failures,"completed_freezes":[]}))
    }
    pub(super) fn freeze_shared(
        &mut self,
        cutoff: u64,
        now: u64,
    ) -> Result<Vec<StructuredPhaseFreezeReceipt>, ExportError> {
        let reports = self
            .children
            .iter()
            .map(|c| c.coverage())
            .collect::<Result<Vec<_>, _>>()?;
        let source = self.shared.as_mut().unwrap();
        source.record(&serde_json::json!({"kind":"coverage","phase":self.children[0].phase,"reports":reports}))?;
        source.flush()?;
        let bytes = source.bytes();
        let digest = source.prefix_digest();
        for child in &mut self.children {
            child.source.set_shared_prefix(bytes, digest);
        }
        // No source phase freeze exists until every child independently succeeds.
        let mut receipts = Vec::with_capacity(self.children.len());
        for (index, child) in self.children.iter_mut().enumerate() {
            match child.freeze_at(cutoff, now) {
                Ok(receipt) => receipts.push(receipt),
                Err(error) => {
                    self.shared.as_mut().unwrap().record(&serde_json::json!({"kind":"phase_failed","reason":error.to_string(),
                        "child_failures":[{"child":index,"capture_identity":child.binding.identity(),"reason":error.to_string()}],
                        "completed_freezes":receipts}))?;
                    return Err(error);
                }
            }
        }
        for (child, receipt) in self.children.iter_mut().zip(&receipts) {
            let records = child.source.take_staged();
            if records != vec![serde_json::json!({"kind":"phase_freeze","receipt":receipt})] {
                return Err(ExportError::Source("shared child freeze differs"));
            }
        }
        self.shared
            .as_mut()
            .unwrap()
            .record(&serde_json::json!({"kind":"phase_freeze","receipts":receipts}))?;
        Ok(receipts)
    }
    pub(super) fn finish_shared(
        mut self,
        cutoff: u64,
        closing: Option<ExportClockReading>,
    ) -> Result<StructuredCalibrationGroupArtifactV2, ExportError> {
        let first = &self.children[0];
        let offered = first.ledger.offered;
        let last_fifo = first.ledger.last_fifo;
        let fingerprint = first.binding.fingerprint().clone();
        let members = self
            .children
            .iter()
            .map(|c| c.ledger.members)
            .collect::<Vec<_>>();
        let failed_members = self
            .children
            .iter()
            .map(|c| c.ledger.failed.iter().sum::<usize>() as u64)
            .collect::<Vec<_>>();
        let complete = self.children.iter().all(|c| {
            c.ledger.offered == offered
                && c.ledger.last_fifo == last_fifo
                && c.ledger.audit_complete(cutoff)
        });
        if !complete {
            self.invalidate("shared source FIFO closure differs".into());
        }
        let mut children = Vec::with_capacity(self.children.len());
        let mut failure = self.failure;
        for child in self.children {
            let artifact = child.finish_with_closing(cutoff, closing)?;
            if artifact.phase != StructuredCapturePhase::Qualified
                || artifact.model.is_none()
                || artifact.failure.is_some()
            {
                failure.get_or_insert_with(|| "a shared child failed qualification".into());
            }
            children.push(artifact);
        }
        let mut source = self.shared.take().unwrap();
        source.record(&serde_json::json!({"kind":"footer","phase":if failure.is_none(){"qualified"}else{"failed"},
            "failure":failure,"offered":offered,"members":members,"failed_members":failed_members,
            "accepted_fifo_cutoff":cutoff,"last_captured_fifo":last_fifo,"fifo_audit_complete":complete,"closing":closing}))?;
        let file = source.finish()?;
        for child in &mut children {
            child.source_path = file.path.clone();
            child.source_sha256 = file.digest;
            child.source_bytes = file.bytes;
        }
        Ok(StructuredCalibrationGroupArtifactV2 {
            fingerprint,
            children,
            failure,
        })
    }
}
