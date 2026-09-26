//! Source records carry their own dense position, independent of offered waves
//! and accepted observation FIFO positions. A partial write poisons the source.
use super::*;
pub(super) struct StructuredSource {
    raw: Option<RawSource>,
    staged: Vec<serde_json::Value>,
    shared_prefix: (u64, [u8; 32]),
    collect_completed: bool,
    records: u64,
    path: PathBuf,
    poisoned: Option<String>,
}
impl StructuredSource {
    pub fn create(path: &Path, limit: u64) -> Result<Self, ExportError> {
        Ok(Self {
            raw: Some(RawSource::create(path, limit)?),
            staged: Vec::new(),
            shared_prefix: (0, [0; 32]),
            collect_completed: false,
            records: 0,
            path: path.into(),
            poisoned: None,
        })
    }
    /// A shared child retains only bounded declaration/lifecycle metadata.
    /// Physical Prepared and completion payloads are emitted once by the group.
    pub fn shared_child(path: &Path) -> Self {
        Self {
            raw: None,
            staged: Vec::new(),
            shared_prefix: (0, [0; 32]),
            collect_completed: false,
            records: 0,
            path: path.into(),
            poisoned: None,
        }
    }
    pub fn is_shared(&self) -> bool {
        self.raw.is_none()
    }
    pub fn captures_completed(&self) -> bool {
        self.collect_completed
    }
    pub fn collect_completed(&mut self, collect: bool) {
        self.collect_completed = collect;
    }
    pub fn take_staged(&mut self) -> Vec<serde_json::Value> {
        std::mem::take(&mut self.staged)
    }
    pub fn set_shared_prefix(&mut self, bytes: u64, digest: [u8; 32]) {
        self.shared_prefix = (bytes, digest);
    }
    pub fn record(&mut self, value: &impl Serialize) -> Result<(), ExportError> {
        if self.is_shared() {
            // Only small records use this entrypoint. Large Completed/Prepared
            // serialization uses record_borrowed and is controlled by the group.
            self.staged.push(serde_json::to_value(value)?);
            return Ok(());
        }
        self.record_borrowed(value)
    }
    /// Serialize borrowed common wave data directly into the bounded writer,
    /// without building N complete JSON Values for a multi-child source.
    pub fn record_borrowed<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), ExportError> {
        if self.is_shared() {
            if self.collect_completed {
                self.staged.push(serde_json::to_value(value)?);
            }
            return Ok(());
        }
        if self.poisoned.is_some() {
            return Err(ExportError::Source(
                "structured raw source is already incomplete",
            ));
        }
        let ordinal = self
            .records
            .checked_add(1)
            .ok_or(ExportError::Source("structured source ordinal exhausted"))?;
        #[derive(Serialize)]
        struct Envelope<'a, T: Serialize + ?Sized> {
            source_record_ordinal: u64,
            record: &'a T,
        }
        let result = self.raw.as_mut().unwrap().record(&Envelope {
            source_record_ordinal: ordinal,
            record: value,
        });
        match result {
            Ok(()) => {
                self.records = ordinal;
                Ok(())
            }
            Err(error) => {
                self.poisoned = Some(error.to_string());
                Err(error)
            }
        }
    }
    pub fn flush(&mut self) -> Result<(), ExportError> {
        let Some(raw) = &mut self.raw else {
            return Ok(());
        };
        if let Err(error) = raw.flush() {
            self.poisoned = Some(error.to_string());
            return Err(error.into());
        }
        Ok(())
    }
    pub fn bytes(&self) -> u64 {
        self.raw
            .as_ref()
            .map_or(self.shared_prefix.0, RawSource::bytes)
    }
    pub fn prefix_digest(&self) -> [u8; 32] {
        self.raw
            .as_ref()
            .map_or(self.shared_prefix.1, RawSource::prefix_digest)
    }
    pub fn finish(self) -> Result<PublishedFile, ExportError> {
        if let Some(reason) = self.poisoned {
            return Err(ExportError::SourceOnly {
                path: self.path,
                reason,
            });
        }
        let Some(raw) = self.raw else {
            // This private interim value cannot escape the group: finish() replaces
            // all child locations/digests after the common footer is durably closed.
            return Ok(PublishedFile {
                path: self.path,
                bytes: self.shared_prefix.0,
                digest: self.shared_prefix.1,
                sha256: String::new(),
            });
        };
        raw.finish().map_err(|error| ExportError::SourceOnly {
            path: self.path,
            reason: error.to_string(),
        })
    }
    pub fn incomplete_error(&self) -> Option<ExportError> {
        self.poisoned
            .as_ref()
            .map(|reason| ExportError::SourceOnly {
                path: self.path.clone(),
                reason: reason.clone(),
            })
    }
}
