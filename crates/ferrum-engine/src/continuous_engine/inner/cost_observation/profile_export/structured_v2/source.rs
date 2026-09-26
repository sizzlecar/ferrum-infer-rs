//! Source records carry their own dense position, independent of offered waves
//! and accepted observation FIFO positions. A partial write poisons the source.
use super::*;
pub(super) struct StructuredSource {
    raw: RawSource,
    records: u64,
    path: PathBuf,
    poisoned: Option<String>,
}
impl StructuredSource {
    pub fn create(path: &Path, limit: u64) -> Result<Self, ExportError> {
        Ok(Self {
            raw: RawSource::create(path, limit)?,
            records: 0,
            path: path.into(),
            poisoned: None,
        })
    }
    pub fn record(&mut self, value: &impl Serialize) -> Result<(), ExportError> {
        self.record_borrowed(value)
    }
    /// Serialize borrowed common wave data directly into the bounded writer,
    /// without building N complete JSON Values for a multi-child source.
    pub fn record_borrowed<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), ExportError> {
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
        let result = self.raw.record(&Envelope {
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
        if let Err(error) = self.raw.flush() {
            self.poisoned = Some(error.to_string());
            return Err(error.into());
        }
        Ok(())
    }
    pub fn bytes(&self) -> u64 {
        self.raw.bytes()
    }
    pub fn prefix_digest(&self) -> [u8; 32] {
        self.raw.prefix_digest()
    }
    pub fn finish(self) -> Result<PublishedFile, ExportError> {
        if let Some(reason) = self.poisoned {
            return Err(ExportError::SourceOnly {
                path: self.path,
                reason,
            });
        }
        self.raw.finish().map_err(|error| ExportError::SourceOnly {
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
