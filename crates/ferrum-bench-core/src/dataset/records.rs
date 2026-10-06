//! Shared source-record semantics for selection and replay of a frozen selection.

use ferrum_types::{FerrumError, Result};
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer as _, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::io::{BufRead, BufReader, Read};
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShareGptReadLimits {
    pub maximum_source_bytes: NonZeroU64,
    /// Includes separators/whitespace consumed before the next array item, or
    /// a complete JSONL line. The parser may retain one byte of lookahead.
    pub maximum_record_bytes: NonZeroUsize,
    pub maximum_records: NonZeroU64,
}

impl Default for ShareGptReadLimits {
    fn default() -> Self {
        Self {
            maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024 * 1024).unwrap(),
            maximum_record_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
            maximum_records: NonZeroU64::new(10_000_000).unwrap(),
        }
    }
}

impl ShareGptReadLimits {
    pub fn validate(self) -> Result<()> {
        let hard = Self::default();
        if self.maximum_source_bytes > hard.maximum_source_bytes
            || self.maximum_record_bytes > hard.maximum_record_bytes
            || self.maximum_records > hard.maximum_records
        {
            return Err(FerrumError::config(
                "ShareGPT source read limits exceed hard bounds",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ShareGptRecord {
    Value(Value),
    /// A nonblank malformed JSONL row occupies an index, just as in selection.
    MalformedJsonLine,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShareGptReadReceipt {
    pub source_format: &'static str,
    pub source_sha256: String,
    pub source_bytes: u64,
    pub records: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ShareGptFirstPair<'a> {
    pub prompt: &'a str,
    pub assistant: &'a str,
}

pub fn sharegpt_first_pair(value: &Value) -> Option<ShareGptFirstPair<'_>> {
    let turns = value.get("conversations")?.as_array()?;
    let human = turns.first()?;
    let assistant = turns.get(1)?;
    if human.get("from")?.as_str()? != "human" || assistant.get("from")?.as_str()? != "gpt" {
        return None;
    }
    Some(ShareGptFirstPair {
        prompt: human.get("value")?.as_str()?,
        assistant: assistant.get("value")?.as_str()?,
    })
}

pub fn sharegpt_original_id(value: &Value) -> Option<String> {
    value.get("id").filter(|id| !id.is_null()).map(|id| {
        id.as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| id.to_string())
    })
}

/// Visits every source record in order; neither filtering nor sampling occurs.
/// Success hashes the entire raw source, including blanks and trailing bytes.
pub fn visit_sharegpt_records(
    input: impl Read,
    limits: ShareGptReadLimits,
    mut visit: impl FnMut(u64, ShareGptRecord) -> Result<()>,
) -> Result<ShareGptReadReceipt> {
    limits.validate()?;
    let mut reader = BufReader::new(SourceReader {
        input,
        digest: Sha256::new(),
        bytes: 0,
        maximum: limits.maximum_source_bytes.get(),
    });
    let first = loop {
        let buffer = reader.fill_buf().map_err(read_error)?;
        if let Some(byte) = buffer.iter().find(|byte| !byte.is_ascii_whitespace()) {
            break Some(*byte);
        }
        let bytes = buffer.len();
        if bytes == 0 {
            break None;
        }
        reader.consume(bytes);
    };
    let mut records = 0;
    let mut emit = |record| {
        if records == limits.maximum_records.get() {
            return Err(FerrumError::config("ShareGPT source exceeds record limit"));
        }
        visit(records, record)?;
        records += 1;
        Ok(())
    };
    let format = if first == Some(b'[') {
        let remaining = Cell::new(limits.maximum_record_bytes.get());
        let capped = RecordReader {
            input: &mut reader,
            remaining: &remaining,
        };
        let mut deserializer = serde_json::Deserializer::from_reader(capped);
        deserializer
            .deserialize_seq(ArrayVisitor {
                emit: &mut emit,
                remaining: &remaining,
                maximum: limits.maximum_record_bytes.get(),
            })
            .and_then(|()| {
                remaining.set(limits.maximum_record_bytes.get());
                deserializer.end()
            })
            .map_err(|error| FerrumError::model(format!("parse ShareGPT JSON array: {error}")))?;
        "json_array"
    } else {
        let mut line = Vec::new();
        while read_bounded_line(&mut reader, &mut line, limits.maximum_record_bytes.get())? {
            let text = std::str::from_utf8(&line).map_err(|error| {
                FerrumError::model(format!("read ShareGPT JSONL UTF-8: {error}"))
            })?;
            if text.trim().is_empty() {
                continue;
            }
            emit(match serde_json::from_str(text) {
                Ok(value) => ShareGptRecord::Value(value),
                Err(_) => ShareGptRecord::MalformedJsonLine,
            })?;
        }
        "jsonl"
    };
    let source = reader.into_inner();
    Ok(ShareGptReadReceipt {
        source_format: format,
        source_sha256: format!("{:x}", source.digest.finalize()),
        source_bytes: source.bytes,
        records,
    })
}

struct ArrayVisitor<'a, F> {
    emit: &'a mut F,
    remaining: &'a Cell<usize>,
    maximum: usize,
}

impl<'de, F: FnMut(ShareGptRecord) -> Result<()>> Visitor<'de> for ArrayVisitor<'_, F> {
    type Value = ();
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a ShareGPT array of conversation records")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> std::result::Result<(), A::Error> {
        loop {
            self.remaining.set(self.maximum);
            let Some(value) = sequence.next_element::<Value>()? else {
                return Ok(());
            };
            (self.emit)(ShareGptRecord::Value(value)).map_err(serde::de::Error::custom)?;
        }
    }
}

struct RecordReader<'a, R> {
    input: R,
    remaining: &'a Cell<usize>,
}

impl<R: Read> Read for RecordReader<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let allowed = bytes.len().min(self.remaining.get());
        if allowed == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ShareGPT record exceeds byte limit",
            ));
        }
        let count = self.input.read(&mut bytes[..allowed])?;
        self.remaining.set(self.remaining.get() - count);
        Ok(count)
    }
}

struct SourceReader<R> {
    input: R,
    digest: Sha256,
    bytes: u64,
    maximum: u64,
}

impl<R: Read> Read for SourceReader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let remaining = self.maximum - self.bytes;
        let allowed = bytes
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        if allowed == 0 {
            let mut probe = [0];
            return if self.input.read(&mut probe)? == 0 {
                Ok(0)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "ShareGPT source exceeds byte limit",
                ))
            };
        }
        let count = self.input.read(&mut bytes[..allowed])?;
        self.digest.update(&bytes[..count]);
        self.bytes += count as u64;
        Ok(count)
    }
}

fn read_bounded_line(
    reader: &mut impl BufRead,
    line: &mut Vec<u8>,
    maximum: usize,
) -> Result<bool> {
    line.clear();
    loop {
        let bytes = reader.fill_buf().map_err(read_error)?;
        if bytes.is_empty() {
            return Ok(!line.is_empty());
        }
        let end = bytes
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|index| index + 1);
        let count = end.unwrap_or(bytes.len());
        if count > maximum - line.len() {
            return Err(FerrumError::config("ShareGPT JSONL row exceeds byte limit"));
        }
        line.try_reserve_exact(count)
            .map_err(|_| FerrumError::resource_exhausted("ShareGPT row allocation failed"))?;
        line.extend_from_slice(&bytes[..count]);
        reader.consume(count);
        if end.is_some() {
            return Ok(true);
        }
    }
}

fn read_error(error: std::io::Error) -> FerrumError {
    FerrumError::model(format!("read ShareGPT: {error}"))
}

#[cfg(test)]
mod tests;
