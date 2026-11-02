//! Lossless bounded records for one unusually large execution-facts event.
//! Normal records retain their wire format. Fragment pieces use JSON pointers
//! into the original event; containers precede children, and every value occurs
//! once. Consumers must assemble all indexed parts before treating them as a
//! complete fact set. Fragment envelopes do not repeat work or command counts.

use super::*;
use serde_json::{json, Value};
use std::io::{self, Write};

struct ByteCount(usize);
impl Write for ByteCount {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self.0.checked_add(bytes.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "execution facts size overflow")
        })?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encoded_bytes(
    value: &impl serde::Serialize,
) -> std::result::Result<usize, ExecutionEventSinkError> {
    let mut count = ByteCount(0);
    serde_json::to_writer(&mut count, value)
        .map_err(|error| ExecutionEventSinkError::contract(error.to_string()))?;
    Ok(count.0)
}

fn pointer_child(parent: &str, child: &str) -> String {
    format!("{parent}/{}", child.replace('~', "~0").replace('/', "~1"))
}

pub(in crate::continuous_engine::profile) fn bounded_records(
    mut event: FerrumProfileEvent,
    maximum_bytes: usize,
) -> std::result::Result<Vec<FerrumProfileEvent>, ExecutionEventSinkError> {
    if encoded_bytes(&event)?.saturating_add(1) <= maximum_bytes {
        return Ok(vec![event]);
    }
    let original = serde_json::to_value(&event)
        .map_err(|error| ExecutionEventSinkError::contract(error.to_string()))?;
    let original_id = event.event_id.clone();
    event.shape.clear();
    event.shape.insert("fragment".to_owned(), json!(true));
    event.attributes.clear();
    event.duration_us = None;
    event.memory = None;
    event.resource = None;
    event.replay = None;
    event.event_id = format!("{original_id}-part-{}", usize::MAX);
    event.attributes.insert(
        "execution_facts_fragment".to_owned(),
        json!({
            "schema_version": 1,
            "original_event_id": original_id,
            "index": usize::MAX,
            "count": usize::MAX,
            "pieces": [],
        }),
    );
    // Reserve the maximum decimal width for indexes before packing. The JSONL
    // newline and each comma are included; final indexes can only be smaller.
    let overhead = encoded_bytes(&event)?.saturating_add(1);
    let available = maximum_bytes
        .checked_sub(overhead)
        .filter(|n| *n > 0)
        .ok_or_else(|| {
            ExecutionEventSinkError::transport(
                "execution facts fragment envelope exceeds the journal record limit",
            )
        })?;
    let mut pending = vec![(String::new(), original)];
    let mut groups = Vec::new();
    let mut pieces = Vec::new();
    let mut used = 0usize;
    while let Some((path, value)) = pending.pop() {
        let mut piece = json!({"path": path, "value": value});
        let mut bytes = encoded_bytes(&piece)?;
        if bytes > available {
            let path = piece["path"].as_str().unwrap().to_owned();
            match piece["value"].take() {
                Value::Array(values) => {
                    for (index, value) in values.into_iter().enumerate().rev() {
                        pending.push((pointer_child(&path, &index.to_string()), value));
                    }
                    piece["value"] = json!([]);
                }
                Value::Object(values) => {
                    for (key, value) in values.into_iter().rev() {
                        pending.push((pointer_child(&path, &key), value));
                    }
                    piece["value"] = json!({});
                }
                _ => {
                    return Err(ExecutionEventSinkError::transport(
                        "indivisible execution facts value exceeds the journal record limit",
                    ))
                }
            }
            bytes = encoded_bytes(&piece)?;
            if bytes > available {
                return Err(ExecutionEventSinkError::transport(
                    "execution facts pointer exceeds the journal record limit",
                ));
            }
        }
        let comma = usize::from(!pieces.is_empty());
        if used.saturating_add(comma).saturating_add(bytes) > available {
            groups.push(std::mem::take(&mut pieces));
            used = 0;
        }
        used += usize::from(!pieces.is_empty()) + bytes;
        pieces.push(piece);
    }
    if !pieces.is_empty() {
        groups.push(pieces);
    }
    let count = groups.len();
    let mut records = Vec::with_capacity(count);
    for (index, pieces) in groups.into_iter().enumerate() {
        let mut record = event.clone();
        record.event_id = format!("{original_id}-part-{index}");
        let fragment = record
            .attributes
            .get_mut("execution_facts_fragment")
            .unwrap();
        fragment["index"] = json!(index);
        fragment["count"] = json!(count);
        fragment["pieces"] = json!(pieces);
        record
            .validate()
            .map_err(ExecutionEventSinkError::contract)?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests;
