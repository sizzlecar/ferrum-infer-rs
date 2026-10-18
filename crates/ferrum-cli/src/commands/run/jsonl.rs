//! Borrow generated payloads through the synchronous JSONL write.

use super::*;
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct DeltaRecord<'a> {
    schema_version: u32,
    event: &'static str,
    session_id: &'a str,
    history_epoch: usize,
    request_id: &'a str,
    turn: usize,
    index: usize,
    raw_text_delta: &'a str,
    utf8_bytes: usize,
    token_id: Option<u32>,
}

pub(super) fn jsonl_assistant_delta_record<'a>(
    session_id: &'a str,
    history_epoch: usize,
    request_id: &'a str,
    turn: usize,
    index: usize,
    raw_text_delta: &'a str,
    token_id: Option<u32>,
) -> DeltaRecord<'a> {
    DeltaRecord {
        schema_version: RUN_JSONL_SCHEMA_VERSION,
        event: "assistant_delta",
        session_id,
        history_epoch,
        request_id,
        turn,
        index,
        raw_text_delta,
        utf8_bytes: raw_text_delta.len(),
        token_id,
    }
}

#[derive(Serialize)]
pub(super) struct AssistantRecord<'a> {
    schema_version: u32,
    event: &'static str,
    session_id: &'a str,
    history_epoch: usize,
    request_id: &'a str,
    turn: usize,
    content: &'a str,
    reasoning: Option<&'a str>,
    history_before: serde_json::Value,
    finish_reason: Option<&'static str>,
    usage: Option<&'a TokenUsage>,
    n_tokens: usize,
    chunk_count: usize,
    raw_text_sha256: String,
    ms: f64,
}

pub(super) fn jsonl_assistant_record<'a>(
    session_id: &'a str,
    history_epoch: usize,
    request_id: &'a str,
    turn: usize,
    content: &'a str,
    reasoning: Option<&'a str>,
    history: &[RunHistoryMessage],
    finish_reason: Option<FinishReason>,
    usage: Option<&'a TokenUsage>,
    n_tokens: usize,
    chunk_count: usize,
    raw_text: &str,
    ms: f64,
) -> AssistantRecord<'a> {
    AssistantRecord {
        schema_version: RUN_JSONL_SCHEMA_VERSION,
        event: "assistant",
        session_id,
        history_epoch,
        request_id,
        turn,
        content,
        reasoning,
        history_before: history_evidence(history),
        finish_reason: finish_reason.map(finish_reason_str),
        usage,
        n_tokens,
        chunk_count,
        raw_text_sha256: sha256_text(raw_text),
        ms,
    }
}

pub(super) fn hash_history(history: &[RunHistoryMessage]) -> String {
    struct RawHistory<'a>(&'a [RunHistoryMessage]);
    impl Serialize for RawHistory<'_> {
        fn serialize<S: serde::Serializer>(
            &self,
            serializer: S,
        ) -> std::result::Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
            for message in self.0 {
                sequence.serialize_element(&(&message.prompt.role, &message.raw_content))?;
            }
            sequence.end()
        }
    }
    struct HashWriter(Sha256);
    impl Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut sink = HashWriter(Sha256::new());
    serde_json::to_writer(&mut sink, &RawHistory(history))
        .expect("run history serialization cannot fail");
    format!("{:x}", sink.0.finalize())
}
