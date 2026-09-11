//! Forward bounded model progress to CI without streaming model output or errors.
//! Reports and process exit status remain the only execution evidence.
use serde::Deserialize;
use std::{
    fs::File,
    io::Read,
    path::Path,
    time::{Duration, Instant},
};

const PREFIX: &[u8] = b"FERRUM_PROGRESS ";
const MAX_LINE: usize = 2048;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    case: String,
    status: Status,
    elapsed_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Started,
    Passed,
    Failed,
}

#[derive(Default)]
struct Decoder {
    line: Vec<u8>,
    discarded: bool,
}

impl Decoder {
    fn accept(&mut self, bytes: &[u8], mut emit: impl FnMut(Event)) {
        for &byte in bytes {
            if byte == b'\n' {
                if !self.discarded {
                    if let Some(json) = self.line.strip_prefix(PREFIX) {
                        if let Ok(event) = serde_json::from_slice::<Event>(json) {
                            if !event.case.is_empty()
                                && event.case.len() <= 128
                                && event
                                    .case
                                    .bytes()
                                    .all(|b| b.is_ascii_graphic() || b == b' ')
                            {
                                emit(event);
                            }
                        }
                    }
                }
                self.line.clear();
                self.discarded = false;
            } else if !self.discarded {
                if self.line.len() == MAX_LINE {
                    self.line.clear();
                    self.discarded = true;
                } else {
                    self.line.push(byte);
                }
            }
        }
    }
}

pub(super) struct Follow {
    file: Option<File>,
    decoder: Decoder,
    label: String,
    started: Instant,
    last_heartbeat: Instant,
}

impl Follow {
    pub(super) fn new(path: &Path, label: &str) -> Self {
        Self {
            file: File::open(path).ok(),
            decoder: Decoder::default(),
            label: label.into(),
            started: Instant::now(),
            last_heartbeat: Instant::now(),
        }
    }

    pub(super) fn flush(&mut self) {
        // Limit work per tick even if a child writes large raw diagnostics.
        let mut buffer = [0u8; 65536];
        if let Some(file) = &mut self.file {
            if let Ok(count) = file.read(&mut buffer) {
                self.decoder.accept(&buffer[..count], |event| {
                    eprintln!(
                        "[model] {}: {:?} ({} ms)",
                        event.case, event.status, event.elapsed_ms
                    );
                });
            }
        }
    }

    pub(super) async fn watch(&mut self) {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            self.flush();
            if self.last_heartbeat.elapsed() >= Duration::from_secs(30) {
                eprintln!(
                    "[progress] {} still running after {} s; waiting for the next stage result",
                    self.label,
                    self.started.elapsed().as_secs()
                );
                self.last_heartbeat = Instant::now();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwards_fragmented_events_without_forwarding_raw_output() {
        let mut decoder = Decoder::default();
        let mut events = Vec::new();
        decoder.accept(b"model response or error\nFERRUM_PRO", |event| {
            events.push(event)
        });
        assert!(events.is_empty());
        decoder.accept(
            b"GRESS {\"case\":\"serve.basic\",\"status\":\"passed\",\"elapsed_ms\":321}\r\n",
            |event| events.push(event),
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].case, "serve.basic");
        assert_eq!(events[0].elapsed_ms, 321);
        assert!(matches!(events[0].status, Status::Passed));
    }

    #[test]
    fn rejects_oversized_malformed_or_control_characters_and_recovers() {
        let mut decoder = Decoder::default();
        let mut events = Vec::new();
        decoder.accept(&vec![b'x'; MAX_LINE * 3], |event| events.push(event));
        assert!(decoder.line.len() <= MAX_LINE);
        decoder.accept(b"\nFERRUM_PROGRESS broken\n", |event| events.push(event));
        for case in ["line\nbreak", "\u{1b}[31m", ""] {
            let line = format!(
                "FERRUM_PROGRESS {}\n",
                serde_json::json!({"case":case,"status":"started","elapsed_ms":0})
            );
            decoder.accept(line.as_bytes(), |event| events.push(event));
        }
        assert!(events.is_empty());
        decoder.accept(
            b"FERRUM_PROGRESS {\"case\":\"version\",\"status\":\"started\",\"elapsed_ms\":0}\n",
            |event| events.push(event),
        );
        assert_eq!(events.len(), 1);
    }
}
