//! Independent identity/outcome salvage. A malformed or duplicated field does
//! not erase a different, unique failure field. Ambiguous fields prove nothing.
use super::{io, ExpectedRequest, RawFailureKind};
use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;
use std::fmt;

#[derive(Default)]
struct Field {
    raw: Option<Box<RawValue>>,
    seen: bool,
}
impl Field {
    fn put(&mut self, value: Box<RawValue>) {
        if self.seen {
            self.raw = None;
        } else {
            self.raw = Some(value);
        }
        self.seen = true;
    }
    fn get<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        io::parse(self.raw.as_ref()?.get().as_bytes()).ok()
    }
}

#[derive(Default)]
pub(super) struct Projection {
    index: Field,
    scenario: Field,
    source: Field,
    cap: Field,
    scheduled: Field,
    success: Field,
    timed_out: Field,
    error: Field,
}
impl Projection {
    pub fn index(&self) -> Option<usize> {
        self.index.get()
    }
    pub fn success(&self) -> Option<bool> {
        self.success.get()
    }
    pub fn timed_out(&self) -> Option<bool> {
        self.timed_out.get()
    }
    pub fn failure(&self) -> Option<RawFailureKind> {
        if self.timed_out() == Some(true) {
            Some(RawFailureKind::Timeout)
        } else if self.success() == Some(false) || self.error.get::<String>().is_some() {
            Some(RawFailureKind::RecordedFailure)
        } else {
            None
        }
    }
    pub fn matches(&self, expected: &ExpectedRequest) -> bool {
        self.index() == Some(expected.key.index)
            && self.scenario.get::<String>().as_ref() == Some(&expected.scenario)
            && self.cap.get::<u64>() == Some(expected.key.output_cap)
            && self.scheduled.get::<u64>() == Some(expected.key.scheduled_ms)
            && self
                .source
                .get()
                .is_some_and(|v| io::semantic_eq(&v, &expected.source))
    }
}
impl<'de> Deserialize<'de> for Projection {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Projection;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a request record object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Projection, A::Error> {
                let mut out = Projection::default();
                while let Some(key) = map.next_key::<String>()? {
                    let field = match key.as_str() {
                        "index" => Some(&mut out.index),
                        "scenario" => Some(&mut out.scenario),
                        "source" => Some(&mut out.source),
                        "requested_max_tokens" => Some(&mut out.cap),
                        "scheduled_ms" => Some(&mut out.scheduled),
                        "success" => Some(&mut out.success),
                        "timed_out" => Some(&mut out.timed_out),
                        "error" => Some(&mut out.error),
                        _ => None,
                    };
                    if let Some(field) = field {
                        field.put(map.next_value()?);
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(out)
            }
        }
        d.deserialize_map(V)
    }
}
