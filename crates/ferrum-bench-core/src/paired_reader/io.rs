use super::{Artifact, ReaderIssueKind as Kind, ReaderLimits};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::io::Read;

pub(super) struct Reader {
    pub limits: ReaderLimits,
    remaining: u64,
}
impl Reader {
    pub fn new(limits: ReaderLimits) -> Self {
        Self {
            remaining: limits.max_total_read_bytes,
            limits,
        }
    }
    pub fn artifact(&mut self, a: &Artifact) -> Result<Vec<u8>, Kind> {
        let n = usize::try_from(a.bytes).map_err(|_| Kind::Budget)?;
        let charge = a.bytes.checked_add(1).ok_or(Kind::Budget)?;
        if n > self.limits.max_artifact_bytes || charge > self.remaining {
            return Err(Kind::Budget);
        }
        self.remaining -= charge;
        let mut f = File::open(&a.path).map_err(|_| Kind::Io)?;
        if f.metadata().map_err(|_| Kind::Io)?.len() != a.bytes {
            return Err(Kind::Size);
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(n).map_err(|_| Kind::Budget)?;
        bytes.resize(n, 0);
        f.read_exact(&mut bytes).map_err(|_| Kind::Io)?;
        let mut tail = [0];
        if f.read(&mut tail).map_err(|_| Kind::Io)? != 0 {
            return Err(Kind::Size);
        }
        if digest(&bytes) != a.sha256 {
            return Err(Kind::Hash);
        }
        Ok(bytes)
    }
    pub fn json<T: serde::de::DeserializeOwned>(&mut self, a: &Artifact) -> Result<T, Kind> {
        parse(&self.artifact(a)?)
    }
}

pub(super) fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
pub(super) fn hex_digest(v: &Value) -> Option<[u8; 32]> {
    let s = v.as_str()?;
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut out = [0; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// Reject duplicate fields at every nesting level, including source/provenance.
/// The scanner retains keys only for open objects, not strings/response bodies.
pub(super) fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, Kind> {
    let mut d = serde_json::Deserializer::from_slice(bytes);
    Scan::deserialize(&mut d).map_err(|e| {
        if e.to_string().contains("duplicate JSON field") {
            Kind::DuplicateJsonField
        } else {
            Kind::Json
        }
    })?;
    d.end().map_err(|_| Kind::Json)?;
    serde_json::from_slice(bytes).map_err(|_| Kind::Json)
}
struct Scan;
impl<'de> Deserialize<'de> for Scan {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Scan;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON")
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Scan, E> {
                Ok(Scan)
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Scan, E> {
                Ok(Scan)
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Scan, E> {
                Ok(Scan)
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Scan, E> {
                Ok(Scan)
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Scan, E> {
                Ok(Scan)
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Scan, E> {
                Ok(Scan)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Scan, A::Error> {
                while a.next_element::<Scan>()?.is_some() {}
                Ok(Scan)
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Scan, A::Error> {
                let mut keys = BTreeSet::new();
                while let Some(k) = a.next_key::<String>()? {
                    if !keys.insert(k) {
                        return Err(serde::de::Error::custom("duplicate JSON field"));
                    }
                    a.next_value::<Scan>()?;
                }
                Ok(Scan)
            }
        }
        d.deserialize_any(V)
    }
}

/// Preserve exact integer identities (including seeds > 2^53). Permit a
/// floating literal such as 1.0 to match 1 only in its exact safe integer range.
pub(super) fn semantic_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let ai = a
                .as_i64()
                .map(i128::from)
                .or_else(|| a.as_u64().map(i128::from));
            let bi = b
                .as_i64()
                .map(i128::from)
                .or_else(|| b.as_u64().map(i128::from));
            match (ai, bi) {
                (Some(a), Some(b)) => a == b,
                (Some(i), None) => i.abs() <= (1_i128 << 53) && b.as_f64() == Some(i as f64),
                (None, Some(i)) => i.abs() <= (1_i128 << 53) && a.as_f64() == Some(i as f64),
                (None, None) => a.as_f64() == b.as_f64(),
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| semantic_eq(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, a)| b.get(k).is_some_and(|b| semantic_eq(a, b)))
        }
        _ => a == b,
    }
}
