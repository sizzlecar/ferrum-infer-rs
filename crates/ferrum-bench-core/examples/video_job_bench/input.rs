use super::{Arguments, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn file_sha256(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut input = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub async fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    tokio::fs::write(&temporary, serde_json::to_vec_pretty(value)?).await?;
    tokio::fs::rename(&temporary, path).await?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expected {
    pub width: u64,
    pub height: u64,
    pub frames: u64,
    pub fps: f64,
}

impl Expected {
    fn from_request(body: &Value) -> Result<Self> {
        // Both public request forms declare the same geometry. Validate their
        // agreement rather than selecting behavior by server/model identity.
        let dimensions = match (body.get("width"), body.get("height")) {
            (None, None) => None,
            (Some(width), Some(height)) => Some((
                width.as_u64().ok_or("width must be an unsigned integer")?,
                height
                    .as_u64()
                    .ok_or("height must be an unsigned integer")?,
            )),
            _ => return Err("width and height must be declared together".into()),
        };
        let size = body
            .get("size")
            .map(|size| -> Result<(u64, u64)> {
                let size = size.as_str().ok_or("size must be WIDTHxHEIGHT")?;
                let (width, height) = size.split_once('x').ok_or("size must be WIDTHxHEIGHT")?;
                Ok((width.parse()?, height.parse()?))
            })
            .transpose()?;
        let (width, height) = match (dimensions, size) {
            (Some(dimensions), Some(size)) if dimensions != size => {
                return Err("explicit width/height and size disagree".into())
            }
            (Some(dimensions), _) | (_, Some(dimensions)) => dimensions,
            _ => return Err("request requires width/height or size".into()),
        };
        let value = Self {
            width,
            height,
            frames: body
                .get("num_frames")
                .and_then(Value::as_u64)
                .ok_or("request requires explicit num_frames")?,
            fps: body
                .get("fps")
                .and_then(Value::as_f64)
                .ok_or("request requires explicit fps")?,
        };
        if value.width == 0
            || value.height == 0
            || value.frames == 0
            || !value.fps.is_finite()
            || value.fps <= 0.0
        {
            return Err("dimensions, frames and fps must be positive".into());
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Request {
    pub path: PathBuf,
    pub sha256: String,
    pub body: Value,
    pub expected: Expected,
}

#[derive(Serialize)]
pub struct ConfigIdentity {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: usize,
}

#[derive(Serialize)]
pub struct Inputs {
    pub provenance: Value,
    pub provenance_sha256: String,
    pub configurations: Vec<ConfigIdentity>,
    pub requests: Vec<Request>,
}

pub(super) fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(format!("{} exceeds {limit} bytes", path.display()).into());
    }
    Ok(bytes)
}

impl Inputs {
    pub fn load(arguments: &Arguments) -> Result<Self> {
        let provenance = read_bounded(&arguments.provenance, arguments.maximum_json_bytes)?;
        let requests = arguments
            .request_json
            .iter()
            .map(|path| {
                let bytes = read_bounded(path, arguments.maximum_json_bytes)?;
                let body: Value = serde_json::from_slice(&bytes)?;
                if !body.is_object() {
                    return Err("request JSON must be an object".into());
                }
                Ok(Request {
                    path: path.clone(),
                    sha256: sha256(&bytes),
                    expected: Expected::from_request(&body)?,
                    body,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let configurations = arguments
            .server_config
            .iter()
            .map(|path| {
                let bytes = read_bounded(path, arguments.maximum_json_bytes)?;
                Ok(ConfigIdentity {
                    path: path.clone(),
                    sha256: sha256(&bytes),
                    bytes: bytes.len(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            provenance: serde_json::from_slice(&provenance)?,
            provenance_sha256: sha256(&provenance),
            configurations,
            requests,
        })
    }
}

impl Arguments {
    pub fn validate(&self) -> Result<()> {
        if self.concurrency.is_empty()
            || self.concurrency.contains(&0)
            || self.repetitions == 0
            || self.requests_per_repeat == 0
            || self.timeout_seconds == 0
            || self.poll_interval_ms == 0
            || self.validation_timeout_seconds == 0
            || self.maximum_artifact_bytes == 0
            || self.maximum_json_bytes == 0
            || !self.latency_slo_p99_seconds.is_finite()
            || self.latency_slo_p99_seconds <= 0.0
        {
            return Err("counts, limits and SLO must be positive and finite".into());
        }
        if self.output.exists() {
            return Err(
                "output directory already exists; choose a fresh artifact directory".into(),
            );
        }
        let unique: std::collections::BTreeSet<_> = self.concurrency.iter().collect();
        if unique.len() != self.concurrency.len() {
            return Err("concurrency values must be distinct".into());
        }
        if self
            .concurrency
            .iter()
            .any(|width| *width > self.requests_per_repeat)
        {
            return Err("requests-per-repeat must cover the largest client concurrency".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expected_requires_real_frame_count_not_rounded_seconds() {
        let body = serde_json::json!({"size":"832x480", "seconds":"5", "num_frames":124, "fps":24});
        assert_eq!(Expected::from_request(&body).unwrap().frames, 124);
        let mut missing = body;
        missing.as_object_mut().unwrap().remove("num_frames");
        assert!(Expected::from_request(&missing).is_err());
    }

    #[test]
    fn explicit_dimensions_and_legacy_size_have_one_unambiguous_expected_geometry() {
        let mut body = serde_json::json!({"width":832,"height":480,"num_frames":124,"fps":24});
        let expected = Expected::from_request(&body).unwrap();
        assert_eq!(
            (
                expected.width,
                expected.height,
                expected.frames,
                expected.fps
            ),
            (832, 480, 124, 24.0)
        );
        body["size"] = serde_json::json!("832x480");
        assert!(Expected::from_request(&body).is_ok());
        body["size"] = serde_json::json!("480x832");
        assert!(Expected::from_request(&body).is_err());
        body.as_object_mut().unwrap().remove("width");
        assert!(Expected::from_request(&body).is_err());
    }

    #[tokio::test]
    async fn failed_atomic_report_update_preserves_previous_complete_report() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("report.json");
        write_json_atomic(&path, &serde_json::json!({"state":"active"}))
            .await
            .unwrap();
        std::fs::create_dir(path.with_extension("json.tmp")).unwrap();
        assert!(
            write_json_atomic(&path, &serde_json::json!({"state":"complete"}))
                .await
                .is_err()
        );
        let retained: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(retained["state"], "active");
    }
}
