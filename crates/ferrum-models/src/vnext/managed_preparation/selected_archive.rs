//! Verify the retained mmap, rather than trusting a subsequently reopened path.
use super::*;
use ferrum_runtime::FileFingerprint;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

pub(super) fn validate(
    prepared: &PreparedProductionModel,
    archive: &SafetensorsArchive,
) -> ferrum_types::Result<()> {
    let bundle = prepared.sources();
    let frozen = &bundle.resolved_sources().weights.files;
    let expected: BTreeMap<String, &FileFingerprint> = match bundle.weights() {
        ProductionWeightArtifact::SafetensorsFile(path) => {
            let [fingerprint] = frozen.as_slice() else {
                return Err(model_error(
                    "single-file source lacks one exact frozen artifact",
                ));
            };
            let basename = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| model_error("selected filename is not UTF-8"))?;
            // HF evidence preserves the snapshot-relative path; the archive
            // records the explicitly opened basename instead.
            if std::path::Path::new(&fingerprint.relative_path)
                .file_name()
                .and_then(|n| n.to_str())
                != Some(basename)
            {
                return Err(model_error("selected filename differs from frozen source"));
            }
            BTreeMap::from([(basename.to_owned(), fingerprint)])
        }
        ProductionWeightArtifact::SafetensorsDirectory(root) => {
            let files = frozen
                .iter()
                .map(|f| (f.relative_path.as_str(), f))
                .collect::<BTreeMap<_, _>>();
            if let Some(file) = files.get("model.safetensors") {
                BTreeMap::from([("model.safetensors".to_owned(), *file)])
            } else {
                let index_name = "model.safetensors.index.json";
                let index = files
                    .get(index_name)
                    .ok_or_else(|| model_error("frozen directory has no selected archive/index"))?;
                let length = index
                    .size_bytes
                    .checked_add(1)
                    .ok_or_else(|| model_error("index extent overflows"))?;
                let file = std::fs::File::open(root.join(index_name)).map_err(model_error)?;
                let mut bytes = Vec::new();
                file.take(length)
                    .read_to_end(&mut bytes)
                    .map_err(model_error)?;
                verify_bytes(index, &bytes)?;
                let parsed: serde_json::Value =
                    serde_json::from_slice(&bytes).map_err(model_error)?;
                let map = parsed
                    .get("weight_map")
                    .and_then(serde_json::Value::as_object)
                    .ok_or_else(|| model_error("frozen safetensors index has no weight_map"))?;
                let names = map
                    .values()
                    .map(|value| {
                        value
                            .as_str()
                            .ok_or_else(|| model_error("index shard name is not a string"))
                    })
                    .collect::<ferrum_types::Result<BTreeSet<_>>>()?;
                if names.is_empty() {
                    return Err(model_error("frozen index selects no shards"));
                }
                names
                    .into_iter()
                    .map(|name| {
                        let fingerprint = files.get(name).ok_or_else(|| {
                            model_error(format!("selected shard `{name}` lacks frozen evidence"))
                        })?;
                        Ok((name.to_owned(), *fingerprint))
                    })
                    .collect::<ferrum_types::Result<_>>()?
            }
        }
        ProductionWeightArtifact::GgufFile(_) => {
            return Err(model_error("managed safetensors source cannot use GGUF"));
        }
    };
    let mut seen = BTreeSet::new();
    for name in archive.tensor_names() {
        let tensor = archive.tensor(name)?;
        if seen.insert(tensor.source_file().to_owned()) {
            let fingerprint = expected.get(tensor.source_file()).ok_or_else(|| {
                model_error("mapped archive is outside the frozen selected source")
            })?;
            verify_bytes(fingerprint, tensor.retained_host_memory().owner_bytes())?;
        }
    }
    if seen != expected.keys().cloned().collect() {
        return Err(model_error(
            "mapped shard set differs from the frozen selection, or contains an empty shard",
        ));
    }
    Ok(())
}

fn verify_bytes(fingerprint: &FileFingerprint, bytes: &[u8]) -> ferrum_types::Result<()> {
    if u64::try_from(bytes.len()).ok() != Some(fingerprint.size_bytes)
        || format!("{:x}", Sha256::digest(bytes)) != fingerprint.sha256
    {
        return Err(model_error(format!(
            "actual mapped source `{}` differs from its frozen size/SHA256",
            fingerprint.relative_path
        )));
    }
    Ok(())
}
