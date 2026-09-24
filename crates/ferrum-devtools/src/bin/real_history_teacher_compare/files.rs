use super::*;

pub(super) fn canonical_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn artifact(directory: &Path, name: &str, size: u64, digest: &str) -> Result<Vec<u8>> {
    ensure!(canonical_sha(digest), "artifact SHA-256 is not canonical");
    let relative = Path::new(name);
    ensure!(
        !name.is_empty()
            && relative
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_))),
        "artifact path is not portable relative path"
    );
    let root = fs::canonicalize(directory)?;
    let path = fs::canonicalize(directory.join(relative))
        .with_context(|| format!("missing artifact {name}"))?;
    ensure!(path.starts_with(root), "artifact escapes capture directory");
    ensure!(
        fs::metadata(&path)?.is_file() && fs::metadata(&path)?.len() == size,
        "artifact {name} byte count differs"
    );
    let bytes = fs::read(&path)?;
    ensure!(sha256(&bytes) == digest, "artifact {name} SHA-256 differs");
    Ok(bytes)
}

pub(super) fn external(
    identity: &VNextTeacherFileIdentity,
    override_path: Option<&Path>,
) -> Result<Value> {
    ensure!(
        canonical_sha(&identity.sha256) && identity.bytes > 0,
        "external file identity has no complete SHA/size"
    );
    let path = override_path.unwrap_or_else(|| Path::new(&identity.path));
    let mut file = fs::File::open(path).with_context(|| {
        format!(
            "missing external provenance file {}; supply an explicit override",
            path.display()
        )
    })?;
    ensure!(
        file.metadata()?.len() == identity.bytes,
        "external provenance file size differs: {}",
        path.display()
    );
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    ensure!(
        format!("{:x}", digest.finalize()) == identity.sha256,
        "external provenance file SHA-256 differs: {}",
        path.display()
    );
    Ok(
        json!({"recorded_path":identity.path,"verified_path":fs::canonicalize(path)?,"bytes":identity.bytes,"sha256":identity.sha256}),
    )
}

pub(super) fn logits(directory: &Path, file: &VNextTeacherLogitArtifact) -> Result<Vec<f32>> {
    ensure!(
        file.encoding == "f32-le"
            && file.elements > 0
            && file.elements.checked_mul(4).map(|n| n as u64) == Some(file.bytes),
        "invalid full-logits layout"
    );
    let bytes = artifact(directory, &file.file, file.bytes, &file.sha256)?;
    decode(&bytes, "f32", file.elements)
}

pub(super) fn decode(bytes: &[u8], dtype: &str, elements: usize) -> Result<Vec<f32>> {
    let width = match dtype {
        "f32" => 4,
        "f16" | "bf16" => 2,
        _ => anyhow::bail!("unsupported raw logit dtype {dtype}"),
    };
    ensure!(
        elements > 0 && elements.checked_mul(width) == Some(bytes.len()),
        "raw readback layout byte count differs"
    );
    let values: Vec<f32> = bytes
        .chunks_exact(width)
        .map(|chunk| {
            if dtype == "f32" {
                return f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
            }
            let bits = u16::from_le_bytes(chunk.try_into().expect("two-byte chunk"));
            if dtype == "bf16" {
                return f32::from_bits(u32::from(bits) << 16);
            }
            let sign = u32::from(bits & 0x8000) << 16;
            let exponent = (bits >> 10) & 31;
            let mantissa = u32::from(bits & 1023);
            let magnitude = match exponent {
                0 => ((mantissa as f32) * 2.0_f32.powi(-24)).to_bits(),
                31 => 0x7f80_0000 | (mantissa << 13),
                _ => ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
            };
            f32::from_bits(sign | magnitude)
        })
        .collect();
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "full vocabulary contains non-finite logits"
    );
    Ok(values)
}

pub(super) fn raw(
    directory: &Path,
    readback: &VNextTeacherReadbackEvidence,
    vocabulary: usize,
) -> Result<Vec<f32>> {
    let file = readback
        .raw_artifact
        .as_ref()
        .context("raw readback sidecar missing")?;
    ensure!(
        file.sha256 == readback.sha256 && file.bytes == readback.byte_count as u64,
        "raw sidecar differs from receipt SHA/size"
    );
    let layout: super::receipt::Layout =
        serde_json::from_value(readback.request["output_layout"].clone())?;
    ensure!(
        layout.element_count == vocabulary as u64,
        "readback is not the complete vocabulary"
    );
    let bytes = artifact(directory, &file.file, file.bytes, &file.sha256)?;
    decode(&bytes, &layout.element_type, vocabulary)
}
