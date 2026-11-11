//! Stock SaveLatent wire compatibility without a second latent-sized buffer.
//! Only an explicit singleton temporal view changes exported file metadata.
use super::*;
use safetensors::tensor::{Dtype, Metadata};
use std::io::{Seek, SeekFrom, Write};

const MAXIMUM_HEADER_BYTES: usize = 1 << 20;

fn native_shape(spec: &ResolvedTensorSpec) -> Result<Vec<usize>> {
    ensure!(
        spec.element_type() == ElementType::F32
            && spec.dimensions().len() == 4
            && spec.layout() == &ResolvedTensorLayout::Contiguous,
        "latent interchange requires the actual F32 NCHW boundary"
    );
    spec.dimensions()
        .iter()
        .map(|&n| usize::try_from(n).map_err(Into::into))
        .collect()
}

fn write_one(path: &Path, latent: &ManagedHostTensor, temporal_view: bool) -> Result<Value> {
    let native_shape = native_shape(latent.tensor())?;
    let mut shape = native_shape.clone();
    if temporal_view {
        shape.insert(2, 1);
    }
    let bytes = latent.bytes().bytes();
    ensure!(
        u64::try_from(bytes.len())? == latent.tensor().minimum_storage_bytes()?,
        "latent extent differs"
    );
    let mut header = serde_json::to_vec(&json!({
        "latent_format_version_0":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},
        "latent_tensor":{"dtype":"F32","shape":shape,"data_offsets":[0,bytes.len()]},
    }))?;
    header.resize((header.len() + 7) & !7, b' ');
    ensure!(
        header.len() <= MAXIMUM_HEADER_BYTES,
        "latent header exceeds diagnostic bound"
    );
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = HashedWriter {
        writer: file,
        sha256: Sha256::new(),
    };
    writer.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    writer.write_all(&header)?;
    writer.write_all(bytes)?;
    writer.writer.sync_all()?;
    let file_bytes = writer.writer.metadata()?.len();
    Ok(
        json!({"path":path,"format":"safetensors_save_latent_v0","dtype":"F32",
        "native_shape":native_shape,"interop_shape":shape,"payload_bytes":bytes.len(),
        "payload_sha256":format!("{:x}",Sha256::digest(bytes)),
        "file_bytes":file_bytes,"file_sha256":format!("{:x}",writer.sha256.finalize()),
        "singleton_temporal_metadata_view":temporal_view,"payload_reordered":false,
        "scaling_applied":false,"version_marker":"latent_format_version_0"}),
    )
}

pub(super) fn write_pair(
    directory: &Path,
    name: &str,
    latent: &ManagedHostTensor,
) -> Result<Value> {
    let native = write_one(&directory.join(format!("{name}.latent")), latent, false)?;
    let comfy = write_one(
        &directory.join(format!("{name}.comfy.latent")),
        latent,
        true,
    )?;
    ensure!(
        native["payload_sha256"] == comfy["payload_sha256"],
        "metadata-only view changed payload"
    );
    Ok(
        json!({"native":native,"comfy":comfy,"payload_identical":true,
        "stock_ph3_decode":"use the explicit .comfy.latent singleton-temporal view; the native 4D export preserves its actual boundary"}),
    )
}

pub(super) fn read(
    host: &HostMemoryAccount,
    path: &Path,
    expected: &ResolvedTensorSpec,
) -> Result<(ManagedHostTensor, Value)> {
    let native_shape = native_shape(expected)?;
    let expected_bytes = usize::try_from(expected.minimum_storage_bytes()?)?;
    let mut file = File::open(path)?;
    let file_metadata = file.metadata()?;
    ensure!(
        file_metadata.is_file(),
        "Comfy latent input must be a regular file"
    );
    let file_bytes = file_metadata.len();
    ensure!(
        file_bytes
            <= u64::try_from(expected_bytes)?
                .checked_add(MAXIMUM_HEADER_BYTES as u64 + 8)
                .context("latent file extent overflow")?,
        "latent file exceeds expected payload plus bounded metadata"
    );
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let header_bytes = usize::try_from(u64::from_le_bytes(prefix))?;
    ensure!(
        header_bytes > 0 && header_bytes <= MAXIMUM_HEADER_BYTES && header_bytes % 8 == 0,
        "latent header must be positive, bounded and aligned"
    );
    let mut header = host.allocate_zeroed(HostMemoryClass::InputPreprocess, header_bytes)?;
    file.read_exact(header.as_mut())?;
    let metadata: Metadata =
        serde_json::from_slice(header.as_mut()).context("invalid latent safetensors header")?;
    ensure!(
        metadata.tensors().len() == 2,
        "latent file must contain exactly the sample and version marker"
    );
    let marker = metadata.info("latent_format_version_0").context("legacy latent without version marker is unsupported; implicit 1/0.18215 scaling is forbidden")?;
    ensure!(
        marker.dtype == Dtype::F32
            && marker.shape == [0]
            && marker.data_offsets.0 == marker.data_offsets.1
            && [0, expected_bytes].contains(&marker.data_offsets.0),
        "invalid empty latent version marker"
    );
    let tensor = metadata
        .info("latent_tensor")
        .context("missing latent_tensor")?;
    ensure!(
        tensor.dtype == Dtype::F32,
        "Comfy latent dtype must be F32; this diagnostic never converts F16/BF16"
    );
    let mut interop_shape = native_shape.clone();
    interop_shape.insert(2, 1);
    ensure!(
        tensor.shape == native_shape || tensor.shape == interop_shape,
        "Comfy latent shape must match the native boundary or its T=1 temporal view"
    );
    // With exactly two tensors and an empty marker at a boundary, the sole
    // payload must cover the entire data section. No holes, overlap or hidden
    // trailing bytes are accepted, even though Metadata::validate is private.
    ensure!(
        tensor.data_offsets == (0, expected_bytes),
        "latent offsets differ from exact F32 shape extent"
    );
    let data_start = 8u64
        .checked_add(u64::try_from(header_bytes)?)
        .context("latent data offset overflow")?;
    ensure!(
        data_start.checked_add(u64::try_from(expected_bytes)?) == Some(file_bytes),
        "latent file length differs from header offsets"
    );
    let source_shape = tensor.shape.clone();
    let mut digest = Sha256::new();
    digest.update(prefix);
    digest.update(header.as_mut());
    drop(header);
    let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, expected_bytes)?;
    file.seek(SeekFrom::Start(data_start))?;
    file.read_exact(bytes.as_mut())?;
    ensure!(
        file.read(&mut [0u8; 1])? == 0 && file.metadata()?.len() == file_bytes,
        "latent file changed length during bounded read"
    );
    ensure!(
        bytes
            .as_mut()
            .chunks_exact(4)
            .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
        "nonfinite Comfy latent cannot enter native decode"
    );
    digest.update(bytes.as_mut());
    let record = json!({"path":path,"dtype":"F32","source_shape":source_shape,
        "native_shape":native_shape,"payload_bytes":expected_bytes,
        "payload_sha256":format!("{:x}",Sha256::digest(bytes.as_mut())),
        "file_bytes":file_bytes,"file_sha256":format!("{:x}",digest.finalize()),
        "removed_singleton_temporal_dimension":source_shape != native_shape,
        "payload_reordered":false,"scaling_applied":false,"version_marker":"latent_format_version_0",
        "metadata_prompt_recorded":false});
    let tensor = ManagedHostTensor::new(
        id("value.vae_roundtrip.comfy_saved_latent")?,
        expected.clone(),
        bytes.freeze(),
    )?;
    Ok((tensor, record))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(input: usize) -> HostMemoryAccount {
        HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                input.max(1),
                HostMemoryClassLimits {
                    input_preprocess_bytes: input,
                    output_bytes: 0,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
            8,
        )
        .unwrap()
    }
    fn spec() -> ResolvedTensorSpec {
        ResolvedTensorSpec::new(
            vec![1, 2, 1, 2],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    }
    fn raw() -> Vec<u8> {
        [0x3f80_0001u32, 0x8000_0000, 0x0000_0001, 0x7f7f_ffff]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect()
    }
    fn tensor(host: &HostMemoryAccount) -> ManagedHostTensor {
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 16)
            .unwrap();
        bytes.as_mut().copy_from_slice(&raw());
        ManagedHostTensor::new(
            id("value.latent_io.fixture").unwrap(),
            spec(),
            bytes.freeze(),
        )
        .unwrap()
    }
    fn header() -> Value {
        json!({"latent_format_version_0":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},
            "latent_tensor":{"dtype":"F32","shape":[1,2,1,1,2],"data_offsets":[0,16]}})
    }
    fn wire(path: &Path, header: &Value, payload: &[u8]) {
        let mut bytes = serde_json::to_vec(header).unwrap();
        bytes.resize((bytes.len() + 7) & !7, b' ');
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.write_all(payload).unwrap();
    }

    #[test]
    fn latent_interop_exports_original_shape_and_comfy_view_with_identical_bits() {
        let directory = tempfile::tempdir().unwrap();
        let host = host(4096);
        let latent = tensor(&host);
        let original = latent.bytes().bytes().as_ptr();
        let before = host.usage().unwrap().allocated_bytes();
        let artifact = write_pair(directory.path(), "mean", &latent).unwrap();
        assert_eq!(artifact["native"]["native_shape"], json!([1, 2, 1, 2]));
        assert_eq!(artifact["native"]["interop_shape"], json!([1, 2, 1, 2]));
        assert_eq!(artifact["comfy"]["interop_shape"], json!([1, 2, 1, 1, 2]));
        assert_eq!(
            artifact["native"]["payload_sha256"],
            artifact["comfy"]["payload_sha256"]
        );
        for (file, shape) in [
            ("mean.latent", vec![1, 2, 1, 2]),
            ("mean.comfy.latent", vec![1, 2, 1, 1, 2]),
        ] {
            let bytes = fs::read(directory.path().join(file)).unwrap();
            let safe = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            assert_eq!(safe.tensors().len(), 2);
            let samples = safe.tensor("latent_tensor").unwrap();
            assert_eq!(samples.dtype(), Dtype::F32);
            assert_eq!(samples.shape(), shape);
            assert_eq!(samples.data(), raw());
            assert_eq!(
                safe.tensor("latent_format_version_0").unwrap().shape(),
                &[0]
            );
        }
        assert!(
            write_pair(directory.path(), "mean", &latent).is_err(),
            "existing artifacts must not be overwritten"
        );
        assert_eq!(latent.tensor(), &spec());
        assert_eq!(latent.bytes().bytes().as_ptr(), original);
        assert_eq!(latent.bytes().bytes(), raw());
        assert_eq!(host.usage().unwrap().allocated_bytes(), before);
        drop(latent);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn latent_interop_read_uses_exact_original_scope_and_refunds_one_byte_short() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("official.latent");
        wire(&file, &header(), &raw());
        let mut input = File::open(&file).unwrap();
        let mut prefix = [0u8; 8];
        input.read_exact(&mut prefix).unwrap();
        let header_bytes = u64::from_le_bytes(prefix) as usize;
        let sizing = host(4096);
        let peak = sizing.backing_bytes(header_bytes.max(16)).unwrap();
        let parent = host(peak);
        let scope = parent
            .reserve_scope(&[(HostMemoryClass::InputPreprocess, peak)])
            .unwrap();
        let (latent, report) = read(&scope.account(), &file, &spec()).unwrap();
        assert_eq!(latent.bytes().bytes(), raw());
        assert_eq!(latent.tensor(), &spec());
        assert_eq!(report["removed_singleton_temporal_dimension"], true);
        assert_eq!(report["scaling_applied"], false);
        assert_eq!(report["metadata_prompt_recorded"], false);
        scope.close().unwrap();
        assert_eq!(
            parent.usage().unwrap().allocated_bytes(),
            parent.backing_bytes(16).unwrap()
        );
        drop(latent);
        assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
        let denied = host(peak - 1);
        assert!(read(&denied, &file, &spec()).is_err());
        assert_eq!(denied.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn latent_interop_rejects_ambiguous_scale_dtype_shape_offsets_and_nonfinite() {
        let directory = tempfile::tempdir().unwrap();
        let mut invalid = Vec::new();
        let mut h = header();
        h.as_object_mut().unwrap().remove("latent_format_version_0");
        invalid.push(h);
        let mut h = header();
        h["latent_tensor"]["dtype"] = json!("F16");
        invalid.push(h);
        let mut h = header();
        h["latent_tensor"]["shape"] = json!([1, 2, 2, 1, 2]);
        invalid.push(h);
        let mut h = header();
        h["latent_tensor"]["data_offsets"] = json!([4, 20]);
        invalid.push(h);
        let mut h = header();
        h["latent_format_version_0"]["data_offsets"] = json!([0, 4]);
        invalid.push(h);
        let mut h = header();
        h["latent_format_version_0"]["data_offsets"] = json!([1, 1]);
        invalid.push(h);
        let mut h = header();
        h["extra"] = json!({"dtype":"F32","shape":[0],"data_offsets":[0,0]});
        invalid.push(h);
        let host = host(4096);
        for (index, h) in invalid.iter().enumerate() {
            let file = directory.path().join(format!("invalid-{index}.latent"));
            wire(&file, h, &raw());
            assert!(read(&host, &file, &spec()).is_err(), "{index}");
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
        let file = directory.path().join("nonfinite.latent");
        let mut payload = raw();
        payload[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        wire(&file, &header(), &payload);
        assert!(read(&host, &file, &spec()).is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn latent_interop_rejects_truncated_trailing_and_oversized_header_without_owner() {
        let directory = tempfile::tempdir().unwrap();
        let host = host(4096);
        for (name, payload) in [
            ("truncated", raw()[..15].to_vec()),
            ("trailing", [raw(), vec![0]].concat()),
        ] {
            let file = directory.path().join(format!("{name}.latent"));
            wire(&file, &header(), &payload);
            assert!(read(&host, &file, &spec()).is_err());
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        }
        let file = directory.path().join("huge-header.latent");
        fs::write(&file, ((MAXIMUM_HEADER_BYTES + 8) as u64).to_le_bytes()).unwrap();
        assert!(read(&host, &file, &spec()).is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
