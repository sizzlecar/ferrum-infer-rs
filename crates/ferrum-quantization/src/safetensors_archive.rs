use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    CanonicalRational, ElementType, RetainedHostMemoryRegion, StableHostMemory, VNextError,
    WeightComponentPayload, WeightComponentSource, WeightComponentSpec, WeightEncoding,
};
use ferrum_types::{FerrumError, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

const MAX_DENSE_TRANSCODE_WORKERS: usize = 8;
const PARALLEL_DENSE_TRANSCODE_MIN_ELEMENTS: usize = 256 * 1024;

#[derive(Debug, Clone)]
struct TensorMeta {
    shard: usize,
    dtype: Dtype,
    shape: Vec<u64>,
    data_start: usize,
    data_end: usize,
}

struct SafetensorsShard {
    relative_path: String,
    mmap: Mmap,
}

// SAFETY: SafetensorsShard owns an immutable Mmap whose allocation, address,
// length, and contents remain fixed for the lifetime of the shard.
unsafe impl StableHostMemory for SafetensorsShard {
    fn stable_bytes(&self) -> &[u8] {
        &self.mmap
    }
}

/// Mmap-backed, once-indexed safetensors archive shared by vNext model
/// packages. Tensor payload access does not reparse shard headers.
pub struct SafetensorsArchive {
    shards: Vec<Arc<SafetensorsShard>>,
    tensors: BTreeMap<String, TensorMeta>,
}

pub struct SafetensorsTensor<'archive> {
    external_name: &'archive str,
    source_file: &'archive str,
    dtype: Dtype,
    shape: &'archive [u64],
    bytes: &'archive [u8],
    retained_host_memory: RetainedHostMemoryRegion,
}

impl<'archive> SafetensorsTensor<'archive> {
    pub fn external_name(&self) -> &'archive str {
        self.external_name
    }

    pub fn source_file(&self) -> &'archive str {
        self.source_file
    }

    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn element_type(&self) -> Option<ElementType> {
        element_type(self.dtype)
    }

    pub fn shape(&self) -> &'archive [u64] {
        self.shape
    }

    pub fn bytes(&self) -> &'archive [u8] {
        self.bytes
    }

    /// Stable mmap owner and exact byte range for this tensor payload.
    pub fn retained_host_memory(&self) -> &RetainedHostMemoryRegion {
        &self.retained_host_memory
    }
}

impl SafetensorsArchive {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let shard_paths = discover_shards(model_dir)?;
        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut tensors = BTreeMap::new();
        for (shard_index, (relative_path, path)) in shard_paths.into_iter().enumerate() {
            let file = File::open(&path)
                .map_err(|error| FerrumError::io(format!("open {path:?}: {error}")))?;
            let mmap = unsafe {
                Mmap::map(&file)
                    .map_err(|error| FerrumError::io(format!("mmap {path:?}: {error}")))?
            };
            let parsed = SafeTensors::deserialize(&mmap)
                .map_err(|error| FerrumError::model(format!("parse {path:?}: {error}")))?;
            for name in parsed.names() {
                let view = parsed.tensor(name).map_err(|error| {
                    FerrumError::model(format!("read tensor {name:?} in {path:?}: {error}"))
                })?;
                let data_start = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
                let data_end = data_start.checked_add(view.data().len()).ok_or_else(|| {
                    FerrumError::model(format!("tensor {name:?} byte range overflows"))
                })?;
                let metadata = TensorMeta {
                    shard: shard_index,
                    dtype: view.dtype(),
                    shape: view.shape().iter().map(|extent| *extent as u64).collect(),
                    data_start,
                    data_end,
                };
                if tensors.insert(name.to_owned(), metadata).is_some() {
                    return Err(FerrumError::model(format!(
                        "tensor {name:?} appears in multiple safetensors shards"
                    )));
                }
            }
            shards.push(Arc::new(SafetensorsShard {
                relative_path,
                mmap,
            }));
        }
        if tensors.is_empty() {
            return Err(FerrumError::model(
                "safetensors archive contains no tensors",
            ));
        }
        Ok(Self { shards, tensors })
    }

    pub fn tensor(&self, external_name: &str) -> Result<SafetensorsTensor<'_>> {
        let (external_name, metadata) =
            self.tensors.get_key_value(external_name).ok_or_else(|| {
                FerrumError::model(format!(
                    "tensor {external_name:?} is absent from safetensors"
                ))
            })?;
        let shard = &self.shards[metadata.shard];
        let bytes = shard
            .mmap
            .get(metadata.data_start..metadata.data_end)
            .ok_or_else(|| {
                FerrumError::model(format!(
                    "tensor {external_name:?} has an invalid safetensors byte range"
                ))
            })?;
        let retained_host_memory = RetainedHostMemoryRegion::new(
            Arc::clone(shard),
            metadata.data_start,
            metadata.data_end - metadata.data_start,
        )
        .map_err(|error| {
            FerrumError::model(format!(
                "tensor {external_name:?} cannot retain its safetensors byte range: {error}"
            ))
        })?;
        Ok(SafetensorsTensor {
            external_name,
            source_file: &shard.relative_path,
            dtype: metadata.dtype,
            shape: &metadata.shape,
            bytes,
            retained_host_memory,
        })
    }

    pub fn contains(&self, external_name: &str) -> bool {
        self.tensors.contains_key(external_name)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

impl WeightComponentSource for SafetensorsArchive {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let (expected_element_type, affine) = match component.encoding {
            WeightEncoding::Dense { element_type } => (element_type, None),
            WeightEncoding::DenseAffine {
                element_type,
                scale,
                bias,
            } => (element_type, Some((scale, bias))),
            WeightEncoding::Quantized(_) | WeightEncoding::BlockQuantized(_) => {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "raw safetensors source cannot decode quantized component `{}` without a format adapter",
                        component.id
                    ),
                })
            }
        };

        let [external_name] = component.external_names.as_slice() else {
            if component.external_names.len() < 2 {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "multi-source component `{}` has fewer than two sources",
                        component.id
                    ),
                });
            }
            let tensors = component
                .external_names
                .iter()
                .map(|external_name| {
                    self.tensor(external_name)
                        .map_err(|error| VNextError::InvalidExecutionPlan {
                            reason: error.to_string(),
                        })
                })
                .collect::<std::result::Result<Vec<_>, VNextError>>()?;
            let stacked = component.dimensions.first().copied()
                == u64::try_from(component.external_names.len()).ok()
                && tensors
                    .iter()
                    .all(|tensor| tensor.shape() == &component.dimensions[1..]);
            let row_concatenated = !component.dimensions.is_empty()
                && tensors.iter().all(|tensor| {
                    tensor.shape().len() == component.dimensions.len()
                        && tensor.shape()[1..] == component.dimensions[1..]
                })
                && tensors.iter().try_fold(0_u64, |rows, tensor| {
                    u64::try_from(tensor.shape()[0])
                        .ok()
                        .and_then(|source_rows| rows.checked_add(source_rows))
                }) == component.dimensions.first().copied();
            if !stacked && !row_concatenated {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "multi-source component `{}` is neither an exact stack nor an axis-0 concatenation",
                        component.id
                    ),
                });
            }
            let mut source_files = Vec::with_capacity(component.external_names.len());
            let mut bytes =
                Vec::with_capacity(usize::try_from(component.physical_bytes()?).map_err(|_| {
                    VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "multi-source component `{}` exceeds host address space",
                            component.id
                        ),
                    }
                })?);
            for (external_name, tensor) in component.external_names.iter().zip(tensors) {
                let actual_element_type = element_type(tensor.dtype()).ok_or_else(|| {
                    VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "tensor {external_name:?} has unsupported safetensors dtype {:?}",
                            tensor.dtype()
                        ),
                    }
                })?;
                let materialized = transcode_dense_bytes(
                    tensor.bytes(),
                    actual_element_type,
                    expected_element_type,
                    external_name,
                    affine,
                )?;
                source_files.push(tensor.source_file().to_owned());
                bytes.extend_from_slice(&materialized);
            }
            return WeightComponentPayload::from_ordered_sources(
                component,
                component.external_names.clone(),
                source_files,
                component.dimensions.clone(),
                expected_element_type,
                bytes,
            );
        };

        let tensor =
            self.tensor(external_name)
                .map_err(|error| VNextError::InvalidExecutionPlan {
                    reason: error.to_string(),
                })?;
        let actual_element_type =
            element_type(tensor.dtype()).ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: format!(
                    "tensor {external_name:?} has unsupported safetensors dtype {:?}",
                    tensor.dtype()
                ),
            })?;
        if tensor.shape() != component.dimensions {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "tensor {external_name:?} shape differs from component `{}`",
                    component.id
                ),
            });
        }
        let bytes = transcode_dense_bytes(
            tensor.bytes(),
            actual_element_type,
            expected_element_type,
            external_name,
            affine,
        )?;
        let retained_host_memory =
            matches!(&bytes, Cow::Borrowed(_)).then(|| tensor.retained_host_memory().clone());
        let payload = WeightComponentPayload::new(
            component,
            tensor.external_name(),
            tensor.source_file(),
            component.dimensions.clone(),
            expected_element_type,
            bytes,
        )?;
        match retained_host_memory {
            Some(retained_host_memory) => payload.with_retained_host_memory(retained_host_memory),
            None => Ok(payload),
        }
    }
}

pub(crate) fn transcode_dense_bytes<'source>(
    bytes: &'source [u8],
    source: ElementType,
    destination: ElementType,
    external_name: &str,
    affine: Option<(CanonicalRational, CanonicalRational)>,
) -> std::result::Result<Cow<'source, [u8]>, VNextError> {
    if source == destination && affine.is_none() {
        return Ok(Cow::Borrowed(bytes));
    }
    let source_float = matches!(
        source,
        ElementType::F16 | ElementType::Bf16 | ElementType::F32
    );
    let destination_float = matches!(
        destination,
        ElementType::F16 | ElementType::Bf16 | ElementType::F32
    );
    if !source_float || !destination_float || bytes.len() % source.size_bytes() as usize != 0 {
        return Err(VNextError::InvalidExecutionPlan {
            reason: format!(
                "tensor {external_name:?} cannot be materialized from {source:?} as {destination:?}"
            ),
        });
    }

    if affine.is_none() {
        return transcode_float_bytes(bytes, source, destination, external_name).map(Cow::Owned);
    }

    let element_count = bytes.len() / source.size_bytes() as usize;
    let mut materialized = Vec::with_capacity(element_count * destination.size_bytes() as usize);
    let affine = affine.map(|(scale, bias)| {
        (
            scale.numerator() as f64 / scale.denominator() as f64,
            bias.numerator() as f64 / bias.denominator() as f64,
        )
    });
    for index in 0..element_count {
        let mut value = read_float(bytes, source, index);
        if let Some((scale, bias)) = affine {
            value = (f64::from(value) * scale + bias) as f32;
            if !value.is_finite() {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "tensor {external_name:?} affine materialization produced a non-finite value at element {index}"
                    ),
                });
            }
        }
        match destination {
            ElementType::F16 => {
                materialized.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes())
            }
            ElementType::Bf16 => {
                materialized.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
            }
            ElementType::F32 => materialized.extend_from_slice(&value.to_le_bytes()),
            _ => unreachable!("non-floating destination was rejected above"),
        }
    }
    Ok(Cow::Owned(materialized))
}

fn transcode_float_bytes(
    bytes: &[u8],
    source: ElementType,
    destination: ElementType,
    external_name: &str,
) -> std::result::Result<Vec<u8>, VNextError> {
    match (source, destination) {
        (ElementType::F16, ElementType::F16) => {
            transcode_float_bytes_typed::<f16, f16>(bytes, source, destination, external_name)
        }
        (ElementType::F16, ElementType::Bf16) => {
            transcode_float_bytes_typed::<f16, bf16>(bytes, source, destination, external_name)
        }
        (ElementType::F16, ElementType::F32) => {
            transcode_float_bytes_typed::<f16, f32>(bytes, source, destination, external_name)
        }
        (ElementType::Bf16, ElementType::F16) => {
            transcode_float_bytes_typed::<bf16, f16>(bytes, source, destination, external_name)
        }
        (ElementType::Bf16, ElementType::Bf16) => {
            transcode_float_bytes_typed::<bf16, bf16>(bytes, source, destination, external_name)
        }
        (ElementType::Bf16, ElementType::F32) => {
            transcode_float_bytes_typed::<bf16, f32>(bytes, source, destination, external_name)
        }
        (ElementType::F32, ElementType::F16) => {
            transcode_float_bytes_typed::<f32, f16>(bytes, source, destination, external_name)
        }
        (ElementType::F32, ElementType::Bf16) => {
            transcode_float_bytes_typed::<f32, bf16>(bytes, source, destination, external_name)
        }
        (ElementType::F32, ElementType::F32) => {
            transcode_float_bytes_typed::<f32, f32>(bytes, source, destination, external_name)
        }
        _ => unreachable!("non-floating element types were rejected before transcoding"),
    }
}

fn transcode_float_bytes_typed<Source: FloatByteCodec, Destination: FloatByteCodec>(
    bytes: &[u8],
    source: ElementType,
    destination: ElementType,
    external_name: &str,
) -> std::result::Result<Vec<u8>, VNextError> {
    debug_assert_eq!(Source::WIDTH, source.size_bytes() as usize);
    debug_assert_eq!(Destination::WIDTH, destination.size_bytes() as usize);
    debug_assert!(bytes.len().is_multiple_of(Source::WIDTH));
    let element_count = bytes.len() / Source::WIDTH;
    let output_bytes = element_count.checked_mul(Destination::WIDTH).ok_or_else(|| {
        VNextError::InvalidExecutionPlan {
            reason: format!(
                "tensor {external_name:?} {source:?}-to-{destination:?} materialization size overflow"
            ),
        }
    })?;
    let mut materialized = Vec::new();
    materialized
        .try_reserve_exact(output_bytes)
        .map_err(|_| VNextError::InvalidExecutionPlan {
            reason: format!(
                "tensor {external_name:?} {source:?}-to-{destination:?} materialization cannot reserve {output_bytes} bytes"
            ),
        })?;
    materialized.resize(output_bytes, 0);

    let worker_count = bounded_dense_transcode_worker_count(element_count);
    if worker_count == 1 {
        transcode_float_partition::<Source, Destination>(bytes, &mut materialized);
        return Ok(materialized);
    }

    let elements_per_worker = element_count.div_ceil(worker_count);
    std::thread::scope(|scope| -> std::result::Result<(), VNextError> {
        let mut remaining_output = materialized.as_mut_slice();
        let mut handles = Vec::with_capacity(worker_count);
        let mut spawn_error = None;
        for worker in 0..worker_count {
            let start = worker * elements_per_worker;
            let end = (start + elements_per_worker).min(element_count);
            if start == end {
                break;
            }
            let output_chunk_bytes = (end - start) * Destination::WIDTH;
            let (output, tail) = remaining_output.split_at_mut(output_chunk_bytes);
            remaining_output = tail;
            let input = &bytes[start * Source::WIDTH..end * Source::WIDTH];
            match std::thread::Builder::new()
                .name(format!("dense-float-transcode-{worker}"))
                .spawn_scoped(scope, move || {
                    transcode_float_partition::<Source, Destination>(input, output)
                }) {
                Ok(handle) => handles.push((worker, handle)),
                Err(error) => {
                    spawn_error = Some((worker, error.to_string()));
                    break;
                }
            }
        }

        let mut panic_worker = None;
        for (worker, handle) in handles {
            if handle.join().is_err() && panic_worker.is_none() {
                panic_worker = Some(worker);
            }
        }
        if let Some((worker, reason)) = spawn_error {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "tensor {external_name:?} {source:?}-to-{destination:?} worker {worker}/{worker_count} could not start: {reason}"
                ),
            });
        }
        if let Some(worker) = panic_worker {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "tensor {external_name:?} {source:?}-to-{destination:?} worker {worker}/{worker_count} panicked"
                ),
            });
        }
        Ok(())
    })?;
    Ok(materialized)
}

fn bounded_dense_transcode_worker_count(element_count: usize) -> usize {
    if element_count < PARALLEL_DENSE_TRANSCODE_MIN_ELEMENTS {
        return 1;
    }
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(MAX_DENSE_TRANSCODE_WORKERS)
        .min(element_count.max(1))
}

fn transcode_float_partition<Source: FloatByteCodec, Destination: FloatByteCodec>(
    input: &[u8],
    output: &mut [u8],
) {
    debug_assert!(input.len().is_multiple_of(Source::WIDTH));
    debug_assert_eq!(
        output.len(),
        input.len() / Source::WIDTH * Destination::WIDTH
    );
    for (source, destination) in input
        .chunks_exact(Source::WIDTH)
        .zip(output.chunks_exact_mut(Destination::WIDTH))
    {
        Destination::encode(Source::decode(source), destination);
    }
}

trait FloatByteCodec {
    const WIDTH: usize;

    fn decode(bytes: &[u8]) -> f32;
    fn encode(value: f32, bytes: &mut [u8]);
}

impl FloatByteCodec for f16 {
    const WIDTH: usize = 2;

    fn decode(bytes: &[u8]) -> f32 {
        f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
    }

    fn encode(value: f32, bytes: &mut [u8]) {
        bytes.copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }
}

impl FloatByteCodec for bf16 {
    const WIDTH: usize = 2;

    fn decode(bytes: &[u8]) -> f32 {
        bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
    }

    fn encode(value: f32, bytes: &mut [u8]) {
        bytes.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
}

impl FloatByteCodec for f32 {
    const WIDTH: usize = 4;

    fn decode(bytes: &[u8]) -> f32 {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn encode(value: f32, bytes: &mut [u8]) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

#[cfg(test)]
fn write_float(bytes: &mut [u8], element_type: ElementType, index: usize, value: f32) {
    let offset = index * element_type.size_bytes() as usize;
    match element_type {
        ElementType::F16 => {
            bytes[offset..offset + 2]
                .copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
        }
        ElementType::Bf16 => {
            bytes[offset..offset + 2]
                .copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        ElementType::F32 => bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes()),
        _ => unreachable!("non-floating destination was rejected before encoding"),
    }
}

fn read_float(bytes: &[u8], element_type: ElementType, index: usize) -> f32 {
    match element_type {
        ElementType::F16 => {
            let offset = index * 2;
            f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
        }
        ElementType::Bf16 => {
            let offset = index * 2;
            bf16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
        }
        ElementType::F32 => {
            let offset = index * 4;
            f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ])
        }
        _ => unreachable!("non-floating source was rejected before decoding"),
    }
}

fn discover_shards(model_dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let single_name = "model.safetensors";
    let single = model_dir.join(single_name);
    if single.is_file() {
        return Ok(vec![(single_name.to_owned(), single)]);
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    let raw = std::fs::read_to_string(&index_path)
        .map_err(|error| FerrumError::io(format!("read {index_path:?}: {error}")))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| FerrumError::serialization(format!("parse {index_path:?}: {error}")))?;
    let weight_map = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| FerrumError::model(format!("{index_path:?} missing weight_map")))?;
    let names = weight_map
        .values()
        .map(|value| {
            value
                .as_str()
                .filter(|name| valid_relative_safetensors_path(name))
                .map(str::to_owned)
                .ok_or_else(|| {
                    FerrumError::model(format!(
                        "{index_path:?} contains an invalid safetensors shard path"
                    ))
                })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if names.is_empty() {
        return Err(FerrumError::model(format!(
            "{index_path:?} contains no safetensors shards"
        )));
    }
    names
        .into_iter()
        .map(|name| {
            let path = model_dir.join(&name);
            if !path.is_file() {
                return Err(FerrumError::model(format!(
                    "safetensors shard {path:?} is missing"
                )));
            }
            Ok((name, path))
        })
        .collect()
}

fn valid_relative_safetensors_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path.ends_with(".safetensors")
        && path
            .split('/')
            .all(|component| !matches!(component, "" | "." | ".."))
}

fn element_type(dtype: Dtype) -> Option<ElementType> {
    match dtype {
        Dtype::BOOL => Some(ElementType::Bool),
        Dtype::U8 => Some(ElementType::U8),
        Dtype::U32 => Some(ElementType::U32),
        Dtype::I8 => Some(ElementType::I8),
        Dtype::I32 => Some(ElementType::I32),
        Dtype::F16 => Some(ElementType::F16),
        Dtype::BF16 => Some(ElementType::Bf16),
        Dtype::F32 => Some(ElementType::F32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use safetensors::tensor::{serialize_to_file, TensorView};
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn tensor_retains_its_exact_mmap_range_after_archive_drop() {
        let directory = tempdir().unwrap();
        let tensor_bytes = [1_u8, 2, 3, 4];
        let tensors = BTreeMap::from([(
            "weight",
            TensorView::new(Dtype::U8, vec![2, 2], &tensor_bytes).unwrap(),
        )]);
        serialize_to_file(tensors, &None, &directory.path().join("model.safetensors")).unwrap();

        let archive = SafetensorsArchive::open(directory.path()).unwrap();
        let tensor = archive.tensor("weight").unwrap();
        assert!(std::ptr::eq(
            tensor.bytes().as_ptr(),
            tensor.retained_host_memory().bytes().as_ptr()
        ));
        let retained = tensor.retained_host_memory().clone();
        drop(tensor);
        drop(archive);
        assert_eq!(retained.bytes(), tensor_bytes);
    }

    #[test]
    fn dense_materialization_converts_bf16_and_f32_to_f16() {
        let bf16_bytes = [
            bf16::from_f32(1.5).to_bits().to_le_bytes(),
            bf16::from_f32(-2.25).to_bits().to_le_bytes(),
        ]
        .concat();
        let converted = transcode_dense_bytes(
            &bf16_bytes,
            ElementType::Bf16,
            ElementType::F16,
            "weight",
            None,
        )
        .unwrap();
        let expected = [
            f16::from_f32(1.5).to_bits().to_le_bytes(),
            f16::from_f32(-2.25).to_bits().to_le_bytes(),
        ]
        .concat();
        assert_eq!(converted.as_ref(), expected);

        let f32_bytes = [1.5_f32.to_le_bytes(), (-2.25_f32).to_le_bytes()].concat();
        let converted = transcode_dense_bytes(
            &f32_bytes,
            ElementType::F32,
            ElementType::F16,
            "weight",
            None,
        )
        .unwrap();
        assert_eq!(converted.as_ref(), expected);
    }

    #[test]
    fn parallel_float_transcode_matches_scalar_for_supported_pairs() {
        let bf16_bytes = (0_u16..=u16::MAX)
            .cycle()
            .take(PARALLEL_DENSE_TRANSCODE_MIN_ELEMENTS)
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let actual = transcode_dense_bytes(
            &bf16_bytes,
            ElementType::Bf16,
            ElementType::F16,
            "large.weight",
            None,
        )
        .unwrap();
        let expected = bf16_bytes
            .chunks_exact(2)
            .flat_map(|source| {
                let value = bf16::from_bits(u16::from_le_bytes([source[0], source[1]])).to_f32();
                f16::from_f32(value).to_bits().to_le_bytes()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.as_ref(), expected);

        let f16_bytes = (0_u16..=u16::MAX)
            .cycle()
            .take(PARALLEL_DENSE_TRANSCODE_MIN_ELEMENTS)
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let f32_bytes = (0..PARALLEL_DENSE_TRANSCODE_MIN_ELEMENTS)
            .flat_map(|index| {
                let value =
                    (index as f32 - PARALLEL_DENSE_TRANSCODE_MIN_ELEMENTS as f32 / 2.0) / 17.0;
                value.to_le_bytes()
            })
            .collect::<Vec<_>>();
        for (source, bytes) in [
            (ElementType::F16, f16_bytes.as_slice()),
            (ElementType::Bf16, bf16_bytes.as_slice()),
            (ElementType::F32, f32_bytes.as_slice()),
        ] {
            for destination in [ElementType::F16, ElementType::Bf16, ElementType::F32] {
                if source == destination {
                    continue;
                }
                let actual =
                    transcode_dense_bytes(bytes, source, destination, "large.weight", None)
                        .unwrap();
                let element_count = bytes.len() / source.size_bytes() as usize;
                let mut expected = vec![0; element_count * destination.size_bytes() as usize];
                for index in 0..element_count {
                    write_float(
                        &mut expected,
                        destination,
                        index,
                        read_float(bytes, source, index),
                    );
                }
                assert_eq!(actual.as_ref(), expected, "{source:?} -> {destination:?}");
            }
        }

        assert_eq!(bounded_dense_transcode_worker_count(0), 1);
        let workers = bounded_dense_transcode_worker_count(usize::MAX);
        assert!((1..=MAX_DENSE_TRANSCODE_WORKERS).contains(&workers));
    }

    #[test]
    fn dense_materialization_borrows_matching_storage() {
        let bytes = f16::from_f32(1.0).to_bits().to_le_bytes();
        let converted =
            transcode_dense_bytes(&bytes, ElementType::F16, ElementType::F16, "weight", None)
                .unwrap();
        assert!(matches!(converted, Cow::Borrowed(_)));
    }

    #[test]
    fn affine_dense_materialization_applies_logical_transform() {
        let bytes = [(-0.5_f32).to_le_bytes(), 2.0_f32.to_le_bytes()].concat();
        let converted = transcode_dense_bytes(
            &bytes,
            ElementType::F32,
            ElementType::F16,
            "norm.weight",
            Some((
                CanonicalRational::new(1, 1).unwrap(),
                CanonicalRational::new(1, 1).unwrap(),
            )),
        )
        .unwrap();
        let actual = converted
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(actual, [0.5, 3.0]);
        assert!(matches!(converted, Cow::Owned(_)));
    }

    #[test]
    fn affine_dense_materialization_rejects_non_finite_results() {
        let bytes = f32::MAX.to_le_bytes();
        let error = transcode_dense_bytes(
            &bytes,
            ElementType::F32,
            ElementType::F32,
            "weight",
            Some((
                CanonicalRational::new(i64::MAX, 1).unwrap(),
                CanonicalRational::new(0, 1).unwrap(),
            )),
        )
        .expect_err("overflowing affine values must not enter device storage");
        assert!(error.to_string().contains("non-finite"), "{error}");
    }
}
