use super::*;
use anyhow::{ensure, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use half::bf16;
use sha2::{Digest, Sha256};
use std::sync::Arc;
mod publication;
pub(super) use publication::PreparedRefinedTextCaches;

/// Request and prompt identity plus the actual immutable weight installation.
/// Construction hashes the BF16 input seen by condition_proj, without a large
/// temporary Vec. A path/name alone cannot authenticate a cached refinement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PH3ImageDiTConditioningKey {
    request: RequestIdentity,
    snapshot: Arc<str>,
    rows: u64,
    text_digest: [u8; 32],
    refiner_digest: [u8; 32],
}
impl PH3ImageDiTConditioningKey {
    pub fn request_identity(&self) -> &RequestIdentity {
        &self.request
    }
    /// Check the real immutable installation before admitting a cached request
    /// to a continuation. This exposes no cache construction or mutation.
    pub fn matches_snapshot(&self, evidence: &SharedComponentWeightsEvidence) -> bool {
        self.snapshot.as_ref() == evidence.snapshot_fingerprint()
    }
    pub(crate) fn from_connector(
        c: &PH3ImageDiTConfig,
        connector: &ManagedHostTensor,
        requests: &[RequestIdentity],
        snapshot: &SharedComponentWeightsEvidence,
        metadata_capacity: usize,
    ) -> Result<Vec<Self>> {
        c.validate()?;
        ensure!(
            connector.tensor().dimensions() == [c.text_rows()?, c.text_dim]
                && connector.tensor().element_type() == ElementType::F32
                && connector.tensor().layout() == &ResolvedTensorLayout::Contiguous,
            "conditioning key requires exact F32 connector storage"
        );
        ensure!(
            requests.len() == c.requests.len()
                && requests.iter().collect::<BTreeSet<_>>().len() == requests.len(),
            "request identities must be distinct and cover the batch"
        );
        let refiner_bytes = c
            .requests
            .iter()
            .try_fold(0usize, |sum, r| -> Result<usize> {
                sum.checked_add(refiner_identity(c, r.text_rows)?.len())
                    .context("conditioning metadata overflow")
            })?;
        let bytes = requests.iter().try_fold(
            snapshot
                .snapshot_fingerprint()
                .len()
                .checked_add(refiner_bytes)
                .context("conditioning metadata overflow")?,
            |sum, id| {
                sum.checked_add(std::mem::size_of::<Self>())
                    .and_then(|n| n.checked_add(id.as_str().len()))
                    .context("conditioning metadata overflow")
            },
        )?;
        ensure!(
            bytes <= metadata_capacity,
            "conditioning keys exceed declared metadata capacity"
        );
        let snapshot: Arc<str> = Arc::from(snapshot.snapshot_fingerprint());
        let mut offset = 0usize;
        c.requests
            .iter()
            .zip(requests)
            .map(|(r, request)| {
                let elements = usize::try_from(
                    r.text_rows
                        .checked_mul(c.text_dim)
                        .context("text extent overflow")?,
                )?;
                let digest = digest(
                    c.text_dim,
                    r.text_rows,
                    &connector.bytes().bytes()[4 * offset..4 * (offset + elements)],
                )?;
                offset += elements;
                Ok(Self {
                    request: request.clone(),
                    snapshot: snapshot.clone(),
                    rows: r.text_rows,
                    text_digest: digest,
                    refiner_digest: current_refiner_digest(c, r.text_rows)?,
                })
            })
            .collect()
    }
}
fn current_refiner_digest(c: &PH3ImageDiTConfig, rows: u64) -> Result<[u8; 32]> {
    Ok(Sha256::digest(refiner_identity(c, rows)?).into())
}
fn refiner_identity(c: &PH3ImageDiTConfig, rows: u64) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(
        c.attention_policy,
        c.refiner.for_request_rows(rows)?,
    ))?)
}
fn digest(width: u64, rows: u64, source: &[u8]) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(width.to_le_bytes());
    hash.update(rows.to_le_bytes());
    for b in source.chunks_exact(4) {
        let v = bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap()));
        ensure!(
            v.is_finite(),
            "connector must remain finite at BF16 conditioning boundary"
        );
        hash.update(v.to_bits().to_le_bytes());
    }
    Ok(hash.finalize().into())
}
#[cfg(test)]
pub(super) fn fixture_keys(
    c: &PH3ImageDiTConfig,
    connector: &ManagedHostTensor,
) -> Vec<PH3ImageDiTConditioningKey> {
    let mut offset = 0usize;
    c.requests
        .iter()
        .enumerate()
        .map(|(index, r)| {
            let elements = (r.text_rows * c.text_dim) as usize;
            let key = PH3ImageDiTConditioningKey {
                request: RequestIdentity::new(format!("request.dit.{index}")).unwrap(),
                snapshot: Arc::from("test.installation"),
                rows: r.text_rows,
                text_digest: digest(
                    c.text_dim,
                    r.text_rows,
                    &connector.bytes().bytes()[4 * offset..4 * (offset + elements)],
                )
                .unwrap(),
                refiner_digest: current_refiner_digest(c, r.text_rows).unwrap(),
            };
            offset += elements;
            key
        })
        .collect()
}
pub(super) fn validate_keys(
    c: &PH3ImageDiTConfig,
    keys: &[PH3ImageDiTConditioningKey],
) -> Result<()> {
    ensure!(
        keys.len() == c.requests.len()
            && !keys.is_empty()
            && keys
                .iter()
                .map(|k| &k.request)
                .collect::<BTreeSet<_>>()
                .len()
                == keys.len(),
        "conditioning key arity or request identity differs"
    );
    for (key, r) in keys.iter().zip(&c.requests) {
        ensure!(
            key.rows == r.text_rows
                && key.snapshot == keys[0].snapshot
                && key.refiner_digest == current_refiner_digest(c, r.text_rows)?,
            "conditioning shape, refiner configuration or shared weight installation differs"
        );
    }
    Ok(())
}
pub(super) fn validate_connector_keys(
    c: &PH3ImageDiTConfig,
    connector: &ManagedHostTensor,
    keys: &[PH3ImageDiTConditioningKey],
) -> Result<()> {
    validate_keys(c, keys)?;
    let mut offset = 0usize;
    for (r, key) in c.requests.iter().zip(keys) {
        let elements = usize::try_from(
            r.text_rows
                .checked_mul(c.text_dim)
                .context("text extent overflow")?,
        )?;
        ensure!(
            digest(
                c.text_dim,
                r.text_rows,
                &connector.bytes().bytes()[4 * offset..4 * (offset + elements)]
            )? == key.text_digest,
            "prompt differs from conditioning key"
        );
        offset += elements;
    }
    Ok(())
}

/// An InputPreprocess copy, independently accounted for the request lifetime.
/// It never turns an Output owner into an Input owner by relabeling its class.
#[derive(Clone)]
pub struct PH3ImageRefinedTextCache {
    key: PH3ImageDiTConditioningKey,
    tensor: ManagedHostTensor,
}
impl PH3ImageRefinedTextCache {
    pub fn key(&self) -> &PH3ImageDiTConditioningKey {
        &self.key
    }
    /// The caller pairs this completed output with its submitted first-phase
    /// Prepared inputs. All allocations complete before any cache is returned.
    pub(crate) fn publish(
        c: &PH3ImageDiTConfig,
        host: &HostMemoryAccount,
        prepared: &PreparedPH3ImageDiTInputs,
        declared: &CompiledTensorPort,
        output: &TensorExecutionOutput,
    ) -> Result<Vec<Self>> {
        ensure!(
            c.phase == PH3ImageDiTPhase::WithTextRefiner,
            "only the first phase publishes refinement"
        );
        let keys = prepared
            .conditioning
            .as_ref()
            .context("prepared first-phase conditioning manifest is missing")?;
        validate_keys(c, keys)?;
        ensure!(
            declared.value_id().as_str() == PH3ImageDiTFamily::REFINED_TEXT_ID
                && output.value_id() == declared.value_id()
                && output.tensor() == declared.tensor()
                && output.tensor().dimensions() == [c.text_rows()?, c.hidden]
                && output.tensor().element_type() == ElementType::Bf16
                && output.tensor().layout() == &ResolvedTensorLayout::Contiguous
                && output.bytes().class() == HostMemoryClass::Output
                && host.owns(output.bytes()),
            "refined cache output port, storage or Host parent differs"
        );
        ensure!(
            output
                .bytes()
                .bytes()
                .chunks_exact(2)
                .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()),
            "refined cache values must be finite"
        );
        let mut result = Vec::with_capacity(keys.len());
        let mut offset = 0usize;
        for (r, key) in c.requests.iter().zip(keys) {
            let bytes = usize::try_from(
                r.text_rows
                    .checked_mul(c.hidden)
                    .and_then(|n| n.checked_mul(2))
                    .context("cache extent overflow")?,
            )?;
            let mut allocation = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
            allocation
                .as_mut()
                .copy_from_slice(&output.bytes().bytes()[offset..offset + bytes]);
            offset += bytes;
            result.push(Self {
                key: key.clone(),
                tensor: inputs::managed(
                    PH3ImageDiTFamily::REFINED_TEXT_ID,
                    config::tensor(vec![r.text_rows, c.hidden], ElementType::Bf16),
                    allocation.freeze(),
                )?,
            });
        }
        Ok(result)
    }
}
pub(super) fn concatenate(
    c: &PH3ImageDiTConfig,
    host: &HostMemoryAccount,
    caches: &[PH3ImageRefinedTextCache],
    keys: &[PH3ImageDiTConditioningKey],
) -> Result<ManagedHostTensor> {
    validate_keys(c, keys)?;
    ensure!(caches.len() == keys.len(), "refined cache count differs");
    for ((cache, key), r) in caches.iter().zip(keys).zip(&c.requests) {
        ensure!(
            &cache.key == key,
            "refined cache prompt, request or installed weights differs"
        );
        inputs::check(
            host,
            &cache.tensor,
            &[r.text_rows, c.hidden],
            ElementType::Bf16,
        )?;
    }
    let bytes = usize::try_from(
        c.text_rows()?
            .checked_mul(c.hidden)
            .and_then(|n| n.checked_mul(2))
            .context("batched cache extent overflow")?,
    )?;
    let mut out = host.allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)?;
    let mut offset = 0;
    for cache in caches {
        let bytes = cache.tensor.bytes().bytes();
        out.as_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
        offset += bytes.len();
    }
    inputs::managed(
        PH3ImageDiTFamily::REFINED_TEXT_ID,
        config::tensor(vec![c.text_rows()?, c.hidden], ElementType::Bf16),
        out.freeze(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn account() -> HostMemoryAccount {
        HostMemoryAccount::new(
            HostMemoryLimits::new(
                1 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 1 << 20,
                    output_bytes: 0,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        )
    }
    #[test]
    fn cached_text_retains_request_prompt_weight_identity_and_host_capacity() {
        let c = super::super::tests::config(2).with_phase(PH3ImageDiTPhase::RefinedText);
        let host = account();
        let keys = c
            .requests
            .iter()
            .enumerate()
            .map(|(i, r)| PH3ImageDiTConditioningKey {
                request: RequestIdentity::new(format!("request.cache.{i}")).unwrap(),
                snapshot: Arc::from("test.installed.snapshot"),
                rows: r.text_rows,
                text_digest: [i as u8; 32],
                refiner_digest: current_refiner_digest(&c, r.text_rows).unwrap(),
            })
            .collect::<Vec<_>>();
        let caches = keys
            .iter()
            .enumerate()
            .map(|(index, key)| {
                let mut a = host
                    .allocate_zeroed(
                        HostMemoryClass::InputPreprocess,
                        (key.rows * c.hidden * 2) as usize,
                    )
                    .unwrap();
                for b in a.as_mut().chunks_exact_mut(2) {
                    b.copy_from_slice(&bf16::from_f32(index as f32 + 0.5).to_bits().to_le_bytes());
                }
                PH3ImageRefinedTextCache {
                    key: key.clone(),
                    tensor: inputs::managed(
                        PH3ImageDiTFamily::REFINED_TEXT_ID,
                        config::tensor(vec![key.rows, c.hidden], ElementType::Bf16),
                        a.freeze(),
                    )
                    .unwrap(),
                }
            })
            .collect::<Vec<_>>();
        let before = host.usage().unwrap().committed_bytes();
        let tensor = concatenate(&c, &host, &caches, &keys).unwrap();
        assert_eq!(tensor.bytes().len(), before);
        assert_eq!(host.usage().unwrap().committed_bytes(), 2 * before);
        let row_bytes = (keys[0].rows * c.hidden * 2) as usize;
        assert_eq!(
            &tensor.bytes().bytes()[..2],
            &bf16::from_f32(0.5).to_bits().to_le_bytes()
        );
        assert_eq!(
            &tensor.bytes().bytes()[row_bytes..row_bytes + 2],
            &bf16::from_f32(1.5).to_bits().to_le_bytes()
        );
        drop(tensor);
        for field in 0..3 {
            let mut bad = keys.clone();
            match field {
                0 => bad[0].request = RequestIdentity::new("request.different").unwrap(),
                1 => bad[0].snapshot = Arc::from("different.snapshot"),
                _ => bad[0].text_digest[0] ^= 1,
            };
            assert!(concatenate(&c, &host, &caches, &bad).is_err());
            assert_eq!(host.usage().unwrap().committed_bytes(), before);
        }
        assert!(concatenate(&c, &account(), &caches, &keys).is_err());
        let one = PH3ImageDiTConfig::new(
            &super::super::tests::transformer(),
            PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
            3,
            vec![c.requests[1].clone()],
            PH3ImageDiTPhase::RefinedText,
            PH3ImageStackProjectionStorage::default(),
        )
        .unwrap();
        let regrouped = concatenate(
            &one,
            &host,
            std::slice::from_ref(&caches[1]),
            std::slice::from_ref(&keys[1]),
        )
        .unwrap();
        assert_eq!(
            &regrouped.bytes().bytes()[..2],
            &bf16::from_f32(1.5).to_bits().to_le_bytes()
        );
        drop(regrouped);
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        // Same raw weights, prompt, request and tensor dimensions, but a
        // different refiner normalization is a different cache computation.
        let mut changed = serde_json::to_value(&c).unwrap();
        changed["refiner"]["norm_eps"] =
            serde_json::to_value(CanonicalRational::from_decimal_str("0.0002").unwrap()).unwrap();
        let changed: PH3ImageDiTConfig = serde_json::from_value(changed).unwrap();
        changed.validate().unwrap();
        assert!(concatenate(&changed, &host, &caches, &keys).is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        let mut changed = serde_json::to_value(&c).unwrap();
        changed["refiner"]["heads"] = serde_json::json!(2);
        let changed: PH3ImageDiTConfig = serde_json::from_value(changed).unwrap();
        changed.validate().unwrap();
        assert!(concatenate(&changed, &host, &caches, &keys).is_err());
        drop(caches);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    #[test]
    fn own_conditioning_key_survives_other_prompt_changes_and_batch_removal() {
        let c = super::super::tests::config(2);
        let host = account();
        let tensor = |shape: Vec<u64>, values: Vec<f32>| {
            let mut a = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
                .unwrap();
            for (b, value) in a.as_mut().chunks_exact_mut(4).zip(values) {
                b.copy_from_slice(&value.to_le_bytes());
            }
            inputs::managed(
                "value.connector",
                config::tensor(shape, ElementType::F32),
                a.freeze(),
            )
            .unwrap()
        };
        let original = tensor(vec![6, 4], vec![0.25; 24]);
        let mut changed = vec![0.25; 24];
        changed[..12].fill(0.75);
        let changed = tensor(vec![6, 4], changed);
        let original_keys = fixture_keys(&c, &original);
        let changed_keys = fixture_keys(&c, &changed);
        assert_ne!(original_keys[0], changed_keys[0]);
        assert_eq!(original_keys[1], changed_keys[1]);
        let one = PH3ImageDiTConfig::new(
            &super::super::tests::transformer(),
            PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
            3,
            vec![c.requests[1].clone()],
            PH3ImageDiTPhase::WithTextRefiner,
            PH3ImageStackProjectionStorage::default(),
        )
        .unwrap();
        let single = tensor(vec![3, 4], vec![0.25; 12]);
        let mut single_keys = fixture_keys(&one, &single);
        single_keys[0].request = original_keys[1].request.clone();
        assert_eq!(single_keys[0], original_keys[1]);
    }
}
