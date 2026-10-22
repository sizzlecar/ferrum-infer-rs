//! Bounded compilation metadata only. No session, allocation or tensor payload
//! is retained here. Each cache belongs to one immutable pipeline composition.
use super::*;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImagePlanCachePolicy {
    Disabled,
    /// At most one compiled plan for each of the four pipeline stages. A shape,
    /// mask, profile or compile-option change replaces that stage's old entry.
    #[default]
    LastPerStage,
}

struct CachedStage<R: DeviceRuntime> {
    options: ProgramPlanCompileOptions,
    component: Arc<CompiledTensorComponent<R>>,
    receipt: ImageStagePlanReceipt,
}

pub(super) struct StagePlanCache<R: DeviceRuntime> {
    entries: Mutex<[Option<CachedStage<R>>; 4]>,
}

impl<R: DeviceRuntime> StagePlanCache<R> {
    pub(super) fn new() -> Self {
        Self {
            entries: Mutex::new(std::array::from_fn(|_| None)),
        }
    }

    pub(super) fn get(
        &self,
        stage: ImageStage,
        batch_size: usize,
        family_fingerprint: &str,
        options: &ProgramPlanCompileOptions,
    ) -> anyhow::Result<Option<(Arc<CompiledTensorComponent<R>>, ImageStagePlanReceipt)>> {
        let entries = self
            .entries
            .lock()
            .map_err(|_| anyhow::anyhow!("image plan cache poisoned"))?;
        Ok(entries[index(stage)]
            .as_ref()
            .filter(|entry| {
                entry.receipt.batch_size == batch_size
                    && entry.receipt.family_fingerprint == family_fingerprint
                    && &entry.options == options
            })
            .map(|entry| (entry.component.clone(), entry.receipt.clone())))
    }

    pub(super) fn insert(
        &self,
        component: Arc<CompiledTensorComponent<R>>,
        receipt: ImageStagePlanReceipt,
        options: ProgramPlanCompileOptions,
    ) -> anyhow::Result<()> {
        let stage = receipt.stage;
        let old = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| anyhow::anyhow!("image plan cache poisoned"))?;
            entries[index(stage)].replace(CachedStage {
                options,
                component,
                receipt,
            })
        };
        drop(old);
        Ok(())
    }
}

fn index(stage: ImageStage) -> usize {
    match stage {
        ImageStage::TextEncoder => 0,
        ImageStage::Noise => 1,
        ImageStage::Denoiser => 2,
        ImageStage::Decoder => 3,
    }
}
