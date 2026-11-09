//! One cohort-owned, immutable main-vision readback. No device owner or
//! conditioning tensor is cached; each consumer builds its own text inputs.
use super::*;
use ferrum_execution::TensorExecutionOutput;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageMainVisionReuse {
    #[default]
    Off,
    ExactImageWithinCohort,
}

#[derive(Clone, Debug, Serialize)]
pub(in crate::image_generation::p_h3) struct MainVisionBindingReceipt {
    pub cohort_first_job: u64,
    pub producer_job: u64,
    pub producer_request: String,
    pub reused: bool,
    pub retained_output_bytes: usize,
}

struct Retained {
    encoded: ManagedHostBytes,
    family: String,
    dimensions: (u32, u32),
    outputs: Vec<TensorExecutionOutput>,
    receipt: MainVisionBindingReceipt,
}
#[derive(Default)]
struct Slot {
    retained: Option<Retained>,
    scope: Option<HostMemoryScope>,
}

pub(in crate::image_generation::p_h3) struct CohortMainVisionReuse {
    parent: HostMemoryAccount,
    domain: Arc<()>,
    capacity: usize,
    first_job: u64,
    slot: Mutex<Slot>,
}

#[derive(Clone)]
pub(in crate::image_generation::p_h3) struct MainVisionReuseBinding {
    pub cohort: Arc<CohortMainVisionReuse>,
    pub encoded: ManagedHostBytes,
}

impl CohortMainVisionReuse {
    pub fn new(
        parent: &HostMemoryAccount,
        domain: Arc<()>,
        capacity: usize,
        first_job: u64,
    ) -> Result<Arc<Self>> {
        ensure!(
            capacity > 0,
            "main vision reuse requires declared output capacity"
        );
        parent.check_scope_capacity(&[(HostMemoryClass::Output, capacity)])?;
        Ok(Arc::new(Self {
            parent: parent.clone(),
            domain,
            capacity,
            first_job,
            slot: Mutex::new(Slot::default()),
        }))
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn unreserved_capacity(&self) -> Result<usize> {
        let slot = self
            .slot
            .lock()
            .map_err(|_| anyhow::anyhow!("main vision reuse poisoned"))?;
        Ok(if slot.scope.is_some() {
            0
        } else {
            self.capacity
        })
    }
    pub fn try_admit(&self, host: &HostMemoryAccount, domain: &Arc<()>) -> Result<bool> {
        ensure!(
            self.parent.same_parent(host) && Arc::ptr_eq(&self.domain, domain),
            "foreign main vision reuse owner"
        );
        let mut slot = self
            .slot
            .lock()
            .map_err(|_| anyhow::anyhow!("main vision reuse poisoned"))?;
        if slot.scope.is_some() {
            return Ok(true);
        }
        match self
            .parent
            .reserve_scope(&[(HostMemoryClass::Output, self.capacity)])
        {
            Ok(scope) => {
                slot.scope = Some(scope);
                Ok(true)
            }
            Err(e) if e.is_temporary_capacity() => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
    pub fn output_account(&self) -> Result<HostMemoryAccount> {
        let slot = self
            .slot
            .lock()
            .map_err(|_| anyhow::anyhow!("main vision reuse poisoned"))?;
        Ok(slot
            .scope
            .as_ref()
            .context("main vision reuse was not admitted")?
            .account())
    }
}

impl MainVisionReuseBinding {
    /// `request_peak` already includes this request's original ingress and
    /// the cohort Output grant. A distinct producer PNG is also self-owned by
    /// this binding until lookup, so it cannot be treated as external pressure
    /// that might release while the request waits for admission.
    pub fn check_request_capacity(
        &self,
        host: &HostMemoryAccount,
        request_peak: &[(HostMemoryClass, usize)],
        diagnostic_output_bytes: usize,
    ) -> Result<()> {
        ensure!(
            self.cohort.parent.same_parent(host) && host.owns(&self.encoded),
            "foreign main vision admission owner"
        );
        let slot = self
            .cohort
            .slot
            .lock()
            .map_err(|_| anyhow::anyhow!("main vision reuse poisoned"))?;
        let extra = slot
            .retained
            .as_ref()
            .filter(|saved| !saved.encoded.same_region(&self.encoded))
            .map_or(0, |saved| saved.encoded.backing_bytes());
        let mut whole = request_peak.to_vec();
        whole.push((HostMemoryClass::InputPreprocess, extra));
        whole.push((HostMemoryClass::Output, diagnostic_output_bytes));
        host.check_scope_capacity(&whole)?;
        Ok(())
    }

    pub(super) fn lookup(
        &self,
        domain: &Arc<()>,
        recipe: &PreparedVision,
    ) -> Result<Option<(Vec<TensorExecutionOutput>, MainVisionBindingReceipt)>> {
        ensure!(
            Arc::ptr_eq(domain, &self.cohort.domain)
                && self.cohort.parent.owns(&self.encoded)
                && self.encoded.class() == HostMemoryClass::InputPreprocess,
            "foreign main vision reuse input"
        );
        let family = recipe.family.fingerprint()?;
        let mut slot = self
            .cohort
            .slot
            .lock()
            .map_err(|_| anyhow::anyhow!("main vision reuse poisoned"))?;
        ensure!(slot.scope.is_some(), "main vision reuse was not admitted");
        if let Some(saved) = &slot.retained {
            // Same original PNG bytes plus the exact pixel-plan/profile family
            // and immutable pipeline domain imply the same normalized patches.
            // Prompt text, file names, and hashes alone never authorize a hit.
            if saved.family == family
                && saved.dimensions == (recipe.source_width, recipe.source_height)
                && saved.encoded.bytes() == self.encoded.bytes()
            {
                let mut receipt = saved.receipt.clone();
                receipt.reused = true;
                return Ok(Some((saved.outputs.clone(), receipt)));
            }
        }
        // A miss releases the only old output set before the new producer.
        slot.retained = None;
        Ok(None)
    }
    pub(super) fn publish(
        &self,
        recipe: &PreparedVision,
        outputs: Vec<TensorExecutionOutput>,
        job: u64,
        request: &RequestIdentity,
    ) -> Result<MainVisionBindingReceipt> {
        ensure!(
            outputs.len() == 4
                && outputs.iter().all(|o| self.cohort.parent.owns(o.bytes())
                    && o.bytes().class() == HostMemoryClass::Output
                    && o.tensor().element_type() == ElementType::F32),
            "invalid main vision retained outputs"
        );
        let bytes = outputs.iter().try_fold(0usize, |sum, o| {
            sum.checked_add(o.bytes().len())
                .context("main vision payload overflows")
        })?;
        ensure!(
            bytes > 0 && bytes <= self.cohort.capacity,
            "main vision payload exceeds its original output grant"
        );
        let receipt = MainVisionBindingReceipt {
            cohort_first_job: self.cohort.first_job,
            producer_job: job,
            producer_request: request.as_str().into(),
            reused: false,
            retained_output_bytes: bytes,
        };
        let mut slot = self
            .cohort
            .slot
            .lock()
            .map_err(|_| anyhow::anyhow!("main vision reuse poisoned"))?;
        ensure!(
            slot.scope.is_some() && slot.retained.is_none(),
            "main vision output publication requires an admitted empty slot"
        );
        slot.retained = Some(Retained {
            encoded: self.encoded.clone(),
            family: recipe.family.fingerprint()?,
            dimensions: (recipe.source_width, recipe.source_height),
            outputs,
            receipt: receipt.clone(),
        });
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests;
