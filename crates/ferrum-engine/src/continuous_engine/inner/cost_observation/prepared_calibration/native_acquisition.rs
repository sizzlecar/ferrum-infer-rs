//! Actual ACK receipt producer is private to startup acquisition. Diagnostic
//! JSON cannot call this adapter or construct that receipt.
use super::*;
impl PreparedOwnerCalibration {
    pub(in crate::continuous_engine::inner) fn native_prefix_restored(
        &mut self,
        receipt: &AcknowledgedProbePrefixRestore,
    ) -> Result<()> {
        let result = (|| {
            self.now()?;
            if self.pending.is_some() {
                return Err(error("native restore crossed an issued inference call"));
            }
            let cohort = self
                .active
                .as_ref()
                .ok_or_else(|| error("native restore outside cohort"))?;
            if cohort.admitted != cohort.slots.len() {
                return Err(error(
                    "native restore requires the complete admitted cohort",
                ));
            }
            let before = receipt.before();
            let slot = cohort
                .slots
                .iter()
                .position(|slot| {
                    slot.id.as_ref() == Some(&before.request_id)
                        && slot.owner == Some(before.owner_incarnation)
                        && slot.generated == 0
                        && !slot.released
                        && !slot.completed
                })
                .ok_or_else(|| error("actual native restore owner differs from admitted slot"))?;
            let (pass, ordinal) = (cohort.pass, cohort.ordinal);
            let native = self
                .declaration
                .native_prefix_acquisition
                .as_ref()
                .and_then(|plan| plan.phases[pass].get(ordinal))
                .and_then(Option::as_ref)
                .ok_or_else(|| error("native restore was not frozen before the source"))?;
            if !receipt.matches_declaration(native) {
                return Err(error(
                    "actual native restore differs from frozen acquisition",
                ));
            }
            self.charge(receipt.retained_payload_bytes()?)?;
            self.preparation_event(receipt.event(phase(pass)?, ordinal, slot)?)
        })();
        self.checked(result)
    }
}
