use super::*;

#[must_use = "a resource lease is the batch owner of committed buffers"]
pub struct StaticProvisioningLease<R>
where
    R: DeviceRuntime,
{
    pub(super) slots: Vec<OwnedLeaseSlot<R::Buffer>>,
    pub(super) identity: ResourceTransactionIdentity,
    pub(super) admission: StaticProvisioningBinding,
    // Backend context drops after all static and dynamic buffers.
    pub(super) runtime: Arc<R>,
}

impl<R> StaticProvisioningLease<R>
where
    R: DeviceRuntime,
{
    pub(super) fn new(
        runtime: Arc<R>,
        identity: &ResourceTransactionIdentity,
        admission: &StaticProvisioningBinding,
        reservations: &ResourceReservationBatch,
    ) -> Self {
        Self {
            slots: reservations
                .reservations()
                .iter()
                .map(OwnedLeaseSlot::new)
                .collect(),
            identity: identity.clone(),
            admission: admission.clone(),
            runtime,
        }
    }

    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub fn state(&self) -> ResourceLeaseState {
        let mut states = self
            .slots
            .iter()
            .filter(|slot| slot.buffer.is_some())
            .map(|slot| slot.entry.state);
        let Some(first) = states.next() else {
            return ResourceLeaseState::Cancelled;
        };
        if states.all(|state| state == first) {
            first
        } else {
            ResourceLeaseState::Mixed
        }
    }

    pub fn entries(&self) -> impl Iterator<Item = &ResourceLeaseEntry> {
        self.slots
            .iter()
            .filter(|slot| slot.buffer.is_some())
            .map(|slot| &slot.entry)
    }

    pub fn plan_static_entries(&self) -> impl Iterator<Item = &ResourceLeaseEntry> {
        self.slots
            .iter()
            .filter(|slot| slot.buffer.is_some())
            .map(|slot| &slot.entry)
    }

    pub(crate) fn view(
        &self,
        resource_id: &ResourceId,
        generation: u64,
    ) -> Result<LeasedBufferView<'_, R::Buffer>, VNextError> {
        let slot = self
            .slots
            .iter()
            .find(|slot| {
                slot.entry.resource_id == *resource_id && slot.entry.generation == generation
            })
            .ok_or_else(|| invalid_resource("lease does not contain that resource generation"))?;
        if slot.entry.state != ResourceLeaseState::Active {
            return Err(VNextError::InvalidLeaseTransition {
                lease_id: self.identity.transaction_id.to_string(),
                from: slot.entry.state.as_str(),
                action: "borrow_live_buffer",
            });
        }
        let descriptor = slot
            .descriptor
            .as_ref()
            .ok_or_else(|| invalid_resource("lease resource is not committed"))?;
        let buffer = slot
            .buffer
            .as_ref()
            .ok_or_else(|| invalid_resource("lease resource buffer is not live"))?;
        Ok(LeasedBufferView {
            identity: &self.identity,
            admission: &self.admission,
            resource_id: &slot.entry.resource_id,
            generation: slot.entry.generation,
            descriptor,
            buffer,
        })
    }

    pub(super) fn buffer(&self, order: usize) -> Option<&R::Buffer> {
        self.slots.get(order).and_then(|slot| slot.buffer.as_ref())
    }

    pub(super) fn install(&mut self, order: usize, allocation: CoreOwnedAllocation<R::Buffer>) {
        self.slots[order].install(allocation);
    }

    pub(super) fn clear(&mut self, order: usize) {
        self.slots[order].clear();
    }

    pub(super) fn transition_subset(
        &mut self,
        orders: &[usize],
        action: ResourceLeaseAction,
    ) -> Result<
        (
            ResourceLeaseState,
            ResourceLeaseState,
            Vec<ResourceLeaseEntry>,
        ),
        VNextError,
    > {
        if orders.is_empty() {
            return Err(invalid_resource(
                "lease transition subset must not be empty",
            ));
        }
        let mut unique = BTreeSet::new();
        let mut common_before = None;
        for &order in orders {
            if !unique.insert(order) {
                return Err(invalid_resource("lease transition subset is duplicated"));
            }
            let slot = self
                .slots
                .get(order)
                .ok_or_else(|| invalid_resource("lease transition order is out of bounds"))?;
            if slot.buffer.is_none() {
                return Err(invalid_resource(
                    "lease transition targets a non-live buffer",
                ));
            }
            let before = slot.entry.state;
            if common_before.is_some_and(|common| common != before) {
                return Err(invalid_resource(
                    "one lease receipt cannot hide heterogeneous before states",
                ));
            }
            if expected_lease_transition(action, before).is_none() {
                return Err(VNextError::InvalidLeaseTransition {
                    lease_id: self.identity.transaction_id.to_string(),
                    from: before.as_str(),
                    action: action.as_str(),
                });
            }
            common_before = Some(before);
        }
        let before = common_before.expect("non-empty subset has a before state");
        let after = expected_lease_transition(action, before)
            .expect("lease subset was preflight validated");
        for &order in orders {
            self.slots[order].entry.state = after;
        }
        Ok((
            before,
            after,
            orders
                .iter()
                .map(|&order| self.slots[order].entry.clone())
                .collect(),
        ))
    }

    pub(super) fn take_owned_buffers(
        &mut self,
        reservations: &ResourceReservationBatch,
    ) -> Vec<ResourceOwnedBuffer<R::Buffer>> {
        self.slots
            .iter_mut()
            .zip(reservations.reservations())
            .enumerate()
            .filter_map(|(order, (slot, reservation))| {
                let allocation = slot.take_allocation()?;
                Some(ResourceOwnedBuffer {
                    order,
                    expected_resource_id: reservation.resource_id.clone(),
                    actual_resource_id: allocation.resource_id,
                    expected_generation: reservation.generation,
                    actual_generation: allocation.generation,
                    expected_descriptor: BufferDescriptor {
                        resource_id: reservation.resource_id.clone(),
                        size_bytes: reservation.size_bytes,
                        alignment_bytes: reservation.alignment_bytes,
                        usage: reservation.usage,
                        element_type: reservation.element_type,
                    },
                    actual_descriptor: allocation.descriptor,
                    buffer: allocation.buffer,
                })
            })
            .collect()
    }

    pub(super) fn restore_owned_buffers(&mut self, buffers: Vec<ResourceOwnedBuffer<R::Buffer>>) {
        for buffer in buffers {
            let (order, allocation) = buffer.into_allocation();
            self.slots[order].restore_allocation(allocation);
        }
    }
}

macro_rules! scoped_resource_admission_request {
    ($name:ident, $single_sequence:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $name {
            pub(super) work_shape: ResourceWorkShape,
            pub(super) fit_policy: AdmissionFitPolicy,
            pub(super) pressure_action: AdmissionPressureAction,
        }

        impl $name {
            pub fn new(
                work_shape: ResourceWorkShape,
                fit_policy: AdmissionFitPolicy,
                pressure_action: AdmissionPressureAction,
            ) -> Result<Self, VNextError> {
                if $single_sequence
                    && (work_shape.immediate_sequences() != 1 || work_shape.fit_sequences() != 1)
                {
                    return Err(invalid_resource(
                        "sequence resource admission requires a single-sequence shape",
                    ));
                }
                Ok(Self {
                    work_shape,
                    fit_policy,
                    pressure_action,
                })
            }

            pub fn work_shape(&self) -> &ResourceWorkShape {
                &self.work_shape
            }

            pub(crate) const fn immediate_shape(&self) -> DynamicResourceShape {
                self.work_shape.immediate_shape()
            }

            pub(crate) const fn fit_shape(&self) -> DynamicResourceShape {
                match self.fit_policy {
                    AdmissionFitPolicy::ImmediateOnly => self.work_shape.immediate_shape(),
                    AdmissionFitPolicy::FullInputMustFit => self.work_shape.fit_shape(),
                }
            }

            pub const fn fit_policy(&self) -> AdmissionFitPolicy {
                self.fit_policy
            }

            pub const fn pressure_action(&self) -> AdmissionPressureAction {
                self.pressure_action
            }
        }
    };
}

scoped_resource_admission_request!(RequestResourceAdmissionRequest, false);
scoped_resource_admission_request!(SequenceResourceAdmissionRequest, true);
