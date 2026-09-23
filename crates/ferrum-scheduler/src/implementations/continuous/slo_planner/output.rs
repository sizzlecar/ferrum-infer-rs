use super::types::{OutputByteBacking, OutputCreditView, PlanningUnknownReason};
use ferrum_interfaces::output_flow::PrepaidOutputCapacityView;

impl TryFrom<PrepaidOutputCapacityView> for OutputCreditView {
    type Error = PlanningUnknownReason;

    fn try_from(view: PrepaidOutputCapacityView) -> Result<Self, Self::Error> {
        let overflow = PlanningUnknownReason::ArithmeticOverflow;
        Ok(Self {
            available_token_commands: view
                .no_drain_token_commands()
                .try_into()
                .map_err(|_| overflow)?,
            byte_backing: OutputByteBacking::PrepaidLifetime {
                remaining_token_commands: view
                    .remaining_token_commands()
                    .try_into()
                    .map_err(|_| overflow)?,
                remaining_wire_bytes: view
                    .remaining_wire_bytes()
                    .try_into()
                    .map_err(|_| overflow)?,
            },
        })
    }
}

impl OutputCreditView {
    /// Advance one hypothetical output command, returning only the additional
    /// shared bytes it needs. This view never owns a grant or refills capacity.
    pub(super) fn after_token(self) -> Result<(Self, u64), PlanningUnknownReason> {
        let blocked = PlanningUnknownReason::OutputOrResourceBlocked;
        let available_token_commands = self
            .available_token_commands
            .checked_sub(1)
            .ok_or(blocked)?;
        let (byte_backing, shared_bytes) = match self.byte_backing {
            OutputByteBacking::Incremental {
                available_bytes,
                bytes_per_token_upper_bound,
            } => {
                let bytes = bytes_per_token_upper_bound.ok_or(blocked)?.get();
                (
                    OutputByteBacking::Incremental {
                        available_bytes: available_bytes.checked_sub(bytes).ok_or(blocked)?,
                        bytes_per_token_upper_bound,
                    },
                    bytes,
                )
            }
            OutputByteBacking::PrepaidLifetime {
                remaining_token_commands,
                remaining_wire_bytes,
            } => (
                OutputByteBacking::PrepaidLifetime {
                    remaining_token_commands: remaining_token_commands
                        .checked_sub(1)
                        .ok_or(blocked)?,
                    // Actual wire lengths are unknowable before decoding. The
                    // reservoir covers the whole remaining producer contract;
                    // do not invent per-token lengths or consumer releases.
                    remaining_wire_bytes,
                },
                0,
            ),
        };
        Ok((
            Self {
                available_token_commands,
                byte_backing,
            },
            shared_bytes,
        ))
    }
}
