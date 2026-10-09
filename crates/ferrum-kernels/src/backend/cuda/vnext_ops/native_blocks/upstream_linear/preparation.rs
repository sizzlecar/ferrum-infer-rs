//! Encoding-local metadata ownership. Both purposes validate the same wave;
//! only a binding-only strict fallback can omit an unconsumed replay digest.
use ferrum_interfaces::vnext::{
    PreparedProjectionNumerics, PreparedUpstreamProjectionRoute, PreparedUpstreamProjectionWave,
    UpstreamNativePlanFacts, UpstreamProjectionWaveFacts,
};
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProjectionPreparation {
    Full,
    BindingsOnly,
}

pub(crate) struct ReplayFingerprint(Option<String>);

impl ReplayFingerprint {
    pub(crate) fn required(&self) -> Result<&str, String> {
        self.0
            .as_deref()
            .ok_or_else(|| "binding-only strict wave has no compute replay fingerprint".into())
    }
}

impl ProjectionPreparation {
    pub(crate) fn numerics<'a>(
        self,
        numerics: &'a PreparedProjectionNumerics,
    ) -> Cow<'a, PreparedProjectionNumerics> {
        match self {
            Self::Full => Cow::Owned(numerics.clone()),
            Self::BindingsOnly => Cow::Borrowed(numerics),
        }
    }

    pub(crate) fn prepare_wave(
        self,
        prepared: &PreparedProjectionNumerics,
        facts: &UpstreamProjectionWaveFacts,
        native: Option<&UpstreamNativePlanFacts>,
    ) -> Result<(PreparedUpstreamProjectionWave, ReplayFingerprint), String> {
        // Never infer a strict route before rebuilding the physical wave proof.
        // Invalid shape, range, contract or native facts still fail in both modes.
        let decision = PreparedUpstreamProjectionWave::prepare(prepared, facts, native)?;
        let omit = self == Self::BindingsOnly
            && matches!(
                decision.route(),
                PreparedUpstreamProjectionRoute::StrictBase { .. }
            );
        let fingerprint = if omit {
            None
        } else {
            Some(decision.fingerprint()?)
        };
        Ok((decision, ReplayFingerprint(fingerprint)))
    }
}
