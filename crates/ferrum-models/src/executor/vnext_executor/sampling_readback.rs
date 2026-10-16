//! Select product readbacks without changing the complete device execution plan.
use super::*;

#[cfg(test)]
mod runtime_tests;

/// Uniform waves retain their existing allocation-free mode representation.
/// Participant modes exist only when a pure decode wave can safely avoid
/// escalating exact greedy rows to full host sampling.
pub(super) enum ProductReadbackModes {
    Uniform {
        mode: VNextProductOutputMode,
        participant_count: usize,
    },
    Participant(Vec<VNextProductOutputMode>),
}

/// Classify borrowed roles before resource preparation without allocating the
/// per-participant modes that are resolved after diagnostic capture admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ReadbackDecision {
    Uniform(VNextProductOutputMode),
    Participant,
}

fn participant_greedy_eligible(role: &VNextParticipantOutputRole, vocabulary_size: usize) -> bool {
    matches!(role, VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax { token_mask, .. })
        if token_mask.as_ref().is_none_or(|mask| mask.valid_token_mask.iter().take(vocabulary_size).any(|flag| *flag != 0)))
        // Keep host repetition arithmetic for newly separated scalar rows.
        // Existing homogeneous device sampling remains unchanged.
        && !product_repetition_input(role.logits_policy(), VNextProductOutputMode::GreedyToken).is_active()
}

impl ReadbackDecision {
    pub(super) fn classify<'a>(
        strategy: SamplingReadbackMode,
        kind: VNextExecutionWaveKind,
        logits_type: ElementType,
        vocabulary_size: usize,
        diagnostic_capture: bool,
        teacher_forced: bool,
        roles: impl ExactSizeIterator<Item = &'a VNextParticipantOutputRole> + Clone,
    ) -> Self {
        let uniform = if teacher_forced {
            VNextProductOutputMode::FullLogits
        } else {
            product_output_mode_for_roles(kind, roles.clone())
        };
        let fallback = Self::Uniform(uniform);
        if strategy != SamplingReadbackMode::Participant
            || kind != VNextExecutionWaveKind::Decode
            || diagnostic_capture
            || teacher_forced
            || !matches!(logits_type, ElementType::F32 | ElementType::F16)
            || uniform != VNextProductOutputMode::FullLogits
        {
            return fallback;
        }
        if !roles
            .clone()
            .any(|role| participant_greedy_eligible(role, vocabulary_size))
        {
            return fallback;
        }
        Self::Participant
    }

    pub(super) fn allows_resident(
        self,
        kind: VNextExecutionWaveKind,
        diagnostic_capture: bool,
    ) -> bool {
        kind == VNextExecutionWaveKind::Decode
            && !diagnostic_capture
            && matches!(
                self,
                Self::Uniform(VNextProductOutputMode::GreedyToken) | Self::Participant
            )
    }
}

impl ProductReadbackModes {
    pub(super) fn resolve<'a>(
        strategy: SamplingReadbackMode,
        kind: VNextExecutionWaveKind,
        logits_type: ElementType,
        vocabulary_size: usize,
        diagnostic_capture: bool,
        teacher_forced: bool,
        roles: impl ExactSizeIterator<Item = &'a VNextParticipantOutputRole> + Clone,
    ) -> Self {
        match ReadbackDecision::classify(
            strategy,
            kind,
            logits_type,
            vocabulary_size,
            diagnostic_capture,
            teacher_forced,
            roles.clone(),
        ) {
            ReadbackDecision::Uniform(mode) => Self::Uniform {
                mode,
                participant_count: roles.len(),
            },
            ReadbackDecision::Participant => Self::Participant(
                roles
                    .map(|role| {
                        if participant_greedy_eligible(role, vocabulary_size) {
                            VNextProductOutputMode::GreedyToken
                        } else {
                            VNextProductOutputMode::FullLogits
                        }
                    })
                    .collect(),
            ),
        }
    }

    pub(super) fn decision(&self) -> ReadbackDecision {
        match self {
            Self::Uniform { mode, .. } => ReadbackDecision::Uniform(*mode),
            Self::Participant(_) => ReadbackDecision::Participant,
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Uniform {
                participant_count, ..
            } => *participant_count,
            Self::Participant(modes) => modes.len(),
        }
    }

    pub(super) fn at(&self, index: usize) -> Option<VNextProductOutputMode> {
        match self {
            Self::Uniform {
                mode,
                participant_count,
            } => (index < *participant_count).then_some(*mode),
            Self::Participant(modes) => modes.get(index).copied(),
        }
    }

    pub(super) fn uniform(&self) -> Option<VNextProductOutputMode> {
        match self {
            Self::Uniform { mode, .. } => Some(*mode),
            Self::Participant(_) => None,
        }
    }

    pub(super) fn full_participants(&self) -> usize {
        match self {
            Self::Uniform {
                mode: VNextProductOutputMode::FullLogits,
                participant_count,
            } => *participant_count,
            Self::Uniform {
                mode: VNextProductOutputMode::GreedyToken,
                ..
            } => 0,
            Self::Participant(modes) => modes
                .iter()
                .filter(|mode| **mode == VNextProductOutputMode::FullLogits)
                .count(),
        }
    }

    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = VNextProductOutputMode> + '_ {
        (0..self.len()).map(move |index| match self {
            Self::Uniform { mode, .. } => *mode,
            Self::Participant(modes) => modes[index],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::model_executor::{GreedyRepetitionPenalty, TokenSelectionMask};

    fn greedy() -> VNextParticipantOutputRole {
        VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
            token_mask: Some(TokenSelectionMask::new(vec![1, 0, 1])),
            repetition_penalty: None,
        })
    }

    #[test]
    fn participant_readback_keeps_each_requests_mask_and_neutral_full_inputs() {
        let roles = [
            greedy(),
            VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits),
            greedy(),
        ];
        for logits_type in [ElementType::F32, ElementType::F16] {
            let decision = ReadbackDecision::classify(
                SamplingReadbackMode::Participant,
                VNextExecutionWaveKind::Decode,
                logits_type,
                3,
                false,
                false,
                roles.iter(),
            );
            assert_eq!(decision, ReadbackDecision::Participant);
            assert!(decision.allows_resident(VNextExecutionWaveKind::Decode, false));
            let modes = ProductReadbackModes::resolve(
                SamplingReadbackMode::Participant,
                VNextExecutionWaveKind::Decode,
                logits_type,
                3,
                false,
                false,
                roles.iter(),
            );
            assert_eq!(modes.decision(), decision);
            assert_eq!(
                modes.iter().collect::<Vec<_>>(),
                [
                    VNextProductOutputMode::GreedyToken,
                    VNextProductOutputMode::FullLogits,
                    VNextProductOutputMode::GreedyToken
                ]
            );
            assert_eq!(modes.at(3), None);
            for (role, mode) in roles.iter().zip(modes.iter()) {
                assert_eq!(
                    normalized_product_token_mask(role.logits_policy(), mode, 3),
                    if mode == VNextProductOutputMode::GreedyToken {
                        vec![1, 0, 1]
                    } else {
                        vec![1, 1, 1]
                    }
                );
                assert!(!product_repetition_input(role.logits_policy(), mode).is_active());
            }
        }
    }

    #[test]
    fn participant_readback_preserves_precision_capture_and_wave_boundaries() {
        let roles = [
            greedy(),
            VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits),
        ];
        for (strategy, kind, dtype, capture, teacher) in [
            (
                SamplingReadbackMode::Uniform,
                VNextExecutionWaveKind::Decode,
                ElementType::F32,
                false,
                false,
            ),
            (
                SamplingReadbackMode::Participant,
                VNextExecutionWaveKind::Prefill,
                ElementType::F32,
                false,
                false,
            ),
            (
                SamplingReadbackMode::Participant,
                VNextExecutionWaveKind::Mixed,
                ElementType::F32,
                false,
                false,
            ),
            (
                SamplingReadbackMode::Participant,
                VNextExecutionWaveKind::Decode,
                ElementType::Bf16,
                false,
                false,
            ),
            (
                SamplingReadbackMode::Participant,
                VNextExecutionWaveKind::Decode,
                ElementType::F32,
                true,
                false,
            ),
            (
                SamplingReadbackMode::Participant,
                VNextExecutionWaveKind::Decode,
                ElementType::F32,
                false,
                true,
            ),
        ] {
            let decision = ReadbackDecision::classify(
                strategy,
                kind,
                dtype,
                3,
                capture,
                teacher,
                roles.iter(),
            );
            assert_eq!(
                decision,
                ReadbackDecision::Uniform(VNextProductOutputMode::FullLogits)
            );
            assert!(!decision.allows_resident(kind, capture));
            let modes = ProductReadbackModes::resolve(
                strategy,
                kind,
                dtype,
                3,
                capture,
                teacher,
                roles.iter(),
            );
            assert_eq!(modes.uniform(), Some(VNextProductOutputMode::FullLogits));
            assert_eq!(modes.decision(), decision);
        }
    }

    #[test]
    fn participant_readback_does_not_change_repetition_arithmetic_for_new_scalar_rows() {
        let repetition = VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: Some(GreedyRepetitionPenalty::new(1.1, vec![1, 2])),
        });
        let roles = [
            greedy(),
            repetition.clone(),
            VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits),
        ];
        let modes = ProductReadbackModes::resolve(
            SamplingReadbackMode::Participant,
            VNextExecutionWaveKind::Decode,
            ElementType::F32,
            3,
            false,
            false,
            roles.iter(),
        );
        assert_eq!(
            modes.iter().collect::<Vec<_>>(),
            [
                VNextProductOutputMode::GreedyToken,
                VNextProductOutputMode::FullLogits,
                VNextProductOutputMode::FullLogits
            ]
        );
        assert!(modes
            .decision()
            .allows_resident(VNextExecutionWaveKind::Decode, false));
        let homogeneous = [greedy(), repetition];
        let modes = ProductReadbackModes::resolve(
            SamplingReadbackMode::Participant,
            VNextExecutionWaveKind::Decode,
            ElementType::F16,
            3,
            false,
            false,
            homogeneous.iter(),
        );
        assert_eq!(modes.uniform(), Some(VNextProductOutputMode::GreedyToken));
        assert!(
            product_repetition_input(homogeneous[1].logits_policy(), modes.at(1).unwrap())
                .is_active()
        );
    }

    #[test]
    fn participant_readback_keeps_statically_empty_vocabulary_masks_on_host() {
        for mask in [vec![], vec![0, 0, 0], vec![0, 0, 0, 1]] {
            let roles = [
                greedy(),
                VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
                    token_mask: Some(TokenSelectionMask::new(mask)),
                    repetition_penalty: None,
                }),
                VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits),
            ];
            let modes = ProductReadbackModes::resolve(
                SamplingReadbackMode::Participant,
                VNextExecutionWaveKind::Decode,
                ElementType::F32,
                3,
                false,
                false,
                roles.iter(),
            );
            assert_eq!(modes.at(1), Some(VNextProductOutputMode::FullLogits));
            assert_eq!(modes.decision(), ReadbackDecision::Participant);
        }
    }

    #[test]
    fn resident_readback_rejects_mixed_without_an_eligible_scalar_row() {
        let mut ineligible = [vec![], vec![0, 0, 0], vec![0, 0, 0, 1]]
            .into_iter()
            .map(|mask| {
                VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
                    token_mask: Some(TokenSelectionMask::new(mask)),
                    repetition_penalty: None,
                })
            })
            .collect::<Vec<_>>();
        ineligible.push(VNextParticipantOutputRole::Decode(
            LogitsReturnPolicy::GreedyArgmax {
                token_mask: None,
                repetition_penalty: Some(GreedyRepetitionPenalty::new(1.1, vec![1, 2])),
            },
        ));
        for role in ineligible {
            let roles = [
                role,
                VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits),
            ];
            for dtype in [ElementType::F16, ElementType::F32] {
                let decision = ReadbackDecision::classify(
                    SamplingReadbackMode::Participant,
                    VNextExecutionWaveKind::Decode,
                    dtype,
                    3,
                    false,
                    false,
                    roles.iter(),
                );
                assert_eq!(
                    decision,
                    ReadbackDecision::Uniform(VNextProductOutputMode::FullLogits)
                );
                assert!(!decision.allows_resident(VNextExecutionWaveKind::Decode, false));
                let modes = ProductReadbackModes::resolve(
                    SamplingReadbackMode::Participant,
                    VNextExecutionWaveKind::Decode,
                    dtype,
                    3,
                    false,
                    false,
                    roles.iter(),
                );
                assert_eq!(
                    modes.iter().collect::<Vec<_>>(),
                    [VNextProductOutputMode::FullLogits; 2]
                );
                assert_eq!(modes.decision(), decision);
            }
        }
        let empty = ReadbackDecision::classify(
            SamplingReadbackMode::Participant,
            VNextExecutionWaveKind::Decode,
            ElementType::F16,
            3,
            false,
            false,
            std::iter::empty(),
        );
        assert_eq!(
            empty,
            ReadbackDecision::Uniform(VNextProductOutputMode::FullLogits)
        );
        assert!(!empty.allows_resident(VNextExecutionWaveKind::Decode, false));
    }

    #[test]
    fn resident_readback_preserves_uniform_greedy_but_excludes_capture_and_teacher() {
        let roles = [
            greedy(),
            VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
                token_mask: Some(TokenSelectionMask::new(vec![])),
                repetition_penalty: Some(GreedyRepetitionPenalty::new(1.1, vec![1, 2])),
            }),
        ];
        for strategy in [
            SamplingReadbackMode::Uniform,
            SamplingReadbackMode::Participant,
        ] {
            for dtype in [ElementType::F16, ElementType::F32, ElementType::Bf16] {
                for (capture, teacher) in [(false, false), (true, false), (true, true)] {
                    let decision = ReadbackDecision::classify(
                        strategy,
                        VNextExecutionWaveKind::Decode,
                        dtype,
                        3,
                        capture,
                        teacher,
                        roles.iter(),
                    );
                    let expected = if teacher {
                        VNextProductOutputMode::FullLogits
                    } else {
                        VNextProductOutputMode::GreedyToken
                    };
                    assert_eq!(decision, ReadbackDecision::Uniform(expected));
                    assert_eq!(
                        decision.allows_resident(VNextExecutionWaveKind::Decode, capture),
                        !capture && !teacher
                    );
                    assert!(!decision.allows_resident(VNextExecutionWaveKind::Prefill, capture));
                    assert!(!decision.allows_resident(VNextExecutionWaveKind::Mixed, capture));
                    let modes = ProductReadbackModes::resolve(
                        strategy,
                        VNextExecutionWaveKind::Decode,
                        dtype,
                        3,
                        capture,
                        teacher,
                        roles.iter(),
                    );
                    assert_eq!(modes.uniform(), Some(expected));
                    assert_eq!(modes.decision(), decision);
                }
            }
        }
    }
}
