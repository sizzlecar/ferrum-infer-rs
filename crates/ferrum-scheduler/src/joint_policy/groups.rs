//! Contiguous invocation work with parallel request identities. Validation is
//! shared by ordinary adapter members and private canonical replay views.

use super::*;

#[derive(Clone, Copy)]
pub(crate) struct GroupView<'a> {
    pub(in crate::joint_policy) requests: &'a [usize],
    pub(in crate::joint_policy) work: &'a [CostWork],
}

impl<'a> GroupView<'a> {
    pub(crate) fn len(self) -> usize {
        self.work.len()
    }

    pub(crate) fn iter(self) -> impl ExactSizeIterator<Item = GroupMember> + Clone + 'a {
        self.requests
            .iter()
            .copied()
            .zip(self.work.iter().copied())
            .map(|(request, work)| GroupMember { request, work })
    }
}

impl PlanBuilder {
    /// Storage is reused within one simulation, never topology across waves.
    pub(crate) fn with_members(
        &mut self,
        build: impl FnOnce(&mut Vec<GroupMember>, &mut Self) -> Result<(), Unsupported>,
    ) -> Result<(), Unsupported> {
        let mut members = std::mem::take(&mut self.members);
        members.clear();
        let result = build(&mut members, self);
        members.clear();
        self.members = members;
        result
    }

    pub fn push_group(&mut self, members: &[GroupMember]) -> Result<(), Unsupported> {
        self.push_matching(members, None)
    }

    pub(crate) fn push_group_kind(
        &mut self,
        members: &[GroupMember],
        kind: AttemptWorkKind,
    ) -> Result<(), Unsupported> {
        self.push_matching(members, Some(kind))
    }

    /// This mask is frozen before any group in this wave executes.
    pub(crate) fn active_mask(&self) -> Option<&[bool]> {
        self.active.as_deref()
    }

    pub(super) fn group(&self, index: usize) -> Option<GroupView<'_>> {
        if index >= self.group_count {
            return None;
        }
        let start = if index == 0 {
            0
        } else {
            self.group_ends[index - 1]
        };
        let end = self.group_ends[index];
        Some(GroupView {
            requests: &self.requests[start..end],
            work: &self.work[start..end],
        })
    }

    pub(super) fn groups(&self) -> impl ExactSizeIterator<Item = GroupView<'_>> + Clone {
        (0..self.group_count).map(move |index| self.group(index).expect("accepted group"))
    }

    #[cfg(test)]
    pub(crate) fn groups_for_test(&self) -> impl ExactSizeIterator<Item = GroupView<'_>> + Clone {
        self.groups()
    }

    pub(super) fn clear_groups(&mut self) {
        self.requests.clear();
        self.work.clear();
        self.group_count = 0;
    }

    /// A replay can fall back before installing any group if storage cannot be
    /// reserved. This does not validate or publish any member or group.
    pub(super) fn try_reserve_members(&mut self, count: usize) -> bool {
        self.requests.try_reserve_exact(count).is_ok() && self.work.try_reserve_exact(count).is_ok()
    }

    fn validate_group(
        &self,
        members: impl Iterator<Item = GroupMember> + Clone,
        count: usize,
        kind: Option<AttemptWorkKind>,
    ) -> Result<bool, Unsupported> {
        // Filtered absence is a no-op even when all other bounds are exhausted.
        if kind.is_some() && count == 0 {
            return Ok(false);
        }
        if count == 0
            || self.group_count >= 2
            || !CostQuery::fits(count, self.query_limits)
            || self
                .requests
                .len()
                .checked_add(count)
                .is_none_or(|count| count > self.participant_limit)
        {
            return Err(Unsupported::ParticipantBudget);
        }
        if self.execution == CostExecution::SingleInvocation && self.group_count != 0 {
            return Err(Unsupported::InvalidPlan);
        }
        if self.execution == CostExecution::SequentialPrefillThenDecode {
            let first_kind =
                kind.unwrap_or_else(|| members.clone().next().expect("nonempty group").work.kind);
            if (kind.is_none() && members.clone().any(|member| member.work.kind != first_kind))
                || self.group(0).is_some_and(|group| {
                    group.work[0].kind != AttemptWorkKind::Prefill
                        || first_kind != AttemptWorkKind::Decode
                })
            {
                return Err(Unsupported::InvalidPlan);
            }
        }
        // The machine word is a fast path, never a participant/index bound.
        let mut seen = 0usize;
        for &request in &self.requests {
            if request < usize::BITS as usize {
                seen |= 1usize << request;
            }
        }
        for (position, member) in members.clone().enumerate() {
            if self
                .active
                .as_ref()
                .is_some_and(|active| !active.get(member.request).copied().unwrap_or(false))
            {
                return Err(Unsupported::WaitingResources);
            }
            let duplicate = if member.request < usize::BITS as usize {
                let bit = 1usize << member.request;
                let duplicate = seen & bit != 0;
                seen |= bit;
                duplicate
            } else {
                members
                    .clone()
                    .take(position)
                    .any(|prior| prior.request == member.request)
                    || self.requests.contains(&member.request)
            };
            if duplicate {
                return Err(Unsupported::InvalidPlan);
            }
        }
        Ok(true)
    }

    fn finish_group(&mut self) {
        debug_assert_eq!(self.requests.len(), self.work.len());
        self.group_ends[self.group_count] = self.work.len();
        self.group_count += 1;
    }

    fn push_matching(
        &mut self,
        members: &[GroupMember],
        kind: Option<AttemptWorkKind>,
    ) -> Result<(), Unsupported> {
        let selected = members
            .iter()
            .copied()
            .filter(|member| kind.is_none_or(|kind| member.work.kind == kind));
        let count = if kind.is_some() {
            selected.clone().count()
        } else {
            members.len()
        };
        if !self.validate_group(selected.clone(), count, kind)? {
            return Ok(());
        }
        // Both reservations and all validation precede any logical mutation.
        self.requests.reserve(count);
        self.work.reserve(count);
        if count == members.len() {
            self.requests
                .extend(members.iter().map(|member| member.request));
            self.work.extend(members.iter().map(|member| member.work));
        } else {
            self.requests
                .extend(selected.clone().map(|member| member.request));
            self.work.extend(selected.map(|member| member.work));
        }
        self.finish_group();
        Ok(())
    }

    pub(super) fn push_group_view(&mut self, group: GroupView<'_>) -> Result<(), Unsupported> {
        // Canonical work and identities must stay aligned; zip truncation is
        // not an admissible plan, even though the production memo stores both.
        if group.requests.len() != group.work.len() {
            return Err(Unsupported::InvalidPlan);
        }
        self.validate_group(group.iter(), group.len(), None)?;
        self.requests.reserve(group.len());
        self.work.reserve(group.len());
        self.requests.extend_from_slice(group.requests);
        self.work.extend_from_slice(group.work);
        self.finish_group();
        Ok(())
    }
}
