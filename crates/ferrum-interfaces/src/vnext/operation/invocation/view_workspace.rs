use std::ops::{Deref, Range};

use super::*;

/// Storage only: a borrowed facade never grants authority to a resource.
pub(super) enum InvocationViews<'a, B> {
    Owned(Vec<OperationBufferView<'a, B>>),
    Borrowed(&'a [OperationBufferView<'a, B>]),
}

impl<'a, B> Deref for InvocationViews<'a, B> {
    type Target = [OperationBufferView<'a, B>];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(views) => views,
            Self::Borrowed(views) => views,
        }
    }
}

pub(super) enum BuiltInvocationViews<'a, B> {
    Owned(Vec<OperationBufferView<'a, B>>),
    Appended(Range<usize>),
}

pub(super) enum InvocationViewBuilder<'a, 'storage, B> {
    Owned(Vec<OperationBufferView<'a, B>>),
    Appended {
        views: &'storage mut Vec<OperationBufferView<'a, B>>,
        start: usize,
    },
}

impl<'a, 'storage, B> InvocationViewBuilder<'a, 'storage, B> {
    pub(super) fn new(
        arena: Option<&'storage mut Vec<OperationBufferView<'a, B>>>,
        capacity: usize,
    ) -> Self {
        match arena {
            None => Self::Owned(Vec::with_capacity(capacity)),
            Some(views) => {
                let start = views.len();
                views.reserve(capacity);
                Self::Appended { views, start }
            }
        }
    }

    pub(super) fn push(&mut self, view: OperationBufferView<'a, B>) {
        match self {
            Self::Owned(views) => views.push(view),
            Self::Appended { views, .. } => views.push(view),
        }
    }

    pub(super) fn finish(self) -> BuiltInvocationViews<'a, B> {
        match self {
            Self::Owned(views) => BuiltInvocationViews::Owned(views),
            Self::Appended { views, start } => BuiltInvocationViews::Appended(start..views.len()),
        }
    }
}

impl<'a, B> Deref for InvocationViewBuilder<'a, '_, B> {
    type Target = [OperationBufferView<'a, B>];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(views) => views,
            Self::Appended { views, start } => &views[*start..],
        }
    }
}

/// One dispatch's scratch capacity. Every node rebuilds and revalidates its
/// views; neither physical backing nor identity/dependency proofs are cached.
pub(in crate::vnext::operation) struct WaveInvocationWorkspace<'wave, B> {
    views: Vec<OperationBufferView<'wave, B>>,
    headers: Vec<(InvocationHeader<'wave>, Range<usize>)>,
}

impl<'wave, B> WaveInvocationWorkspace<'wave, B> {
    pub(in crate::vnext::operation) fn new() -> Self {
        Self {
            views: Vec::new(),
            headers: Vec::new(),
        }
    }

    /// The callback cannot return a reference into this node's facade. The
    /// original provider API returns owned commands/retentions, as before.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::vnext::operation) fn with_reusable_wave_node<'binding, R, I, T>(
        &mut self,
        runtime: &R,
        resolved: &'wave dyn ExecutablePlanView,
        prepared: &PreparedOperationDispatchBinding,
        batch_identity: &'wave BatchOperationIdentity,
        node_identity: &'wave BatchOperationNodeIdentity,
        wave: &'wave PreparedStepSubmissionWave<R>,
        node_index: usize,
        active_bindings: I,
        encode: impl for<'node> FnOnce(BatchedOperationInvocation<'node, B>) -> T,
    ) -> Result<T, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
        I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
    {
        // Drop runs after the callback's borrows on success, Err, and unwind.
        // Only empty storage capacity can be reused by the next node.
        let reset = WorkspaceReset(self);
        let workspace = &mut *reset.0;
        debug_assert!(workspace.views.is_empty() && workspace.headers.is_empty());
        let resources = OperationInvocationResources::Wave { wave, node_index };
        let (participant_count, node, operation) = BatchedOperationInvocation::validate_resources(
            resolved,
            prepared,
            batch_identity,
            node_identity,
            resources,
            &active_bindings,
        )?;
        let mut shared_backings = if participant_count > 1 {
            (0..prepared.resources.len())
                .map(|_| None)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut device_agreements = [
            DeviceDescriptorAgreement::default(),
            DeviceDescriptorAgreement::default(),
        ];
        for (index, (participant, active_binding)) in node_identity
            .participants()
            .iter()
            .zip(active_bindings)
            .enumerate()
        {
            let (header, views) = OperationInvocation::prepare_with_storage(
                runtime,
                resolved,
                prepared,
                node,
                operation,
                participant.identity(),
                node_identity.node_id(),
                resources,
                active_binding,
                index,
                true,
                &mut shared_backings,
                &mut device_agreements,
                Some(&mut workspace.views),
            )?;
            let BuiltInvocationViews::Appended(range) = views else {
                unreachable!("workspace preparation cannot return owned views")
            };
            workspace.headers.push((header, range));
        }
        let program_binding = resources.program_binding_node();
        let retained_dependency_scope = Arc::new(());
        let retained_persistent_preserve =
            node.provider_resources().persistent().is_some_and(|p| {
                p.scope() == crate::vnext::ProviderWorkspaceScope::Plan
                    && p.reuse_policy() == crate::vnext::ProviderWorkspaceReusePolicy::Preserve
            });
        // Match the old constructor boundary: its temporary shared table is
        // gone before the provider receives the invocation.
        drop(shared_backings);
        let participants = workspace
            .headers
            .iter()
            .map(|(header, range)| OperationInvocation {
                header: *header,
                views: InvocationViews::Borrowed(&workspace.views[range.clone()]),
            })
            .collect();
        let invocation = BatchedOperationInvocation {
            batch_identity,
            node_identity,
            participants,
            program_binding,
            retained_dependency_scope,
            retained_persistent_preserve,
        };
        Ok(encode(invocation))
    }

    #[cfg(test)]
    pub(in crate::vnext::operation) fn is_empty(&self) -> bool {
        self.views.is_empty() && self.headers.is_empty()
    }

    #[cfg(test)]
    pub(in crate::vnext::operation) fn view_capacity(&self) -> usize {
        self.views.capacity()
    }
}

struct WorkspaceReset<'storage, 'wave, B>(&'storage mut WaveInvocationWorkspace<'wave, B>);
impl<B> Drop for WorkspaceReset<'_, '_, B> {
    fn drop(&mut self) {
        self.0.headers.clear();
        self.0.views.clear();
    }
}
