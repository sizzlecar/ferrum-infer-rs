use super::*;
use ferrum_models::p_h3_image::{PH3ImageModality, PH3ImageReferenceImage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{sync::mpsc, task::Wake};

fn geometry(id: &str) -> RequestGeometry {
    let tensor = |shape| {
        ResolvedTensorSpec::new(shape, ElementType::F32, ResolvedTensorLayout::Contiguous).unwrap()
    };
    RequestGeometry {
        identity: RequestIdentity::new(id).unwrap(),
        target: PH3ImageReferenceImage::new(2, 2).unwrap(),
        references: vec![PH3ImageReferenceImage::new(2, 4).unwrap()],
        tags: vec![PH3ImageModality::Text],
        image_sample: tensor(vec![1, 4, 1, 2, 2]),
        audio_sample: tensor(vec![1, 4, 2, 2]),
    }
}

struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn pending(members: Vec<RequestGeometry>, drops: Arc<AtomicUsize>) -> PendingMembership<Owner> {
    let slot = Arc::new(PreparationSlot::default());
    let task = PreparationPermit::try_acquire(&slot, None)
        .unwrap()
        .spawn(move || Ok(Owner(drops)))
        .unwrap();
    PendingMembership {
        geometry: members.into(),
        task,
    }
}

#[test]
fn near_partial_completion_prepares_the_same_survivors_across_two_waves() {
    assert_eq!(
        survivor_indices([7, 3, 7, 0].into_iter(), 8),
        Some(vec![1, 3])
    );
    assert_eq!(survivor_indices([3, 0].into_iter(), 8), None);
    assert_eq!(survivor_indices([5, 3].into_iter(), 8), None);
    assert_eq!(
        survivor_indices([6, 5, 6, 0].into_iter(), 8),
        Some(vec![1, 3])
    );
    assert_eq!(
        survivor_indices([7, 6, 7, 1].into_iter(), 8),
        Some(vec![1, 3])
    );
    assert_eq!(survivor_indices([6, 6].into_iter(), 8), None);
    assert_eq!(survivor_indices([7, 7].into_iter(), 8), None);
    assert_eq!(survivor_indices([0].into_iter(), 0), None);
    assert_eq!(survivor_indices([].into_iter(), 8), None);
    assert_eq!(survivor_indices([8, 3].into_iter(), 8), None);
    assert_eq!(survivor_indices([usize::MAX, 3].into_iter(), 8), None);
}

#[test]
fn exact_survivors_join_and_transfer_the_prepared_owner_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let a = geometry("request.a");
    let b = geometry("request.b");
    let prepared = pending(vec![a.clone(), b.clone()], drops.clone())
        .finish_for(&[&a, &b])
        .unwrap()
        .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(prepared);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancelled_joined_reordered_or_changed_geometry_retires_the_speculative_owner() {
    let a = geometry("request.a");
    let b = geometry("request.b");
    let joined = geometry("request.joined");
    let mut changed = a.clone();
    changed
        .references
        .push(PH3ImageReferenceImage::new(4, 2).unwrap());
    for proposal in [
        vec![&a],
        vec![&a, &b, &joined],
        vec![&b, &a],
        vec![&changed, &b],
    ] {
        let drops = Arc::new(AtomicUsize::new(0));
        assert!(pending(vec![a.clone(), b.clone()], drops.clone())
            .finish_for(&proposal)
            .unwrap()
            .is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn withdrawn_join_proposal_retires_at_membership_replacement() {
    let a = geometry("request.running.a");
    let b = geometry("request.running.b");
    let joined = geometry("request.initializing");
    let drops = Arc::new(AtomicUsize::new(0));
    assert!(pending(vec![a.clone(), b.clone(), joined], drops.clone())
        .finish_for(&[&a, &b])
        .unwrap()
        .is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

struct CompletionSignal(mpsc::SyncSender<()>);
impl Wake for CompletionSignal {
    fn wake(self: Arc<Self>) {
        let _ = self.0.try_send(());
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.try_send(());
    }
}

fn notified_pending(
    members: Vec<RequestGeometry>,
    work: impl FnOnce() -> Result<Owner> + Send + 'static,
) -> (
    PendingMembership<Owner>,
    mpsc::Receiver<()>,
    Arc<PreparationSlot>,
) {
    let slot = Arc::new(PreparationSlot::default());
    let (signal, awakened) = mpsc::sync_channel(1);
    let wake = std::task::Waker::from(Arc::new(CompletionSignal(signal)));
    let task = PreparationPermit::try_acquire(&slot, Some(&wake))
        .unwrap()
        .spawn(work)
        .unwrap();
    (
        PendingMembership {
            geometry: members.into(),
            task,
        },
        awakened,
        slot,
    )
}

#[test]
fn obsolete_survivor_retires_only_after_completion_and_frees_join_preparation() {
    let a = geometry("request.running");
    let b = geometry("request.survivor");
    let joining = geometry("request.joining");
    let drops = Arc::new(AtomicUsize::new(0));
    let owner = Owner(drops.clone());
    let (release, blocked) = mpsc::sync_channel(0);
    let (proposal, awakened, slot) = notified_pending(vec![b.clone()], move || {
        blocked.recv().unwrap();
        Ok(owner)
    });
    let mut proposal = Some(proposal);
    let joined_geometry = [&a, &b, &joining];
    PendingMembership::retire_completed_mismatch(&mut proposal, &joined_geometry);
    let retained_unfinished = proposal.is_some();
    let slot_still_busy = PreparationPermit::try_acquire(&slot, None).is_none();
    release.send(()).unwrap();
    awakened
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert!(retained_unfinished && slot_still_busy);

    // Completion alone cannot consume a still-useful result.
    PendingMembership::retire_completed_mismatch(&mut proposal, &[&b]);
    assert!(proposal.is_some());
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    PendingMembership::retire_completed_mismatch(&mut proposal, &joined_geometry);
    assert!(proposal.is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let next_owner = Owner(drops.clone());
    let next = PendingMembership {
        geometry: vec![a.clone(), b.clone(), joining.clone()].into(),
        task: PreparationPermit::try_acquire(&slot, None)
            .unwrap()
            .spawn(move || Ok(next_owner))
            .unwrap(),
    };
    let admitted = next.finish_for(&joined_geometry).unwrap().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(admitted);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn obsolete_preparation_failure_retires_but_matching_failure_reaches_admission() {
    let a = geometry("request.a");
    let b = geometry("request.b");
    for matching in [false, true] {
        let (proposal, awakened, slot) = notified_pending(vec![a.clone()], || {
            Err(anyhow::anyhow!("metadata preparation failed"))
        });
        awakened
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let mut proposal = Some(proposal);
        PendingMembership::retire_completed_mismatch(
            &mut proposal,
            &[if matching { &a } else { &b }],
        );
        if matching {
            assert!(proposal.take().unwrap().finish_for(&[&a]).is_err());
        } else {
            assert!(proposal.is_none());
        }
        assert!(PreparationPermit::try_acquire(&slot, None).is_some());
    }
}
