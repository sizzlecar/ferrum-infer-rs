use super::*;
use ferrum_models::p_h3_image::{PH3ImageModality, PH3ImageReferenceImage};
use std::sync::atomic::{AtomicUsize, Ordering};

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

fn pending(members: Vec<RequestGeometry>, drops: Arc<AtomicUsize>) -> PendingSurvivors<Owner> {
    let slot = Arc::new(PreparationSlot::default());
    let task = PreparationPermit::try_acquire(&slot, None)
        .unwrap()
        .spawn(move || Ok(Owner(drops)))
        .unwrap();
    PendingSurvivors {
        geometry: members.into(),
        task,
    }
}

#[test]
fn only_a_partial_last_wave_prepares_survivors_in_original_order() {
    assert_eq!(
        survivor_indices([7, 3, 7, 0].into_iter(), 8),
        Some(vec![1, 3])
    );
    assert_eq!(survivor_indices([3, 0].into_iter(), 8), None);
    assert_eq!(survivor_indices([7, 7].into_iter(), 8), None);
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
fn cancelled_reordered_or_changed_geometry_retires_the_speculative_owner() {
    let a = geometry("request.a");
    let b = geometry("request.b");
    let mut changed = a.clone();
    changed
        .references
        .push(PH3ImageReferenceImage::new(4, 2).unwrap());
    for proposal in [vec![&a], vec![&b, &a], vec![&changed, &b]] {
        let drops = Arc::new(AtomicUsize::new(0));
        assert!(pending(vec![a.clone(), b.clone()], drops.clone())
            .finish_for(&proposal)
            .unwrap()
            .is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
