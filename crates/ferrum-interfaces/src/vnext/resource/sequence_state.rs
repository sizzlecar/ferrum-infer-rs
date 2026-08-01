pub(super) const fn sequence_slot_active(epoch: u64) -> u64 {
    (epoch << 2) | 1
}

pub(super) const fn sequence_slot_poisoned_drained(epoch: u64) -> u64 {
    (epoch << 2) | 2
}

pub(super) const fn sequence_slot_poisoned_undrained(epoch: u64) -> u64 {
    (epoch << 2) | 3
}

pub(super) const fn sequence_slot_is_poisoned(state: u64) -> bool {
    matches!(state & 3, 2 | 3)
}
