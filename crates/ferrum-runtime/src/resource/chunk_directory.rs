//! Controlled resident directory mutation. Ordinary claim accounting cannot
//! replace backing; removal or replacement invalidates issued directory proofs.
use super::dynamic_pool::ResidentChunkState;
use std::collections::BTreeMap;
use std::ops::Deref;

pub(super) struct ChunkDirectory<B> {
    chunks: BTreeMap<u32, ResidentChunkState<B>>,
    epoch: u64,
}

impl<B> ChunkDirectory<B> {
    pub(super) fn new() -> Self {
        Self {
            chunks: BTreeMap::new(),
            epoch: 1,
        }
    }

    /// Zero permanently disables proof reuse after counter exhaustion.
    pub(super) fn epoch(&self) -> u64 {
        self.epoch
    }

    fn invalidate(&mut self) {
        if self.epoch != 0 {
            self.epoch = self.epoch.checked_add(1).unwrap_or(0);
        }
    }

    pub(super) fn insert(
        &mut self,
        ordinal: u32,
        chunk: ResidentChunkState<B>,
    ) -> Option<ResidentChunkState<B>> {
        if self.chunks.contains_key(&ordinal) {
            self.invalidate();
        }
        self.chunks.insert(ordinal, chunk)
    }

    pub(super) fn remove(&mut self, ordinal: &u32) -> Option<ResidentChunkState<B>> {
        if self.chunks.contains_key(ordinal) {
            self.invalidate();
        }
        self.chunks.remove(ordinal)
    }

    pub(super) fn live_segments_mut(&mut self, ordinal: &u32) -> Option<&mut u64> {
        self.chunks
            .get_mut(ordinal)
            .map(|chunk| &mut chunk.live_segments)
    }
}

// No DerefMut: a mutable backing reference would bypass invalidation.
impl<B> Deref for ChunkDirectory<B> {
    type Target = BTreeMap<u32, ResidentChunkState<B>>;
    fn deref(&self) -> &Self::Target {
        &self.chunks
    }
}

impl<'a, B> IntoIterator for &'a ChunkDirectory<B> {
    type Item = (&'a u32, &'a ResidentChunkState<B>);
    type IntoIter = std::collections::btree_map::Iter<'a, u32, ResidentChunkState<B>>;
    fn into_iter(self) -> Self::IntoIter {
        self.chunks.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::ChunkDirectory;

    #[test]
    fn chunk_directory_epoch_exhaustion_permanently_disables_reuse() {
        let mut directory = ChunkDirectory::<()>::new();
        let initial = directory.epoch();
        assert!(directory.remove(&7).is_none());
        assert_eq!(directory.epoch(), initial);
        directory.epoch = u64::MAX;
        directory.invalidate();
        assert_eq!(directory.epoch(), 0);
        directory.invalidate();
        assert_eq!(directory.epoch(), 0);
    }
}
