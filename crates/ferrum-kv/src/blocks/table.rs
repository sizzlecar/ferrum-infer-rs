//! Block table implementation for logical to physical mapping

use ferrum_interfaces::BlockTable;
use smallvec::SmallVec;

/// Default implementation of block table
#[derive(Debug, Clone, Default)]
pub struct DefaultBlockTable {
    /// Physical block usage bitmap
    pub physical: SmallVec<[u32; 8]>,
    /// Logical to physical block mapping
    pub logical_to_physical: SmallVec<[u32; 8]>,
    /// Sequence length
    pub seq_len: usize,
}

impl DefaultBlockTable {
    /// Create new block table
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            physical: SmallVec::with_capacity(capacity),
            logical_to_physical: SmallVec::with_capacity(capacity),
            seq_len: 0,
        }
    }

    /// Add logical to physical mapping
    pub fn map_block(&mut self, logical_id: u32, physical_id: u32) {
        // Ensure capacity
        while self.logical_to_physical.len() <= logical_id as usize {
            self.logical_to_physical.push(0);
        }
        while self.physical.len() <= physical_id as usize {
            self.physical.push(0);
        }

        self.logical_to_physical[logical_id as usize] = physical_id;
        self.physical[physical_id as usize] = 1; // Mark as used
    }

    /// Remove mapping for logical block
    pub fn unmap_block(&mut self, logical_id: u32) -> Option<u32> {
        if logical_id as usize < self.logical_to_physical.len() {
            let physical_id = self.logical_to_physical[logical_id as usize];
            if physical_id > 0 && (physical_id as usize) < self.physical.len() {
                self.physical[physical_id as usize] = 0; // Mark as free
                self.logical_to_physical[logical_id as usize] = 0;
                Some(physical_id)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get physical block ID for logical block
    pub fn get_physical(&self, logical_id: u32) -> Option<u32> {
        if logical_id as usize < self.logical_to_physical.len() {
            let physical_id = self.logical_to_physical[logical_id as usize];
            if physical_id > 0 {
                Some(physical_id)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Check if physical block is used
    pub fn is_physical_used(&self, physical_id: u32) -> bool {
        if physical_id as usize < self.physical.len() {
            self.physical[physical_id as usize] != 0
        } else {
            false
        }
    }

    /// Get number of logical blocks
    pub fn num_logical_blocks(&self) -> usize {
        self.logical_to_physical.len()
    }

    /// Get number of physical blocks allocated
    pub fn num_physical_blocks(&self) -> usize {
        self.physical.len()
    }

    /// Get number of used physical blocks
    pub fn num_used_blocks(&self) -> usize {
        self.physical.iter().filter(|&&used| used != 0).count()
    }

    /// Clear all mappings
    pub fn clear(&mut self) {
        self.physical.clear();
        self.logical_to_physical.clear();
        self.seq_len = 0;
    }

    /// Shrink block table to remove unused entries
    pub fn shrink(&mut self) {
        // Remove trailing zeros from logical_to_physical
        while let Some(&last) = self.logical_to_physical.last() {
            if last == 0 {
                self.logical_to_physical.pop();
            } else {
                break;
            }
        }

        // Remove trailing zeros from physical
        while let Some(&last) = self.physical.last() {
            if last == 0 {
                self.physical.pop();
            } else {
                break;
            }
        }
    }

    /// Get all used physical block IDs
    pub fn used_physical_blocks(&self) -> Vec<u32> {
        self.physical
            .iter()
            .enumerate()
            .filter_map(|(id, &used)| {
                if used != 0 {
                    Some(id as u32)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Create a copy with extended capacity
    pub fn extend(&self, additional_logical: usize, additional_physical: usize) -> Self {
        let mut new_table = self.clone();
        new_table.logical_to_physical.reserve(additional_logical);
        new_table.physical.reserve(additional_physical);
        new_table
    }
}

impl From<DefaultBlockTable> for BlockTable {
    fn from(table: DefaultBlockTable) -> Self {
        BlockTable {
            physical: table.physical,
            logical_to_physical: table.logical_to_physical,
            seq_len: table.seq_len,
        }
    }
}

impl From<BlockTable> for DefaultBlockTable {
    fn from(table: BlockTable) -> Self {
        Self {
            physical: table.physical,
            logical_to_physical: table.logical_to_physical,
            seq_len: table.seq_len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_table_creation() {
        let table = DefaultBlockTable::new();
        assert_eq!(table.num_logical_blocks(), 0);
        assert_eq!(table.num_physical_blocks(), 0);
        assert_eq!(table.seq_len, 0);
    }

    #[test]
    fn test_block_mapping() {
        let mut table = DefaultBlockTable::new();
        
        // Map logical block 0 to physical block 5
        table.map_block(0, 5);
        
        assert_eq!(table.get_physical(0), Some(5));
        assert!(table.is_physical_used(5));
        assert_eq!(table.num_used_blocks(), 1);
    }

    #[test]
    fn test_block_unmapping() {
        let mut table = DefaultBlockTable::new();
        
        table.map_block(0, 5);
        assert_eq!(table.get_physical(0), Some(5));
        
        let unmapped = table.unmap_block(0);
        assert_eq!(unmapped, Some(5));
        assert_eq!(table.get_physical(0), None);
        assert!(!table.is_physical_used(5));
        assert_eq!(table.num_used_blocks(), 0);
    }

    #[test]
    fn test_multiple_mappings() {
        let mut table = DefaultBlockTable::new();
        
        table.map_block(0, 10);
        table.map_block(1, 20);
        table.map_block(2, 30);
        
        assert_eq!(table.get_physical(0), Some(10));
        assert_eq!(table.get_physical(1), Some(20));
        assert_eq!(table.get_physical(2), Some(30));
        assert_eq!(table.num_used_blocks(), 3);
        
        let used_blocks = table.used_physical_blocks();
        assert!(used_blocks.contains(&10));
        assert!(used_blocks.contains(&20));
        assert!(used_blocks.contains(&30));
    }

    #[test]
    fn test_table_shrink() {
        let mut table = DefaultBlockTable::new();
        
        table.map_block(0, 5);
        table.map_block(1, 6);
        table.unmap_block(1); // This should leave trailing zeros
        
        assert!(table.logical_to_physical.len() >= 2);
        table.shrink();
        
        // After shrinking, trailing zeros should be removed
        assert_eq!(table.logical_to_physical.len(), 1);
        assert_eq!(table.get_physical(0), Some(5));
    }

    #[test]
    fn test_table_clear() {
        let mut table = DefaultBlockTable::new();
        
        table.map_block(0, 5);
        table.map_block(1, 6);
        table.seq_len = 100;
        
        assert!(table.num_logical_blocks() > 0);
        assert!(table.num_physical_blocks() > 0);
        
        table.clear();
        
        assert_eq!(table.num_logical_blocks(), 0);
        assert_eq!(table.num_physical_blocks(), 0);
        assert_eq!(table.seq_len, 0);
    }

    #[test]
    fn test_with_capacity() {
        let table = DefaultBlockTable::with_capacity(10);
        assert!(table.logical_to_physical.capacity() >= 10);
        assert!(table.physical.capacity() >= 10);
    }

    #[test]
    fn test_extend() {
        let table = DefaultBlockTable::new();
        let extended = table.extend(5, 10);
        
        assert!(extended.logical_to_physical.capacity() >= 5);
        assert!(extended.physical.capacity() >= 10);
    }

    #[test] 
    fn test_conversion() {
        let mut default_table = DefaultBlockTable::new();
        default_table.map_block(0, 5);
        default_table.seq_len = 50;
        
        // Convert to BlockTable
        let block_table: BlockTable = default_table.clone().into();
        assert_eq!(block_table.seq_len, 50);
        assert_eq!(block_table.logical_to_physical[0], 5);
        
        // Convert back to DefaultBlockTable
        let back_to_default: DefaultBlockTable = block_table.into();
        assert_eq!(back_to_default.seq_len, 50);
        assert_eq!(back_to_default.get_physical(0), Some(5));
    }
}
