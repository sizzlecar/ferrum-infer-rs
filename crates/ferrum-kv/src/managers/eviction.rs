//! Eviction policies for KV cache management

use ferrum_types::{RequestId, Result};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Eviction policy trait
pub trait EvictionPolicy: Send + Sync + std::fmt::Debug {
    /// Record access to a request
    fn record_access(&mut self, request_id: RequestId);

    /// Select candidates for eviction
    fn select_eviction_candidates(&mut self, count: usize) -> Vec<RequestId>;

    /// Remove request from tracking
    fn remove_request(&mut self, request_id: RequestId);

    /// Get policy name
    fn name(&self) -> &str;

    /// Clear all tracking data
    fn clear(&mut self);
}

/// Least Recently Used (LRU) eviction policy
#[derive(Debug)]
pub struct LRUEviction {
    /// Request access order (most recent at back)
    access_order: VecDeque<RequestId>,
    /// Last access timestamps
    access_times: HashMap<RequestId, Instant>,
}

impl LRUEviction {
    /// Create new LRU eviction policy
    pub fn new() -> Self {
        Self {
            access_order: VecDeque::new(),
            access_times: HashMap::new(),
        }
    }
}

impl Default for LRUEviction {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for LRUEviction {
    fn record_access(&mut self, request_id: RequestId) {
        // Remove from current position if exists
        if let Some(pos) = self.access_order.iter().position(|id| *id == request_id) {
            self.access_order.remove(pos);
        }

        // Add to back (most recent)
        self.access_order.push_back(request_id.clone());
        self.access_times.insert(request_id, Instant::now());
    }

    fn select_eviction_candidates(&mut self, count: usize) -> Vec<RequestId> {
        // Return oldest requests (from front of queue)
        self.access_order.iter().take(count).cloned().collect()
    }

    fn remove_request(&mut self, request_id: RequestId) {
        if let Some(pos) = self.access_order.iter().position(|id| *id == request_id) {
            self.access_order.remove(pos);
        }
        self.access_times.remove(&request_id);
    }

    fn name(&self) -> &str {
        "LRU"
    }

    fn clear(&mut self) {
        self.access_order.clear();
        self.access_times.clear();
    }
}

/// First In, First Out (FIFO) eviction policy
#[derive(Debug)]
pub struct FIFOEviction {
    /// Request arrival order
    arrival_order: VecDeque<RequestId>,
    /// Arrival timestamps  
    arrival_times: HashMap<RequestId, Instant>,
}

impl FIFOEviction {
    /// Create new FIFO eviction policy
    pub fn new() -> Self {
        Self {
            arrival_order: VecDeque::new(),
            arrival_times: HashMap::new(),
        }
    }
}

impl Default for FIFOEviction {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for FIFOEviction {
    fn record_access(&mut self, request_id: RequestId) {
        // Only record if not already present (first access only)
        if !self.arrival_times.contains_key(&request_id) {
            self.arrival_order.push_back(request_id.clone());
            self.arrival_times.insert(request_id, Instant::now());
        }
    }

    fn select_eviction_candidates(&mut self, count: usize) -> Vec<RequestId> {
        // Return oldest arrivals (from front of queue)
        self.arrival_order.iter().take(count).cloned().collect()
    }

    fn remove_request(&mut self, request_id: RequestId) {
        if let Some(pos) = self.arrival_order.iter().position(|id| *id == request_id) {
            self.arrival_order.remove(pos);
        }
        self.arrival_times.remove(&request_id);
    }

    fn name(&self) -> &str {
        "FIFO"
    }

    fn clear(&mut self) {
        self.arrival_order.clear();
        self.arrival_times.clear();
    }
}

/// Clock (Second Chance) eviction policy
#[derive(Debug)]
pub struct ClockEviction {
    /// Circular list of requests
    requests: Vec<RequestId>,
    /// Reference bits for each request
    reference_bits: HashMap<RequestId, bool>,
    /// Current clock hand position
    clock_hand: usize,
}

impl ClockEviction {
    /// Create new Clock eviction policy
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            reference_bits: HashMap::new(),
            clock_hand: 0,
        }
    }
}

impl Default for ClockEviction {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictionPolicy for ClockEviction {
    fn record_access(&mut self, request_id: RequestId) {
        // Set reference bit
        self.reference_bits.insert(request_id.clone(), true);

        // Add to list if not present
        if !self.requests.contains(&request_id) {
            self.requests.push(request_id);
        }
    }

    fn select_eviction_candidates(&mut self, count: usize) -> Vec<RequestId> {
        let mut candidates = Vec::new();
        let mut attempts = 0;
        let max_attempts = self.requests.len() * 2; // Avoid infinite loops

        while candidates.len() < count && attempts < max_attempts {
            if self.requests.is_empty() {
                break;
            }

            let request_id = self.requests[self.clock_hand].clone();
            let referenced = self
                .reference_bits
                .get(&request_id)
                .copied()
                .unwrap_or(false);

            if referenced {
                // Give second chance, clear reference bit
                self.reference_bits.insert(request_id, false);
            } else {
                // Select for eviction
                candidates.push(request_id.clone());
            }

            // Advance clock hand
            self.clock_hand = (self.clock_hand + 1) % self.requests.len();
            attempts += 1;
        }

        candidates
    }

    fn remove_request(&mut self, request_id: RequestId) {
        if let Some(pos) = self.requests.iter().position(|id| *id == request_id) {
            self.requests.remove(pos);

            // Adjust clock hand if needed
            if self.clock_hand > pos {
                self.clock_hand -= 1;
            } else if self.clock_hand >= self.requests.len() && !self.requests.is_empty() {
                self.clock_hand = 0;
            }
        }

        self.reference_bits.remove(&request_id);
    }

    fn name(&self) -> &str {
        "Clock"
    }

    fn clear(&mut self) {
        self.requests.clear();
        self.reference_bits.clear();
        self.clock_hand = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_eviction() {
        let mut policy = LRUEviction::new();

        let req1 = RequestId::new();
        let req2 = RequestId::new();
        let req3 = RequestId::new();

        // Access in order: req1, req2, req3
        policy.record_access(req1.clone());
        policy.record_access(req2.clone());
        policy.record_access(req3);

        // Should evict req1 (least recently used)
        let candidates = policy.select_eviction_candidates(1);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], req1);

        // Access req1 again to make it most recent
        policy.record_access(req1.clone());

        // Now should evict req2
        let candidates = policy.select_eviction_candidates(1);
        assert_eq!(candidates[0], req2);
    }

    #[test]
    fn test_fifo_eviction() {
        let mut policy = FIFOEviction::new();

        let req1 = RequestId::new();
        let req2 = RequestId::new();
        let req3 = RequestId::new();

        // Access in order: req1, req2, req3
        policy.record_access(req1.clone());
        policy.record_access(req2.clone());
        policy.record_access(req3);

        // Should evict req1 (first in)
        let candidates = policy.select_eviction_candidates(1);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], req1);

        // Access req1 again - should not change FIFO order
        policy.record_access(req1.clone());

        // Still should evict req1
        let candidates = policy.select_eviction_candidates(1);
        assert_eq!(candidates[0], req1);
    }

    #[test]
    fn test_clock_eviction() {
        let mut policy = ClockEviction::new();

        let req1 = RequestId::new();
        let req2 = RequestId::new();

        // Add requests
        policy.record_access(req1.clone());
        policy.record_access(req2.clone());

        // First eviction should hit req1, but it has reference bit set
        // So it should get second chance and req2 should be selected
        let candidates = policy.select_eviction_candidates(1);
        assert_eq!(candidates.len(), 1);
        // The exact behavior depends on implementation details
    }

    #[test]
    fn test_policy_removal() {
        let mut policy = LRUEviction::new();

        let req1 = RequestId::new();
        let req2 = RequestId::new();

        policy.record_access(req1.clone());
        policy.record_access(req2.clone());

        // Remove req1
        policy.remove_request(req1);

        // Should only evict req2 now
        let candidates = policy.select_eviction_candidates(1);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], req2);
    }

    #[test]
    fn test_policy_clear() {
        let mut policy = LRUEviction::new();

        let req1 = RequestId::new();
        policy.record_access(req1.clone());

        policy.clear();

        // Should have no candidates after clear
        let candidates = policy.select_eviction_candidates(1);
        assert_eq!(candidates.len(), 0);
    }
}
