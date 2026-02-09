//! Queue management utilities for schedulers

use ferrum_types::{InferenceRequest, Priority, RequestId};
use std::collections::VecDeque;

/// Simple FIFO request queue
#[derive(Debug, Default)]
pub struct FifoQueue {
    requests: VecDeque<InferenceRequest>,
    max_size: usize,
}

impl FifoQueue {
    /// Create new FIFO queue with capacity limit
    pub fn new(max_size: usize) -> Self {
        Self {
            requests: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add request to back of queue
    pub fn push(&mut self, request: InferenceRequest) -> Result<(), InferenceRequest> {
        if self.requests.len() >= self.max_size {
            Err(request)
        } else {
            self.requests.push_back(request);
            Ok(())
        }
    }

    /// Remove request from front of queue
    pub fn pop(&mut self) -> Option<InferenceRequest> {
        self.requests.pop_front()
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Check if queue is full
    pub fn is_full(&self) -> bool {
        self.requests.len() >= self.max_size
    }

    /// Remove specific request from queue
    pub fn remove(&mut self, request_id: &RequestId) -> Option<InferenceRequest> {
        if let Some(pos) = self.requests.iter().position(|req| &req.id == request_id) {
            self.requests.remove(pos)
        } else {
            None
        }
    }

    /// Get reference to request by ID
    pub fn get(&self, request_id: &RequestId) -> Option<&InferenceRequest> {
        self.requests.iter().find(|req| &req.id == request_id)
    }

    /// Get mutable reference to request by ID
    pub fn get_mut(&mut self, request_id: &RequestId) -> Option<&mut InferenceRequest> {
        self.requests.iter_mut().find(|req| &req.id == request_id)
    }

    /// Clear all requests
    pub fn clear(&mut self) {
        self.requests.clear();
    }

    /// Get iterator over requests
    pub fn iter(&self) -> impl Iterator<Item = &InferenceRequest> {
        self.requests.iter()
    }
}

/// Request queue stats
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Current queue length
    pub length: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Utilization percentage (0.0 - 1.0)
    pub utilization: f32,
    /// Priority distribution
    pub priority_distribution: PriorityDistribution,
}

/// Priority distribution in queue
#[derive(Debug, Clone, Default)]
pub struct PriorityDistribution {
    pub critical: usize,
    pub high: usize,
    pub normal: usize,
    pub low: usize,
}

impl PriorityDistribution {
    /// Calculate priority distribution from requests
    pub fn from_requests<'a, I>(requests: I) -> Self
    where
        I: Iterator<Item = &'a InferenceRequest>,
    {
        let mut dist = PriorityDistribution::default();

        for request in requests {
            match request.priority {
                Priority::Critical => dist.critical += 1,
                Priority::High => dist.high += 1,
                Priority::Normal => dist.normal += 1,
                Priority::Low => dist.low += 1,
            }
        }

        dist
    }

    /// Get total count
    pub fn total(&self) -> usize {
        self.critical + self.high + self.normal + self.low
    }

    /// Get percentage for each priority
    pub fn percentages(&self) -> (f32, f32, f32, f32) {
        let total = self.total() as f32;
        if total == 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        (
            self.critical as f32 / total,
            self.high as f32 / total,
            self.normal as f32 / total,
            self.low as f32 / total,
        )
    }
}

impl FifoQueue {
    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        let utilization = if self.max_size > 0 {
            self.requests.len() as f32 / self.max_size as f32
        } else {
            0.0
        };

        QueueStats {
            length: self.requests.len(),
            capacity: self.max_size,
            utilization,
            priority_distribution: PriorityDistribution::from_requests(self.requests.iter()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::ModelId;

    fn create_test_request(_id: u64, priority: Priority) -> InferenceRequest {
        InferenceRequest::new("test prompt", ModelId::new("test-model")).with_priority(priority)
    }

    #[test]
    fn test_fifo_queue_creation() {
        let queue = FifoQueue::new(10);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
        assert!(!queue.is_full());
        assert_eq!(queue.max_size, 10);
    }

    #[test]
    fn test_fifo_queue_push_pop() {
        let mut queue = FifoQueue::new(3);
        let req1 = create_test_request(1, Priority::Normal);
        let req2 = create_test_request(2, Priority::High);
        let req3 = create_test_request(3, Priority::Low);

        // Push requests
        assert!(queue.push(req1.clone()).is_ok());
        assert_eq!(queue.len(), 1);
        assert!(queue.push(req2.clone()).is_ok());
        assert_eq!(queue.len(), 2);
        assert!(queue.push(req3.clone()).is_ok());
        assert_eq!(queue.len(), 3);
        assert!(queue.is_full());

        // Try to push when full
        let req4 = create_test_request(4, Priority::Normal);
        assert!(queue.push(req4).is_err());

        // Pop requests in FIFO order
        let popped1 = queue.pop().unwrap();
        assert_eq!(popped1.id, req1.id);
        assert_eq!(queue.len(), 2);

        let popped2 = queue.pop().unwrap();
        assert_eq!(popped2.id, req2.id);
        assert_eq!(queue.len(), 1);

        let popped3 = queue.pop().unwrap();
        assert_eq!(popped3.id, req3.id);
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());

        // Pop from empty queue
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_fifo_queue_remove() {
        let mut queue = FifoQueue::new(5);
        let req1 = create_test_request(1, Priority::Normal);
        let req2 = create_test_request(2, Priority::High);
        let req3 = create_test_request(3, Priority::Low);

        let id1 = req1.id.clone();
        let id2 = req2.id.clone();

        queue.push(req1).ok();
        queue.push(req2).ok();
        queue.push(req3).ok();

        assert_eq!(queue.len(), 3);

        // Remove middle element
        let removed = queue.remove(&id2);
        assert!(removed.is_some());
        assert_eq!(queue.len(), 2);

        // Try to remove non-existent
        let removed = queue.remove(&id2);
        assert!(removed.is_none());
        assert_eq!(queue.len(), 2);

        // Remove first element
        let removed = queue.remove(&id1);
        assert!(removed.is_some());
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_fifo_queue_get() {
        let mut queue = FifoQueue::new(5);
        let req = create_test_request(1, Priority::High);
        let req_id = req.id.clone();

        queue.push(req).ok();

        // Get immutable reference
        let found = queue.get(&req_id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, req_id);

        // Get mutable reference
        let found_mut = queue.get_mut(&req_id);
        assert!(found_mut.is_some());
        assert_eq!(found_mut.unwrap().id, req_id);

        // Try to get non-existent
        let not_found = queue.get(&RequestId::new());
        assert!(not_found.is_none());
    }

    #[test]
    fn test_fifo_queue_clear() {
        let mut queue = FifoQueue::new(5);
        queue.push(create_test_request(1, Priority::Normal)).ok();
        queue.push(create_test_request(2, Priority::High)).ok();
        assert_eq!(queue.len(), 2);

        queue.clear();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_fifo_queue_iter() {
        let mut queue = FifoQueue::new(5);
        let req1 = create_test_request(1, Priority::Normal);
        let req2 = create_test_request(2, Priority::High);

        let id1 = req1.id.clone();
        let id2 = req2.id.clone();

        queue.push(req1).ok();
        queue.push(req2).ok();

        let ids: Vec<RequestId> = queue.iter().map(|r| r.id.clone()).collect();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], id1);
        assert_eq!(ids[1], id2);
    }

    #[test]
    fn test_fifo_queue_stats() {
        let mut queue = FifoQueue::new(10);

        // Empty queue
        let stats = queue.stats();
        assert_eq!(stats.length, 0);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.utilization, 0.0);

        // Add requests with different priorities
        queue.push(create_test_request(1, Priority::Critical)).ok();
        queue.push(create_test_request(2, Priority::High)).ok();
        queue.push(create_test_request(3, Priority::High)).ok();
        queue.push(create_test_request(4, Priority::Normal)).ok();
        queue.push(create_test_request(5, Priority::Low)).ok();

        let stats = queue.stats();
        assert_eq!(stats.length, 5);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.utilization, 0.5);
        assert_eq!(stats.priority_distribution.critical, 1);
        assert_eq!(stats.priority_distribution.high, 2);
        assert_eq!(stats.priority_distribution.normal, 1);
        assert_eq!(stats.priority_distribution.low, 1);
    }

    #[test]
    fn test_priority_distribution_from_requests() {
        let requests = vec![
            create_test_request(1, Priority::Critical),
            create_test_request(2, Priority::Critical),
            create_test_request(3, Priority::High),
            create_test_request(4, Priority::Normal),
            create_test_request(5, Priority::Normal),
            create_test_request(6, Priority::Normal),
            create_test_request(7, Priority::Low),
        ];

        let dist = PriorityDistribution::from_requests(requests.iter());
        assert_eq!(dist.critical, 2);
        assert_eq!(dist.high, 1);
        assert_eq!(dist.normal, 3);
        assert_eq!(dist.low, 1);
        assert_eq!(dist.total(), 7);
    }

    #[test]
    fn test_priority_distribution_percentages() {
        let mut dist = PriorityDistribution::default();
        dist.critical = 1;
        dist.high = 2;
        dist.normal = 3;
        dist.low = 4;

        let (critical_pct, high_pct, normal_pct, low_pct) = dist.percentages();
        assert!((critical_pct - 0.1).abs() < 0.01); // 1/10
        assert!((high_pct - 0.2).abs() < 0.01); // 2/10
        assert!((normal_pct - 0.3).abs() < 0.01); // 3/10
        assert!((low_pct - 0.4).abs() < 0.01); // 4/10

        // Test empty distribution
        let empty_dist = PriorityDistribution::default();
        let (c, h, n, l) = empty_dist.percentages();
        assert_eq!(c, 0.0);
        assert_eq!(h, 0.0);
        assert_eq!(n, 0.0);
        assert_eq!(l, 0.0);
    }

    #[test]
    fn test_priority_distribution_total() {
        let mut dist = PriorityDistribution::default();
        assert_eq!(dist.total(), 0);

        dist.critical = 5;
        dist.high = 10;
        dist.normal = 15;
        dist.low = 20;
        assert_eq!(dist.total(), 50);
    }
}
