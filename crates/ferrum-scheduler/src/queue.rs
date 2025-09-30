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
        if let Some(pos) = self
            .requests
            .iter()
            .position(|req| &req.id == request_id)
        {
            self.requests.remove(pos)
        } else {
            None
        }
    }

    /// Get reference to request by ID
    pub fn get(&self, request_id: &RequestId) -> Option<&InferenceRequest> {
        self.requests
            .iter()
            .find(|req| &req.id == request_id)
    }

    /// Get mutable reference to request by ID
    pub fn get_mut(&mut self, request_id: &RequestId) -> Option<&mut InferenceRequest> {
        self.requests
            .iter_mut()
            .find(|req| &req.id == request_id)
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
