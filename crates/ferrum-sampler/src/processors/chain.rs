//! Logits processor chain for composing multiple processors

use crate::LogitsProcessorInterface;
use ferrum_types::Result;

/// Chain of logits processors
///
/// Applies multiple processors in sequence. Processors are applied
/// in the order they were added to the chain.
#[derive(Debug, Default)]
pub struct ProcessorChain {
    processors: Vec<Box<dyn LogitsProcessorInterface + Send + Sync>>,
}

impl ProcessorChain {
    /// Create new empty processor chain
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add a processor to the chain
    pub fn add<P>(&mut self, processor: P) -> &mut Self
    where
        P: LogitsProcessorInterface + Send + Sync + 'static,
    {
        self.processors.push(Box::new(processor));
        self
    }

    /// Add a boxed processor to the chain
    pub fn add_boxed(
        &mut self,
        processor: Box<dyn LogitsProcessorInterface + Send + Sync>,
    ) -> &mut Self {
        self.processors.push(processor);
        self
    }

    /// Get number of processors in chain
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Clear all processors from chain
    pub fn clear(&mut self) {
        self.processors.clear();
    }

    /// Remove processor at index
    pub fn remove(
        &mut self,
        index: usize,
    ) -> Option<Box<dyn LogitsProcessorInterface + Send + Sync>> {
        if index < self.processors.len() {
            Some(self.processors.remove(index))
        } else {
            None
        }
    }

    /// Get processor names in order
    pub fn processor_names(&self) -> Vec<&str> {
        self.processors.iter().map(|p| p.name()).collect()
    }

    /// Check if any processor is stateful
    pub fn has_stateful_processors(&self) -> bool {
        self.processors.iter().any(|p| p.is_stateful())
    }
}

impl LogitsProcessorInterface for ProcessorChain {
    fn process(&self, logits: &mut [f32]) -> Result<()> {
        for processor in &self.processors {
            processor.process(logits)?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "processor_chain"
    }

    fn is_stateful(&self) -> bool {
        self.has_stateful_processors()
    }

    fn reset(&mut self) -> Result<()> {
        for processor in &mut self.processors {
            processor.reset()?;
        }
        Ok(())
    }
}

impl Clone for ProcessorChain {
    fn clone(&self) -> Self {
        // Note: This is a simplified clone that creates a new empty chain.
        // Full cloning of trait objects requires additional infrastructure.
        Self::new()
    }
}

/// Builder for processor chain
pub struct ProcessorChainBuilder {
    chain: ProcessorChain,
}

impl ProcessorChainBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            chain: ProcessorChain::new(),
        }
    }

    /// Add processor to chain
    pub fn with<P>(mut self, processor: P) -> Self
    where
        P: LogitsProcessorInterface + Send + Sync + 'static,
    {
        self.chain.add(processor);
        self
    }

    /// Add boxed processor to chain
    pub fn with_boxed(
        mut self,
        processor: Box<dyn LogitsProcessorInterface + Send + Sync>,
    ) -> Self {
        self.chain.add_boxed(processor);
        self
    }

    /// Build the processor chain
    pub fn build(self) -> ProcessorChain {
        self.chain
    }
}

impl Default for ProcessorChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::TemperatureProcessor;

    #[test]
    fn test_empty_chain() {
        let mut chain = ProcessorChain::new();
        let mut logits = vec![1.0, 2.0, 3.0];

        chain.process(&mut logits).unwrap();

        // Empty chain should not modify logits
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_single_processor_chain() {
        let mut chain = ProcessorChain::new();
        chain.add(TemperatureProcessor::from_value(2.0).unwrap());

        let mut logits = vec![2.0, 4.0, 6.0];
        chain.process(&mut logits).unwrap();

        // Temperature processor should scale by 1/temperature
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_multiple_processor_chain() {
        let mut chain = ProcessorChain::new();
        chain.add(TemperatureProcessor::from_value(2.0).unwrap());
        chain.add(TemperatureProcessor::from_value(0.5).unwrap());

        let mut logits = vec![2.0, 4.0, 6.0];
        chain.process(&mut logits).unwrap();

        // First processor: scale by 1/2, then by 1/0.5 = 2
        // Net effect: no change
        assert_eq!(logits, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_chain_builder() {
        let chain = ProcessorChainBuilder::new()
            .with(TemperatureProcessor::from_value(2.0).unwrap())
            .build();

        assert_eq!(chain.len(), 1);
        assert_eq!(chain.processor_names(), vec!["temperature"]);
    }

    #[test]
    fn test_chain_management() {
        let mut chain = ProcessorChain::new();
        assert!(chain.is_empty());

        chain.add(TemperatureProcessor::from_value(1.0).unwrap());
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());

        let removed = chain.remove(0);
        assert!(removed.is_some());
        assert!(chain.is_empty());
    }
}
