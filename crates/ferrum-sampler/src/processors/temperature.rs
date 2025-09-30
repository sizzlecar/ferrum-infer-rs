//! Temperature scaling processor

use crate::LogitsProcessorInterface;
use ferrum_types::{Result, Temperature};

/// Temperature scaling processor
///
/// Scales logits by dividing by temperature value.
/// Higher temperature (> 1.0) makes output more random.
/// Lower temperature (< 1.0) makes output more deterministic.
#[derive(Debug, Clone)]
pub struct TemperatureProcessor {
    temperature: f32,
}

impl TemperatureProcessor {
    /// Create new temperature processor
    pub fn new(temperature: Temperature) -> Self {
        Self {
            temperature: temperature.value(),
        }
    }

    /// Create from raw value
    pub fn from_value(temperature: f32) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(ferrum_types::FerrumError::config(
                "Temperature must be positive",
            ));
        }
        Ok(Self { temperature })
    }

    /// Get temperature value
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Set temperature value
    pub fn set_temperature(&mut self, temperature: f32) -> Result<()> {
        if temperature <= 0.0 {
            return Err(ferrum_types::FerrumError::config(
                "Temperature must be positive",
            ));
        }
        self.temperature = temperature;
        Ok(())
    }
}

impl LogitsProcessorInterface for TemperatureProcessor {
    fn process(&self, logits: &mut [f32]) -> Result<()> {
        if self.temperature != 1.0 {
            let inv_temp = 1.0 / self.temperature;
            for logit in logits.iter_mut() {
                *logit *= inv_temp;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "temperature"
    }

    fn is_stateful(&self) -> bool {
        false
    }

    fn reset(&mut self) -> Result<()> {
        // Temperature processor is stateless
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_scaling() {
        let mut processor = TemperatureProcessor::from_value(2.0).unwrap();
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];

        processor.process(&mut logits).unwrap();

        // Logits should be scaled by 1/temperature
        assert_eq!(logits, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_temperature_one() {
        let mut processor = TemperatureProcessor::from_value(1.0).unwrap();
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut logits = original.clone();

        processor.process(&mut logits).unwrap();

        // Logits should remain unchanged when temperature = 1.0
        assert_eq!(logits, original);
    }

    #[test]
    fn test_invalid_temperature() {
        assert!(TemperatureProcessor::from_value(0.0).is_err());
        assert!(TemperatureProcessor::from_value(-1.0).is_err());
    }
}
