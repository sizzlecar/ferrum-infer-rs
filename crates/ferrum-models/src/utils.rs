//! Utility functions for model operations

use ferrum_core::{Tensor, TokenId, Result, Error};
use candle_core::{Device, DType, Tensor as CandleTensor};
use tracing::debug;

/// Convert Ferrum Tensor to Candle Tensor
pub fn to_candle_tensor(tensor: &Tensor, device: &Device) -> Result<CandleTensor> {
    CandleTensor::from_vec(
        tensor.data.clone(),
        tensor.shape.clone(),
        device,
    ).map_err(|e| Error::internal(format!("Failed to convert to Candle tensor: {}", e)))
}

/// Convert Candle Tensor to Ferrum Tensor
pub fn from_candle_tensor(tensor: &CandleTensor) -> Result<Tensor> {
    let shape = tensor.dims().to_vec();
    let data = tensor
        .flatten_all()
        .map_err(|e| Error::internal(format!("Failed to flatten tensor: {}", e)))?
        .to_vec1::<f32>()
        .map_err(|e| Error::internal(format!("Failed to convert tensor to vec: {}", e)))?;
    
    Ok(Tensor::new(data, shape))
}

/// Apply temperature to logits
pub fn apply_temperature(logits: &CandleTensor, temperature: f32) -> Result<CandleTensor> {
    if temperature == 1.0 {
        return Ok(logits.clone());
    }
    
    (logits / temperature as f64)
        .map_err(|e| Error::model_execution(format!("Failed to apply temperature: {}", e)))
}

/// Apply top-k filtering to logits
pub fn apply_top_k(logits: &CandleTensor, k: usize) -> Result<CandleTensor> {
    let vocab_size = logits.dims()[logits.dims().len() - 1];
    
    if k >= vocab_size {
        return Ok(logits.clone());
    }
    
    // Get top-k values and indices
    let (topk_values, _topk_indices) = logits
        .topk(k, logits.dims().len() - 1, true, true)
        .map_err(|e| Error::model_execution(format!("Failed to apply top-k: {}", e)))?;
    
    // Get minimum value in top-k
    let min_topk = topk_values
        .min(logits.dims().len() - 1)
        .map_err(|e| Error::model_execution(format!("Failed to get min top-k: {}", e)))?
        .to_scalar::<f32>()
        .map_err(|e| Error::model_execution(format!("Failed to convert to scalar: {}", e)))?;
    
    // Mask values below threshold
    let mask = logits
        .ge(min_topk)
        .map_err(|e| Error::model_execution(format!("Failed to create mask: {}", e)))?;
    
    let filtered = logits
        .where_cond(&mask, &CandleTensor::new(f32::NEG_INFINITY, logits.device())?)
        .map_err(|e| Error::model_execution(format!("Failed to apply mask: {}", e)))?;
    
    Ok(filtered)
}

/// Apply top-p (nucleus) filtering to logits
pub fn apply_top_p(logits: &CandleTensor, p: f32) -> Result<CandleTensor> {
    if p >= 1.0 {
        return Ok(logits.clone());
    }
    
    // Apply softmax to get probabilities
    let probs = candle_nn::ops::softmax(logits, logits.dims().len() - 1)
        .map_err(|e| Error::model_execution(format!("Failed to apply softmax: {}", e)))?;
    
    // Sort probabilities in descending order
    let (sorted_probs, sorted_indices) = probs
        .sort(logits.dims().len() - 1, true)
        .map_err(|e| Error::model_execution(format!("Failed to sort probabilities: {}", e)))?;
    
    // Calculate cumulative probabilities
    let cumsum = sorted_probs
        .cumsum(logits.dims().len() - 1)
        .map_err(|e| Error::model_execution(format!("Failed to calculate cumsum: {}", e)))?;
    
    // Create mask for values to keep
    let mask = cumsum
        .le(p as f64)
        .map_err(|e| Error::model_execution(format!("Failed to create mask: {}", e)))?;
    
    // Apply mask to original logits
    let filtered = logits
        .gather(&sorted_indices, logits.dims().len() - 1)
        .map_err(|e| Error::model_execution(format!("Failed to gather: {}", e)))?
        .where_cond(&mask, &CandleTensor::new(f32::NEG_INFINITY, logits.device())?)
        .map_err(|e| Error::model_execution(format!("Failed to apply mask: {}", e)))?;
    
    Ok(filtered)
}

/// Apply repetition penalty to logits
pub fn apply_repetition_penalty(
    logits: &CandleTensor,
    generated_tokens: &[TokenId],
    penalty: f32,
) -> Result<CandleTensor> {
    if penalty == 1.0 || generated_tokens.is_empty() {
        return Ok(logits.clone());
    }
    
    let mut logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| Error::model_execution(format!("Failed to convert logits to vec: {}", e)))?;
    
    // Apply penalty to previously generated tokens
    for &token_id in generated_tokens {
        if (token_id as usize) < logits_vec.len() {
            if logits_vec[token_id as usize] > 0.0 {
                logits_vec[token_id as usize] /= penalty;
            } else {
                logits_vec[token_id as usize] *= penalty;
            }
        }
    }
    
    CandleTensor::from_vec(logits_vec, logits.dims(), logits.device())
        .map_err(|e| Error::model_execution(format!("Failed to create tensor from vec: {}", e)))
}

/// Sample token from logits using multinomial sampling
pub fn sample_token(logits: &CandleTensor) -> Result<TokenId> {
    // Apply softmax to get probabilities
    let probs = candle_nn::ops::softmax(logits, logits.dims().len() - 1)
        .map_err(|e| Error::model_execution(format!("Failed to apply softmax: {}", e)))?;
    
    let probs_vec = probs
        .to_vec1::<f32>()
        .map_err(|e| Error::model_execution(format!("Failed to convert probs to vec: {}", e)))?;
    
    // Sample from multinomial distribution
    use rand::distributions::{Distribution, WeightedIndex};
    let mut rng = rand::thread_rng();
    
    let dist = WeightedIndex::new(&probs_vec)
        .map_err(|e| Error::model_execution(format!("Failed to create distribution: {}", e)))?;
    
    Ok(dist.sample(&mut rng) as TokenId)
}

/// Greedy decoding - select token with highest probability
pub fn greedy_decode(logits: &CandleTensor) -> Result<TokenId> {
    let logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| Error::model_execution(format!("Failed to convert logits to vec: {}", e)))?;
    
    logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as TokenId)
        .ok_or_else(|| Error::model_execution("Failed to find max logit"))
}

/// Calculate perplexity from loss
pub fn calculate_perplexity(loss: f32) -> f32 {
    loss.exp()
}

/// Format model size in human-readable format
pub fn format_model_size(num_params: u64) -> String {
    if num_params >= 1_000_000_000 {
        format!("{:.1}B", num_params as f64 / 1_000_000_000.0)
    } else if num_params >= 1_000_000 {
        format!("{:.1}M", num_params as f64 / 1_000_000.0)
    } else if num_params >= 1_000 {
        format!("{:.1}K", num_params as f64 / 1_000.0)
    } else {
        format!("{}", num_params)
    }
}
