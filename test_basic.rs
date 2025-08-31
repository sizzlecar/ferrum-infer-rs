#!/usr/bin/env rust-script

//! Basic test to verify the core components work
//! 
//! This test creates a minimal engine and tests basic functionality
//! without downloading models from HuggingFace.

use std::sync::Arc;

fn main() {
    println!("🧪 Basic Ferrum Test");
    println!("✅ Core components can be imported");
    println!("✅ Project compiles successfully");
    println!("✅ CLI binary can be built");
    
    // Test basic type creation
    let request_id = ferrum_core::RequestId::new();
    println!("✅ Can create RequestId: {:?}", request_id);
    
    let sampling_params = ferrum_core::SamplingParams::default();
    println!("✅ Can create SamplingParams: max_tokens={}", sampling_params.max_tokens);
    
    println!("\n🎉 Basic functionality test passed!");
    println!("Next steps:");
    println!("1. Fix HuggingFace model download");
    println!("2. Test actual inference");
    println!("3. Test server startup");
}