#!/usr/bin/env rust-script

//! MVP test script to verify basic functionality

use std::process::Command;

fn main() {
    println!("🧪 Ferrum MVP Test Suite");
    println!("========================");
    
    // Test 1: Check if binary builds
    println!("\n1️⃣ Testing binary build...");
    let output = Command::new("cargo")
        .args(&["build", "-p", "ferrum-cli"])
        .output()
        .expect("Failed to run cargo build");
    
    if output.status.success() {
        println!("✅ Binary builds successfully");
    } else {
        println!("❌ Binary build failed");
        println!("Error: {}", String::from_utf8_lossy(&output.stderr));
        return;
    }
    
    // Test 2: Check CLI help
    println!("\n2️⃣ Testing CLI help...");
    let output = Command::new("./target/debug/ferrum")
        .args(&["--help"])
        .output()
        .expect("Failed to run ferrum --help");
    
    if output.status.success() {
        println!("✅ CLI help works");
    } else {
        println!("❌ CLI help failed");
        return;
    }
    
    // Test 3: Check serve command help
    println!("\n3️⃣ Testing serve command help...");
    let output = Command::new("./target/debug/ferrum")
        .args(&["serve", "--help"])
        .output()
        .expect("Failed to run ferrum serve --help");
    
    if output.status.success() {
        println!("✅ Serve command help works");
    } else {
        println!("❌ Serve command help failed");
        return;
    }
    
    // Test 4: Check infer command help
    println!("\n4️⃣ Testing infer command help...");
    let output = Command::new("./target/debug/ferrum")
        .args(&["infer", "--help"])
        .output()
        .expect("Failed to run ferrum infer --help");
    
    if output.status.success() {
        println!("✅ Infer command help works");
    } else {
        println!("❌ Infer command help failed");
        return;
    }
    
    println!("\n🎉 All basic tests passed!");
    println!("📝 Next steps:");
    println!("   - Fix model download issue");
    println!("   - Test actual inference");
    println!("   - Test server startup");
}