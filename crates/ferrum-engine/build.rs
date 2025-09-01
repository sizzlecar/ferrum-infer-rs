use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if cfg!(all(feature = "metal", any(target_os = "macos", target_os = "ios"))) {
        println!("cargo:rerun-if-env-changed=OPT_LEVEL");
        compile_metal_shaders();
    } else {
        write_empty_metallib();
    }
}

fn compile_metal_shaders() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    if target_os != "macos" && target_os != "ios" {
        println!("cargo:warning=Not an Apple platform, skipping Metal shader compilation");
        write_empty_metallib();
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let metal_dir = manifest_dir.join("src/metal/kernel");
    
    if !metal_dir.exists() {
        println!("cargo:warning=Metal kernel directory not found, skipping shader compilation");
        write_empty_metallib();
        return;
    }
    
    let metal_files = find_metal_files(&metal_dir);
    if metal_files.is_empty() {
        println!("cargo:warning=No Metal shader files found");
        write_empty_metallib();
        return;
    }

    println!("cargo:info=Found {} Metal shader files", metal_files.len());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let metallib_path = out_dir.join("ferrum_shaders.metallib");

    // Check if xcrun is available
    if !Command::new("xcrun")
        .arg("--version")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        println!("cargo:warning=xcrun not found, skipping Metal shader compilation");
        write_empty_metallib();
        return;
    }

    compile_metal_files(&metal_files, &metallib_path, &metal_dir);
    
    // Read metallib and embed as bytes
    let metallib_data = fs::read(&metallib_path)
        .unwrap_or_else(|_| {
            println!("cargo:warning=Failed to read compiled metallib, using empty library");
            Vec::new()
        });
    
    write_metallib_as_bytes(&metallib_data, &out_dir.join("metal_lib.rs"));

    // Tell Cargo when to recompile
    for file in &metal_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}

fn find_metal_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_metal_files(&path));
            } else if path.extension().and_then(|s| s.to_str()) == Some("metal") {
                files.push(path);
            }
        }
    }
    files
}

fn compile_metal_files(metal_files: &[PathBuf], output_path: &Path, include_dir: &Path) {
    let opt = env::var("OPT_LEVEL").unwrap_or_else(|_| "0".into());
    let metal_opt = match opt.as_str() {
        "0" => "-O0",
        "1" => "-O1", 
        _ => "-O2",
    };

    // Compile all .metal files to .air
    let temp_dir = output_path.parent().unwrap().join("air_temp");
    fs::create_dir_all(&temp_dir).unwrap();
    
    let mut air_files = Vec::new();
    
    for metal_file in metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_file = temp_dir.join(format!("{}.air", stem));
        air_files.push(air_file.clone());
        
        let mut cmd = Command::new("xcrun");
        cmd.args(&["-sdk", "macosx", "metal"])
            .arg(metal_opt)
            .arg("-std=metal3.1")
            .arg("-mmacosx-version-min=14.0")
            .args(&["-I", include_dir.to_str().unwrap()])
            .args(&["-c", metal_file.to_str().unwrap()])
            .args(&["-o", air_file.to_str().unwrap()]);

        println!("cargo:info=Compiling Metal shader: {}", metal_file.display());
        
        if !cmd.status().expect("Failed to run Metal compiler").success() {
            panic!("Metal shader compilation failed for {}", metal_file.display());
        }
    }

    // Link .air files into .metallib
    let mut lib_cmd = Command::new("xcrun");
    lib_cmd.args(&["-sdk", "macosx", "metallib"]);
    for air_file in &air_files {
        lib_cmd.arg(air_file);
    }
    lib_cmd.args(&["-o", output_path.to_str().unwrap()]);

    println!("cargo:info=Linking Metal library: {}", output_path.display());
    
    if !lib_cmd.status().expect("Failed to run metallib").success() {
        panic!("Metal library linking failed");
    }

    // Clean up temporary .air files
    for air_file in air_files {
        let _ = fs::remove_file(air_file);
    }
    let _ = fs::remove_dir(&temp_dir);
    
    println!("cargo:info=Metal shaders compiled successfully");
}

fn write_metallib_as_bytes(metallib_data: &[u8], out_path: &Path) {
    let mut f = fs::File::create(out_path).expect("Cannot create metal_lib.rs");
    writeln!(f, "// AUTO-GENERATED Metal library byte array").unwrap();
    writeln!(f, "pub const METAL_LIBRARY_DATA: &[u8] = &[").unwrap();
    
    for chunk in metallib_data.chunks(16) {
        write!(f, "    ").unwrap();
        for byte in chunk {
            write!(f, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(f).unwrap();
    }
    
    writeln!(f, "];").unwrap();
}

fn write_empty_metallib() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let metal_lib_rs = out_dir.join("metal_lib.rs");
    
    fs::write(metal_lib_rs, b"pub const METAL_LIBRARY_DATA: &[u8] = &[];\n")
        .expect("Cannot create stub metal_lib.rs");
}