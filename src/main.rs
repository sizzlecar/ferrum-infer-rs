use clap::{Arg, Command};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    name: String,
    version: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            name: "rust-project".to_string(),
            version: "0.1.0".to_string(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("rust-project")
        .version("0.1.0")
        .author("Your Name <your.email@example.com>")
        .about("A Rust project with comprehensive development workflow")
        .arg(
            Arg::new("name")
                .short('n')
                .long("name")
                .value_name("NAME")
                .help("Sets a custom name")
                .action(clap::ArgAction::Set),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Turn debugging information on")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let config = Config::default();

    if let Some(name) = matches.get_one::<String>("name") {
        println!("Hello, {}!", name);
    } else {
        println!("Hello, {}!", config.name);
    }

    if matches.get_flag("verbose") {
        println!("Debug info: {:?}", config);
    }

    Ok(())
}
