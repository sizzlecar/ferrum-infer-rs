//! # Rust Project
//!
//! 这是一个包含完整开发流程的 Rust 项目示例。
//!
//! ## 功能特性
//!
//! - 代码格式化和 lint 检查
//! - 自动化测试和基准测试
//! - Git hooks 和 CI/CD 流程
//! - 完整的项目文档

use serde::{Deserialize, Serialize};
use std::fmt;

/// 应用程序配置结构
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    /// 应用程序名称
    pub name: String,
    /// 版本号
    pub version: String,
    /// 调试模式
    pub debug: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            name: "rust-project".to_string(),
            version: "0.1.0".to_string(),
            debug: false,
        }
    }
}

impl fmt::Display for AppConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} v{}", self.name, self.version)
    }
}

/// 计算器模块
pub mod calculator {
    /// 加法运算
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_project::calculator::add;
    /// assert_eq!(add(2, 3), 5);
    /// ```
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }

    /// 减法运算
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_project::calculator::subtract;
    /// assert_eq!(subtract(5, 3), 2);
    /// ```
    pub fn subtract(a: i32, b: i32) -> i32 {
        a - b
    }

    /// 乘法运算
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_project::calculator::multiply;
    /// assert_eq!(multiply(4, 3), 12);
    /// ```
    pub fn multiply(a: i32, b: i32) -> i32 {
        a * b
    }

    /// 除法运算
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_project::calculator::divide;
    /// assert_eq!(divide(10, 2), Ok(5));
    /// assert_eq!(divide(10, 0), Err("Division by zero"));
    /// ```
    pub fn divide(a: i32, b: i32) -> Result<i32, &'static str> {
        if b == 0 {
            Err("Division by zero")
        } else {
            Ok(a / b)
        }
    }
}

/// 实用工具模块
pub mod utils {
    use super::AppConfig;

    /// 创建默认配置
    pub fn create_default_config() -> AppConfig {
        AppConfig::default()
    }

    /// 验证配置
    pub fn validate_config(config: &AppConfig) -> Result<(), String> {
        if config.name.is_empty() {
            return Err("Name cannot be empty".to_string());
        }
        if config.version.is_empty() {
            return Err("Version cannot be empty".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_config_default() {
        let config = AppConfig::default();
        assert_eq!(config.name, "rust-project");
        assert_eq!(config.version, "0.1.0");
        assert!(!config.debug);
    }

    #[test]
    fn test_app_config_display() {
        let config = AppConfig::default();
        assert_eq!(format!("{}", config), "rust-project v0.1.0");
    }

    #[test]
    fn test_calculator_operations() {
        use calculator::*;

        assert_eq!(add(2, 3), 5);
        assert_eq!(subtract(5, 3), 2);
        assert_eq!(multiply(4, 3), 12);
        assert_eq!(divide(10, 2), Ok(5));
        assert_eq!(divide(10, 0), Err("Division by zero"));
    }

    #[test]
    fn test_utils() {
        use utils::*;

        let config = create_default_config();
        assert!(validate_config(&config).is_ok());

        let invalid_config = AppConfig {
            name: "".to_string(),
            version: "1.0.0".to_string(),
            debug: false,
        };
        assert!(validate_config(&invalid_config).is_err());
    }
}
