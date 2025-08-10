# Rust Project Template

[![CI/CD Pipeline](https://github.com/username/rust-project/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/username/rust-project/actions)
[![codecov](https://codecov.io/gh/username/rust-project/branch/main/graph/badge.svg)](https://codecov.io/gh/username/rust-project)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)

一个包含完整开发流程的 Rust 项目模板，包含代码质量检查、自动化测试、CI/CD 流程等最佳实践。

## 📋 功能特性

- ✅ **代码质量保证**
  - Rustfmt 自动格式化
  - Clippy lint 检查
  - 预提交钩子（pre-commit hooks）
  - 提交信息格式验证

- 🔄 **CI/CD 流程**
  - GitHub Actions 自动化构建
  - 多平台编译支持
  - 自动化测试和基准测试
  - 代码覆盖率报告
  - 安全漏洞扫描

- 📚 **完整文档**
  - API 文档自动生成
  - 使用示例和教程
  - 开发指南和贡献指导

- 🛠️ **开发工具**
  - 配置文件模板
  - 实用工具函数
  - 性能基准测试

## 🚀 快速开始

### 前置要求

- Rust 1.70 或更高版本
- Git

### 安装

```bash
# 克隆项目
git clone https://github.com/username/rust-project.git
cd rust-project

# 安装依赖
cargo build

# 运行测试
cargo test

# 运行程序
cargo run -- --help
```

### 使用示例

```bash
# 基本使用
cargo run

# 带参数运行
cargo run -- --name "World" --verbose

# 运行测试
cargo test

# 运行基准测试
cargo bench

# 生成文档
cargo doc --open

# 代码格式化
cargo fmt

# 代码检查
cargo clippy
```

## 📖 项目结构

```
rust-project/
├── .github/workflows/     # GitHub Actions 工作流
├── .git/hooks/           # Git 钩子脚本
├── benches/              # 基准测试
├── src/                  # 源代码
│   ├── lib.rs           # 库入口
│   └── main.rs          # 二进制入口
├── Cargo.toml           # 项目配置
├── rustfmt.toml         # 代码格式化配置
├── clippy.toml          # Clippy 配置
└── README.md            # 项目文档
```

## 🔧 开发指南

### 代码规范

项目遵循以下代码规范：

1. **格式化**: 使用 `rustfmt` 进行代码格式化
2. **Lint**: 使用 `clippy` 进行代码检查
3. **测试**: 所有功能都需要有相应的单元测试
4. **文档**: 公共 API 需要有完整的文档注释

### Git 工作流

1. **分支策略**: 使用 Git Flow 分支模型
   - `main`: 主分支，用于发布
   - `develop`: 开发分支
   - `feature/*`: 功能分支
   - `hotfix/*`: 热修复分支

2. **提交规范**: 使用 Conventional Commits 格式
   ```
   <type>[optional scope]: <description>
   
   [optional body]
   
   [optional footer(s)]
   ```

   类型说明：
   - `feat`: 新功能
   - `fix`: 修复 bug
   - `docs`: 文档更新
   - `style`: 代码风格调整
   - `refactor`: 重构
   - `perf`: 性能优化
   - `test`: 测试相关
   - `chore`: 构建工具或辅助工具变动

### 提交前检查

项目配置了 Git hooks，在提交前会自动执行：

1. 代码格式化检查
2. Clippy lint 检查
3. 单元测试
4. 编译检查
5. 提交信息格式验证

### 持续集成

每次推送和 Pull Request 都会触发 CI/CD 流程：

1. **质量检查**
   - 代码格式验证
   - Clippy lint
   - 单元测试
   - 文档测试

2. **安全检查**
   - 依赖漏洞扫描
   - 代码安全审计

3. **性能测试**
   - 基准测试
   - 性能回归检测

4. **多平台构建**
   - Linux (x86_64)
   - Windows (x86_64)
   - macOS (x86_64, aarch64)

## 📝 API 文档

详细的 API 文档可以通过以下命令生成：

```bash
cargo doc --open
```

或查看在线文档：[项目 API 文档](https://docs.rs/rust-project)

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_name

# 显示测试输出
cargo test -- --nocapture

# 运行文档测试
cargo test --doc
```

### 代码覆盖率

```bash
# 安装 cargo-llvm-cov
cargo install cargo-llvm-cov

# 生成覆盖率报告
cargo llvm-cov --open
```

### 基准测试

```bash
# 运行基准测试
cargo bench

# 查看基准测试报告
open target/criterion/report/index.html
```

## 🚀 部署

### 构建发布版本

```bash
# 优化构建
cargo build --release

# 交叉编译
cargo build --release --target x86_64-pc-windows-gnu
```

### 发布流程

1. 更新版本号在 `Cargo.toml`
2. 更新 CHANGELOG.md
3. 创建 git tag: `git tag v0.1.0`
4. 推送 tag: `git push origin v0.1.0`
5. GitHub Actions 会自动创建 release

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 贡献要求

- 代码需要通过所有测试
- 新功能需要添加相应的测试
- 遵循项目的代码规范
- 提交信息需要符合 Conventional Commits 格式

## 📄 许可证

本项目使用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Rust 社区](https://www.rust-lang.org/community)
- 所有贡献者

## 📞 联系方式

- 作者：Your Name
- 邮箱：your.email@example.com
- 项目主页：https://github.com/username/rust-project

---

如果这个项目对你有帮助，请给一个 ⭐️！