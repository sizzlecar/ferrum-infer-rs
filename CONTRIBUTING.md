# 贡献指南

感谢您对本项目的关注！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 Bug 报告
- 💡 功能建议
- 📝 文档改进
- 🔧 代码贡献
- 🧪 测试用例

## 📋 贡献准则

### 行为准则

请遵循我们的行为准则，确保社区环境友好和包容：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同情

### 开发环境设置

1. **Fork 项目**
   ```bash
   git clone https://github.com/your-username/rust-project.git
   cd rust-project
   ```

2. **安装依赖**
   ```bash
   # 确保 Rust 版本 >= 1.70
   rustc --version
   
   # 安装项目依赖
   cargo build
   ```

3. **验证环境**
   ```bash
   # 运行测试
   cargo test
   
   # 检查代码格式
   cargo fmt --check
   
   # 运行 clippy
   cargo clippy
   ```

### 开发工作流

1. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/issue-number
   ```

2. **进行开发**
   - 编写代码
   - 添加测试
   - 更新文档
   - 确保代码通过所有检查

3. **提交更改**
   ```bash
   # 使用 conventional commits 格式
   git commit -m "feat: add new feature"
   git commit -m "fix: resolve issue with parser"
   git commit -m "docs: update API documentation"
   ```

4. **推送分支**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 填写 PR 模板
   - 等待代码审查

## 📝 代码规范

### Rust 代码风格

项目使用标准的 Rust 代码风格：

- 使用 `rustfmt` 进行代码格式化
- 遵循 `clippy` 的建议
- 变量和函数名使用 `snake_case`
- 常量使用 `SCREAMING_SNAKE_CASE`
- 类型名使用 `PascalCase`

### 文档规范

- 所有公共 API 必须有文档注释
- 使用 `///` 为函数和结构体添加文档
- 包含使用示例：
  ```rust
  /// 计算两个数的和
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
  ```

### 测试规范

- 新功能必须包含测试
- 测试函数使用描述性名称
- 使用 `#[cfg(test)]` 模块组织测试
- 集成测试放在 `tests/` 目录

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_positive_numbers() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn test_add_negative_numbers() {
        assert_eq!(add(-2, -3), -5);
    }
}
```

## 🔍 提交信息格式

项目使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 类型说明

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 仅文档更改
- `style`: 不影响代码含义的更改（空格、格式化等）
- `refactor`: 既不修复 bug 也不添加功能的代码更改
- `perf`: 提高性能的代码更改
- `test`: 添加或修改测试
- `chore`: 构建过程或辅助工具的更改
- `ci`: CI 配置和脚本的更改
- `build`: 影响构建系统或外部依赖的更改

### 示例

```bash
feat: add user authentication module
fix(parser): handle empty input correctly
docs: update installation instructions
style: format code with rustfmt
refactor: simplify error handling logic
perf: optimize string concatenation
test: add unit tests for calculator
chore: update dependencies
ci: add benchmark workflow
build: configure cross-compilation
```

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定模块测试
cargo test calculator

# 运行集成测试
cargo test --test integration_tests

# 生成测试覆盖率报告
cargo llvm-cov --open
```

### 测试分类

1. **单元测试**: 测试单个函数或方法
2. **集成测试**: 测试模块间的交互
3. **文档测试**: 测试文档中的示例代码
4. **基准测试**: 性能测试

### 测试最佳实践

- 测试应该快速、可靠、独立
- 使用描述性的测试名称
- 每个测试只验证一个行为
- 使用 `pretty_assertions` 获得更好的错误输出

## 📋 Pull Request 检查清单

在提交 PR 之前，请确保：

- [ ] 代码通过所有测试 (`cargo test`)
- [ ] 代码格式化正确 (`cargo fmt`)
- [ ] 通过 Clippy 检查 (`cargo clippy`)
- [ ] 新功能有相应的测试
- [ ] 公共 API 有文档注释
- [ ] 提交信息符合 Conventional Commits 格式
- [ ] PR 描述清晰，包含变更说明
- [ ] 如果是 breaking change，已在 PR 中说明

## 📄 PR 模板

创建 PR 时请填写以下信息：

```markdown
## 变更说明

简要描述这个 PR 的目的和主要变更。

## 变更类型

- [ ] Bug 修复
- [ ] 新功能
- [ ] Breaking change
- [ ] 文档更新
- [ ] 性能优化
- [ ] 重构
- [ ] 其他 (请说明):

## 测试

- [ ] 新增测试覆盖新功能
- [ ] 现有测试仍然通过
- [ ] 手动测试已完成

## 检查清单

- [ ] 代码符合项目规范
- [ ] 自我审查代码
- [ ] 添加了必要的注释
- [ ] 文档已更新
- [ ] 无新的警告产生

## 其他信息

添加任何其他相关信息、截图或 GIF。
```

## 🐛 Bug 报告

发现 Bug 时，请创建 Issue 并包含：

1. **问题描述**: 清晰描述遇到的问题
2. **复现步骤**: 详细的复现步骤
3. **期望行为**: 描述期望的正确行为
4. **实际行为**: 描述实际发生的情况
5. **环境信息**: 
   - OS 版本
   - Rust 版本
   - 项目版本
6. **额外信息**: 日志、截图等

## 💡 功能建议

提出功能建议时，请说明：

1. **功能描述**: 详细描述建议的功能
2. **使用场景**: 这个功能解决什么问题
3. **替代方案**: 是否考虑过其他解决方案
4. **实现建议**: 如果有想法，可以提供实现建议

## 📞 获取帮助

如果您在贡献过程中遇到问题，可以通过以下方式获取帮助：

- 创建 GitHub Issue
- 在项目讨论区提问
- 发送邮件到 your.email@example.com

## 🙏 致谢

感谢每一位贡献者！您的贡献让这个项目变得更好。

---

Happy coding! 🎉