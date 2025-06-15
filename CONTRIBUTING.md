# Contributing to Athena Cybersecurity Platform

Thank you for your interest in contributing to Athena! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Agent Development](#agent-development)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## 🚀 Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/athena-platform.git
   cd athena-platform
   ```

2. **Set Up Development Environment**
   ```bash
   # Install Rust and WASM toolchain
   rustup target add wasm32-wasi
   cargo install cargo-component
   cargo install spin-cli
   
   # Install development tools
   cargo install cargo-watch
   cargo install cargo-nextest
   ```

3. **Build the Platform**
   ```bash
   cd athena-platform
   cargo build --workspace
   ```

## 💻 Development Process

### Branch Naming

- `feature/` - New features (e.g., `feature/add-yara-scanner`)
- `fix/` - Bug fixes (e.g., `fix/memory-leak-in-wasi-nn`)
- `docs/` - Documentation updates (e.g., `docs/update-api-guide`)
- `refactor/` - Code refactoring (e.g., `refactor/agent-communication`)

### Commit Messages

Follow the conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(owl): add vulnerability scoring model

- Integrate CVSS v3.1 scoring algorithm
- Add WASI-NN model for exploit prediction
- Update test coverage for scoring module

Closes #123
```

## 📝 Coding Standards

### Rust Style Guide

- Follow the official [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/)
- Use `cargo fmt` before committing
- Ensure `cargo clippy` passes with no warnings
- Add documentation comments for public APIs

### Example Code Style

```rust
/// Analyzes a binary for potential vulnerabilities
///
/// # Arguments
/// * `binary_path` - Path to the binary file
/// * `options` - Analysis configuration options
///
/// # Returns
/// * `Result<VulnerabilityReport>` - Analysis results or error
pub async fn analyze_binary(
    binary_path: &Path,
    options: AnalysisOptions,
) -> Result<VulnerabilityReport> {
    // Implementation
}
```

### Error Handling

- Use `anyhow::Result` for application errors
- Provide context with `.context()` method
- Create custom error types for domain-specific errors

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests** - Test individual functions and modules
2. **Integration Tests** - Test agent interactions
3. **Performance Tests** - Benchmark critical paths
4. **Security Tests** - Validate security controls

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_vulnerability_detection() {
        // Arrange
        let engine = create_test_engine().await;
        let sample = load_test_sample("malicious.exe");
        
        // Act
        let result = engine.analyze(sample).await.unwrap();
        
        // Assert
        assert!(result.is_malicious);
        assert_eq!(result.confidence, 0.95);
    }
}
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Run tests before submitting PR:
  ```bash
  cargo test --workspace
  cargo nextest run
  ```

## 🔄 Submitting Changes

### Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write code following our standards
   - Add/update tests
   - Update documentation

3. **Validate Changes**
   ```bash
   cargo fmt
   cargo clippy -- -D warnings
   cargo test
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

## 🤖 Agent Development

### Creating a New Agent

1. **Use the Template**
   ```bash
   ./tools/create-agent.sh --name my-agent --type security-tester
   ```

2. **Implement Core Interfaces**
   - `MessageHandler` for communication
   - `Analyzer` for domain logic
   - `Processor` for data transformation

3. **Add ML Models**
   - Place ONNX models in `models/`
   - Document model inputs/outputs
   - Add inference benchmarks

4. **Document Agent Capabilities**
   - Update agent README
   - Add API documentation
   - Include usage examples

### Agent Review Criteria

- Security isolation maintained
- Performance within targets (<1μs cold start)
- Proper error handling
- Cross-agent communication tested
- Documentation complete

## 📚 Resources

- [WebAssembly Component Model](https://github.com/WebAssembly/component-model)
- [WASI-NN Specification](https://github.com/WebAssembly/wasi-nn)
- [Spin Documentation](https://developer.fermyon.com/spin)
- [Rust WASM Book](https://rustwasm.github.io/docs/book/)

## ❓ Questions?

- Open a [Discussion](https://github.com/your-org/athena-platform/discussions)
- Join our [Discord](https://discord.gg/athena-platform)
- Email: contributors@athena-platform.io

Thank you for contributing to Athena! 🛡️