# Contributing Guidelines

## Code Standards

### Style Guide

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use Ruff for linting
- Use type hints for all function signatures
- Keep files under 500 lines

### Documentation

- Write concise, professional docstrings
- No AI-style verbose comments
- Document complex algorithms with brief explanations
- Update README for new features

### Example

```python
def calculate_latency(venue: str, features: dict) -> float:
    """Calculate predicted latency for venue.

    Args:
        venue: Trading venue identifier
        features: Market features dict

    Returns:
        Predicted latency in microseconds
    """
    # Implementation
    pass
```

## Testing

### Requirements

- Unit tests for all new functions
- Integration tests for new components
- Maintain >80% code coverage
- All tests must pass before PR

### Running Tests

```bash
make test           # All tests
make test-unit      # Unit only
make test-cov       # With coverage
```

### Writing Tests

```python
import pytest

class TestLatencyPredictor:
    """Test latency prediction."""

    def test_prediction_accuracy(self):
        """Test prediction within bounds."""
        predictor = LatencyPredictor(['NYSE'])
        result = predictor.predict({'volume': 1000})

        assert result > 0
        assert result < 10000
```

## Code Quality

### Pre-commit Checks

Install hooks:
```bash
make setup
```

Manual run:
```bash
make quality
```

### Checks

1. **Black** - Code formatting
2. **Ruff** - Linting
3. **Mypy** - Type checking
4. **Tests** - All passing

## Pull Request Process

1. Create feature branch: `feature/description`
2. Write code following standards
3. Add tests
4. Run quality checks: `make quality`
5. Ensure tests pass: `make test`
6. Update documentation
7. Submit PR with description
8. Address review comments

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] All tests passing

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings
```

## Architecture

### File Organization

- Keep modules under 500 LOC
- One class per file (exceptions allowed)
- Clear separation of concerns
- Logical grouping in directories

### Module Structure

```
src/
├── core/          # Core types, config, logging
├── infra/         # Infrastructure (Phase 1)
├── ml/            # ML models (Phase 2)
├── execution/     # Trading execution (Phase 3)
└── strategies/    # Trading strategies
```

## Git Workflow

### Branches

- `main` - Production-ready code
- `develop` - Integration branch
- `feature/*` - New features
- `fix/*` - Bug fixes
- `refactor/*` - Code refactoring

### Commits

Use conventional commits:

```
feat: add latency prediction caching
fix: correct position size calculation
refactor: split large orchestrator file
test: add unit tests for config manager
docs: update README with new CLI flags
```

### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## Performance Guidelines

### Critical Paths

- Order execution: <1ms target
- ML prediction: <100μs target
- Tick processing: >10k/sec target

### Optimization Rules

1. Profile before optimizing
2. Use async for I/O operations
3. Avoid premature optimization
4. Document performance-critical code
5. Add performance tests for hot paths

## Security

### Secrets Management

- Never commit secrets
- Use environment variables
- Store keys in `.env` (not in git)
- Use secrets manager in production

### Code Review Focus

- Input validation
- Error handling
- Resource cleanup
- Type safety
- Security vulnerabilities

## Questions?

Contact the trading systems team or open an issue.
