# Task Completion Checklist

## Before Marking a Task Complete

### 1. Code Quality Checks
```bash
# Format code
uv run --frozen black src tests

# Lint code
uv run --frozen ruff check src tests

# Type checking
uv run --frozen mypy src
```

### 2. Testing Requirements
```bash
# Run all tests
uv run --frozen pytest

# Check coverage (must be >= 80%)
uv run --frozen pytest --cov=src --cov-report=term-missing

# Run specific test for new feature
uv run --frozen pytest tests/unit/test_<feature>.py -v
```

### 3. Documentation
- [ ] Docstrings added for all public functions/classes
- [ ] Type hints complete and accurate
- [ ] Complex logic has inline comments
- [ ] README updated if new feature added

### 4. Git Workflow
```bash
# Stage changes
git add .

# Review changes
git diff --staged

# Commit with descriptive message
git commit -m "feat: <description>"
# or
git commit -m "fix: <description>"
# or
git commit -m "test: <description>"
```

### 5. Validation Checklist
- [ ] All tests pass
- [ ] Coverage >= 80%
- [ ] No linting errors
- [ ] No type checking errors
- [ ] Code follows project conventions
- [ ] Float32 constraint applied where needed
- [ ] Polars used for DataFrames (not Pandas)
- [ ] Async pattern used for I/O operations

### 6. File Updates
- [ ] Update docs/context.md with completion status
- [ ] Update docs/plan.md with next steps
- [ ] Mark step as completed in tracking

### 7. Common Issues to Check
- [ ] No hardcoded values (use config)
- [ ] Error handling implemented
- [ ] Edge cases tested
- [ ] Performance considerations addressed
- [ ] Security best practices followed

## Post-Completion
1. Verify all checks passed
2. Update progress tracking
3. Prepare for next task
4. Document any technical debt or TODOs

## Quick Command Sequence
```bash
# Full validation sequence
uv run --frozen black src tests && \
uv run --frozen ruff check src tests && \
uv run --frozen mypy src && \
uv run --frozen pytest --cov=src

# If all pass, commit
git add . && git commit -m "feat: <description>"
```