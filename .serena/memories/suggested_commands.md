# Suggested Development Commands

## Package Management (uv)
```bash
# Install dependencies
uv sync

# Run with frozen dependencies
uv run --frozen <command>

# Add new dependency
uv add <package>

# Add dev dependency
uv add --dev <package>
```

## Testing
```bash
# Run all tests with coverage
uv run --frozen pytest

# Run specific test file
uv run --frozen pytest tests/unit/test_tick_model.py

# Run with verbose output
uv run --frozen pytest -v

# Run tests with specific marker
uv run --frozen pytest -m unit
uv run --frozen pytest -m "not slow"

# Generate coverage report
uv run --frozen pytest --cov=src --cov-report=html
```

## Code Quality
```bash
# Format code with Black
uv run --frozen black src tests

# Lint with Ruff
uv run --frozen ruff check src tests

# Fix linting issues
uv run --frozen ruff check --fix src tests

# Type checking with MyPy
uv run --frozen mypy src

# Run all quality checks
uv run --frozen black src tests && uv run --frozen ruff check src tests && uv run --frozen mypy src
```

## Git Commands (Windows)
```bash
# Check status
git status

# View diff
git diff

# Stage changes
git add <file>
git add .

# Commit with message
git commit -m "message"

# View log
git log --oneline -10

# Push to remote
git push -u origin task_1

# Create branch
git checkout -b <branch-name>

# Switch branch
git checkout <branch-name>
```

## Development Workflow
```bash
# Before committing
uv run --frozen black src tests
uv run --frozen ruff check src tests
uv run --frozen mypy src
uv run --frozen pytest

# Quick test during development
uv run --frozen pytest tests/unit/test_<current_file>.py -v

# Watch tests (if pytest-watch installed)
uv run --frozen ptw tests/unit/ -- -v
```

## Project-Specific Commands
```bash
# Run main application
uv run --frozen python -m src.main

# Interactive Python with project context
uv run --frozen ipython

# Jupyter notebook
uv run --frozen jupyter notebook
```

## File Operations (Windows)
```bash
# List files
dir
ls  # Also works in Git Bash/PowerShell

# Change directory
cd <path>

# Create directory
mkdir <name>

# View file content
type <file>
cat <file>  # In Git Bash/PowerShell

# Find files
dir /s /b *pattern*
where /r . pattern*
```

## Environment Management
```bash
# Check Python version
python --version

# View installed packages
uv pip list

# Check uv version
uv --version

# View environment variables
set  # Windows CMD
env  # Git Bash/PowerShell
```

## Common Aliases (add to shell profile)
```bash
# Quick commands
alias ll='ls -la'
alias gs='git status'
alias gd='git diff'
alias pytest='uv run --frozen pytest'
alias black='uv run --frozen black'
alias ruff='uv run --frozen ruff'
```