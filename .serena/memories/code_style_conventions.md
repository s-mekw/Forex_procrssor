# Code Style and Conventions

## Python Code Style

### General Rules
- **Python Version**: 3.12
- **Line Length**: 88 characters (Black standard)
- **Formatting**: Black formatter
- **Linting**: Ruff (E, W, F, I, B, C4, UP rules)
- **Type Checking**: MyPy with strict settings

### Type Hints
- **Required**: Type hints are mandatory for all functions and methods
- **MyPy Settings**:
  - check_untyped_defs = true
  - disallow_any_generics = true
  - disallow_incomplete_defs = true
  - disallow_untyped_defs = true
  - no_implicit_optional = true
  - strict_equality = true

### Naming Conventions
- **Classes**: PascalCase (e.g., `DataFetcher`, `StorageHandler`)
- **Functions/Methods**: snake_case (e.g., `fetch_tick_data`, `process_ohlc`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Private Methods**: Leading underscore (e.g., `_validate_data`)
- **Enums**: PascalCase for class, UPPER_SNAKE_CASE for values

### Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Each group alphabetically sorted
- isort handles automatic sorting

### Docstrings
- **Format**: Google-style docstrings
- **Required for**: All public classes, methods, and functions
- **Content**: Description, Args, Returns, Raises sections

### Data Type Standards
- **Numeric Data**: Always use np.float32 for memory efficiency
- **DataFrames**: Polars only (Pandas is prohibited)
- **Timestamps**: datetime objects with timezone awareness
- **Async Operations**: Use async/await pattern for I/O

### Validation
- **Pydantic**: For data models with automatic validation
- **Field Validators**: Custom validators for complex logic
- **Range Constraints**: Explicit min/max values where applicable

### Testing Conventions
- **Framework**: pytest
- **Coverage Target**: 80% minimum
- **Test Files**: `test_*.py` or `*_test.py`
- **Test Classes**: `Test*`
- **Test Functions**: `test_*`
- **Markers**: unit, integration, e2e, slow, mt5, influxdb, gpu

### Error Handling
- **Specific Exceptions**: Use domain-specific exceptions
- **Logging**: Use structlog for structured logging
- **Error Messages**: Clear, actionable error messages

### File Organization
- **Module Structure**: One primary class/concept per file
- **Private Helpers**: Keep in same file as main implementation
- **Constants**: Define at module level
- **Imports**: At file top, properly grouped