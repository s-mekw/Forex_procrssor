# Development Guidelines

This document contains critical information about working with this codebase.
Follow these guidelines precisely.

## Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Follow existing patterns exactly
   - Use Google style for docstring
   - DataFrames: Use Polars。pandas は原則禁止だが、外部ライブラリ境界アダプタ内での一時変換のみ許可（公開インターフェースでpandas型を露出しないこと）
     - 許可される例: NeuralForecast 等、pandas 入力を要求するライブラリに渡す直前で `pl.DataFrame`→`pd.DataFrame` に変換し、戻り値は直ちに `pl.DataFrame` に再変換する
     - 禁止事項: アプリ内での恒常的なpandas利用、pandas型の関数引数/戻り値、pandas依存の処理フロー

3. Testing Requirements
   - Framework: `uv run --frozen pytest`
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Git
   - Follow the Conventional Commits style on commit messages.

## Code Formatting and Linting

1. Ruff
   - Format: `uv run --frozen ruff format .`
   - Check: `uv run --frozen ruff check .`
   - Fix: `uv run --frozen ruff check . --fix`
2. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Ruff (Python)