# Forex Processor Project Overview

## Project Purpose
リアルタイムForex取引データ処理とPatchTST機械学習予測システムの開発

## Tech Stack
- **Language**: Python 3.12
- **Data Processing**: 
  - Polars (primary DataFrame library - Pandas is prohibited)
  - NumPy (Float32 as standard data type for memory optimization)
  - PyArrow
- **Machine Learning**: PyTorch, scikit-learn, SHAP, Numba
- **Trading Platform**: MetaTrader5
- **Database**: InfluxDB, FastParquet
- **Web Framework**: Dash, FastAPI, Plotly
- **Async**: aiohttp, websockets, asyncio
- **Configuration**: Pydantic, pydantic-settings, TOML
- **Logging**: structlog, rich

## Project Structure
```
Forex_procrssor/
├── src/
│   ├── common/          # Common models and interfaces
│   │   ├── models.py    # Pydantic models (Tick, OHLC, Prediction, Alert)
│   │   ├── interfaces.py # Abstract base classes
│   │   └── config.py    # Configuration management
│   ├── app/            # Application layer
│   ├── data_processing/ # Data processing logic
│   ├── mt5_data_acquisition/ # MT5 integration
│   ├── patchTST_model/ # ML model implementation
│   ├── production/     # Production deployment
│   └── storage/        # Storage handlers
├── tests/
│   ├── unit/          # Unit tests
│   └── common/        # Common component tests
├── docs/              # Documentation
│   ├── context.md     # Workflow context (UTF-8)
│   └── plan.md        # Execution plan (UTF-8)
└── agents_docs/       # Agent documentation
```

## Key Design Principles
- **Float32 Constraint**: All numeric data uses np.float32 for memory optimization
- **Polars-First**: All DataFrame operations use Polars (Pandas prohibited)
- **Type Safety**: Strict type hints throughout the codebase
- **Async Pattern**: async/await for I/O operations
- **Protocol-Based Design**: typing.Protocol for flexible type system
- **ABC Enforcement**: Abstract base classes for interface contracts

## Data Models
1. **Tick**: Real-time market tick data (timestamp, symbol, bid, ask, volume)
2. **OHLC**: Candlestick data with TimeFrame enum (M1, M5, M15, H1, H4, D1, W1, MN)
3. **Prediction**: ML predictions with confidence intervals
4. **Alert**: Trading alerts with severity levels (INFO, WARNING, CRITICAL)

## Core Interfaces
1. **DataFetcher**: Abstract data acquisition
2. **DataProcessor**: Data transformation logic
3. **StorageHandler**: Data persistence operations
4. **Predictor**: ML model abstractions

## Current State
- Branch: task_1
- Main branch: main
- Active task: Task 2 - Common data models and interfaces
- Progress: Step 4/8 completed
- Test coverage: 86.29% (target: 80%+)