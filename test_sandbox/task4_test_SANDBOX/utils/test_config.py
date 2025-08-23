"""テスト用設定ファイル"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# MT5デモアカウント設定
MT5_CONFIG = {
    "login": int(os.getenv("FOREX_MT5_LOGIN", "20046505")),  # デモアカウント
    "password": os.getenv("FOREX_MT5_PASSWORD", ""),
    "server": os.getenv("FOREX_MT5_SERVER", "Axiory-Demo"),
    "timeout": int(os.getenv("FOREX_MT5_TIMEOUT", "60000")),
    "path": os.getenv("FOREX_MT5_PATH"),  # 環境変数からMT5パスを取得
}

# テスト用シンボル設定
TEST_SYMBOLS = {
    "major": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
    "minor": ["EURJPY", "GBPJPY", "AUDUSD", "NZDUSD"],
    "default": "EURJPY",
}

# ストリーミング設定
STREAMING_CONFIG = {
    "buffer_size": 1000,
    "spike_threshold_percent": 0.1,  # spike_threshold → spike_threshold_percent (価格変動率%)
    "backpressure_threshold": 0.8,
}

# TickDataStreamer用の追加設定（StreamerConfigに含まれないパラメータ）
ADDITIONAL_STREAMER_CONFIG = {
    "warmup_samples": 20,
    "max_retries": 3,
    "retry_delay": 1.0,
    "circuit_breaker_threshold": 5,
    "circuit_breaker_timeout": 30,
}

# 表示設定
DISPLAY_CONFIG = {
    "refresh_rate": 0.5,  # 秒
    "max_display_rows": 20,
    "enable_colors": True,
    "show_timestamps": True,
    "decimal_places": 5,
}

# Marimoダッシュボード設定
MARIMO_CONFIG = {
    "update_interval": 1000,  # ミリ秒
    "chart_points": 100,
    "chart_height": 400,
    "enable_animations": True,
}

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
TEST_PATH = PROJECT_ROOT / "test_sandbox" / "task4_test_SANDBOX"
LOG_PATH = TEST_PATH / "logs"

# ログディレクトリを作成
LOG_PATH.mkdir(parents=True, exist_ok=True)

def get_mt5_credentials() -> dict:
    """MT5認証情報を取得"""
    return {
        "account": MT5_CONFIG["login"],
        "password": MT5_CONFIG["password"],
        "server": MT5_CONFIG["server"],
        "timeout": MT5_CONFIG["timeout"],
        "path": MT5_CONFIG["path"],  # 環境変数からMT5パスを取得
    }

def get_mt5_config() -> dict:
    """MT5ConnectionManager用の設定を取得"""
    return {
        "account": MT5_CONFIG["login"],
        "password": MT5_CONFIG["password"],
        "server": MT5_CONFIG["server"],
        "timeout": MT5_CONFIG["timeout"],
        "path": MT5_CONFIG["path"],  # 環境変数からMT5パスを取得
    }

def get_test_symbol(category: str = "default") -> str:
    """テスト用シンボルを取得"""
    if category == "default":
        return TEST_SYMBOLS["default"]
    elif category in TEST_SYMBOLS:
        return TEST_SYMBOLS[category][0]
    return TEST_SYMBOLS["default"]

def create_tick_streamer(symbol: str, mt5_client):
    """TickDataStreamerを作成するヘルパー関数"""
    from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
    
    # TickDataStreamerが受け入れるパラメータのみを渡す
    streamer_params = {
        "symbol": symbol,
        "mt5_client": mt5_client,
        **STREAMING_CONFIG,  # buffer_size, spike_threshold_percent, backpressure_threshold
        "max_retries": ADDITIONAL_STREAMER_CONFIG["max_retries"],
        "circuit_breaker_threshold": ADDITIONAL_STREAMER_CONFIG["circuit_breaker_threshold"],
        "circuit_breaker_timeout": ADDITIONAL_STREAMER_CONFIG["circuit_breaker_timeout"],
    }
    
    return TickDataStreamer(**streamer_params)