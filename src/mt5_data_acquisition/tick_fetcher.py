"""
TickDataStreamerクラス - リアルタイムティックデータ取得モジュール

MetaTrader 5からリアルタイムでティックデータを取得し、
非同期ストリーミングで処理します。
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

# structlogを使用（可能な場合）
try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    # structlogが利用できない場合は標準のloggingを使用
    import logging

    logger = logging.getLogger(__name__)

from mt5_data_acquisition.mt5_client import MT5ConnectionManager


@dataclass
class StreamerConfig:
    """TickDataStreamerの設定クラス

    ストリーミング設定を管理するデータクラスです。

    Attributes:
        symbol: 通貨ペアシンボル（例: USDJPY, EURUSD）
        buffer_size: リングバッファのサイズ（デフォルト: 10000）
        spike_threshold: スパイク検出の閾値（標準偏差の倍数、デフォルト: 3.0）
        backpressure_threshold: バックプレッシャー発動閾値（バッファ使用率、デフォルト: 0.8）
        stats_window_size: 統計計算用のウィンドウサイズ（デフォルト: 1000）
    """

    symbol: str
    buffer_size: int = 10000
    spike_threshold: float = 3.0
    backpressure_threshold: float = 0.8
    stats_window_size: int = 1000


class TickDataStreamer:
    """リアルタイムティックデータストリーマー

    MT5からリアルタイムでティックデータを取得し、
    非同期ストリーミングで配信します。

    Features:
        - 非同期ティックストリーミング
        - リングバッファによるメモリ管理
        - スパイクフィルター（3σルール）
        - バックプレッシャー制御
        - 自動再購読機能
    """

    def __init__(
        self,
        symbol: str,
        buffer_size: int = 10000,
        spike_threshold: float = 3.0,
        backpressure_threshold: float = 0.8,
        stats_window_size: int = 1000,
        mt5_client: MT5ConnectionManager | None = None,
    ):
        """初期化メソッド

        Args:
            symbol: 通貨ペアシンボル（例: USDJPY, EURUSD）
            buffer_size: リングバッファのサイズ（デフォルト: 10000）
            spike_threshold: スパイク検出の閾値（標準偏差の倍数、デフォルト: 3.0）
            backpressure_threshold: バックプレッシャー発動閾値（バッファ使用率、デフォルト: 0.8）
            stats_window_size: 統計計算用のウィンドウサイズ（デフォルト: 1000）
            mt5_client: MT5接続マネージャー（オプション）

        Raises:
            ValueError: 無効なパラメータが指定された場合
        """
        # パラメータのバリデーション
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid symbol: Symbol must be a non-empty string")

        if buffer_size <= 0:
            raise ValueError("Buffer size must be positive")

        if spike_threshold <= 0:
            raise ValueError("Spike threshold must be positive")

        if backpressure_threshold <= 0 or backpressure_threshold > 1:
            raise ValueError("Backpressure threshold must be between 0 and 1")

        if stats_window_size <= 0:
            raise ValueError("Stats window size must be positive")

        # 設定の作成と保存
        self.config = StreamerConfig(
            symbol=symbol,
            buffer_size=buffer_size,
            spike_threshold=spike_threshold,
            backpressure_threshold=backpressure_threshold,
            stats_window_size=stats_window_size,
        )

        # プロパティとして直接アクセス可能にする
        self.backpressure_threshold = backpressure_threshold

        # MT5接続マネージャー
        self.connection_manager = mt5_client or MT5ConnectionManager()

        # リングバッファの初期化
        self.buffer: deque = deque(maxlen=buffer_size)

        # 統計情報の初期化
        self.stats: dict[str, Any] = {
            "mean": 0.0,
            "std": 0.0,
            "sample_count": 0,
            "spike_count": 0,
            "last_update": None,
        }

        # 状態管理フラグ
        self.is_streaming: bool = False
        self.is_subscribed: bool = False

        # ロガーの設定
        self.logger = logger

        # 内部管理用
        self._stats_buffer: deque = deque(maxlen=stats_window_size)
        self._current_task: asyncio.Task | None = None

        self.logger.info(
            "TickDataStreamer初期化",
            symbol=symbol,
            buffer_size=buffer_size,
            spike_threshold=spike_threshold,
        )

    @property
    def buffer_usage(self) -> float:
        """バッファ使用率を取得

        Returns:
            float: バッファ使用率（0.0～1.0）
        """
        if self.config.buffer_size == 0:
            return 0.0
        return len(self.buffer) / self.config.buffer_size

    @property
    def current_stats(self) -> dict[str, Any]:
        """現在の統計情報を取得

        Returns:
            Dict[str, Any]: 統計情報の辞書
                - mean: 平均値
                - std: 標準偏差
                - sample_count: サンプル数
                - spike_count: スパイク検出数
                - last_update: 最終更新時刻
        """
        return self.stats.copy()

    @property
    def is_connected(self) -> bool:
        """MT5接続状態を取得

        Returns:
            bool: 接続中の場合True
        """
        if self.connection_manager is None:
            return False
        return self.connection_manager.is_connected()

    @property
    def symbol(self) -> str:
        """シンボル名を取得

        Returns:
            str: 通貨ペアシンボル
        """
        return self.config.symbol

    @property
    def buffer_size(self) -> int:
        """バッファサイズを取得

        Returns:
            int: リングバッファのサイズ
        """
        return self.config.buffer_size

    @property
    def spike_threshold(self) -> float:
        """スパイク検出閾値を取得

        Returns:
            float: スパイク検出の標準偏差倍数
        """
        return self.config.spike_threshold

    def __repr__(self) -> str:
        """文字列表現

        Returns:
            str: オブジェクトの文字列表現
        """
        return (
            f"TickDataStreamer("
            f"symbol={self.config.symbol}, "
            f"buffer_usage={self.buffer_usage:.2%}, "
            f"is_streaming={self.is_streaming}, "
            f"is_connected={self.is_connected}"
            f")"
        )
