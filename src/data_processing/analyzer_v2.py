"""
マルチタイムフレーム分析エンジン（改良版）

MultiTimeframeManagerを使用してMT5から直接マルチタイムフレームデータを取得し、
各タイムフレームのRCIを計算します。

主な改善点:
- MT5から直接各タイムフレームのOHLCを取得
- 形成中バー[0]のリアルタイム更新
- 完成バー[1]の正確性保証
- 複数タイムフレーム（M1, M5, M15, M30, H1）の同時サポート
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional
import polars as pl
import numpy as np

from src.data_processing.rci import RCICalculatorEngine
from src.data_processing.multiframe_manager import MultiTimeframeManager

logger = logging.getLogger(__name__)


class MultiTimeframeAnalysisError(Exception):
    """マルチタイムフレーム分析固有のエラー"""
    pass


class AnalysisConfigurationError(MultiTimeframeAnalysisError):
    """分析設定エラー"""
    pass


class MultiTimeframeAnalyzerV2:
    """
    改良版マルチタイムフレーム分析クラス
    
    MT5から直接各タイムフレームのデータを取得し、RCI分析を実行します。
    """
    
    # デフォルトのRCI期間設定（各タイムフレーム用）
    DEFAULT_RCI_PERIODS = {
        "M1": [9, 13, 24, 33, 48, 66, 108],      # 1分足用
        "M5": [24, 33, 48, 66, 108],             # 5分足用
        "M15": [24, 33, 48, 66],                 # 15分足用
        "M30": [24, 33, 48],                      # 30分足用
        "H1": [24, 33],                           # 1時間足用
    }
    
    def __init__(
        self,
        symbol: str,
        timeframes: List[str] = None,
        rci_periods: Dict[str, List[int]] = None,
        initial_bars: int = 200,
        max_history_bars: int = 5000,
        use_parallel: bool = True,
        max_workers: int = None
    ):
        """
        初期化
        
        Args:
            symbol: 取引シンボル
            timeframes: 分析するタイムフレーム（デフォルト: ["M1", "M5"]）
            rci_periods: 各タイムフレームのRCI期間設定
            initial_bars: 初期化時に取得するバー数
            max_history_bars: 保持する最大履歴バー数
            use_parallel: 並列処理を使用するか
            max_workers: 並列処理の最大ワーカー数
        """
        self.symbol = symbol
        self.timeframes = timeframes or ["M1", "M5"]
        self.rci_periods = rci_periods or self.DEFAULT_RCI_PERIODS
        self.initial_bars = initial_bars
        self.max_history_bars = max_history_bars
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        
        # MultiTimeframeManagerを初期化
        self.manager = MultiTimeframeManager(
            symbol=symbol,
            timeframes=self.timeframes,
            initial_bars=initial_bars,
            max_bars=max_history_bars
        )
        
        # RCI計算エンジン
        self.rci_engine = RCICalculatorEngine()
        
        # 各タイムフレームの最新RCI値を保持
        self.latest_rci: Dict[str, Dict[int, float]] = {tf: {} for tf in self.timeframes}
        
        # 統計情報
        self.stats = {
            "ticks_processed": 0,
            "bars_completed": {tf: 0 for tf in self.timeframes},
            "rci_calculations": 0,
            "last_update": None
        }
        
        logger.info(f"MultiTimeframeAnalyzerV2 initialized for {symbol} with timeframes: {self.timeframes}")
    
    def initialize(self) -> bool:
        """
        初期データを取得して分析器を初期化
        
        Returns:
            成功した場合True
        """
        # マネージャーの初期化
        if not self.manager.initialize_data():
            logger.error("Failed to initialize MultiTimeframeManager")
            return False
        
        # 各タイムフレームの初期RCIを計算
        for tf in self.timeframes:
            self._calculate_initial_rci(tf)
        
        # バー完成時のコールバックを設定
        for tf in self.timeframes:
            self.manager.set_bar_complete_callback(tf, self._on_bar_complete)
        
        logger.info("MultiTimeframeAnalyzerV2 initialization complete")
        return True
    
    def _calculate_initial_rci(self, timeframe: str):
        """
        初期RCIを計算
        
        Args:
            timeframe: タイムフレーム
        """
        df = self.manager.get_completed_bars(timeframe)
        if df is None or df.is_empty():
            logger.warning(f"No data available for {timeframe} RCI calculation")
            return
        
        periods = self.rci_periods.get(timeframe, [])
        if not periods:
            return
        
        try:
            # RCI計算
            result = self.rci_engine.calculate_multiple(
                data=df,
                periods=periods,
                column_name="close",
                mode="batch",
                add_reliability=True
            )
            
            # 最新値を保存
            for period in periods:
                rci_col = f"rci_{period}"
                if rci_col in result.columns:
                    values = result[rci_col].to_list()
                    # 最後の有効な値を取得
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        self.latest_rci[timeframe][period] = float(valid_values[-1])
                        logger.debug(f"{timeframe} RCI[{period}] initial value: {valid_values[-1]:.2f}")
        
        except Exception as e:
            logger.error(f"Failed to calculate initial RCI for {timeframe}: {e}")
    
    def _on_bar_complete(self, timeframe: str, bar_data: Dict[str, Any]):
        """
        バー完成時のコールバック
        
        Args:
            timeframe: タイムフレーム
            bar_data: 完成したバーのデータ
        """
        self.stats["bars_completed"][timeframe] += 1
        logger.debug(f"{timeframe} bar completed: {bar_data['timestamp']}")
        
        # RCIを再計算
        self._update_rci_for_timeframe(timeframe)
    
    def _update_rci_for_timeframe(self, timeframe: str):
        """
        指定タイムフレームのRCIを更新
        
        Args:
            timeframe: タイムフレーム
        """
        periods = self.rci_periods.get(timeframe, [])
        if not periods:
            return
        
        # 必要な最大期間分のデータを取得
        max_period = max(periods)
        df = self.manager.get_completed_bars(timeframe, limit=max_period + 10)
        
        if df is None or len(df) < max_period:
            logger.debug(f"Insufficient data for {timeframe} RCI update")
            return
        
        try:
            # 各期間のRCIを計算（最新値のみ）
            for period in periods:
                if len(df) >= period:
                    recent_data = df.tail(period)
                    rci_value = self._calculate_single_rci(
                        recent_data["close"].to_numpy(), period
                    )
                    self.latest_rci[timeframe][period] = rci_value
                    self.stats["rci_calculations"] += 1
        
        except Exception as e:
            logger.error(f"Failed to update RCI for {timeframe}: {e}")
    
    def _calculate_single_rci(self, prices: np.ndarray, period: int) -> float:
        """
        単一期間のRCIを計算（簡易版）
        
        Args:
            prices: 価格データ
            period: 計算期間
        
        Returns:
            RCI値（-100 to 100）
        """
        n = len(prices)
        if n != period:
            raise ValueError(f"データ長が期間と一致しません: {n} != {period}")
        
        # 価格の順位を計算
        price_ranks = np.argsort(np.argsort(prices)) + 1
        
        # 時間の順位（1, 2, ..., n）
        time_ranks = np.arange(1, n + 1)
        
        # 順位の差の二乗和
        d_squared = np.sum((price_ranks - time_ranks) ** 2)
        
        # RCI計算
        rci = (1 - 6 * d_squared / (n * (n**2 - 1))) * 100
        
        return float(rci)
    
    def analyze_tick(self, tick: Any) -> Dict[str, Any]:
        """
        ティックを分析して各タイムフレームを更新
        
        Args:
            tick: MT5のティックオブジェクト
        
        Returns:
            分析結果
        """
        # マネージャーでティックを処理
        update_results = self.manager.process_tick(tick)
        
        self.stats["ticks_processed"] += 1
        self.stats["last_update"] = datetime.now()
        
        # 分析結果を構築
        analysis_result = {
            "timestamp": datetime.fromtimestamp(tick.time) if hasattr(tick, 'time') else datetime.now(),
            "price": float(tick.bid) if hasattr(tick, 'bid') else float(tick.last),
            "timeframes": {}
        }
        
        # 各タイムフレームの結果を追加
        for tf_name, tf_result in update_results.items():
            analysis_result["timeframes"][tf_name] = {
                "new_bar": tf_result.get("new_bar", False),
                "current_bar": tf_result.get("current_bar", {}),
                "rci": self.latest_rci.get(tf_name, {})
            }
            
            # 新しいバーが完成した場合
            if tf_result.get("new_bar"):
                analysis_result["timeframes"][tf_name]["completed_bar"] = tf_result.get("completed_bar")
        
        return analysis_result
    
    def get_dataframe_for_timeframe(self, timeframe: str, limit: int = None) -> Optional[pl.DataFrame]:
        """
        指定タイムフレームのDataFrameを取得
        
        Args:
            timeframe: タイムフレーム
            limit: 取得する最大バー数
        
        Returns:
            OHLCデータのDataFrame
        """
        return self.manager.get_completed_bars(timeframe, limit)
    
    def get_current_bars(self) -> Dict[str, Dict[str, Any]]:
        """
        全タイムフレームの現在形成中のバーを取得
        
        Returns:
            タイムフレーム名をキーとした現在バーの辞書
        """
        return self.manager.get_all_current_bars()
    
    def get_latest_rci(self, timeframe: str = None) -> Dict:
        """
        最新のRCI値を取得
        
        Args:
            timeframe: 特定のタイムフレーム（Noneで全て）
        
        Returns:
            RCI値の辞書
        """
        if timeframe:
            return self.latest_rci.get(timeframe, {})
        return self.latest_rci
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        分析メトリクスを取得
        
        Returns:
            統計情報とメトリクス
        """
        manager_metrics = self.manager.get_metrics()
        
        return {
            **self.stats,
            "manager": manager_metrics,
            "rci_values": {
                tf: {f"rci_{p}": v for p, v in rci.items()}
                for tf, rci in self.latest_rci.items()
            }
        }
    
    def is_ready(self) -> bool:
        """
        分析器が準備完了かチェック
        
        Returns:
            準備完了の場合True
        """
        return self.manager.is_initialized