"""
RCI（Rank Correlation Index）計算エンジン

高速な差分更新アルゴリズムによるRCI計算を提供します。
Float32精度による メモリ効率化と、O(n)の計算複雑度を実現。
"""

from collections import deque
from typing import Optional, List, Dict, Any, Literal, Tuple, Union
import numpy as np
import polars as pl
import logging
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import os

logger = logging.getLogger(__name__)


class RCICalculationError(Exception):
    """RCI計算固有のエラー"""
    pass


class InvalidPeriodError(RCICalculationError):
    """無効な期間パラメータエラー"""
    pass


class InsufficientDataError(RCICalculationError):
    """データ不足エラー"""
    pass

class DifferentialRCICalculator:
    """単一期間のストリーミングRCI計算クラス
    
    差分更新アルゴリズムによる高速RCI計算を実装。
    dequeによるスライディングウィンドウとFloat32精度により、
    メモリ効率的かつ高速な計算を実現します。
    
    Attributes:
        period (int): RCI計算期間（3〜200）
        prices (deque): 価格データのスライディングウィンドウ
        time_ranks (np.ndarray): 時間順位（事前計算）
        denominator (float): RCI式の分母（事前計算）
    
    Methods:
        add(price: float) -> Optional[float]: 
            新価格を追加してRCIを計算
        _optimized_ranking(prices: np.ndarray) -> np.ndarray:
            Float32精度での最適化ランキング計算
    """
    
    MIN_PERIOD = 3
    MAX_PERIOD = 200
    
    def __init__(self, period: int):
        """Initialize the calculator for a specific period.
        
        Args:
            period: RCI計算期間（MIN_PERIOD以上、MAX_PERIOD以下）
            
        Raises:
            ValueError: 期間が有効範囲外の場合
        """
        if period < self.MIN_PERIOD:
            raise ValueError(
                f"Period must be at least {self.MIN_PERIOD}, got {period}"
            )
        if period > self.MAX_PERIOD:
            raise ValueError(
                f"Period must be at most {self.MAX_PERIOD}, got {period}"
            )
            
        self.period = period
        self.prices = deque(maxlen=period)
        
        # 時間順位の事前計算（0-based、Float32精度）
        self.time_ranks = np.arange(0, period, dtype=np.float32)
        
        # RCI分母の事前計算（Float32精度）
        self.denominator = np.float32(period * (period**2 - 1))
        
        # 統計情報
        self._calculation_count = 0
        self._last_rci = None
        
        # ランキングキャッシュ（頻出パターン用）
        self._ranking_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 100  # キャッシュサイズの上限
        
        logger.debug(f"Initialized DifferentialRCICalculator with period={period}")
        
    def add(self, price: float) -> Optional[float]:
        """新しい価格を追加してRCIを計算
        
        Args:
            price: 新しい価格値
            
        Returns:
            ウィンドウがフルの場合はRCI値、そうでない場合はNone
            
        Note:
            RCI値は-100から+100の範囲に正規化されます。
            +100に近いほど上昇トレンド、-100に近いほど下降トレンドを示します。
        """
        # NaN/Inf値のチェック
        if not np.isfinite(price):
            logger.warning(f"Invalid price value: {price}")
            return None
            
        self.prices.append(price)
        
        # ウィンドウがフルでない場合はNoneを返す
        if len(self.prices) < self.period:
            return None
            
        # Float32精度での価格配列作成（メモリ効率のため in-place 変換）
        prices_array = np.array(self.prices, dtype=np.float32)
        
        # 適応的なランキング手法の選択
        # キャッシュを優先的に使用（メモリ効率と高速化）
        if self._calculation_count > 100:  # ウォームアップ後
            price_ranks = self._cached_ranking(prices_array)
        elif self._check_for_ties(prices_array):
            price_ranks = self._optimized_ranking(prices_array)
        else:
            price_ranks = self._fast_ranking_no_ties(prices_array)
        
        # RCI計算（Float32精度）
        # RCI = (1 - 6 * Σd²/(n(n²-1))) * 100
        # d = 時間順位 - 価格順位
        d_squared_sum = np.sum(
            (self.time_ranks - price_ranks) ** 2, 
            dtype=np.float32
        )
        rci = (1.0 - (6.0 * d_squared_sum) / self.denominator) * 100.0
        
        # 統計情報の更新
        self._calculation_count += 1
        self._last_rci = float(rci)
        
        return self._last_rci
    
    def _optimized_ranking(self, prices: np.ndarray) -> np.ndarray:
        """Float32精度での最適化ランキング計算
        
        scipy.stats.rankdataを使用して正確なランキングを計算。
        同値（タイ）の場合は平均順位を割り当てます。
        scipyが利用できない場合はNumPyベースの実装にフォールバック。
        
        Args:
            prices: ランク付けする価格配列（Float32）
            
        Returns:
            0ベースのランク配列（Float32）
            最小価格が0、最大価格がperiod-1
        """
        try:
            # scipy.stats.rankdataによる正確なランキング
            from scipy.stats import rankdata
            
            # rankdataは1ベースのランクを返すので、0ベースに変換
            # method='average'により、同値には平均順位を割り当て
            ranks = rankdata(prices, method='average') - 1
            return ranks.astype(np.float32)
            
        except ImportError:
            # scipyが利用できない場合のフォールバック実装
            logger.debug("scipy not available, using numpy-based ranking")
            
            # NumPyベースの同値対応ランキング実装を使用
            return self._numpy_ranking_with_ties(prices)
    
    def _numpy_ranking_with_ties(self, prices: np.ndarray) -> np.ndarray:
        """NumPyベースの同値対応ランキング実装（メモリ最適化版）
        
        scipyが利用できない場合の高精度フォールバック実装。
        同値（タイ）に対して平均順位を割り当てます。
        in-place操作を活用してメモリ使用量を最小化。
        
        Args:
            prices: ランク付けする価格配列（Float32）
            
        Returns:
            0ベースのランク配列（Float32）、同値には平均順位
        """
        n = len(prices)
        # ソート順のインデックスを取得
        order = np.argsort(prices)
        
        # ソートされた価格（ビューを使用してメモリ節約）
        sorted_prices = prices[order]
        
        # 基本ランク（0ベース）
        ranks = np.arange(n, dtype=np.float32)
        
        # 同値グループの処理（in-place更新）
        i = 0
        while i < n:
            # 同じ値のグループを見つける
            j = i + 1
            while j < n and np.abs(sorted_prices[j] - sorted_prices[i]) < 1e-9:
                j += 1
            
            # 同値グループがある場合、平均順位を割り当て（in-place）
            if j > i + 1:
                ranks[i:j] = np.mean(ranks[i:j])
            
            i = j
        
        # 元の順序に戻す（メモリ効率的な方法）
        result = np.empty(n, dtype=np.float32)
        result[order] = ranks
        return result
    
    def _fast_ranking_no_ties(self, prices: np.ndarray) -> np.ndarray:
        """同値がない場合の高速ランキング
        
        FX価格では同値は稀なため、このケースに最適化。
        同値チェックを省略して高速化。
        
        Args:
            prices: ランク付けする価格配列（Float32）
            
        Returns:
            0ベースのランク配列（Float32）
        """
        # argsortを2回使用する高速実装
        order = np.argsort(prices)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(prices), dtype=np.float32)
        return ranks
    
    def _check_for_ties(self, prices: np.ndarray) -> bool:
        """価格配列に同値があるかチェック
        
        Args:
            prices: チェックする価格配列
            
        Returns:
            同値が存在する場合True
        """
        # ソートして隣接要素を比較
        sorted_prices = np.sort(prices)
        diffs = np.diff(sorted_prices)
        # 浮動小数点の精度を考慮
        return np.any(np.abs(diffs) < 1e-9)
    
    def _get_price_hash(self, prices: np.ndarray) -> str:
        """価格配列のハッシュ値を計算
        
        Args:
            prices: ハッシュ化する価格配列
            
        Returns:
            価格配列のハッシュ値（文字列）
        """
        # Float32精度で丸めてからハッシュ化
        rounded = np.round(prices, decimals=5).tobytes()
        return hashlib.md5(rounded).hexdigest()
    
    def _cached_ranking(self, prices: np.ndarray) -> np.ndarray:
        """キャッシュ付きランキング計算
        
        頻出パターンの結果をキャッシュして再利用。
        キャッシュがない場合は通常のランキングを実行。
        
        Args:
            prices: ランク付けする価格配列（Float32）
            
        Returns:
            0ベースのランク配列（Float32）
        """
        # 価格パターンのハッシュを生成
        price_hash = self._get_price_hash(prices)
        
        # キャッシュをチェック
        if price_hash in self._ranking_cache:
            self._cache_hits += 1
            return self._ranking_cache[price_hash].copy()
        
        # キャッシュミス - 通常のランキング実行
        self._cache_misses += 1
        
        # 同値の有無でアルゴリズムを選択
        if self._check_for_ties(prices):
            ranks = self._optimized_ranking(prices)
        else:
            ranks = self._fast_ranking_no_ties(prices)
        
        # キャッシュに保存（サイズ制限あり）
        if len(self._ranking_cache) < self._max_cache_size:
            self._ranking_cache[price_hash] = ranks.copy()
        elif len(self._ranking_cache) >= self._max_cache_size:
            # 古いエントリを削除（FIFO）
            first_key = next(iter(self._ranking_cache))
            del self._ranking_cache[first_key]
            self._ranking_cache[price_hash] = ranks.copy()
        
        return ranks
    
    def reset(self):
        """計算器の状態をリセット"""
        self.prices.clear()
        self._calculation_count = 0
        self._last_rci = None
        self._ranking_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug(f"Reset DifferentialRCICalculator (period={self.period})")
    
    @property
    def is_ready(self) -> bool:
        """RCI計算の準備ができているかを確認
        
        Returns:
            ウィンドウがフルで計算可能な場合True
        """
        return len(self.prices) >= self.period
    
    @property
    def calculation_count(self) -> int:
        """これまでのRCI計算回数を取得"""
        return self._calculation_count
    
    @property
    def last_rci(self) -> Optional[float]:
        """最後に計算されたRCI値を取得"""
        return self._last_rci
    
    def get_buffer_state(self) -> Dict[str, Any]:
        """内部バッファの状態を取得（デバッグ用）
        
        Returns:
            バッファ状態を含む辞書
        """
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0 else 0
        )
        
        return {
            'period': self.period,
            'buffer_size': len(self.prices),
            'is_ready': self.is_ready,
            'calculation_count': self._calculation_count,
            'last_rci': self._last_rci,
            'current_prices': list(self.prices) if self.prices else [],
            'cache_stats': {
                'cache_size': len(self._ranking_cache),
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'max_cache_size': self._max_cache_size
            }
        }


class RCICalculatorEngine:
    """複数期間対応の汎用RCI計算エンジン
    
    複数期間のRCI計算を並列または逐次的に実行し、
    バッチ処理とストリーミング処理の両方をサポートします。
    DifferentialRCICalculatorを内部で使用し、効率的な計算を実現。
    
    Attributes:
        DEFAULT_PERIODS: デフォルトのRCI計算期間リスト
        MIN_PERIOD: 最小計算期間
        MAX_PERIOD: 最大計算期間
        calculators: 各期間用のDifferentialRCICalculatorインスタンス
    
    Methods:
        calculate_multiple: 複数期間のRCI計算（メインインターフェース）
        validate_periods: 期間パラメータのバリデーション
    """
    
    DEFAULT_PERIODS = [9, 13, 24, 33, 48, 66, 108]
    MIN_PERIOD = 3
    MAX_PERIOD = 200
    DEFAULT_CHUNK_SIZE = 100_000  # デフォルトチャンクサイズ
    AUTO_MODE_THRESHOLD = 50_000  # 自動モード切り替え閾値
    MAX_PARALLEL_WORKERS = 4  # 並列処理の最大ワーカー数
    
    def __init__(self, chunk_size: Optional[int] = None):
        """エンジンを初期化
        
        Args:
            chunk_size: バッチ処理時のチャンクサイズ（デフォルト: 100,000行）
        """
        self.calculators: Dict[int, DifferentialRCICalculator] = {}
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self._statistics = {
            'batch_calculations': 0,
            'streaming_calculations': 0,
            'total_periods_calculated': 0,
            'validation_errors': 0,
            'chunked_calculations': 0,
            'parallel_calculations': 0,
            'auto_mode_selections': 0
        }
        # メモリ監視用
        self._process = psutil.Process(os.getpid())
        logger.info(f"RCICalculatorEngine initialized with chunk_size={self.chunk_size}")
    
    def validate_periods(self, periods: List[int]) -> List[int]:
        """期間パラメータのバリデーション
        
        Args:
            periods: 検証する期間リスト
            
        Returns:
            検証済みの期間リスト（重複除去・ソート済み）
            
        Raises:
            InvalidPeriodError: 無効な期間が含まれる場合
        """
        if not periods:
            raise InvalidPeriodError("Periods list cannot be empty")
        
        # 期間の型チェックと範囲チェック
        validated_periods = []
        invalid_periods = []
        
        for period in periods:
            try:
                period_int = int(period)
                if self.MIN_PERIOD <= period_int <= self.MAX_PERIOD:
                    validated_periods.append(period_int)
                else:
                    invalid_periods.append(period)
            except (TypeError, ValueError):
                invalid_periods.append(period)
        
        if invalid_periods:
            self._statistics['validation_errors'] += 1
            raise InvalidPeriodError(
                f"Invalid periods: {invalid_periods}. "
                f"Periods must be integers between {self.MIN_PERIOD} and {self.MAX_PERIOD}"
            )
        
        # 重複除去とソート
        unique_periods = sorted(set(validated_periods))
        
        if len(unique_periods) < len(validated_periods):
            logger.warning(
                f"Duplicate periods removed. Original: {validated_periods}, "
                f"Unique: {unique_periods}"
            )
        
        return unique_periods
    
    def calculate_multiple(
        self,
        data: pl.DataFrame,
        periods: Optional[List[int]] = None,
        column_name: str = 'close',
        mode: Literal['batch', 'streaming', 'auto'] = 'auto',
        add_reliability: bool = True
    ) -> pl.DataFrame:
        """複数期間のRCI計算（メインインターフェース）
        
        Args:
            data: 入力データフレーム
            periods: 計算する期間リスト（Noneの場合はDEFAULT_PERIODS使用）
            column_name: 価格データのカラム名
            mode: 計算モード（'batch' または 'streaming'）
            add_reliability: 信頼性フラグを追加するか
            
        Returns:
            RCI値と信頼性フラグが追加されたDataFrame
            
        Raises:
            InvalidPeriodError: 無効な期間が指定された場合
            InsufficientDataError: データが不足している場合
            ValueError: 指定されたカラムが存在しない場合
        """
        # カラムの存在チェック
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        # 期間リストの準備と検証
        if periods is None:
            periods = self.DEFAULT_PERIODS
        periods = self.validate_periods(periods)
        
        # データ長のチェック
        data_length = len(data)
        max_period = max(periods)
        if data_length < max_period:
            raise InsufficientDataError(
                f"Insufficient data: {data_length} rows, "
                f"but maximum period {max_period} requires at least {max_period} rows"
            )
        
        # モードの自動選択
        if mode == 'auto':
            mode = self._determine_best_mode(data_length, len(periods))
            self._statistics['auto_mode_selections'] += 1
            logger.info(f"Auto mode selected: {mode} for {data_length} rows")
        
        # モードに応じた処理
        if mode == 'batch':
            # 大規模データの場合はチャンク処理を使用
            if data_length > self.chunk_size:
                result = self._process_batch_chunked(data, periods, column_name)
                self._statistics['chunked_calculations'] += 1
            else:
                result = self._process_batch(data, periods, column_name)
                self._statistics['batch_calculations'] += 1
        elif mode == 'streaming':
            result = self._process_streaming(data, periods, column_name)
            self._statistics['streaming_calculations'] += 1
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'batch', 'streaming', or 'auto'")
        
        # 信頼性フラグの追加
        if add_reliability:
            result = self._add_reliability_flags(result, periods, data_length)
        
        self._statistics['total_periods_calculated'] += len(periods)
        
        return result
    
    def _process_batch(
        self,
        data: pl.DataFrame,
        periods: List[int],
        column_name: str
    ) -> pl.DataFrame:
        """バッチモードでのRCI計算（Polars最適化）
        
        Args:
            data: 入力データフレーム
            periods: 計算する期間リスト
            column_name: 価格データのカラム名
            
        Returns:
            RCI値が追加されたDataFrame
        """
        result = data.clone()
        
        for period in periods:
            logger.debug(f"Calculating RCI for period {period} in batch mode")
            
            # Polars式を使用したRCI計算
            rci_values = self._calculate_rci_batch(
                data[column_name].to_numpy(),
                period
            )
            
            # 結果をDataFrameに追加
            rci_column_name = f"rci_{period}"
            result = result.with_columns(
                pl.Series(name=rci_column_name, values=rci_values, dtype=pl.Float32)
            )
        
        return result
    
    def _process_streaming(
        self,
        data: pl.DataFrame,
        periods: List[int],
        column_name: str
    ) -> pl.DataFrame:
        """ストリーミングモードでのRCI計算
        
        Args:
            data: 入力データフレーム
            periods: 計算する期間リスト
            column_name: 価格データのカラム名
            
        Returns:
            RCI値が追加されたDataFrame
        """
        result = data.clone()
        prices = data[column_name].to_numpy()
        
        # 各期間用の計算器を準備
        for period in periods:
            if period not in self.calculators:
                self.calculators[period] = DifferentialRCICalculator(period)
            else:
                self.calculators[period].reset()
        
        # ストリーミング計算
        rci_results = {period: [] for period in periods}
        
        for price in prices:
            for period in periods:
                rci_value = self.calculators[period].add(float(price))
                rci_results[period].append(rci_value)
        
        # 結果をDataFrameに追加
        for period in periods:
            rci_column_name = f"rci_{period}"
            result = result.with_columns(
                pl.Series(
                    name=rci_column_name,
                    values=rci_results[period],
                    dtype=pl.Float32
                )
            )
            logger.debug(f"Added RCI column for period {period} in streaming mode")
        
        return result
    
    def _calculate_rci_batch(
        self,
        prices: np.ndarray,
        period: int
    ) -> List[Optional[float]]:
        """バッチ処理用のRCI計算
        
        Args:
            prices: 価格配列
            period: RCI計算期間
            
        Returns:
            RCI値のリスト（初期値はNone）
        """
        n = len(prices)
        rci_values: List[Optional[float]] = [None] * n
        
        # Float32精度での計算
        prices_float32 = prices.astype(np.float32)
        
        # 時間順位の事前計算
        time_ranks = np.arange(period, dtype=np.float32)
        denominator = np.float32(period * (period**2 - 1))
        
        # スライディングウィンドウでRCI計算
        for i in range(period - 1, n):
            window_prices = prices_float32[i - period + 1:i + 1]
            
            # ランキング計算（scipy使用を試みる）
            try:
                from scipy.stats import rankdata
                price_ranks = rankdata(window_prices, method='average') - 1
            except ImportError:
                # NumPyベースの実装
                order = np.argsort(window_prices)
                price_ranks = np.empty_like(order, dtype=np.float32)
                price_ranks[order] = np.arange(period, dtype=np.float32)
            
            # RCI計算
            d_squared_sum = np.sum((time_ranks - price_ranks) ** 2)
            rci = (1.0 - (6.0 * d_squared_sum) / denominator) * 100.0
            rci_values[i] = float(rci)
        
        return rci_values
    
    def _add_reliability_flags(
        self,
        data: pl.DataFrame,
        periods: List[int],
        data_length: int
    ) -> pl.DataFrame:
        """信頼性フラグを追加
        
        Args:
            data: RCI値が計算されたDataFrame
            periods: 計算された期間リスト
            data_length: データの長さ
            
        Returns:
            信頼性フラグが追加されたDataFrame
        """
        result = data
        
        for period in periods:
            rci_column = f"rci_{period}"
            reliability_column = f"rci_{period}_reliable"
            
            # 信頼性の判定
            # - 最初のperiod-1個はNoneなので信頼性なし
            # - それ以降は信頼性あり
            reliability_values = [False] * (period - 1) + [True] * (data_length - period + 1)
            
            # 長さ調整（念のため）
            if len(reliability_values) < data_length:
                reliability_values.extend([False] * (data_length - len(reliability_values)))
            elif len(reliability_values) > data_length:
                reliability_values = reliability_values[:data_length]
            
            result = result.with_columns(
                pl.Series(name=reliability_column, values=reliability_values, dtype=pl.Boolean)
            )
            
            logger.debug(f"Added reliability flag for RCI period {period}")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """エンジンの統計情報を取得
        
        Returns:
            統計情報を含む辞書
        """
        stats = self._statistics.copy()
        stats['active_calculators'] = list(self.calculators.keys())
        stats['calculator_details'] = {}
        
        for period, calc in self.calculators.items():
            stats['calculator_details'][period] = {
                'is_ready': calc.is_ready,
                'calculation_count': calc.calculation_count,
                'last_rci': calc.last_rci
            }
        
        return stats
    
    def reset(self):
        """エンジンの状態をリセット"""
        for calc in self.calculators.values():
            calc.reset()
        self.calculators.clear()
        self._statistics = {
            'batch_calculations': 0,
            'streaming_calculations': 0,
            'total_periods_calculated': 0,
            'validation_errors': 0,
            'chunked_calculations': 0,
            'parallel_calculations': 0,
            'auto_mode_selections': 0
        }
        logger.info("RCICalculatorEngine reset")

    def _determine_best_mode(self, data_length: int, num_periods: int) -> str:
        """データサイズと期間数に基づいて最適なモードを決定
        
        Args:
            data_length: データの行数
            num_periods: 計算する期間の数
            
        Returns:
            'batch' または 'streaming'
        """
        # メモリ使用量をチェック
        memory_usage = self._monitor_memory_usage()
        
        # 決定ロジック
        if data_length < self.AUTO_MODE_THRESHOLD:
            # 小規模データはストリーミングが効率的
            return 'streaming'
        elif memory_usage > 70:
            # メモリ使用率が高い場合はチャンク処理を伴うバッチ
            return 'batch'
        elif num_periods > 5:
            # 多数の期間を計算する場合はバッチ（並列処理可能）
            return 'batch'
        else:
            # デフォルトはバッチ
            return 'batch'
    
    def _process_batch_chunked(
        self,
        data: pl.DataFrame,
        periods: List[int],
        column_name: str,
        chunk_size: Optional[int] = None
    ) -> pl.DataFrame:
        """チャンク処理による大規模データ対応バッチ処理
        
        Args:
            data: 入力データフレーム
            periods: 計算する期間リスト
            column_name: 価格データのカラム名
            chunk_size: カスタムチャンクサイズ
            
        Returns:
            RCI値が追加されたDataFrame
        """
        chunk_size = chunk_size or self.chunk_size
        n_rows = len(data)
        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        
        logger.info(
            f"Processing {n_rows:,} rows in {n_chunks} chunks of {chunk_size:,} rows"
        )
        
        # 結果を格納する辞書
        all_rci_values = {period: [] for period in periods}
        
        # チャンクごとに処理
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_rows)
            
            # 各期間のウィンドウサイズを考慮してオーバーラップを追加
            max_period = max(periods)
            overlap_start = max(0, start_idx - max_period + 1)
            
            # チャンクデータを取得（オーバーラップ含む）
            chunk_data = data[overlap_start:end_idx]
            
            # チャンクでRCI計算（並列処理）
            chunk_rci = self._calculate_rci_parallel(
                chunk_data[column_name].to_numpy(),
                periods
            )
            
            # オーバーラップ部分を除いて結果を保存
            skip_count = start_idx - overlap_start if chunk_idx > 0 else 0
            for period in periods:
                values = chunk_rci[period][skip_count:]
                all_rci_values[period].extend(values)
            
            # メモリ使用量をチェックして必要に応じて調整
            if chunk_idx % 5 == 0:  # 5チャンクごとにチェック
                memory_usage = self._monitor_memory_usage()
                if memory_usage > 80:
                    chunk_size = self._adjust_chunk_size_dynamically(memory_usage)
                    logger.warning(f"Adjusted chunk size to {chunk_size:,} due to memory pressure")
            
            # 進捗ログ
            if chunk_idx % 10 == 0 or chunk_idx == n_chunks - 1:
                logger.debug(
                    f"Processed chunk {chunk_idx + 1}/{n_chunks} "
                    f"({(chunk_idx + 1) * 100 // n_chunks}%)"
                )
        
        # 結果をDataFrameに追加
        result = data.clone()
        for period in periods:
            rci_column_name = f"rci_{period}"
            result = result.with_columns(
                pl.Series(name=rci_column_name, values=all_rci_values[period], dtype=pl.Float32)
            )
        
        return result
    
    def _calculate_rci_parallel(
        self,
        prices: np.ndarray,
        periods: List[int]
    ) -> Dict[int, List[Optional[float]]]:
        """複数期間のRCIを並列計算
        
        Args:
            prices: 価格配列
            periods: 計算する期間リスト
            
        Returns:
            各期間のRCI値リスト
        """
        results = {}
        
        # 並列処理を使用
        with ThreadPoolExecutor(max_workers=min(len(periods), self.MAX_PARALLEL_WORKERS)) as executor:
            # 各期間の計算を並列実行
            future_to_period = {
                executor.submit(self._calculate_rci_batch, prices, period): period
                for period in periods
            }
            
            # 結果を収集
            for future in as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    rci_values = future.result()
                    results[period] = rci_values
                    logger.debug(f"Completed RCI calculation for period {period}")
                except Exception as e:
                    logger.error(f"Error calculating RCI for period {period}: {e}")
                    results[period] = [None] * len(prices)
        
        self._statistics['parallel_calculations'] += 1
        return results
    
    def add_incremental(
        self,
        price: float,
        periods: Optional[List[int]] = None
    ) -> Dict[int, Optional[float]]:
        """単一価格の増分更新（リアルタイム用）
        
        Args:
            price: 新しい価格値
            periods: 計算する期間リスト（Noneの場合はDEFAULT_PERIODS）
            
        Returns:
            各期間の最新RCI値
        """
        if periods is None:
            periods = self.DEFAULT_PERIODS
        periods = self.validate_periods(periods)
        
        results = {}
        
        for period in periods:
            # 計算器が存在しない場合は作成
            if period not in self.calculators:
                self.calculators[period] = DifferentialRCICalculator(period)
            
            # 増分更新
            rci_value = self.calculators[period].add(price)
            results[period] = rci_value
        
        return results
    
    def get_latest_rci_values(
        self,
        periods: Optional[List[int]] = None
    ) -> Dict[int, Optional[float]]:
        """最新のRCI値を取得（リアルタイム監視用）
        
        Args:
            periods: 取得する期間リスト（Noneの場合はアクティブな全期間）
            
        Returns:
            各期間の最新RCI値
        """
        if periods is None:
            periods = list(self.calculators.keys())
        
        results = {}
        for period in periods:
            if period in self.calculators:
                results[period] = self.calculators[period].last_rci
            else:
                results[period] = None
        
        return results
    
    def _monitor_memory_usage(self) -> float:
        """メモリ使用量を監視
        
        Returns:
            メモリ使用率（パーセント）
        """
        try:
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
            
            # 詳細ログ（デバッグ用）
            logger.debug(
                f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB "
                f"({memory_percent:.1f}%)"
            )
            
            return memory_percent
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def _adjust_chunk_size_dynamically(self, current_memory: float) -> int:
        """メモリ使用量に基づいてチャンクサイズを動的調整
        
        Args:
            current_memory: 現在のメモリ使用率（パーセント）
            
        Returns:
            調整後のチャンクサイズ
        """
        if current_memory > 80:
            # メモリ使用率が80%以上：チャンクサイズを半分に
            new_size = max(1000, self.chunk_size // 2)
        elif current_memory > 60:
            # メモリ使用率が60-80%：チャンクサイズを75%に
            new_size = max(1000, int(self.chunk_size * 0.75))
        elif current_memory < 30:
            # メモリ使用率が30%未満：チャンクサイズを1.5倍に
            new_size = min(1_000_000, int(self.chunk_size * 1.5))
        else:
            # そのまま
            new_size = self.chunk_size
        
        if new_size != self.chunk_size:
            logger.info(
                f"Adjusting chunk size: {self.chunk_size:,} -> {new_size:,} "
                f"(Memory usage: {current_memory:.1f}%)"
            )
            self.chunk_size = new_size
        
        return new_size


class RCIProcessor:
    """Polars Expression統合用ラッパークラス
    
    RCI計算をPolars Expressionとして提供し、
    DataFrameやLazyFrameとシームレスに統合できるようにします。
    
    Attributes:
        engine (RCICalculatorEngine): 内部で使用するRCI計算エンジン
        use_float32 (bool): Float32精度を使用するかどうか
        
    Methods:
        create_rci_expression: Polars ExpressionとしてRCI計算を定義
        apply_to_dataframe: DataFrameに直接RCI計算を適用
        apply_to_lazyframe: LazyFrameにRCI計算を適用（遅延評価）
        apply_grouped: グループごとのRCI計算（最適化済み）
    """
    
    def __init__(self, use_float32: bool = True):
        """RCIProcessorの初期化
        
        Args:
            use_float32: Float32精度を使用するかどうか（デフォルト: True）
        """
        self.engine = RCICalculatorEngine()
        self.use_float32 = use_float32
        self._expr_cache: Dict[Tuple[str, int], pl.Expr] = {}
        
    def create_rci_expression(
        self,
        column: str,
        period: int
    ) -> pl.Expr:
        """Polars ExpressionとしてRCI計算を定義
        
        Args:
            column: 価格データのカラム名
            period: RCI計算期間
            
        Returns:
            RCI計算を表すPolars Expression
            
        Raises:
            InvalidPeriodError: 無効な期間が指定された場合
        """
        # パラメータ検証
        if period < RCICalculatorEngine.MIN_PERIOD or period > RCICalculatorEngine.MAX_PERIOD:
            raise InvalidPeriodError(
                f"Period must be between {RCICalculatorEngine.MIN_PERIOD} and "
                f"{RCICalculatorEngine.MAX_PERIOD}, got {period}"
            )
        
        # キャッシュチェック
        cache_key = (column, period)
        if cache_key in self._expr_cache:
            return self._expr_cache[cache_key]
        
        # RCI計算Expression作成
        # 時間順位の事前計算
        time_ranks = np.arange(period, dtype=np.float32)
        denominator = np.float32(period * (period**2 - 1))
        
        def _calculate_rci_values(values) -> float:
            """ウィンドウ内のRCI値を計算"""
            # Polars SeriesをNumPy配列に変換
            if hasattr(values, 'to_numpy'):
                values = values.to_numpy()
            
            if len(values) < period:
                return None
            
            # Float32精度での計算
            values_f32 = values.astype(np.float32)
            
            # ランキング計算
            try:
                from scipy.stats import rankdata
                price_ranks = rankdata(values_f32, method='average') - 1
            except ImportError:
                # NumPyフォールバック
                order = np.argsort(values_f32)
                price_ranks = np.empty_like(order, dtype=np.float32)
                price_ranks[order] = np.arange(period, dtype=np.float32)
            
            # RCI計算
            d_squared_sum = np.sum((time_ranks - price_ranks) ** 2)
            rci = (1.0 - (6.0 * d_squared_sum) / denominator) * 100.0
            
            return float(rci)
        
        # Polars Expressionの構築
        expr = (
            pl.col(column)
            .cast(pl.Float32 if self.use_float32 else pl.Float64)
            .rolling_map(
                function=_calculate_rci_values,
                window_size=period,
                min_periods=period
            )
            .alias(f"rci_{period}")
        )
        
        # キャッシュ保存
        self._expr_cache[cache_key] = expr
        
        return expr
    
    def apply_to_dataframe(
        self,
        df: pl.DataFrame,
        periods: Optional[List[int]] = None,
        column_name: str = 'close',
        add_reliability: bool = True
    ) -> pl.DataFrame:
        """DataFrameに直接RCI計算を適用
        
        Args:
            df: 入力DataFrame
            periods: RCI計算期間のリスト（Noneの場合はデフォルト期間）
            column_name: 価格データのカラム名
            add_reliability: 信頼性フラグを追加するかどうか
            
        Returns:
            RCI列が追加されたDataFrame
        """
        if periods is None:
            periods = RCICalculatorEngine.DEFAULT_PERIODS
        
        # 期間の検証
        self.engine.validate_periods(periods)
        
        # 各期間のRCI Expression作成
        expressions = []
        for period in periods:
            expr = self.create_rci_expression(column_name, period)
            expressions.append(expr)
        
        # DataFrameに適用
        result = df.with_columns(expressions)
        
        # 信頼性フラグの追加（別ステップで）
        if add_reliability:
            reliability_expressions = []
            for period in periods:
                reliability_expr = (
                    pl.col(f"rci_{period}")
                    .is_not_null()
                    .alias(f"rci_{period}_reliable")
                )
                reliability_expressions.append(reliability_expr)
            result = result.with_columns(reliability_expressions)
        
        return result
    
    def apply_to_lazyframe(
        self,
        lf: pl.LazyFrame,
        periods: Optional[List[int]] = None,
        column_name: str = 'close',
        add_reliability: bool = True
    ) -> pl.LazyFrame:
        """LazyFrameにRCI計算を適用（遅延評価）
        
        Args:
            lf: 入力LazyFrame
            periods: RCI計算期間のリスト（Noneの場合はデフォルト期間）
            column_name: 価格データのカラム名
            add_reliability: 信頼性フラグを追加するかどうか
            
        Returns:
            RCI列が追加されたLazyFrame（遅延評価）
        """
        if periods is None:
            periods = RCICalculatorEngine.DEFAULT_PERIODS
        
        # 期間の検証
        self.engine.validate_periods(periods)
        
        # 各期間のRCI Expression作成
        expressions = []
        for period in periods:
            expr = self.create_rci_expression(column_name, period)
            expressions.append(expr)
        
        # LazyFrameに適用（遅延評価）
        result = lf.with_columns(expressions)
        
        # 信頼性フラグの追加（別ステップで）
        if add_reliability:
            reliability_expressions = []
            for period in periods:
                reliability_expr = (
                    pl.col(f"rci_{period}")
                    .is_not_null()
                    .alias(f"rci_{period}_reliable")
                )
                reliability_expressions.append(reliability_expr)
            result = result.with_columns(reliability_expressions)
        
        return result
    
    def apply_grouped(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        group_by: List[str],
        periods: Optional[List[int]] = None,
        column_name: str = 'close',
        add_reliability: bool = True,
        parallel: bool = True
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """グループごとのRCI計算（最適化済み）
        
        Args:
            df: 入力DataFrame/LazyFrame
            group_by: グループ化するカラムのリスト
            periods: RCI計算期間のリスト
            column_name: 価格データのカラム名
            add_reliability: 信頼性フラグを追加するかどうか
            parallel: 並列処理を使用するかどうか
            
        Returns:
            グループごとにRCIが計算されたDataFrame/LazyFrame
        """
        if periods is None:
            periods = RCICalculatorEngine.DEFAULT_PERIODS
        
        # 期間の検証
        self.engine.validate_periods(periods)
        
        # グループごとのRCI計算用Expression作成
        expressions = []
        
        for period in periods:
            # グループ内でのRCI計算
            def _group_rci_calc(group_values: np.ndarray) -> List[Optional[float]]:
                """グループ内のRCI計算"""
                calculator = DifferentialRCICalculator(period)
                results = []
                
                for value in group_values:
                    rci = calculator.add(float(value))
                    results.append(rci)
                
                return results
            
            # over()句を使用したグループ計算Expression
            dtype = pl.Float32 if self.use_float32 else pl.Float64
            expr = (
                pl.col(column_name)
                .cast(dtype)
                .map_batches(
                    lambda s: pl.Series(
                        _group_rci_calc(s.to_numpy()),
                        dtype=dtype
                    ),
                    return_dtype=dtype
                )
                .over(group_by)
                .alias(f"rci_{period}")
            )
            expressions.append(expr)
        
        # DataFrameに適用
        result = df.with_columns(expressions)
        
        # 信頼性フラグの追加（別ステップで）
        if add_reliability:
            reliability_expressions = []
            for period in periods:
                reliability_expr = (
                    pl.col(f"rci_{period}")
                    .is_not_null()
                    .over(group_by)
                    .alias(f"rci_{period}_reliable")
                )
                reliability_expressions.append(reliability_expr)
            result = result.with_columns(reliability_expressions)
        
        return result
    
    def create_multi_period_expression(
        self,
        column: str,
        periods: List[int]
    ) -> List[pl.Expr]:
        """複数期間のRCI Expressionを一度に作成
        
        Args:
            column: 価格データのカラム名
            periods: RCI計算期間のリスト
            
        Returns:
            RCI Expressionのリスト
        """
        # 期間の検証
        self.engine.validate_periods(periods)
        
        expressions = []
        for period in periods:
            expr = self.create_rci_expression(column, period)
            expressions.append(expr)
        
        return expressions
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得
        
        Returns:
            Expressionキャッシュのサイズなど
        """
        return {
            "expression_cache_size": len(self._expr_cache),
            "cached_expressions": list(self._expr_cache.keys()),
            "engine_statistics": self.engine.get_statistics()
        }
    
    def clear_cache(self) -> None:
        """Expressionキャッシュをクリア"""
        self._expr_cache.clear()
        logger.info("RCIProcessor expression cache cleared")
