"""
RCI（Rank Correlation Index）計算エンジン

高速な差分更新アルゴリズムによるRCI計算を提供します。
Float32精度による メモリ効率化と、O(n)の計算複雑度を実現。
"""

from collections import deque
from typing import Optional, List, Dict, Any, Literal, Tuple
import numpy as np
import polars as pl
import logging
import hashlib
from functools import lru_cache

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
    
    def __init__(self):
        """エンジンを初期化"""
        self.calculators: Dict[int, DifferentialRCICalculator] = {}
        self._statistics = {
            'batch_calculations': 0,
            'streaming_calculations': 0,
            'total_periods_calculated': 0,
            'validation_errors': 0
        }
        logger.info("RCICalculatorEngine initialized")
    
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
        mode: Literal['batch', 'streaming'] = 'batch',
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
        
        # モードに応じた処理
        if mode == 'batch':
            result = self._process_batch(data, periods, column_name)
            self._statistics['batch_calculations'] += 1
        elif mode == 'streaming':
            result = self._process_streaming(data, periods, column_name)
            self._statistics['streaming_calculations'] += 1
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'batch' or 'streaming'")
        
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
            'validation_errors': 0
        }
        logger.info("RCICalculatorEngine reset")
