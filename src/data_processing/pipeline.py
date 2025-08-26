"""
統合指標パイプライン

RCI、RSI、MACD等のテクニカル指標を統一的に処理するパイプラインを提供。
効率的なバッチ処理とストリーミング処理をサポート。
"""

from typing import Dict, List, Optional, Any, Union, Literal
import polars as pl
import logging
from datetime import datetime

from .processor import PolarsProcessingEngine
from .rci import RCIProcessor, RCICalculatorEngine
from .indicators import TechnicalIndicatorEngine

logger = logging.getLogger(__name__)


class IndicatorPipeline:
    """テクニカル指標の統合パイプライン
    
    すべてのテクニカル指標（RCI、RSI、MACD、Bollinger Bands等）を
    統一的なインターフェースで処理します。
    
    Attributes:
        rci_processor: RCI計算プロセッサ
        indicator_engine: テクニカル指標エンジン
        processing_engine: Polars処理エンジン
        
    Methods:
        calculate_all_indicators: すべての指標を一括計算
        calculate_indicators_lazy: LazyFrameでの遅延評価計算
        calculate_grouped: グループごとの指標計算
    """
    
    def __init__(self, chunk_size: int = 100_000):
        """
        パイプラインの初期化
        
        Args:
            chunk_size: チャンク処理のサイズ
        """
        self.rci_processor = RCIProcessor(use_float32=True)
        self.indicator_engine = TechnicalIndicatorEngine()
        self.processing_engine = PolarsProcessingEngine(chunk_size=chunk_size)
        self._statistics = {
            "total_calculations": 0,
            "total_rows_processed": 0,
            "indicators_calculated": set()
        }
        logger.info(f"IndicatorPipeline initialized with chunk_size={chunk_size:,}")
    
    def calculate_all_indicators(
        self,
        df: pl.DataFrame,
        indicators: Dict[str, Dict[str, Any]],
        price_column: str = 'close',
        validate: bool = True
    ) -> pl.DataFrame:
        """
        すべての指標を一括計算
        
        Args:
            df: 入力DataFrame
            indicators: 指標設定の辞書
                {
                    'rci': {'periods': [9, 13, 24], 'add_reliability': True},
                    'rsi': {'period': 14},
                    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                    'bollinger': {'period': 20, 'num_std': 2}
                }
            price_column: 価格データのカラム名
            validate: 入力データのバリデーション実行
            
        Returns:
            すべての指標が追加されたDataFrame
        """
        if df.is_empty():
            logger.warning("Empty DataFrame provided")
            return df
        
        # バリデーション
        if validate:
            df = self.processing_engine.validate_datatypes(df)
        
        result = df
        start_time = datetime.now()
        
        # RCI計算
        if 'rci' in indicators:
            logger.info("Calculating RCI indicators")
            rci_config = indicators['rci']
            periods = rci_config.get('periods', [9, 13, 24, 33, 48, 66, 108])
            add_reliability = rci_config.get('add_reliability', True)
            
            result = self.rci_processor.apply_to_dataframe(
                result,
                periods=periods,
                column_name=price_column,
                add_reliability=add_reliability
            )
            self._statistics["indicators_calculated"].add('rci')
        
        # RSI計算
        if 'rsi' in indicators:
            logger.info("Calculating RSI indicator")
            rsi_config = indicators['rsi']
            period = rsi_config.get('period', 14)
            
            result = self.indicator_engine.calculate_rsi(
                result,
                period=period,
                price_column=price_column
            )
            self._statistics["indicators_calculated"].add('rsi')
        
        # MACD計算
        if 'macd' in indicators:
            logger.info("Calculating MACD indicator")
            macd_config = indicators['macd']
            fast = macd_config.get('fast', 12)
            slow = macd_config.get('slow', 26)
            signal = macd_config.get('signal', 9)
            
            result = self.indicator_engine.calculate_macd(
                result,
                fast_period=fast,
                slow_period=slow,
                signal_period=signal,
                price_column=price_column
            )
            self._statistics["indicators_calculated"].add('macd')
        
        # Bollinger Bands計算
        if 'bollinger' in indicators:
            logger.info("Calculating Bollinger Bands")
            bb_config = indicators['bollinger']
            period = bb_config.get('period', 20)
            num_std = bb_config.get('num_std', 2)
            
            result = self.indicator_engine.calculate_bollinger_bands(
                result,
                period=period,
                num_std=num_std,
                price_column=price_column
            )
            self._statistics["indicators_calculated"].add('bollinger')
        
        # EMA計算
        if 'ema' in indicators:
            logger.info("Calculating EMA indicators")
            ema_config = indicators['ema']
            periods = ema_config.get('periods', [5, 10, 20, 50, 100, 200])
            
            for period in periods:
                result = self.indicator_engine.calculate_ema(
                    result,
                    period=period,
                    price_column=price_column
                )
            self._statistics["indicators_calculated"].add('ema')
        
        # 統計更新
        self._statistics["total_calculations"] += 1
        self._statistics["total_rows_processed"] += len(result)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"All indicators calculated in {elapsed:.2f}s for {len(result):,} rows. "
            f"Indicators: {list(self._statistics['indicators_calculated'])}"
        )
        
        return result
    
    def calculate_indicators_lazy(
        self,
        lf: pl.LazyFrame,
        indicators: Dict[str, Dict[str, Any]],
        price_column: str = 'close'
    ) -> pl.LazyFrame:
        """
        LazyFrameでの遅延評価による指標計算
        
        メモリ効率的な大規模データ処理に適しています。
        
        Args:
            lf: 入力LazyFrame
            indicators: 指標設定の辞書
            price_column: 価格データのカラム名
            
        Returns:
            指標計算が設定されたLazyFrame（未実行）
        """
        result = lf
        
        # RCI計算（LazyFrame対応）
        if 'rci' in indicators:
            rci_config = indicators['rci']
            periods = rci_config.get('periods', [9, 13, 24])
            add_reliability = rci_config.get('add_reliability', True)
            
            result = self.rci_processor.apply_to_lazyframe(
                result,
                periods=periods,
                column_name=price_column,
                add_reliability=add_reliability
            )
        
        # 他の指標はLazyFrame非対応のため、警告を出す
        non_lazy_indicators = [ind for ind in indicators if ind != 'rci']
        if non_lazy_indicators:
            logger.warning(
                f"LazyFrame mode: Only RCI is fully supported. "
                f"Skipping: {non_lazy_indicators}"
            )
        
        return result
    
    def calculate_grouped(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        group_by: List[str],
        indicators: Dict[str, Dict[str, Any]],
        price_column: str = 'close'
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        グループごとの指標計算
        
        シンボルや時間枠ごとに分けて指標を計算します。
        
        Args:
            df: 入力DataFrame/LazyFrame
            group_by: グループ化するカラムのリスト
            indicators: 指標設定の辞書
            price_column: 価格データのカラム名
            
        Returns:
            グループごとに指標が計算されたDataFrame/LazyFrame
        """
        logger.info(f"Calculating indicators grouped by: {group_by}")
        
        result = df
        
        # RCIのグループ計算
        if 'rci' in indicators:
            rci_config = indicators['rci']
            periods = rci_config.get('periods', [9, 13, 24])
            add_reliability = rci_config.get('add_reliability', True)
            
            result = self.rci_processor.apply_grouped(
                result,
                group_by=group_by,
                periods=periods,
                column_name=price_column,
                add_reliability=add_reliability,
                parallel=True
            )
        
        # 他の指標のグループ計算
        if any(ind in indicators for ind in ['rsi', 'macd', 'bollinger', 'ema']):
            # 指標設定を構築
            indicator_configs = {
                k: v for k, v in indicators.items()
                if k in ['rsi', 'macd', 'bollinger', 'ema']
            }
            
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            
            # グループごとに処理
            result = self.indicator_engine.calculate_all_indicators(
                result,
                include_ema=('ema' in indicator_configs),
                include_rsi=('rsi' in indicator_configs),
                include_macd=('macd' in indicator_configs),
                include_bollinger=('bollinger' in indicator_configs),
                price_column=price_column,
                group_by=group_by if group_by else None
            )
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        パイプラインの統計情報を取得
        
        Returns:
            処理統計の辞書
        """
        return {
            "pipeline_stats": self._statistics,
            "rci_stats": self.rci_processor.get_statistics(),
            "indicator_stats": self.indicator_engine.get_processing_statistics()
        }
    
    def reset_statistics(self) -> None:
        """統計情報をリセット"""
        self._statistics = {
            "total_calculations": 0,
            "total_rows_processed": 0,
            "indicators_calculated": set()
        }
        self.rci_processor.clear_cache()
        logger.info("Pipeline statistics reset")
    
    def validate_and_process(
        self,
        df: pl.DataFrame,
        indicators: Dict[str, Dict[str, Any]],
        price_column: str = 'close',
        optimize_memory: bool = True
    ) -> pl.DataFrame:
        """
        データ検証と最適化を含む完全な処理パイプライン
        
        Args:
            df: 入力DataFrame
            indicators: 指標設定
            price_column: 価格データのカラム名
            optimize_memory: メモリ最適化の実行
            
        Returns:
            処理済みDataFrame
        """
        # データ型の検証と最適化
        df = self.processing_engine.validate_datatypes(df)
        
        if optimize_memory:
            df = self.processing_engine.optimize_dtypes(df)
        
        # メモリチェック
        self.processing_engine.handle_memory_pressure()
        
        # 指標計算
        result = self.calculate_all_indicators(
            df,
            indicators=indicators,
            price_column=price_column,
            validate=False  # 既に検証済み
        )
        
        return result