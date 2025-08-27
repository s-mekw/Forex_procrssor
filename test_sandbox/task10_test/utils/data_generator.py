"""
FX市場データジェネレーター

リアルなFX市場データをシミュレートするためのデータ生成ユーティリティです。
"""

import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from src.data_processing.pipelines import DataPoint


class MarketCondition(Enum):
    """市場状態の定義"""
    
    NORMAL = "normal"  # 通常状態
    HIGH_VOLATILITY = "high_volatility"  # 高ボラティリティ
    NEWS_RELEASE = "news_release"  # ニュースリリース時
    MARKET_OPEN = "market_open"  # 市場オープン時
    MARKET_CLOSE = "market_close"  # 市場クローズ時
    LOW_LIQUIDITY = "low_liquidity"  # 低流動性


class FXDataGenerator:
    """FX市場データのジェネレーター"""
    
    # 主要通貨ペアと基準レート
    CURRENCY_PAIRS = {
        "USDJPY": 150.00,
        "EURUSD": 1.0800,
        "GBPUSD": 1.2500,
        "AUDUSD": 0.6500,
        "USDCAD": 1.3500,
        "USDCHF": 0.9000,
        "NZDUSD": 0.5900,
        "EURJPY": 162.00,
        "GBPJPY": 187.50,
        "AUDJPY": 97.50,
        "EURGBP": 0.8640,  # 追加
    }
    
    def __init__(self, seed: Optional[int] = None):
        """
        初期化
        
        Args:
            seed: 乱数シード（再現性のため）
        """
        if seed is not None:
            random.seed(seed)
        
        self.current_rates = self.CURRENCY_PAIRS.copy()
        self.tick_count = 0
        
    def generate_tick(
        self,
        symbol: str,
        condition: MarketCondition = MarketCondition.NORMAL,
        timestamp: Optional[datetime] = None,
        provider: str = "default"
    ) -> DataPoint:
        """
        単一のティックデータを生成
        
        Args:
            symbol: 通貨ペア
            condition: 市場状態
            timestamp: タイムスタンプ（None時は現在時刻）
            provider: プロバイダー名
            
        Returns:
            DataPoint: 生成されたティックデータ
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.current_rates:
            raise ValueError(f"Unknown currency pair: {symbol}")
        
        # ボラティリティの計算（市場状態による）
        volatility = self._get_volatility(condition)
        
        # レート変動の計算
        change = random.gauss(0, volatility)
        self.current_rates[symbol] *= (1 + change)
        
        # スプレッドの計算
        spread = self._get_spread(symbol, condition)
        
        bid = self.current_rates[symbol]
        ask = bid + spread
        
        # ボリュームの生成（市場状態による）
        volume = self._generate_volume(condition)
        
        self.tick_count += 1
        
        return DataPoint(
            timestamp=timestamp,
            data={
                "symbol": symbol,
                "bid": round(bid, 5),
                "ask": round(ask, 5),
                "mid": round((bid + ask) / 2, 5),
                "volume": volume,
                "spread": round(spread, 5),
                "tick_id": self.tick_count,
                "provider": provider,
            },
            metadata={
                "source": "FXDataGenerator",
                "market_condition": condition.value,
                "generated_at": datetime.now().isoformat(),
            }
        )
    
    def generate_burst(
        self,
        symbol: str,
        count: int,
        interval_ms: int = 100,
        condition: MarketCondition = MarketCondition.NEWS_RELEASE,
        base_timestamp: Optional[datetime] = None,
        provider: str = "default"
    ) -> list[DataPoint]:
        """
        バースト的なデータを生成（ニュースリリース時など）
        
        Args:
            symbol: 通貨ペア
            count: 生成するデータ数
            interval_ms: データ間隔（ミリ秒）
            condition: 市場状態
            base_timestamp: 基準時刻
            provider: プロバイダー名
            
        Returns:
            list[DataPoint]: 生成されたデータリスト
        """
        if base_timestamp is None:
            base_timestamp = datetime.now()
        
        data_points = []
        for i in range(count):
            timestamp = base_timestamp + timedelta(milliseconds=i * interval_ms)
            data_point = self.generate_tick(symbol, condition, timestamp, provider)
            data_points.append(data_point)
        
        return data_points
    
    def generate_multi_symbol_stream(
        self,
        symbols: list[str],
        duration_seconds: int,
        ticks_per_second: int = 10,
        condition: MarketCondition = MarketCondition.NORMAL,
        provider: str = "default"
    ) -> list[DataPoint]:
        """
        複数通貨ペアのストリームデータを生成
        
        Args:
            symbols: 通貨ペアリスト
            duration_seconds: 期間（秒）
            ticks_per_second: 秒あたりのティック数
            condition: 市場状態
            provider: プロバイダー名
            
        Returns:
            list[DataPoint]: 生成されたデータストリーム
        """
        total_ticks = duration_seconds * ticks_per_second
        interval_ms = 1000 // ticks_per_second
        
        base_timestamp = datetime.now()
        data_stream = []
        
        for i in range(total_ticks):
            timestamp = base_timestamp + timedelta(milliseconds=i * interval_ms)
            symbol = random.choice(symbols)
            data_point = self.generate_tick(symbol, condition, timestamp, provider)
            data_stream.append(data_point)
        
        return data_stream
    
    def generate_delayed_data(
        self,
        symbol: str,
        delay_seconds: float,
        count: int = 1,
        provider: str = "delayed_provider"
    ) -> list[DataPoint]:
        """
        遅延のあるデータを生成（アラートテスト用）
        
        Args:
            symbol: 通貨ペア
            delay_seconds: 遅延秒数
            count: 生成数
            provider: プロバイダー名
            
        Returns:
            list[DataPoint]: 遅延タイムスタンプを持つデータ
        """
        data_points = []
        past_timestamp = datetime.now() - timedelta(seconds=delay_seconds)
        
        for i in range(count):
            timestamp = past_timestamp + timedelta(milliseconds=i * 100)
            data_point = self.generate_tick(
                symbol, 
                MarketCondition.NORMAL,
                timestamp,
                provider
            )
            data_points.append(data_point)
        
        return data_points
    
    def _get_volatility(self, condition: MarketCondition) -> float:
        """市場状態に応じたボラティリティを取得"""
        volatilities = {
            MarketCondition.NORMAL: 0.0001,
            MarketCondition.HIGH_VOLATILITY: 0.0010,
            MarketCondition.NEWS_RELEASE: 0.0020,
            MarketCondition.MARKET_OPEN: 0.0008,
            MarketCondition.MARKET_CLOSE: 0.0006,
            MarketCondition.LOW_LIQUIDITY: 0.0003,
        }
        return volatilities.get(condition, 0.0001)
    
    def _get_spread(self, symbol: str, condition: MarketCondition) -> float:
        """通貨ペアと市場状態に応じたスプレッドを取得"""
        # 基本スプレッド（pips）
        base_spreads = {
            "USDJPY": 0.003,
            "EURUSD": 0.00002,
            "GBPUSD": 0.00003,
            "AUDUSD": 0.00003,
            "USDCAD": 0.00004,
            "USDCHF": 0.00003,
            "NZDUSD": 0.00004,
            "EURJPY": 0.004,
            "GBPJPY": 0.005,
            "AUDJPY": 0.004,
        }
        
        # 市場状態による倍率
        multipliers = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.HIGH_VOLATILITY: 2.0,
            MarketCondition.NEWS_RELEASE: 3.0,
            MarketCondition.MARKET_OPEN: 1.5,
            MarketCondition.MARKET_CLOSE: 1.2,
            MarketCondition.LOW_LIQUIDITY: 2.5,
        }
        
        base_spread = base_spreads.get(symbol, 0.00003)
        multiplier = multipliers.get(condition, 1.0)
        
        return base_spread * multiplier
    
    def _generate_volume(self, condition: MarketCondition) -> int:
        """市場状態に応じたボリュームを生成"""
        base_volume = 1000000  # 100万通貨単位
        
        volume_ranges = {
            MarketCondition.NORMAL: (0.5, 1.5),
            MarketCondition.HIGH_VOLATILITY: (2.0, 5.0),
            MarketCondition.NEWS_RELEASE: (3.0, 10.0),
            MarketCondition.MARKET_OPEN: (1.5, 3.0),
            MarketCondition.MARKET_CLOSE: (0.8, 1.2),
            MarketCondition.LOW_LIQUIDITY: (0.1, 0.5),
        }
        
        min_mult, max_mult = volume_ranges.get(condition, (0.5, 1.5))
        multiplier = random.uniform(min_mult, max_mult)
        
        return int(base_volume * multiplier)
    
    def reset(self):
        """レートを初期状態にリセット"""
        self.current_rates = self.CURRENCY_PAIRS.copy()
        self.tick_count = 0