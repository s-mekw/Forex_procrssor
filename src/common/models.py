"""Common data models for Forex Processor.

このモジュールは、Forex Processorで使用される共通データモデルを定義します。
すべてのモデルはPydanticを使用し、Float32による型制約でメモリ効率を最適化します。
"""

from datetime import datetime
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class Tick(BaseModel):
    """Tickデータモデル

    為替レートのティックデータを表現します。
    メモリ効率のため、数値フィールドはFloat32として制約されます。
    """

    timestamp: datetime = Field(
        ...,
        description="ティックのタイムスタンプ（UTC）"
    )
    symbol: str = Field(
        ...,
        min_length=6,
        max_length=10,
        description="通貨ペアシンボル（例: USDJPY, EURUSD）"
    )
    bid: float = Field(
        ...,
        gt=0,
        description="Bid価格（売値）"
    )
    ask: float = Field(
        ...,
        gt=0,
        description="Ask価格（買値）"
    )
    volume: float = Field(
        default=0.0,
        ge=0,
        description="ティックボリューム"
    )

    @field_validator('bid', 'ask', 'volume', mode='before')
    @classmethod
    def ensure_float32(cls, v: float) -> float:
        """Float32型への変換を保証"""
        if v is not None:
            # numpy float32として処理し、Pythonのfloatに戻す
            return float(np.float32(v))
        return v

    @field_validator('ask')
    @classmethod
    def validate_spread(cls, v: float, info: ValidationInfo) -> float:
        """スプレッドの妥当性を検証（ask > bid）"""
        if 'bid' in info.data:
            # Float32精度での比較を行う
            bid_f32 = float(np.float32(info.data['bid']))
            ask_f32 = float(np.float32(v))
            if ask_f32 <= bid_f32:
                raise ValueError('Ask price must be greater than bid price')
        return v

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """シンボルを大文字に正規化"""
        return v.upper()

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-17T12:00:00Z",
                "symbol": "USDJPY",
                "bid": 150.123,
                "ask": 150.125,
                "volume": 1000.0
            }
        }
    )

    @property
    def spread(self) -> float:
        """スプレッドを計算"""
        return float(np.float32(self.ask - self.bid))

    @property
    def mid_price(self) -> float:
        """仲値（中間価格）を計算"""
        return float(np.float32((self.bid + self.ask) / 2))

    def to_float32_dict(self) -> dict:
        """Float32型で数値フィールドを含む辞書を返す"""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "bid": np.float32(self.bid),
            "ask": np.float32(self.ask),
            "volume": np.float32(self.volume)
        }


class TimeFrame(str, Enum):
    """時間足の定義
    
    FX取引で使用される標準的な時間足を定義します。
    """
    M1 = "M1"   # 1分足
    M5 = "M5"   # 5分足
    M15 = "M15" # 15分足
    M30 = "M30" # 30分足
    H1 = "H1"   # 1時間足
    H4 = "H4"   # 4時間足
    D1 = "D1"   # 日足
    W1 = "W1"   # 週足
    MN = "MN"   # 月足


class OHLC(BaseModel):
    """OHLCデータモデル
    
    為替レートのOHLC（四本値）データを表現します。
    メモリ効率のため、数値フィールドはFloat32として制約されます。
    """
    
    timestamp: datetime = Field(
        ...,
        description="ローソク足の開始時刻（UTC）"
    )
    symbol: str = Field(
        ...,
        min_length=6,
        max_length=10,
        description="通貨ペアシンボル（例: USDJPY, EURUSD）"
    )
    timeframe: TimeFrame = Field(
        ...,
        description="時間足"
    )
    open: float = Field(
        ...,
        gt=0,
        description="開値"
    )
    high: float = Field(
        ...,
        gt=0,
        description="高値"
    )
    low: float = Field(
        ...,
        gt=0,
        description="安値"
    )
    close: float = Field(
        ...,
        gt=0,
        description="終値"
    )
    volume: float = Field(
        default=0.0,
        ge=0,
        description="取引量"
    )
    
    @field_validator('open', 'high', 'low', 'close', 'volume', mode='before')
    @classmethod
    def ensure_float32(cls, v: float) -> float:
        """Float32型への変換を保証"""
        if v is not None:
            # numpy float32として処理し、Pythonのfloatに戻す
            return float(np.float32(v))
        return v
    
    @field_validator('high')
    @classmethod
    def validate_high(cls, v: float, info: ValidationInfo) -> float:
        """高値が他の価格以上であることを検証"""
        # すでにFloat32に変換された値を使用
        v_f32 = float(np.float32(v))
        
        # 検証時はvalidation contexではなくvaluesから取得（低、開、終の順に検証）
        if info.data:
            # 低値との比較（lowが先に定義されている場合）
            if 'low' in info.data:
                low_f32 = float(np.float32(info.data['low']))
                if v_f32 < low_f32:
                    raise ValueError('High price must be greater than or equal to low price')
            
            # 開値との比較
            if 'open' in info.data:
                open_f32 = float(np.float32(info.data['open']))
                if v_f32 < open_f32:
                    raise ValueError('High price must be greater than or equal to open price')
            
            # 終値との比較（closeが後に定義される場合は検証しない）
        
        return v
    
    @field_validator('low')
    @classmethod
    def validate_low(cls, v: float, info: ValidationInfo) -> float:
        """安値が他の価格以下であることを検証"""
        # すでにFloat32に変換された値を使用
        v_f32 = float(np.float32(v))
        
        # 検証時はvalidation contextではなくvaluesから取得
        if info.data:
            # 開値との比較
            if 'open' in info.data:
                open_f32 = float(np.float32(info.data['open']))
                if v_f32 > open_f32:
                    raise ValueError('Low price must be less than or equal to open price')
        
        return v
    
    @field_validator('close')
    @classmethod
    def validate_close(cls, v: float, info: ValidationInfo) -> float:
        """終値が高値以下、安値以上であることを検証"""
        # すでにFloat32に変換された値を使用
        v_f32 = float(np.float32(v))
        
        if info.data:
            # 高値との比較
            if 'high' in info.data:
                high_f32 = float(np.float32(info.data['high']))
                if v_f32 > high_f32:
                    raise ValueError('Close price must be less than or equal to high price')
            
            # 安値との比較
            if 'low' in info.data:
                low_f32 = float(np.float32(info.data['low']))
                if v_f32 < low_f32:
                    raise ValueError('Close price must be greater than or equal to low price')
        
        return v
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """シンボルを大文字に正規化"""
        return v.upper()
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        json_schema_extra={
            "example": {
                "timestamp": "2025-01-17T12:00:00Z",
                "symbol": "USDJPY",
                "timeframe": "H1",
                "open": 150.100,
                "high": 150.500,
                "low": 149.900,
                "close": 150.300,
                "volume": 10000.0
            }
        }
    )
    
    @property
    def range(self) -> float:
        """価格レンジ（高値-安値）を計算"""
        return float(np.float32(self.high - self.low))
    
    @property
    def is_bullish(self) -> bool:
        """陽線（bullish）かどうかを判定"""
        # Float32精度での比較
        close_f32 = np.float32(self.close)
        open_f32 = np.float32(self.open)
        return bool(close_f32 > open_f32)  # Pythonのbool型に明示的に変換
    
    @property
    def is_bearish(self) -> bool:
        """陰線（bearish）かどうかを判定"""
        # Float32精度での比較
        close_f32 = np.float32(self.close)
        open_f32 = np.float32(self.open)
        return bool(close_f32 < open_f32)  # Pythonのbool型に明示的に変換
    
    @property
    def body_size(self) -> float:
        """実体サイズ（|終値-開値|）を計算"""
        return float(np.float32(abs(self.close - self.open)))
    
    @property
    def upper_shadow(self) -> float:
        """上髭の長さを計算"""
        body_high = max(self.open, self.close)
        return float(np.float32(self.high - body_high))
    
    @property
    def lower_shadow(self) -> float:
        """下髭の長さを計算"""
        body_low = min(self.open, self.close)
        return float(np.float32(body_low - self.low))
    
    def to_float32_dict(self) -> dict:
        """Float32型で数値フィールドを含む辞書を返す"""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "open": np.float32(self.open),
            "high": np.float32(self.high),
            "low": np.float32(self.low),
            "close": np.float32(self.close),
            "volume": np.float32(self.volume)
        }
