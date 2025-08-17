"""Common data models for Forex Processor.

このモジュールは、Forex Processorで使用される共通データモデルを定義します。
すべてのモデルはPydanticを使用し、Float32による型制約でメモリ効率を最適化します。
"""

from datetime import datetime

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
