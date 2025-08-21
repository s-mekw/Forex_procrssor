"""Common data models for Forex Processor.

このモジュールは、Forex Processorで使用される共通データモデルを定義します。
すべてのモデルはPydanticを使用し、Float32による型制約でメモリ効率を最適化します。
"""

from datetime import datetime
from enum import Enum

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


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

    @property
    def time(self) -> datetime:
        """後方互換性のためのプロパティ
        
        TickToBarConverterとの互換性を保つため、
        time属性でtimestampにアクセスできるようにする。
        
        Note:
            このプロパティはStep 6で削除予定。
            新規コードではtimestamp属性を使用すること。
        """
        return self.timestamp

    @time.setter
    def time(self, value: datetime):
        """後方互換性のためのセッター
        
        Args:
            value: 設定する時刻値
        """
        self.timestamp = value


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


class PredictionType(str, Enum):
    """予測タイプの定義
    
    予測モデルが提供する予測の種類を定義します。
    """
    PRICE = "PRICE"           # 価格予測
    DIRECTION = "DIRECTION"   # 方向性予測（上昇/下降）
    VOLATILITY = "VOLATILITY" # ボラティリティ予測


class AlertType(str, Enum):
    """アラートタイプの定義
    
    システムが生成するアラートの種類を定義します。
    """
    PRICE_THRESHOLD = "PRICE_THRESHOLD"     # 価格閾値アラート
    PATTERN_DETECTED = "PATTERN_DETECTED"   # パターン検出アラート
    RISK_WARNING = "RISK_WARNING"           # リスク警告アラート


class AlertSeverity(str, Enum):
    """アラート重要度の定義
    
    アラートの重要度レベルを定義します。
    """
    INFO = "INFO"         # 情報レベル
    WARNING = "WARNING"   # 警告レベル
    CRITICAL = "CRITICAL" # 緊急レベル


class Prediction(BaseModel):
    """予測データモデル
    
    機械学習モデルによる予測結果を表現します。
    メモリ効率のため、数値フィールドはFloat32として制約されます。
    """

    symbol: str = Field(
        ...,
        min_length=6,
        max_length=10,
        description="通貨ペアシンボル（例: USDJPY, EURUSD）"
    )
    predicted_at: datetime = Field(
        ...,
        description="予測実行時刻（UTC）"
    )
    target_timestamp: datetime = Field(
        ...,
        description="予測対象時刻（UTC）"
    )
    prediction_type: PredictionType = Field(
        ...,
        description="予測タイプ"
    )
    predicted_value: float = Field(
        ...,
        description="予測値"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="信頼度スコア（0.0-1.0）"
    )
    confidence_upper: float | None = Field(
        default=None,
        description="信頼区間上限"
    )
    confidence_lower: float | None = Field(
        default=None,
        description="信頼区間下限"
    )
    model_version: str | None = Field(
        default=None,
        max_length=50,
        description="予測モデルのバージョン"
    )
    metadata: dict | None = Field(
        default=None,
        description="追加メタデータ"
    )

    @field_validator('predicted_value', 'confidence_score', 'confidence_upper', 'confidence_lower', mode='before')
    @classmethod
    def ensure_float32(cls, v: float) -> float:
        """Float32型への変換を保証"""
        if v is not None:
            # numpy float32として処理し、Pythonのfloatに戻す
            return float(np.float32(v))
        return v

    @model_validator(mode='after')
    def validate_confidence_interval(self) -> 'Prediction':
        """信頼区間の妥当性を検証"""
        if self.confidence_upper is not None and self.confidence_lower is not None:
            # Float32精度での比較
            upper_f32 = float(np.float32(self.confidence_upper))
            lower_f32 = float(np.float32(self.confidence_lower))
            if upper_f32 < lower_f32:
                raise ValueError('Confidence upper must be greater than or equal to confidence lower')
        return self

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """シンボルを大文字に正規化"""
        return v.upper()

    @field_validator('target_timestamp')
    @classmethod
    def validate_target_timestamp(cls, v: datetime, info: ValidationInfo) -> datetime:
        """予測対象時刻が予測実行時刻より未来であることを検証"""
        if 'predicted_at' in info.data:
            if v <= info.data['predicted_at']:
                raise ValueError('Target timestamp must be after predicted_at timestamp')
        return v

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        json_schema_extra={
            "example": {
                "symbol": "USDJPY",
                "predicted_at": "2025-01-17T12:00:00Z",
                "target_timestamp": "2025-01-17T13:00:00Z",
                "prediction_type": "PRICE",
                "predicted_value": 150.500,
                "confidence_score": 0.85,
                "confidence_upper": 150.600,
                "confidence_lower": 150.400,
                "model_version": "v1.2.3"
            }
        }
    )

    @property
    def confidence_range(self) -> float | None:
        """信頼区間の幅を計算"""
        if self.confidence_upper is not None and self.confidence_lower is not None:
            return float(np.float32(self.confidence_upper - self.confidence_lower))
        return None

    @property
    def is_high_confidence(self) -> bool:
        """高信頼度（0.7以上）かどうかを判定"""
        return self.confidence_score >= 0.7

    @property
    def prediction_horizon_hours(self) -> float:
        """予測ホライゾン（時間）を計算"""
        delta = self.target_timestamp - self.predicted_at
        return delta.total_seconds() / 3600

    def to_float32_dict(self) -> dict:
        """Float32型で数値フィールドを含む辞書を返す"""
        result = {
            "symbol": self.symbol,
            "predicted_at": self.predicted_at,
            "target_timestamp": self.target_timestamp,
            "prediction_type": self.prediction_type.value,
            "predicted_value": np.float32(self.predicted_value),
            "confidence_score": np.float32(self.confidence_score),
            "model_version": self.model_version,
            "metadata": self.metadata
        }
        if self.confidence_upper is not None:
            result["confidence_upper"] = np.float32(self.confidence_upper)
        if self.confidence_lower is not None:
            result["confidence_lower"] = np.float32(self.confidence_lower)
        return result


class Alert(BaseModel):
    """アラートデータモデル
    
    システムが生成するアラート情報を表現します。
    メモリ効率のため、数値フィールドはFloat32として制約されます。
    """

    symbol: str = Field(
        ...,
        min_length=6,
        max_length=10,
        description="通貨ペアシンボル（例: USDJPY, EURUSD）"
    )
    timestamp: datetime = Field(
        ...,
        description="アラート生成時刻（UTC）"
    )
    alert_type: AlertType = Field(
        ...,
        description="アラートタイプ"
    )
    severity: AlertSeverity = Field(
        ...,
        description="アラート重要度"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="アラートメッセージ"
    )
    threshold_value: float | None = Field(
        default=None,
        description="閾値（価格閾値アラートの場合）"
    )
    current_value: float | None = Field(
        default=None,
        description="現在値（閾値比較用）"
    )
    condition: str | None = Field(
        default=None,
        max_length=200,
        description="トリガー条件の記録"
    )
    metadata: dict | None = Field(
        default=None,
        description="追加メタデータ"
    )
    acknowledged: bool = Field(
        default=False,
        description="確認済みフラグ"
    )

    @field_validator('threshold_value', 'current_value', mode='before')
    @classmethod
    def ensure_float32(cls, v: float) -> float:
        """Float32型への変換を保証"""
        if v is not None:
            # numpy float32として処理し、Pythonのfloatに戻す
            return float(np.float32(v))
        return v

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """シンボルを大文字に正規化"""
        return v.upper()

    @field_validator('current_value')
    @classmethod
    def validate_current_value(cls, v: float | None, info: ValidationInfo) -> float | None:
        """閾値アラートの場合、現在値と閾値の関係を記録"""
        # 実際のアラート生成ロジックはビジネスロジック層で実装
        # ここではデータの整合性のみ確認
        if v is not None and 'alert_type' in info.data:
            if info.data['alert_type'] == AlertType.PRICE_THRESHOLD:
                if 'threshold_value' not in info.data or info.data['threshold_value'] is None:
                    raise ValueError('Threshold value is required for price threshold alerts')
        return v

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        json_schema_extra={
            "example": {
                "symbol": "USDJPY",
                "timestamp": "2025-01-17T12:00:00Z",
                "alert_type": "PRICE_THRESHOLD",
                "severity": "WARNING",
                "message": "USDJPY price exceeded threshold",
                "threshold_value": 150.000,
                "current_value": 150.123,
                "condition": "price > 150.000",
                "acknowledged": False
            }
        }
    )

    @property
    def is_critical(self) -> bool:
        """緊急レベルのアラートかどうかを判定"""
        return self.severity == AlertSeverity.CRITICAL

    @property
    def is_warning(self) -> bool:
        """警告レベルのアラートかどうかを判定"""
        return self.severity == AlertSeverity.WARNING

    @property
    def is_info(self) -> bool:
        """情報レベルのアラートかどうかを判定"""
        return self.severity == AlertSeverity.INFO

    @property
    def threshold_exceeded(self) -> bool | None:
        """閾値超過判定（価格閾値アラートの場合）"""
        if (self.alert_type == AlertType.PRICE_THRESHOLD and
            self.threshold_value is not None and
            self.current_value is not None):
            # Float32精度での比較
            current_f32 = np.float32(self.current_value)
            threshold_f32 = np.float32(self.threshold_value)
            return bool(current_f32 > threshold_f32)
        return None

    @property
    def threshold_difference(self) -> float | None:
        """閾値との差分を計算"""
        if self.threshold_value is not None and self.current_value is not None:
            return float(np.float32(self.current_value - self.threshold_value))
        return None

    def to_float32_dict(self) -> dict:
        """Float32型で数値フィールドを含む辞書を返す"""
        result = {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "condition": self.condition,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged
        }
        if self.threshold_value is not None:
            result["threshold_value"] = np.float32(self.threshold_value)
        if self.current_value is not None:
            result["current_value"] = np.float32(self.current_value)
        return result
