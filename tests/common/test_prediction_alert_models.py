"""Tests for Prediction and Alert data models.

このモジュールは、PredictionとAlertデータモデルのテストを提供します。
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from src.common.models import (
    Alert,
    AlertSeverity,
    AlertType,
    Prediction,
    PredictionType,
)


class TestPrediction:
    """Predictionモデルのテストクラス"""

    def test_prediction_creation_basic(self):
        """基本的なPredictionの作成テスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        prediction = Prediction(
            symbol="USDJPY",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.PRICE,
            predicted_value=150.500,
            confidence_score=0.85
        )
        
        assert prediction.symbol == "USDJPY"
        assert prediction.predicted_at == now
        assert prediction.target_timestamp == future
        assert prediction.prediction_type == PredictionType.PRICE
        assert prediction.predicted_value == pytest.approx(150.500, rel=1e-5)
        assert prediction.confidence_score == pytest.approx(0.85, rel=1e-5)
        assert prediction.confidence_upper is None
        assert prediction.confidence_lower is None

    def test_prediction_with_confidence_interval(self):
        """信頼区間付きPredictionの作成テスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        prediction = Prediction(
            symbol="EURUSD",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.PRICE,
            predicted_value=1.1000,
            confidence_score=0.90,
            confidence_upper=1.1050,
            confidence_lower=1.0950
        )
        
        assert prediction.confidence_upper == pytest.approx(1.1050, rel=1e-5)
        assert prediction.confidence_lower == pytest.approx(1.0950, rel=1e-5)
        assert prediction.confidence_range == pytest.approx(0.01, rel=1e-5)

    def test_prediction_float32_conversion(self):
        """Float32変換のテスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        # 高精度の値を入力
        high_precision_value = 150.123456789
        
        prediction = Prediction(
            symbol="USDJPY",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.PRICE,
            predicted_value=high_precision_value,
            confidence_score=0.85,
            confidence_upper=150.2,
            confidence_lower=150.0
        )
        
        # Float32精度に丸められることを確認
        expected_value = float(np.float32(high_precision_value))
        assert prediction.predicted_value == expected_value
        assert prediction.confidence_upper == float(np.float32(150.2))
        assert prediction.confidence_lower == float(np.float32(150.0))

    def test_prediction_confidence_score_validation(self):
        """信頼度スコアのバリデーションテスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        # 範囲外の信頼度スコアでエラー
        with pytest.raises(ValidationError) as exc_info:
            Prediction(
                symbol="USDJPY",
                predicted_at=now,
                target_timestamp=future,
                prediction_type=PredictionType.PRICE,
                predicted_value=150.0,
                confidence_score=1.5  # 1.0を超える
            )
        assert "less than or equal to 1" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            Prediction(
                symbol="USDJPY",
                predicted_at=now,
                target_timestamp=future,
                prediction_type=PredictionType.PRICE,
                predicted_value=150.0,
                confidence_score=-0.1  # 0.0未満
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_prediction_confidence_interval_validation(self):
        """信頼区間の妥当性検証テスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        # 上限が下限より小さい場合エラー
        with pytest.raises(ValidationError) as exc_info:
            Prediction(
                symbol="USDJPY",
                predicted_at=now,
                target_timestamp=future,
                prediction_type=PredictionType.PRICE,
                predicted_value=150.0,
                confidence_score=0.85,
                confidence_lower=150.1,  # 下限が先に設定される
                confidence_upper=149.9   # 上限が下限より小さい
            )
        assert "Confidence upper must be greater than or equal to confidence lower" in str(exc_info.value)

    def test_prediction_timestamp_validation(self):
        """タイムスタンプの妥当性検証テスト"""
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        
        # 予測対象時刻が過去の場合エラー
        with pytest.raises(ValidationError) as exc_info:
            Prediction(
                symbol="USDJPY",
                predicted_at=now,
                target_timestamp=past,  # 過去の時刻
                prediction_type=PredictionType.PRICE,
                predicted_value=150.0,
                confidence_score=0.85
            )
        assert "Target timestamp must be after predicted_at timestamp" in str(exc_info.value)

    def test_prediction_symbol_normalization(self):
        """シンボルの正規化テスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        prediction = Prediction(
            symbol="usdjpy",  # 小文字で入力
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.PRICE,
            predicted_value=150.0,
            confidence_score=0.85
        )
        
        assert prediction.symbol == "USDJPY"  # 大文字に正規化

    def test_prediction_types(self):
        """異なる予測タイプのテスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        # 価格予測
        price_pred = Prediction(
            symbol="USDJPY",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.PRICE,
            predicted_value=150.0,
            confidence_score=0.85
        )
        assert price_pred.prediction_type == PredictionType.PRICE
        
        # 方向性予測
        direction_pred = Prediction(
            symbol="USDJPY",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.DIRECTION,
            predicted_value=1.0,  # 1: 上昇, -1: 下降
            confidence_score=0.75
        )
        assert direction_pred.prediction_type == PredictionType.DIRECTION
        
        # ボラティリティ予測
        volatility_pred = Prediction(
            symbol="USDJPY",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.VOLATILITY,
            predicted_value=0.015,  # 1.5%のボラティリティ
            confidence_score=0.80
        )
        assert volatility_pred.prediction_type == PredictionType.VOLATILITY

    def test_prediction_properties(self):
        """プロパティメソッドのテスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=3)
        
        prediction = Prediction(
            symbol="USDJPY",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.PRICE,
            predicted_value=150.0,
            confidence_score=0.75,
            confidence_upper=150.1,
            confidence_lower=149.9
        )
        
        # 高信頼度判定
        assert prediction.is_high_confidence is True
        
        # 予測ホライゾン
        assert prediction.prediction_horizon_hours == pytest.approx(3.0, rel=1e-5)
        
        # 信頼区間の幅
        assert prediction.confidence_range == pytest.approx(0.2, rel=1e-3)

    def test_prediction_to_float32_dict(self):
        """Float32辞書変換のテスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        prediction = Prediction(
            symbol="USDJPY",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.PRICE,
            predicted_value=150.123456,
            confidence_score=0.85,
            confidence_upper=150.2,
            confidence_lower=150.0,
            model_version="v1.2.3"
        )
        
        float32_dict = prediction.to_float32_dict()
        
        assert float32_dict["symbol"] == "USDJPY"
        assert float32_dict["predicted_at"] == now
        assert float32_dict["target_timestamp"] == future
        assert float32_dict["prediction_type"] == "PRICE"
        assert isinstance(float32_dict["predicted_value"], np.float32)
        assert isinstance(float32_dict["confidence_score"], np.float32)
        assert isinstance(float32_dict["confidence_upper"], np.float32)
        assert isinstance(float32_dict["confidence_lower"], np.float32)
        assert float32_dict["model_version"] == "v1.2.3"

    def test_prediction_with_metadata(self):
        """メタデータ付きPredictionのテスト"""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        
        metadata = {
            "model_name": "LSTM",
            "features_used": ["price", "volume", "rsi"],
            "training_window": 100
        }
        
        prediction = Prediction(
            symbol="USDJPY",
            predicted_at=now,
            target_timestamp=future,
            prediction_type=PredictionType.PRICE,
            predicted_value=150.0,
            confidence_score=0.85,
            metadata=metadata
        )
        
        assert prediction.metadata == metadata
        assert prediction.metadata["model_name"] == "LSTM"


class TestAlert:
    """Alertモデルのテストクラス"""

    def test_alert_creation_basic(self):
        """基本的なAlertの作成テスト"""
        now = datetime.now(timezone.utc)
        
        alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="USDJPY price exceeded threshold"
        )
        
        assert alert.symbol == "USDJPY"
        assert alert.timestamp == now
        assert alert.alert_type == AlertType.PRICE_THRESHOLD
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "USDJPY price exceeded threshold"
        assert alert.acknowledged is False

    def test_alert_with_threshold(self):
        """閾値付きAlertの作成テスト"""
        now = datetime.now(timezone.utc)
        
        alert = Alert(
            symbol="EURUSD",
            timestamp=now,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="EURUSD price exceeded upper threshold",
            threshold_value=1.1000,
            current_value=1.1050,
            condition="price > 1.1000"
        )
        
        assert alert.threshold_value == pytest.approx(1.1000, rel=1e-5)
        assert alert.current_value == pytest.approx(1.1050, rel=1e-5)
        assert alert.condition == "price > 1.1000"
        assert alert.threshold_exceeded is True
        assert alert.threshold_difference == pytest.approx(0.005, rel=1e-3)

    def test_alert_float32_conversion(self):
        """Float32変換のテスト"""
        now = datetime.now(timezone.utc)
        
        # 高精度の値を入力
        high_precision_threshold = 150.123456789
        high_precision_current = 150.234567890
        
        alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            threshold_value=high_precision_threshold,
            current_value=high_precision_current
        )
        
        # Float32精度に丸められることを確認
        assert alert.threshold_value == float(np.float32(high_precision_threshold))
        assert alert.current_value == float(np.float32(high_precision_current))

    def test_alert_price_threshold_validation(self):
        """価格閾値アラートのバリデーションテスト"""
        now = datetime.now(timezone.utc)
        
        # 価格閾値アラートで閾値がない場合エラー
        with pytest.raises(ValidationError) as exc_info:
            Alert(
                symbol="USDJPY",
                timestamp=now,
                alert_type=AlertType.PRICE_THRESHOLD,
                severity=AlertSeverity.WARNING,
                message="Test alert",
                current_value=150.0  # threshold_valueがない
            )
        assert "Threshold value is required for price threshold alerts" in str(exc_info.value)

    def test_alert_symbol_normalization(self):
        """シンボルの正規化テスト"""
        now = datetime.now(timezone.utc)
        
        alert = Alert(
            symbol="eurusd",  # 小文字で入力
            timestamp=now,
            alert_type=AlertType.PATTERN_DETECTED,
            severity=AlertSeverity.INFO,
            message="Pattern detected"
        )
        
        assert alert.symbol == "EURUSD"  # 大文字に正規化

    def test_alert_types(self):
        """異なるアラートタイプのテスト"""
        now = datetime.now(timezone.utc)
        
        # 価格閾値アラート
        price_alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="Price threshold alert",
            threshold_value=150.0,
            current_value=150.1
        )
        assert price_alert.alert_type == AlertType.PRICE_THRESHOLD
        
        # パターン検出アラート
        pattern_alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PATTERN_DETECTED,
            severity=AlertSeverity.INFO,
            message="Head and shoulders pattern detected"
        )
        assert pattern_alert.alert_type == AlertType.PATTERN_DETECTED
        
        # リスク警告アラート
        risk_alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.RISK_WARNING,
            severity=AlertSeverity.CRITICAL,
            message="High volatility detected"
        )
        assert risk_alert.alert_type == AlertType.RISK_WARNING

    def test_alert_severities(self):
        """異なる重要度レベルのテスト"""
        now = datetime.now(timezone.utc)
        
        # 情報レベル
        info_alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PATTERN_DETECTED,
            severity=AlertSeverity.INFO,
            message="Info alert"
        )
        assert info_alert.is_info is True
        assert info_alert.is_warning is False
        assert info_alert.is_critical is False
        
        # 警告レベル
        warning_alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="Warning alert"
        )
        assert warning_alert.is_info is False
        assert warning_alert.is_warning is True
        assert warning_alert.is_critical is False
        
        # 緊急レベル
        critical_alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.RISK_WARNING,
            severity=AlertSeverity.CRITICAL,
            message="Critical alert"
        )
        assert critical_alert.is_info is False
        assert critical_alert.is_warning is False
        assert critical_alert.is_critical is True

    def test_alert_properties(self):
        """プロパティメソッドのテスト"""
        now = datetime.now(timezone.utc)
        
        # 閾値超過なし
        alert_below = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            threshold_value=150.0,
            current_value=149.9
        )
        assert alert_below.threshold_exceeded is False
        assert alert_below.threshold_difference == pytest.approx(-0.1, rel=1e-3)
        
        # 閾値超過あり
        alert_above = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            threshold_value=150.0,
            current_value=150.1
        )
        assert alert_above.threshold_exceeded is True
        assert alert_above.threshold_difference == pytest.approx(0.1, rel=1e-3)

    def test_alert_to_float32_dict(self):
        """Float32辞書変換のテスト"""
        now = datetime.now(timezone.utc)
        
        alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            threshold_value=150.123456,
            current_value=150.234567,
            condition="price > 150.0",
            acknowledged=True
        )
        
        float32_dict = alert.to_float32_dict()
        
        assert float32_dict["symbol"] == "USDJPY"
        assert float32_dict["timestamp"] == now
        assert float32_dict["alert_type"] == "PRICE_THRESHOLD"
        assert float32_dict["severity"] == "WARNING"
        assert float32_dict["message"] == "Test alert"
        assert isinstance(float32_dict["threshold_value"], np.float32)
        assert isinstance(float32_dict["current_value"], np.float32)
        assert float32_dict["condition"] == "price > 150.0"
        assert float32_dict["acknowledged"] is True

    def test_alert_with_metadata(self):
        """メタデータ付きAlertのテスト"""
        now = datetime.now(timezone.utc)
        
        metadata = {
            "source": "price_monitor",
            "strategy": "breakout",
            "timeframe": "H1"
        }
        
        alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.PATTERN_DETECTED,
            severity=AlertSeverity.INFO,
            message="Breakout pattern detected",
            metadata=metadata
        )
        
        assert alert.metadata == metadata
        assert alert.metadata["source"] == "price_monitor"

    def test_alert_acknowledged_flag(self):
        """確認済みフラグのテスト"""
        now = datetime.now(timezone.utc)
        
        # デフォルトは未確認
        alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.RISK_WARNING,
            severity=AlertSeverity.CRITICAL,
            message="High risk alert"
        )
        assert alert.acknowledged is False
        
        # 確認済みとして作成
        acknowledged_alert = Alert(
            symbol="USDJPY",
            timestamp=now,
            alert_type=AlertType.RISK_WARNING,
            severity=AlertSeverity.CRITICAL,
            message="High risk alert",
            acknowledged=True
        )
        assert acknowledged_alert.acknowledged is True

    def test_alert_message_validation(self):
        """メッセージ長のバリデーションテスト"""
        now = datetime.now(timezone.utc)
        
        # 空のメッセージでエラー
        with pytest.raises(ValidationError) as exc_info:
            Alert(
                symbol="USDJPY",
                timestamp=now,
                alert_type=AlertType.RISK_WARNING,
                severity=AlertSeverity.WARNING,
                message=""  # 空文字列
            )
        assert "at least 1 character" in str(exc_info.value)
        
        # 長すぎるメッセージでエラー
        long_message = "x" * 501
        with pytest.raises(ValidationError) as exc_info:
            Alert(
                symbol="USDJPY",
                timestamp=now,
                alert_type=AlertType.RISK_WARNING,
                severity=AlertSeverity.WARNING,
                message=long_message
            )
        assert "at most 500 characters" in str(exc_info.value)


class TestEnums:
    """Enumクラスのテスト"""

    def test_prediction_type_enum(self):
        """PredictionType Enumのテスト"""
        assert PredictionType.PRICE.value == "PRICE"
        assert PredictionType.DIRECTION.value == "DIRECTION"
        assert PredictionType.VOLATILITY.value == "VOLATILITY"
        
        # Enumメンバーの存在確認
        assert hasattr(PredictionType, "PRICE")
        assert hasattr(PredictionType, "DIRECTION")
        assert hasattr(PredictionType, "VOLATILITY")

    def test_alert_type_enum(self):
        """AlertType Enumのテスト"""
        assert AlertType.PRICE_THRESHOLD.value == "PRICE_THRESHOLD"
        assert AlertType.PATTERN_DETECTED.value == "PATTERN_DETECTED"
        assert AlertType.RISK_WARNING.value == "RISK_WARNING"
        
        # Enumメンバーの存在確認
        assert hasattr(AlertType, "PRICE_THRESHOLD")
        assert hasattr(AlertType, "PATTERN_DETECTED")
        assert hasattr(AlertType, "RISK_WARNING")

    def test_alert_severity_enum(self):
        """AlertSeverity Enumのテスト"""
        assert AlertSeverity.INFO.value == "INFO"
        assert AlertSeverity.WARNING.value == "WARNING"
        assert AlertSeverity.CRITICAL.value == "CRITICAL"
        
        # Enumメンバーの存在確認
        assert hasattr(AlertSeverity, "INFO")
        assert hasattr(AlertSeverity, "WARNING")
        assert hasattr(AlertSeverity, "CRITICAL")