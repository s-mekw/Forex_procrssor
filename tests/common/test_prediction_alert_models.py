"""PredictionとAlertモデルの包括的なテスト

このテストファイルは、PredictionとAlertモデルのバリデーション、
プロパティメソッド、エッジケースを網羅的にテストします。
"""

import pytest
from datetime import datetime, timedelta, timezone
import numpy as np
from pydantic import ValidationError

from src.common.models import (
    Prediction, PredictionType,
    Alert, AlertType, AlertSeverity
)


class TestPredictionModel:
    """Predictionモデルのテストクラス"""
    
    @pytest.fixture
    def valid_prediction_data(self):
        """有効なPredictionデータのフィクスチャ"""
        now = datetime(2025, 1, 17, 12, 0, 0, tzinfo=timezone.utc)
        return {
            "symbol": "usdjpy",  # 小文字でテスト（正規化確認）
            "predicted_at": now,
            "target_timestamp": now + timedelta(hours=1),
            "prediction_type": PredictionType.PRICE,
            "predicted_value": 150.500,
            "confidence_score": 0.85,
            "confidence_upper": 150.600,
            "confidence_lower": 150.400,
            "model_version": "v1.2.3"
        }
    
    def test_prediction_creation_with_valid_data(self, valid_prediction_data):
        """有効なデータでPredictionインスタンスが作成できることを確認"""
        prediction = Prediction(**valid_prediction_data)
        assert prediction.symbol == "USDJPY"  # 大文字に正規化
        assert prediction.predicted_at == valid_prediction_data["predicted_at"]
        assert prediction.target_timestamp == valid_prediction_data["target_timestamp"]
        assert prediction.prediction_type == PredictionType.PRICE
        assert np.isclose(prediction.predicted_value, np.float32(150.500))
        assert np.isclose(prediction.confidence_score, np.float32(0.85))
        assert np.isclose(prediction.confidence_upper, np.float32(150.600))
        assert np.isclose(prediction.confidence_lower, np.float32(150.400))
    
    def test_float32_conversion(self, valid_prediction_data):
        """Float32への変換が正しく行われることを確認"""
        prediction = Prediction(**valid_prediction_data)
        # 高精度の値を設定してFloat32に丸められることを確認
        high_precision_data = valid_prediction_data.copy()
        high_precision_data["predicted_value"] = 150.123456789
        high_precision_data["confidence_score"] = 0.987654321
        prediction = Prediction(**high_precision_data)
        assert prediction.predicted_value == float(np.float32(150.123456789))
        assert prediction.confidence_score == float(np.float32(0.987654321))
    
    def test_symbol_normalization(self, valid_prediction_data):
        """シンボルが大文字に正規化されることを確認"""
        data = valid_prediction_data.copy()
        data["symbol"] = "eurusd"
        prediction = Prediction(**data)
        assert prediction.symbol == "EURUSD"
    
    def test_confidence_score_validation(self, valid_prediction_data):
        """信頼度スコアの範囲検証"""
        # 範囲外の値（> 1.0）
        invalid_data = valid_prediction_data.copy()
        invalid_data["confidence_score"] = 1.5
        
        with pytest.raises(ValidationError) as exc_info:
            Prediction(**invalid_data)
        assert "less than or equal to 1" in str(exc_info.value).lower()
        
        # 範囲外の値（< 0.0）
        invalid_data["confidence_score"] = -0.1
        
        with pytest.raises(ValidationError) as exc_info:
            Prediction(**invalid_data)
        assert "greater than or equal to 0" in str(exc_info.value).lower()
    
    def test_confidence_interval_validation(self, valid_prediction_data):
        """信頼区間の妥当性検証"""
        # upperがlowerより小さい場合
        invalid_data = valid_prediction_data.copy()
        invalid_data["confidence_upper"] = 150.300
        invalid_data["confidence_lower"] = 150.500  # upperより大きい
        
        with pytest.raises(ValidationError) as exc_info:
            Prediction(**invalid_data)
        assert "Confidence upper must be greater than or equal to confidence lower" in str(exc_info.value)
    
    def test_target_timestamp_validation(self, valid_prediction_data):
        """予測対象時刻が予測実行時刻より未来であることの検証"""
        invalid_data = valid_prediction_data.copy()
        # target_timestampをpredicted_atと同じにする
        invalid_data["target_timestamp"] = invalid_data["predicted_at"]
        
        with pytest.raises(ValidationError) as exc_info:
            Prediction(**invalid_data)
        assert "Target timestamp must be after predicted_at timestamp" in str(exc_info.value)
        
        # target_timestampをpredicted_atより過去にする
        invalid_data["target_timestamp"] = invalid_data["predicted_at"] - timedelta(hours=1)
        
        with pytest.raises(ValidationError) as exc_info:
            Prediction(**invalid_data)
        assert "Target timestamp must be after predicted_at timestamp" in str(exc_info.value)
    
    def test_confidence_range_property(self, valid_prediction_data):
        """信頼区間の幅計算が正しいことを確認"""
        prediction = Prediction(**valid_prediction_data)
        expected_range = float(np.float32(150.600 - 150.400))
        assert np.isclose(prediction.confidence_range, expected_range, rtol=1e-3)
        
        # 信頼区間が設定されていない場合
        data = valid_prediction_data.copy()
        del data["confidence_upper"]
        del data["confidence_lower"]
        prediction = Prediction(**data)
        assert prediction.confidence_range is None
    
    def test_is_high_confidence_property(self, valid_prediction_data):
        """高信頼度判定が正しいことを確認"""
        # 高信頼度（>= 0.7）
        prediction = Prediction(**valid_prediction_data)
        assert prediction.is_high_confidence is True
        
        # 低信頼度（< 0.7）
        low_confidence_data = valid_prediction_data.copy()
        low_confidence_data["confidence_score"] = 0.6
        prediction = Prediction(**low_confidence_data)
        assert prediction.is_high_confidence is False
        
        # 境界値（= 0.7）
        boundary_data = valid_prediction_data.copy()
        boundary_data["confidence_score"] = 0.70001  # Float32では0.7以上になる
        prediction = Prediction(**boundary_data)
        assert prediction.is_high_confidence is True
    
    def test_prediction_horizon_hours_property(self, valid_prediction_data):
        """予測ホライゾン計算が正しいことを確認"""
        prediction = Prediction(**valid_prediction_data)
        # 1時間後を予測
        assert prediction.prediction_horizon_hours == 1.0
        
        # 24時間後を予測
        data = valid_prediction_data.copy()
        data["target_timestamp"] = data["predicted_at"] + timedelta(days=1)
        prediction = Prediction(**data)
        assert prediction.prediction_horizon_hours == 24.0
        
        # 30分後を予測
        data["target_timestamp"] = data["predicted_at"] + timedelta(minutes=30)
        prediction = Prediction(**data)
        assert prediction.prediction_horizon_hours == 0.5
    
    def test_all_prediction_types(self, valid_prediction_data):
        """全ての予測タイプが正しく処理されることを確認"""
        for pred_type in PredictionType:
            data = valid_prediction_data.copy()
            data["prediction_type"] = pred_type
            prediction = Prediction(**data)
            assert prediction.prediction_type == pred_type
    
    def test_optional_fields(self, valid_prediction_data):
        """オプショナルフィールドの処理を確認"""
        # 最小限の必須フィールドのみ
        minimal_data = {
            "symbol": "USDJPY",
            "predicted_at": datetime.now(timezone.utc),
            "target_timestamp": datetime.now(timezone.utc) + timedelta(hours=1),
            "prediction_type": PredictionType.DIRECTION,
            "predicted_value": 1.0,  # 1: up, -1: down
            "confidence_score": 0.75
        }
        prediction = Prediction(**minimal_data)
        assert prediction.model_version is None
        assert prediction.metadata is None
        assert prediction.confidence_upper is None
        assert prediction.confidence_lower is None
    
    def test_metadata_field(self, valid_prediction_data):
        """メタデータフィールドの処理を確認"""
        data = valid_prediction_data.copy()
        data["metadata"] = {
            "model_name": "LSTM",
            "features_used": ["price", "volume", "indicators"],
            "training_window": 30
        }
        prediction = Prediction(**data)
        assert prediction.metadata["model_name"] == "LSTM"
        assert len(prediction.metadata["features_used"]) == 3
    
    def test_to_float32_dict(self, valid_prediction_data):
        """Float32辞書への変換が正しいことを確認"""
        prediction = Prediction(**valid_prediction_data)
        float32_dict = prediction.to_float32_dict()
        
        assert float32_dict["symbol"] == "USDJPY"
        assert float32_dict["prediction_type"] == "PRICE"
        assert isinstance(float32_dict["predicted_value"], np.float32)
        assert isinstance(float32_dict["confidence_score"], np.float32)
        assert isinstance(float32_dict["confidence_upper"], np.float32)
        assert isinstance(float32_dict["confidence_lower"], np.float32)
        
        # オプショナルフィールドが含まれていることを確認
        assert "model_version" in float32_dict
        assert "metadata" in float32_dict


class TestAlertModel:
    """Alertモデルのテストクラス"""
    
    @pytest.fixture
    def valid_alert_data(self):
        """有効なAlertデータのフィクスチャ"""
        return {
            "symbol": "usdjpy",  # 小文字でテスト（正規化確認）
            "timestamp": datetime(2025, 1, 17, 12, 0, 0, tzinfo=timezone.utc),
            "alert_type": AlertType.PRICE_THRESHOLD,
            "severity": AlertSeverity.WARNING,
            "message": "USDJPY price exceeded threshold",
            "threshold_value": 150.000,
            "current_value": 150.123,
            "condition": "price > 150.000",
            "acknowledged": False
        }
    
    def test_alert_creation_with_valid_data(self, valid_alert_data):
        """有効なデータでAlertインスタンスが作成できることを確認"""
        alert = Alert(**valid_alert_data)
        assert alert.symbol == "USDJPY"  # 大文字に正規化
        assert alert.timestamp == valid_alert_data["timestamp"]
        assert alert.alert_type == AlertType.PRICE_THRESHOLD
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == valid_alert_data["message"]
        assert np.isclose(alert.threshold_value, np.float32(150.000))
        assert np.isclose(alert.current_value, np.float32(150.123))
        assert alert.acknowledged is False
    
    def test_float32_conversion(self, valid_alert_data):
        """Float32への変換が正しく行われることを確認"""
        alert = Alert(**valid_alert_data)
        # 高精度の値を設定してFloat32に丸められることを確認
        high_precision_data = valid_alert_data.copy()
        high_precision_data["threshold_value"] = 150.123456789
        high_precision_data["current_value"] = 150.987654321
        alert = Alert(**high_precision_data)
        assert alert.threshold_value == float(np.float32(150.123456789))
        assert alert.current_value == float(np.float32(150.987654321))
    
    def test_symbol_normalization(self, valid_alert_data):
        """シンボルが大文字に正規化されることを確認"""
        data = valid_alert_data.copy()
        data["symbol"] = "eurusd"
        alert = Alert(**data)
        assert alert.symbol == "EURUSD"
    
    def test_price_threshold_validation(self, valid_alert_data):
        """価格閾値アラートのバリデーション"""
        # PRICE_THRESHOLDタイプでthreshold_valueがない場合
        invalid_data = valid_alert_data.copy()
        invalid_data["alert_type"] = AlertType.PRICE_THRESHOLD
        invalid_data["current_value"] = 150.123
        del invalid_data["threshold_value"]
        
        with pytest.raises(ValidationError) as exc_info:
            Alert(**invalid_data)
        assert "Threshold value is required for price threshold alerts" in str(exc_info.value)
    
    def test_is_critical_property(self):
        """緊急レベル判定が正しいことを確認"""
        data = {
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "alert_type": AlertType.RISK_WARNING,
            "severity": AlertSeverity.CRITICAL,
            "message": "Critical risk detected"
        }
        alert = Alert(**data)
        assert alert.is_critical is True
        assert alert.is_warning is False
        assert alert.is_info is False
    
    def test_is_warning_property(self):
        """警告レベル判定が正しいことを確認"""
        data = {
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "alert_type": AlertType.PATTERN_DETECTED,
            "severity": AlertSeverity.WARNING,
            "message": "Pattern detected"
        }
        alert = Alert(**data)
        assert alert.is_warning is True
        assert alert.is_critical is False
        assert alert.is_info is False
    
    def test_is_info_property(self):
        """情報レベル判定が正しいことを確認"""
        data = {
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "alert_type": AlertType.PATTERN_DETECTED,
            "severity": AlertSeverity.INFO,
            "message": "Information alert"
        }
        alert = Alert(**data)
        assert alert.is_info is True
        assert alert.is_critical is False
        assert alert.is_warning is False
    
    def test_threshold_exceeded_property(self, valid_alert_data):
        """閾値超過判定が正しいことを確認"""
        # 閾値を超過している場合
        alert = Alert(**valid_alert_data)
        assert alert.threshold_exceeded is True
        
        # 閾値を超過していない場合
        data = valid_alert_data.copy()
        data["current_value"] = 149.999
        alert = Alert(**data)
        assert alert.threshold_exceeded is False
        
        # 閾値と同じ場合
        data["current_value"] = 150.000
        alert = Alert(**data)
        assert alert.threshold_exceeded is False
        
        # PRICE_THRESHOLD以外のタイプの場合
        data["alert_type"] = AlertType.PATTERN_DETECTED
        alert = Alert(**data)
        assert alert.threshold_exceeded is None
    
    def test_threshold_difference_property(self, valid_alert_data):
        """閾値との差分計算が正しいことを確認"""
        alert = Alert(**valid_alert_data)
        expected_diff = float(np.float32(150.123 - 150.000))
        assert np.isclose(alert.threshold_difference, expected_diff, rtol=1e-3)
        
        # 閾値を下回る場合（負の差分）
        data = valid_alert_data.copy()
        data["current_value"] = 149.900
        alert = Alert(**data)
        expected_diff = float(np.float32(149.900 - 150.000))
        assert np.isclose(alert.threshold_difference, expected_diff, rtol=1e-3)
        
        # 閾値情報がない場合
        data = {
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "alert_type": AlertType.PATTERN_DETECTED,
            "severity": AlertSeverity.INFO,
            "message": "Pattern detected"
        }
        alert = Alert(**data)
        assert alert.threshold_difference is None
    
    def test_all_alert_types(self):
        """全てのアラートタイプが正しく処理されることを確認"""
        base_data = {
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "severity": AlertSeverity.INFO,
            "message": "Test alert"
        }
        
        for alert_type in AlertType:
            data = base_data.copy()
            data["alert_type"] = alert_type
            alert = Alert(**data)
            assert alert.alert_type == alert_type
    
    def test_all_severity_levels(self):
        """全ての重要度レベルが正しく処理されることを確認"""
        base_data = {
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "alert_type": AlertType.PATTERN_DETECTED,
            "message": "Test alert"
        }
        
        for severity in AlertSeverity:
            data = base_data.copy()
            data["severity"] = severity
            alert = Alert(**data)
            assert alert.severity == severity
    
    def test_optional_fields(self):
        """オプショナルフィールドの処理を確認"""
        # 最小限の必須フィールドのみ
        minimal_data = {
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "alert_type": AlertType.PATTERN_DETECTED,
            "severity": AlertSeverity.INFO,
            "message": "Pattern detected"
        }
        alert = Alert(**minimal_data)
        assert alert.threshold_value is None
        assert alert.current_value is None
        assert alert.condition is None
        assert alert.metadata is None
        assert alert.acknowledged is False  # デフォルト値
    
    def test_metadata_field(self, valid_alert_data):
        """メタデータフィールドの処理を確認"""
        data = valid_alert_data.copy()
        data["metadata"] = {
            "pattern_name": "Head and Shoulders",
            "confidence": 0.85,
            "timeframe": "H1"
        }
        alert = Alert(**data)
        assert alert.metadata["pattern_name"] == "Head and Shoulders"
        assert alert.metadata["confidence"] == 0.85
    
    def test_acknowledged_flag(self, valid_alert_data):
        """確認済みフラグの処理を確認"""
        # デフォルトはFalse
        data = valid_alert_data.copy()
        del data["acknowledged"]
        alert = Alert(**data)
        assert alert.acknowledged is False
        
        # Trueに設定
        data["acknowledged"] = True
        alert = Alert(**data)
        assert alert.acknowledged is True
    
    def test_message_validation(self):
        """メッセージの長さ制限を確認"""
        data = {
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "alert_type": AlertType.RISK_WARNING,
            "severity": AlertSeverity.WARNING,
            "message": ""  # 空文字列
        }
        
        with pytest.raises(ValidationError) as exc_info:
            Alert(**data)
        assert "at least 1 character" in str(exc_info.value).lower()
        
        # 500文字を超えるメッセージ
        data["message"] = "x" * 501
        with pytest.raises(ValidationError) as exc_info:
            Alert(**data)
        assert "at most 500 character" in str(exc_info.value).lower()
    
    def test_to_float32_dict(self, valid_alert_data):
        """Float32辞書への変換が正しいことを確認"""
        alert = Alert(**valid_alert_data)
        float32_dict = alert.to_float32_dict()
        
        assert float32_dict["symbol"] == "USDJPY"
        assert float32_dict["alert_type"] == "PRICE_THRESHOLD"
        assert float32_dict["severity"] == "WARNING"
        assert isinstance(float32_dict["threshold_value"], np.float32)
        assert isinstance(float32_dict["current_value"], np.float32)
        assert float32_dict["acknowledged"] is False
        
        # オプショナルフィールドが含まれていることを確認
        assert "condition" in float32_dict
        assert "metadata" in float32_dict
    
    def test_complex_alert_scenario(self):
        """複雑なアラートシナリオをテスト"""
        # パターン検出アラート
        pattern_alert = Alert(
            symbol="EURUSD",
            timestamp=datetime.now(timezone.utc),
            alert_type=AlertType.PATTERN_DETECTED,
            severity=AlertSeverity.INFO,
            message="Bullish flag pattern detected on EURUSD H4",
            condition="flag_pattern_confidence > 0.8",
            metadata={
                "pattern": "bullish_flag",
                "confidence": 0.85,
                "timeframe": "H4"
            }
        )
        assert pattern_alert.is_info
        assert pattern_alert.threshold_exceeded is None
        
        # リスク警告アラート
        risk_alert = Alert(
            symbol="GBPJPY",
            timestamp=datetime.now(timezone.utc),
            alert_type=AlertType.RISK_WARNING,
            severity=AlertSeverity.CRITICAL,
            message="High volatility detected - risk level critical",
            metadata={
                "volatility": 2.5,
                "risk_score": 0.95
            },
            acknowledged=False
        )
        assert risk_alert.is_critical
        assert not risk_alert.acknowledged