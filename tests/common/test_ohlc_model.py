"""OHLCモデルの包括的なテスト

このテストファイルは、OHLCモデルのバリデーション、プロパティメソッド、
エッジケースを網羅的にテストします。
"""

import pytest
from datetime import datetime, timezone
import numpy as np
from pydantic import ValidationError

from src.common.models import OHLC, TimeFrame


class TestOHLCModel:
    """OHLCモデルのテストクラス"""
    
    @pytest.fixture
    def valid_ohlc_data(self):
        """有効なOHLCデータのフィクスチャ"""
        return {
            "timestamp": datetime(2025, 1, 17, 12, 0, 0, tzinfo=timezone.utc),
            "symbol": "usdjpy",  # 小文字でテスト（正規化確認）
            "timeframe": TimeFrame.H1,
            "open": 150.100,
            "high": 150.500,
            "low": 149.900,
            "close": 150.300,
            "volume": 10000.0
        }
    
    def test_ohlc_creation_with_valid_data(self, valid_ohlc_data):
        """有効なデータでOHLCインスタンスが作成できることを確認"""
        ohlc = OHLC(**valid_ohlc_data)
        assert ohlc.timestamp == valid_ohlc_data["timestamp"]
        assert ohlc.symbol == "USDJPY"  # 大文字に正規化されることを確認
        assert ohlc.timeframe == TimeFrame.H1
        assert np.isclose(ohlc.open, np.float32(150.100))
        assert np.isclose(ohlc.high, np.float32(150.500))
        assert np.isclose(ohlc.low, np.float32(149.900))
        assert np.isclose(ohlc.close, np.float32(150.300))
        assert np.isclose(ohlc.volume, np.float32(10000.0))
    
    def test_float32_conversion(self, valid_ohlc_data):
        """Float32への変換が正しく行われることを確認"""
        ohlc = OHLC(**valid_ohlc_data)
        # 高精度の値を設定してFloat32に丸められることを確認
        high_precision_data = valid_ohlc_data.copy()
        high_precision_data["open"] = 150.123456789
        ohlc = OHLC(**high_precision_data)
        assert ohlc.open == float(np.float32(150.123456789))
    
    def test_symbol_normalization(self):
        """シンボルが大文字に正規化されることを確認"""
        data = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": "eurusd",
            "timeframe": TimeFrame.M5,
            "open": 1.0850,
            "high": 1.0860,
            "low": 1.0840,
            "close": 1.0855,
            "volume": 5000.0
        }
        ohlc = OHLC(**data)
        assert ohlc.symbol == "EURUSD"
    
    def test_validate_high_greater_than_low(self, valid_ohlc_data):
        """高値が安値より小さい場合にバリデーションエラーが発生することを確認"""
        invalid_data = valid_ohlc_data.copy()
        invalid_data["low"] = 150.000  # lowを先に設定
        invalid_data["high"] = 149.800  # lowより小さい値
        
        with pytest.raises(ValidationError) as exc_info:
            OHLC(**invalid_data)
        # openとの比較で先にエラーになる可能性がある
        error_msg = str(exc_info.value)
        assert ("High price must be greater than or equal to low price" in error_msg or 
                "High price must be greater than or equal to open price" in error_msg)
    
    def test_validate_high_greater_than_open(self, valid_ohlc_data):
        """高値が開値より小さい場合にバリデーションエラーが発生することを確認"""
        invalid_data = valid_ohlc_data.copy()
        invalid_data["open"] = 150.600
        invalid_data["high"] = 150.500  # openより小さい値
        
        with pytest.raises(ValidationError) as exc_info:
            OHLC(**invalid_data)
        assert "High price must be greater than or equal to open price" in str(exc_info.value)
    
    def test_validate_low_less_than_open(self, valid_ohlc_data):
        """安値が開値より大きい場合にバリデーションエラーが発生することを確認"""
        invalid_data = valid_ohlc_data.copy()
        invalid_data["open"] = 150.000
        invalid_data["low"] = 150.100  # openより大きい値
        invalid_data["high"] = 150.200  # 有効な高値
        
        with pytest.raises(ValidationError) as exc_info:
            OHLC(**invalid_data)
        assert "Low price must be less than or equal to open price" in str(exc_info.value)
    
    def test_validate_close_within_range(self, valid_ohlc_data):
        """終値が高値と安値の範囲外の場合にバリデーションエラーが発生することを確認"""
        # 終値が高値より大きい場合
        invalid_data = valid_ohlc_data.copy()
        invalid_data["close"] = 150.600  # highより大きい値
        
        with pytest.raises(ValidationError) as exc_info:
            OHLC(**invalid_data)
        assert "Close price must be less than or equal to high price" in str(exc_info.value)
        
        # 終値が安値より小さい場合
        invalid_data = valid_ohlc_data.copy()
        invalid_data["close"] = 149.800  # lowより小さい値
        
        with pytest.raises(ValidationError) as exc_info:
            OHLC(**invalid_data)
        assert "Close price must be greater than or equal to low price" in str(exc_info.value)
    
    def test_edge_case_all_prices_equal(self):
        """全ての価格が同じ場合（十字線）の処理を確認"""
        data = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": "USDJPY",
            "timeframe": TimeFrame.H1,
            "open": 150.000,
            "high": 150.000,
            "low": 150.000,
            "close": 150.000,
            "volume": 1000.0
        }
        ohlc = OHLC(**data)
        assert ohlc.range == 0.0
        assert not ohlc.is_bullish
        assert not ohlc.is_bearish
        assert ohlc.body_size == 0.0
        assert ohlc.upper_shadow == 0.0
        assert ohlc.lower_shadow == 0.0
    
    def test_range_property(self, valid_ohlc_data):
        """価格レンジ計算が正しいことを確認"""
        ohlc = OHLC(**valid_ohlc_data)
        expected_range = float(np.float32(150.500 - 149.900))
        assert np.isclose(ohlc.range, expected_range, rtol=1e-4)
    
    def test_is_bullish_property(self):
        """陽線判定が正しいことを確認"""
        # 陽線のケース
        bullish_data = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": "USDJPY",
            "timeframe": TimeFrame.H1,
            "open": 150.000,
            "high": 150.200,
            "low": 149.900,
            "close": 150.100,  # close > open
            "volume": 1000.0
        }
        ohlc = OHLC(**bullish_data)
        assert ohlc.is_bullish is True
        assert ohlc.is_bearish is False
    
    def test_is_bearish_property(self):
        """陰線判定が正しいことを確認"""
        # 陰線のケース
        bearish_data = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": "USDJPY",
            "timeframe": TimeFrame.H1,
            "open": 150.100,
            "high": 150.200,
            "low": 149.900,
            "close": 150.000,  # close < open
            "volume": 1000.0
        }
        ohlc = OHLC(**bearish_data)
        assert ohlc.is_bearish is True
        assert ohlc.is_bullish is False
    
    def test_body_size_property(self, valid_ohlc_data):
        """実体サイズ計算が正しいことを確認"""
        ohlc = OHLC(**valid_ohlc_data)
        expected_body = float(np.float32(abs(150.300 - 150.100)))
        assert np.isclose(ohlc.body_size, expected_body, rtol=1e-4)
    
    def test_upper_shadow_property(self):
        """上髭の長さ計算が正しいことを確認"""
        data = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": "USDJPY",
            "timeframe": TimeFrame.H1,
            "open": 150.000,
            "high": 150.500,  # 高値
            "low": 149.900,
            "close": 150.100,  # close > open (陽線)
            "volume": 1000.0
        }
        ohlc = OHLC(**data)
        # 上髭 = high - max(open, close) = 150.500 - 150.100
        expected_upper_shadow = float(np.float32(150.500 - 150.100))
        assert np.isclose(ohlc.upper_shadow, expected_upper_shadow, rtol=1e-4)
    
    def test_lower_shadow_property(self):
        """下髭の長さ計算が正しいことを確認"""
        data = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": "USDJPY",
            "timeframe": TimeFrame.H1,
            "open": 150.100,
            "high": 150.500,
            "low": 149.900,  # 安値
            "close": 150.000,  # close < open (陰線)
            "volume": 1000.0
        }
        ohlc = OHLC(**data)
        # 下髭 = min(open, close) - low = 150.000 - 149.900
        expected_lower_shadow = float(np.float32(150.000 - 149.900))
        assert np.isclose(ohlc.lower_shadow, expected_lower_shadow, rtol=1e-4)
    
    def test_to_float32_dict(self, valid_ohlc_data):
        """Float32辞書への変換が正しいことを確認"""
        ohlc = OHLC(**valid_ohlc_data)
        float32_dict = ohlc.to_float32_dict()
        
        assert float32_dict["symbol"] == "USDJPY"
        assert float32_dict["timeframe"] == "H1"
        assert isinstance(float32_dict["open"], np.float32)
        assert isinstance(float32_dict["high"], np.float32)
        assert isinstance(float32_dict["low"], np.float32)
        assert isinstance(float32_dict["close"], np.float32)
        assert isinstance(float32_dict["volume"], np.float32)
    
    def test_all_timeframes(self):
        """全ての時間足が正しく処理されることを確認"""
        base_data = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": "USDJPY",
            "open": 150.000,
            "high": 150.100,
            "low": 149.900,
            "close": 150.050,
            "volume": 1000.0
        }
        
        for timeframe in TimeFrame:
            data = base_data.copy()
            data["timeframe"] = timeframe
            ohlc = OHLC(**data)
            assert ohlc.timeframe == timeframe
    
    def test_negative_volume_validation(self, valid_ohlc_data):
        """負のボリュームが拒否されることを確認"""
        invalid_data = valid_ohlc_data.copy()
        invalid_data["volume"] = -1000.0
        
        with pytest.raises(ValidationError) as exc_info:
            OHLC(**invalid_data)
        assert "greater than or equal to 0" in str(exc_info.value).lower()
    
    def test_complex_validation_scenario(self):
        """複雑なバリデーションシナリオをテスト"""
        # 有効だが極端な値のケース
        extreme_data = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": "USDJPY",
            "timeframe": TimeFrame.D1,
            "open": 100.000,
            "high": 200.000,  # 極端に高い
            "low": 50.000,    # 極端に低い
            "close": 150.000,  # 範囲内
            "volume": 1000000.0
        }
        ohlc = OHLC(**extreme_data)
        assert ohlc.range == float(np.float32(150.000))
        assert ohlc.is_bullish  # close > open
        
    def test_validation_order_dependency(self):
        """バリデーションの順序依存性をテスト"""
        # フィールドの定義順序に関わらず正しくバリデーションされることを確認
        data = {
            "close": 150.300,  # 先にcloseを指定
            "low": 149.900,
            "high": 150.500,
            "open": 150.100,
            "timeframe": TimeFrame.H1,
            "symbol": "USDJPY",
            "timestamp": datetime.now(timezone.utc),
            "volume": 1000.0
        }
        ohlc = OHLC(**data)
        assert ohlc.close == float(np.float32(150.300))