"""Tests for OHLC model.

OHLCデータモデルのテストを実施します。
TimeFrameの定義、Float32制約、価格の論理的整合性、プロパティメソッドの動作を検証します。
"""

from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError

from src.common.models import OHLC, TimeFrame


class TestTimeFrame:
    """TimeFrame Enumのテスト"""
    
    def test_timeframe_values(self):
        """定義された時間足の値を確認"""
        assert TimeFrame.M1.value == "M1"
        assert TimeFrame.M5.value == "M5"
        assert TimeFrame.M15.value == "M15"
        assert TimeFrame.M30.value == "M30"
        assert TimeFrame.H1.value == "H1"
        assert TimeFrame.H4.value == "H4"
        assert TimeFrame.D1.value == "D1"
        assert TimeFrame.W1.value == "W1"
        assert TimeFrame.MN.value == "MN"
    
    def test_timeframe_from_string(self):
        """文字列からTimeFrameへの変換"""
        assert TimeFrame("M1") == TimeFrame.M1
        assert TimeFrame("H1") == TimeFrame.H1
        assert TimeFrame("D1") == TimeFrame.D1


class TestOHLCModel:
    """OHLCモデルのテスト"""
    
    def test_valid_ohlc_creation(self):
        """有効なOHLCデータの作成"""
        ohlc = OHLC(
            timestamp=datetime(2025, 1, 17, 12, 0, 0),
            symbol="USDJPY",
            timeframe=TimeFrame.H1,
            open=150.100,
            high=150.500,
            low=149.900,
            close=150.300,
            volume=10000.0
        )
        
        assert ohlc.timestamp == datetime(2025, 1, 17, 12, 0, 0)
        assert ohlc.symbol == "USDJPY"
        assert ohlc.timeframe == TimeFrame.H1
        # Float32精度での比較
        assert ohlc.open == float(np.float32(150.100))
        assert ohlc.high == float(np.float32(150.500))
        assert ohlc.low == float(np.float32(149.900))
        assert ohlc.close == float(np.float32(150.300))
        assert ohlc.volume == float(np.float32(10000.0))
    
    def test_float32_constraint(self):
        """Float32型への変換制約"""
        # 大きな精度の値を与える
        high_precision_value = 150.123456789123456
        
        ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="EURUSD",
            timeframe=TimeFrame.M5,
            open=high_precision_value,
            high=high_precision_value + 0.001,
            low=high_precision_value - 0.001,
            close=high_precision_value,
            volume=1000.123456789
        )
        
        # Float32精度に丸められていることを確認
        assert ohlc.open == float(np.float32(high_precision_value))
        assert ohlc.high == float(np.float32(high_precision_value + 0.001))
        assert ohlc.low == float(np.float32(high_precision_value - 0.001))
        assert ohlc.close == float(np.float32(high_precision_value))
        assert ohlc.volume == float(np.float32(1000.123456789))
    
    def test_symbol_normalization(self):
        """シンボルの大文字正規化"""
        ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="eurusd",
            timeframe=TimeFrame.M1,
            open=1.0800,
            high=1.0850,
            low=1.0750,
            close=1.0820
        )
        
        assert ohlc.symbol == "EURUSD"
    
    def test_invalid_high_validation(self):
        """高値の検証エラー"""
        # 高値が開値より低い
        with pytest.raises(ValidationError) as exc_info:
            OHLC(
                timestamp=datetime.now(),
                symbol="USDJPY",
                timeframe=TimeFrame.H1,
                open=150.100,
                high=150.050,  # 開値より低い
                low=149.900,
                close=150.075
            )
        assert "High price must be greater than or equal to open price" in str(exc_info.value)
        
        # 高値が終値より低い（closeのバリデーションで検証）
        with pytest.raises(ValidationError) as exc_info:
            OHLC(
                timestamp=datetime.now(),
                symbol="USDJPY",
                timeframe=TimeFrame.H1,
                open=150.100,
                high=150.200,
                low=149.900,
                close=150.300  # 高値より高い
            )
        assert "Close price must be less than or equal to high price" in str(exc_info.value)
        
        # 高値が安値より低い
        with pytest.raises(ValidationError) as exc_info:
            OHLC(
                timestamp=datetime.now(),
                symbol="USDJPY",
                timeframe=TimeFrame.H1,
                open=150.100,
                low=150.500,  # 開値より高い（まずlowのバリデーションでエラー）
                high=150.200,
                close=150.100
            )
        # lowが先にバリデーションされるため、lowのエラーが発生する
        assert "Low price must be less than or equal to open price" in str(exc_info.value)
    
    def test_invalid_low_validation(self):
        """安値の検証エラー"""
        # 安値が開値より高い
        with pytest.raises(ValidationError) as exc_info:
            OHLC(
                timestamp=datetime.now(),
                symbol="USDJPY",
                timeframe=TimeFrame.H1,
                open=150.100,
                high=150.500,
                low=150.200,  # 開値より高い
                close=150.150
            )
        assert "Low price must be less than or equal to open price" in str(exc_info.value)
        
        # 安値が終値より高い（closeのバリデーションで検証）
        with pytest.raises(ValidationError) as exc_info:
            OHLC(
                timestamp=datetime.now(),
                symbol="USDJPY",
                timeframe=TimeFrame.H1,
                open=150.300,
                high=150.500,
                low=150.200,
                close=150.100  # 安値より低い
            )
        assert "Close price must be greater than or equal to low price" in str(exc_info.value)
    
    def test_range_property(self):
        """価格レンジの計算"""
        ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="GBPUSD",
            timeframe=TimeFrame.H4,
            open=1.2700,
            high=1.2750,
            low=1.2650,
            close=1.2720
        )
        
        # Float32精度で計算される値と比較
        high_f32 = float(np.float32(1.2750))
        low_f32 = float(np.float32(1.2650))
        expected_range = float(np.float32(high_f32 - low_f32))
        
        # rangeはFloat32精度で計算されているため、同じ精度で比較
        assert abs(ohlc.range - expected_range) < 1e-7
        assert isinstance(ohlc.range, float)
    
    def test_is_bullish_property(self):
        """陽線判定"""
        # 陽線（終値 > 開値）
        bullish_ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="USDJPY",
            timeframe=TimeFrame.D1,
            open=150.000,
            high=150.500,
            low=149.900,
            close=150.300
        )
        assert bullish_ohlc.is_bullish is True
        assert bullish_ohlc.is_bearish is False
        
        # 陰線（終値 < 開値）
        bearish_ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="USDJPY",
            timeframe=TimeFrame.D1,
            open=150.300,
            high=150.500,
            low=149.900,
            close=150.000
        )
        assert bearish_ohlc.is_bullish is False
        assert bearish_ohlc.is_bearish is True
        
        # 同値（終値 = 開値）
        doji_ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="USDJPY",
            timeframe=TimeFrame.D1,
            open=150.100,
            high=150.500,
            low=149.900,
            close=150.100
        )
        assert doji_ohlc.is_bullish is False
        assert doji_ohlc.is_bearish is False
    
    def test_body_size_property(self):
        """実体サイズの計算"""
        ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="EURUSD",
            timeframe=TimeFrame.H1,
            open=1.0800,
            high=1.0850,
            low=1.0750,
            close=1.0820
        )
        
        # Float32精度で計算される値と比較
        close_f32 = float(np.float32(1.0820))
        open_f32 = float(np.float32(1.0800))
        expected_body = float(np.float32(abs(close_f32 - open_f32)))
        
        # body_sizeはFloat32精度で計算されているため、同じ精度で比較
        assert abs(ohlc.body_size - expected_body) < 1e-7
        assert isinstance(ohlc.body_size, float)
    
    def test_shadows_properties(self):
        """上髭・下髭の計算"""
        # 陽線のケース
        bullish_ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="GBPJPY",
            timeframe=TimeFrame.M15,
            open=185.000,
            high=185.500,  # 上髭: 185.500 - 185.300 = 0.200
            low=184.900,   # 下髭: 185.000 - 184.900 = 0.100
            close=185.300
        )
        
        # Float32精度での計算を確認
        high_f32 = float(np.float32(185.500))
        close_f32 = float(np.float32(185.300))
        open_f32 = float(np.float32(185.000))
        low_f32 = float(np.float32(184.900))
        
        expected_upper = float(np.float32(high_f32 - close_f32))  # 陽線なのでclose
        expected_lower = float(np.float32(open_f32 - low_f32))    # 陽線なのでopen
        
        assert abs(bullish_ohlc.upper_shadow - expected_upper) < 1e-5
        assert abs(bullish_ohlc.lower_shadow - expected_lower) < 1e-5
        
        # 陰線のケース
        bearish_ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="GBPJPY",
            timeframe=TimeFrame.M15,
            open=185.300,
            high=185.500,  # 上髭: 185.500 - 185.300 = 0.200
            low=184.900,   # 下髭: 185.000 - 184.900 = 0.100
            close=185.000
        )
        
        high_f32 = float(np.float32(185.500))
        open_f32 = float(np.float32(185.300))
        close_f32 = float(np.float32(185.000))
        low_f32 = float(np.float32(184.900))
        
        expected_upper = float(np.float32(high_f32 - open_f32))  # 陰線なのでopen
        expected_lower = float(np.float32(close_f32 - low_f32))  # 陰線なのでclose
        
        assert abs(bearish_ohlc.upper_shadow - expected_upper) < 1e-5
        assert abs(bearish_ohlc.lower_shadow - expected_lower) < 1e-5
    
    def test_to_float32_dict(self):
        """Float32辞書への変換"""
        ohlc = OHLC(
            timestamp=datetime(2025, 1, 17, 12, 0, 0),
            symbol="AUDUSD",
            timeframe=TimeFrame.W1,
            open=0.6500,
            high=0.6550,
            low=0.6450,
            close=0.6520,
            volume=50000.0
        )
        
        result = ohlc.to_float32_dict()
        
        assert result["timestamp"] == datetime(2025, 1, 17, 12, 0, 0)
        assert result["symbol"] == "AUDUSD"
        assert result["timeframe"] == "W1"
        assert isinstance(result["open"], np.float32)
        assert isinstance(result["high"], np.float32)
        assert isinstance(result["low"], np.float32)
        assert isinstance(result["close"], np.float32)
        assert isinstance(result["volume"], np.float32)
        assert result["open"] == np.float32(0.6500)
        assert result["high"] == np.float32(0.6550)
        assert result["low"] == np.float32(0.6450)
        assert result["close"] == np.float32(0.6520)
        assert result["volume"] == np.float32(50000.0)
    
    def test_negative_prices_rejected(self):
        """負の価格値の拒否"""
        with pytest.raises(ValidationError):
            OHLC(
                timestamp=datetime.now(),
                symbol="USDJPY",
                timeframe=TimeFrame.H1,
                open=-150.100,  # 負の値
                high=150.500,
                low=149.900,
                close=150.300
            )
    
    def test_zero_volume_allowed(self):
        """ゼロボリュームの許可"""
        ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="USDJPY",
            timeframe=TimeFrame.MN,
            open=150.100,
            high=150.500,
            low=149.900,
            close=150.300,
            volume=0.0  # ゼロボリューム
        )
        
        assert ohlc.volume == 0.0
    
    def test_extreme_values(self):
        """極端な値のテスト"""
        # 非常に小さい値
        small_ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="BTCUSD",
            timeframe=TimeFrame.M1,
            open=0.00001,
            high=0.00002,
            low=0.000005,
            close=0.000015
        )
        assert small_ohlc.open > 0
        
        # 非常に大きい値
        large_ohlc = OHLC(
            timestamp=datetime.now(),
            symbol="BTCJPY",
            timeframe=TimeFrame.D1,
            open=5000000.0,
            high=5100000.0,
            low=4900000.0,
            close=5050000.0
        )
        assert large_ohlc.high == float(np.float32(5100000.0))