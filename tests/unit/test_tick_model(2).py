"""
Tickモデルのユニットテスト

Tickデータモデルのバリデーション、Float32変換、
プロパティ計算のテストを行います。
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
import numpy as np

from src.common.models import Tick


class TestTickModel:
    """Tickモデルのテストクラス"""
    
    def test_valid_tick_creation(self):
        """正常なTickインスタンスの作成テスト"""
        tick = Tick(
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
            symbol="USDJPY",
            bid=149.500,
            ask=149.505,
            volume=1000.0
        )
        
        assert tick.timestamp == datetime(2025, 1, 15, 10, 30, 0)
        assert tick.symbol == "USDJPY"
        assert tick.bid == pytest.approx(149.500, rel=1e-5)
        assert tick.ask == pytest.approx(149.505, rel=1e-5)
        assert tick.volume == pytest.approx(1000.0, rel=1e-5)
    
    def test_symbol_uppercase_conversion(self):
        """シンボルが大文字に変換されることをテスト"""
        tick = Tick(
            timestamp=datetime.now(),
            symbol="usdjpy",  # 小文字で入力
            bid=149.500,
            ask=149.505,
            volume=1000.0
        )
        
        assert tick.symbol == "USDJPY"  # 大文字に変換されている
    
    def test_float32_precision(self):
        """Float32精度への変換テスト"""
        # 高精度の値を入力
        tick = Tick(
            timestamp=datetime.now(),
            symbol="EURUSD",
            bid=1.234567890123456,
            ask=1.234567890123457,
            volume=999.999999999
        )
        
        # Float32精度に丸められていることを確認
        tick_dict = tick.to_float32_dict()
        assert isinstance(tick_dict["bid"], np.float32)
        assert isinstance(tick_dict["ask"], np.float32)
        assert isinstance(tick_dict["volume"], np.float32)
    
    def test_spread_calculation(self):
        """スプレッド計算のテスト"""
        tick = Tick(
            timestamp=datetime.now(),
            symbol="EURUSD",
            bid=1.12345,
            ask=1.12355,
            volume=1000.0
        )
        
        expected_spread = 1.12355 - 1.12345
        assert tick.spread == pytest.approx(expected_spread, rel=1e-5)
    
    def test_mid_price_calculation(self):
        """仲値計算のテスト"""
        tick = Tick(
            timestamp=datetime.now(),
            symbol="GBPUSD",
            bid=1.30000,
            ask=1.30010,
            volume=500.0
        )
        
        expected_mid = (1.30000 + 1.30010) / 2
        assert tick.mid_price == pytest.approx(expected_mid, rel=1e-5)
    
    def test_invalid_negative_price(self):
        """負の価格でバリデーションエラーになることをテスト"""
        with pytest.raises(ValidationError) as exc_info:
            Tick(
                timestamp=datetime.now(),
                symbol="USDJPY",
                bid=-149.500,  # 負の値
                ask=149.505,
                volume=1000.0
            )
        
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('bid',) for error in errors)
    
    def test_invalid_spread(self):
        """Ask < Bidでバリデーションエラーになることをテスト"""
        with pytest.raises(ValidationError) as exc_info:
            Tick(
                timestamp=datetime.now(),
                symbol="USDJPY",
                bid=149.505,
                ask=149.500,  # Bidより小さい
                volume=1000.0
            )
        
        errors = exc_info.value.errors()
        assert any("Ask価格" in str(error) for error in errors)
    
    def test_invalid_symbol_length(self):
        """シンボル長のバリデーションテスト"""
        # 短すぎるシンボル
        with pytest.raises(ValidationError):
            Tick(
                timestamp=datetime.now(),
                symbol="USD",  # 3文字（最小6文字）
                bid=149.500,
                ask=149.505,
                volume=1000.0
            )
        
        # 長すぎるシンボル
        with pytest.raises(ValidationError):
            Tick(
                timestamp=datetime.now(),
                symbol="USDJPYEXTRA",  # 11文字（最大10文字）
                bid=149.500,
                ask=149.505,
                volume=1000.0
            )
    
    def test_negative_volume(self):
        """負の取引量でバリデーションエラーになることをテスト"""
        with pytest.raises(ValidationError):
            Tick(
                timestamp=datetime.now(),
                symbol="EURUSD",
                bid=1.12345,
                ask=1.12355,
                volume=-100.0  # 負の値
            )
    
    def test_json_serialization(self):
        """JSON変換のテスト"""
        tick = Tick(
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
            symbol="USDJPY",
            bid=149.500,
            ask=149.505,
            volume=1000.0
        )
        
        json_data = tick.model_dump_json()
        assert "2025-01-15T10:30:00" in json_data
        assert "USDJPY" in json_data
        
        # JSONから再構築
        tick2 = Tick.model_validate_json(json_data)
        assert tick2.symbol == tick.symbol
        assert tick2.bid == pytest.approx(tick.bid, rel=1e-5)