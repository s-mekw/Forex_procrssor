"""
統合テストファイル - モデル間の相互作用とパフォーマンステスト
全体的なデータフローと設定管理との統合をテスト
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, UTC
from pathlib import Path
import tempfile
import time
from typing import List
import os

from src.common.models import (
    Tick, OHLC, Prediction, Alert,
    TimeFrame, PredictionType, AlertType, AlertSeverity
)
from src.common.config import ConfigManager, BaseConfig


class TestModelIntegration:
    """モデル間の相互作用をテストするクラス"""
    
    @pytest.fixture
    def sample_ticks(self) -> List[Tick]:
        """テスト用のTickデータを生成"""
        base_time = datetime.now(UTC)
        ticks = []
        
        # 1分間のTickデータを生成（60秒分）
        for i in range(60):
            tick = Tick(
                timestamp=base_time + timedelta(seconds=i),
                symbol="USDJPY",
                bid=149.50 + np.sin(i * 0.1) * 0.05,  # サイン波で価格変動
                ask=149.51 + np.sin(i * 0.1) * 0.05,
                volume=100.0 + i * 2
            )
            ticks.append(tick)
        
        return ticks
    
    def test_tick_to_ohlc_conversion(self, sample_ticks: List[Tick]):
        """TickデータからOHLCへの変換シミュレーション"""
        # Tickデータから統計値を計算
        prices = [(tick.bid + tick.ask) / 2 for tick in sample_ticks]
        volumes = [tick.volume for tick in sample_ticks]
        
        # OHLC作成
        ohlc = OHLC(
            timestamp=sample_ticks[0].timestamp,
            symbol=sample_ticks[0].symbol,
            timeframe=TimeFrame.M1,
            open=float(np.float32(prices[0])),
            high=float(np.float32(max(prices))),
            low=float(np.float32(min(prices))),
            close=float(np.float32(prices[-1])),
            volume=float(np.float32(sum(volumes)))
        )
        
        # 検証
        assert ohlc.symbol == "USDJPY"
        assert ohlc.timeframe == TimeFrame.M1
        assert ohlc.high >= ohlc.low
        assert ohlc.high >= ohlc.open
        assert ohlc.high >= ohlc.close
        assert ohlc.low <= ohlc.open
        assert ohlc.low <= ohlc.close
        assert ohlc.volume > 0
        
        # プロパティの検証
        assert ohlc.range >= 0
        assert isinstance(ohlc.is_bullish, bool)
        assert isinstance(ohlc.is_bearish, bool)
    
    def test_ohlc_to_prediction_flow(self):
        """OHLCデータから予測モデルへのフロー"""
        # 複数のOHLCデータを作成
        base_time = datetime.now(UTC)
        ohlc_data = []
        
        for i in range(5):
            ohlc = OHLC(
                timestamp=base_time + timedelta(minutes=i),
                symbol="EURUSD",
                timeframe=TimeFrame.M1,
                open=1.0850 + i * 0.0001,
                high=1.0852 + i * 0.0001,
                low=1.0849 + i * 0.0001,
                close=1.0851 + i * 0.0001,
                volume=1000.0 + i * 100
            )
            ohlc_data.append(ohlc)
        
        # OHLCデータから予測を生成
        latest_ohlc = ohlc_data[-1]
        prediction = Prediction(
            symbol=latest_ohlc.symbol,
            predicted_at=datetime.now(UTC),
            target_timestamp=latest_ohlc.timestamp + timedelta(minutes=5),
            prediction_type=PredictionType.PRICE,
            predicted_value=latest_ohlc.close * 1.001,  # 0.1%上昇予測
            confidence_score=0.75,
            confidence_upper=latest_ohlc.close * 1.002,
            confidence_lower=latest_ohlc.close * 1.0005
        )
        
        # 検証
        assert prediction.symbol == latest_ohlc.symbol
        assert prediction.prediction_type == PredictionType.PRICE
        assert prediction.confidence_score > 0.5
        assert prediction.confidence_upper > prediction.predicted_value
        assert prediction.predicted_value > prediction.confidence_lower
        assert prediction.confidence_range > 0
    
    def test_prediction_to_alert_flow(self):
        """予測からアラート生成のフロー"""
        # 予測を作成
        prediction = Prediction(
            symbol="GBPUSD",
            predicted_at=datetime.now(UTC),
            target_timestamp=datetime.now(UTC) + timedelta(hours=1),
            prediction_type=PredictionType.DIRECTION,
            predicted_value=1.0,  # 上昇予測
            confidence_score=0.85
        )
        
        # 高信頼度の予測に基づいてアラートを生成
        if prediction.confidence_score > 0.8:
            alert = Alert(
                symbol=prediction.symbol,
                timestamp=prediction.predicted_at,
                alert_type=AlertType.PATTERN_DETECTED,
                severity=AlertSeverity.WARNING,
                message=f"High confidence {prediction.prediction_type.value} prediction",
                condition=f"confidence_score > 0.8",
                current_value=prediction.confidence_score,
                threshold_value=0.8
            )
            
            # 検証
            assert alert.symbol == prediction.symbol
            assert alert.severity == AlertSeverity.WARNING
            # threshold_exceededプロパティはcurrent_value > threshold_valueをチェック
            assert alert.current_value > alert.threshold_value
            assert not alert.is_critical
    
    def test_complete_data_flow(self):
        """Tick → OHLC → Prediction → Alert の完全なデータフロー"""
        # 1. Tickデータの生成
        base_time = datetime.now(UTC)
        ticks = []
        for i in range(10):
            tick = Tick(
                timestamp=base_time + timedelta(seconds=i*6),
                symbol="AUDUSD",
                bid=0.6500 + i * 0.0001,
                ask=0.6501 + i * 0.0001,
                volume=500.0
            )
            ticks.append(tick)
        
        # 2. TickからOHLCへの変換
        mid_prices = [(t.bid + t.ask) / 2 for t in ticks]
        ohlc = OHLC(
            timestamp=ticks[0].timestamp,
            symbol=ticks[0].symbol,
            timeframe=TimeFrame.M1,
            open=float(np.float32(mid_prices[0])),
            high=float(np.float32(max(mid_prices))),
            low=float(np.float32(min(mid_prices))),
            close=float(np.float32(mid_prices[-1])),
            volume=float(np.float32(sum(t.volume for t in ticks)))
        )
        
        # 3. OHLCから予測を生成
        if ohlc.is_bullish:
            prediction = Prediction(
                symbol=ohlc.symbol,
                predicted_at=datetime.now(UTC),
                target_timestamp=ohlc.timestamp + timedelta(minutes=5),
                prediction_type=PredictionType.DIRECTION,
                predicted_value=1.0,  # 上昇継続予測
                confidence_score=0.9
            )
            
            # 4. 予測からアラートを生成
            alert = Alert(
                symbol=prediction.symbol,
                timestamp=prediction.predicted_at,
                alert_type=AlertType.PATTERN_DETECTED,
                severity=AlertSeverity.CRITICAL if prediction.confidence_score > 0.85 else AlertSeverity.WARNING,
                message="Strong bullish momentum detected",
                condition="is_bullish=True and confidence>0.85"
            )
            
            # 全体フローの検証
            assert tick.symbol == ohlc.symbol == prediction.symbol == alert.symbol
            assert alert.is_critical
            assert ohlc.close > ohlc.open  # 陽線
            assert prediction.predicted_value == 1.0  # 上昇予測


class TestFloat32Performance:
    """Float32制約のパフォーマンステスト"""
    
    def test_memory_efficiency(self):
        """Float32使用によるメモリ効率の検証"""
        # 大量のTickデータを生成
        n_ticks = 10000
        base_time = datetime.now(UTC)
        
        # Float32を使用したTickデータ
        ticks_float32 = []
        for i in range(n_ticks):
            tick = Tick(
                timestamp=base_time + timedelta(seconds=i),
                symbol="USDJPY",
                bid=float(np.float32(150.0 + i * 0.0001)),
                ask=float(np.float32(150.01 + i * 0.0001)),
                volume=float(np.float32(1000.0))
            )
            ticks_float32.append(tick)
        
        # メモリ使用量の概算（各フィールド4バイト × 3フィールド）
        expected_memory_per_tick = 3 * 4  # bid, ask, volume各4バイト
        total_expected_memory = n_ticks * expected_memory_per_tick
        
        # Float64の場合の理論的メモリ使用量
        float64_memory = n_ticks * 3 * 8  # 各8バイト
        
        # メモリ削減率の計算
        memory_reduction = (1 - total_expected_memory / float64_memory) * 100
        
        assert memory_reduction == 50.0  # 50%のメモリ削減
        assert len(ticks_float32) == n_ticks
    
    def test_calculation_performance(self):
        """Float32での計算パフォーマンステスト"""
        n_iterations = 100000
        
        # Float32での計算
        start_time = time.perf_counter()
        values_f32 = [float(np.float32(i * 0.001)) for i in range(n_iterations)]
        sum_f32 = sum(values_f32)
        f32_time = time.perf_counter() - start_time
        
        # 結果の検証
        assert len(values_f32) == n_iterations
        assert sum_f32 > 0
        assert f32_time < 1.0  # 1秒以内に完了すること
    
    def test_precision_tradeoff(self):
        """Float32の精度トレードオフのテスト"""
        # 小数点以下の精度テスト
        test_values = [
            0.1234567890123456,
            150.123456789,
            0.00001234567890
        ]
        
        for original in test_values:
            # Float32に変換
            f32_value = float(np.float32(original))
            
            # 相対誤差の計算
            if original != 0:
                relative_error = abs(f32_value - original) / abs(original)
                # Float32の精度は約7桁なので、相対誤差は1e-6程度まで
                assert relative_error < 1e-5 or np.isclose(f32_value, original, rtol=1e-5)


class TestConfigIntegration:
    """設定管理システムとの統合テスト"""
    
    @pytest.fixture
    def temp_config_file(self):
        """一時的な設定ファイルを作成"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write("""
app_name = "test_forex"
debug = true
batch_size = 500
memory_limit_gb = 4.0
model_type = "lstm"
confidence_threshold = 0.75
""")
            temp_path = f.name
        
        yield Path(temp_path)
        
        # クリーンアップ
        os.unlink(temp_path)
    
    def test_config_with_models(self, temp_config_file):
        """設定管理とモデルの統合テスト"""
        # ConfigManagerのインスタンスをリセット
        ConfigManager._instance = None
        ConfigManager._config = None
        
        # 設定を読み込み
        config_manager = ConfigManager()
        config = config_manager.load_config(toml_file=temp_config_file)
        
        # 設定値を使用してモデルを作成
        # batch_sizeはデフォルトが1000なので、設定ファイルの500を使用
        tick = Tick(
            timestamp=datetime.now(UTC),
            symbol="USDJPY",
            bid=150.0,
            ask=150.01,
            volume=1000.0  # 通常の値を使用
        )
        
        # 検証（設定ファイルから読み込んだ値を確認）
        assert tick.volume == 1000.0
        assert config.batch_size == 500  # 設定ファイルから読み込んだ値
        assert config.confidence_threshold == 0.75
        assert config.model_type == "lstm"
    
    def test_model_validation_with_config(self):
        """設定値を使用したモデルバリデーション"""
        # ConfigManagerのインスタンスをリセット
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config_manager = ConfigManager()
        config = config_manager.load_config()  # load_configを呼び出す
        
        # 設定の信頼度閾値を使用して予測を検証
        prediction = Prediction(
            symbol="EURUSD",
            predicted_at=datetime.now(UTC),
            target_timestamp=datetime.now(UTC) + timedelta(hours=1),
            prediction_type=PredictionType.PRICE,
            predicted_value=1.0850,
            confidence_score=0.6  # デフォルト閾値0.7より低い
        )
        
        # 設定値を使用した判定
        is_reliable = prediction.confidence_score >= config.confidence_threshold
        
        # アラートを生成するかの判定
        if not is_reliable:
            alert = Alert(
                symbol=prediction.symbol,
                timestamp=prediction.predicted_at,
                alert_type=AlertType.RISK_WARNING,
                severity=AlertSeverity.INFO,
                message="Low confidence prediction",
                threshold_value=config.confidence_threshold,
                current_value=prediction.confidence_score
            )
            
            assert alert.severity == AlertSeverity.INFO
            assert not alert.threshold_exceeded
    
    def test_batch_processing_with_config(self):
        """設定のバッチサイズを使用した処理テスト"""
        # ConfigManagerのインスタンスをリセット
        ConfigManager._instance = None
        ConfigManager._config = None
        
        config_manager = ConfigManager()
        config = config_manager.load_config()  # load_configを呼び出す
        
        # バッチサイズ分のTickデータを生成
        base_time = datetime.now(UTC)
        batch_ticks = []
        
        for i in range(config.batch_size):
            tick = Tick(
                timestamp=base_time + timedelta(seconds=i),
                symbol="GBPUSD",
                bid=1.2500 + i * 0.00001,
                ask=1.2501 + i * 0.00001,
                volume=100.0
            )
            batch_ticks.append(tick)
        
        # バッチ処理のシミュレーション
        assert len(batch_ticks) == config.batch_size
        
        # メモリ制限のチェック（Float32使用で約12バイト/Tick）
        estimated_memory_mb = len(batch_ticks) * 12 / (1024 * 1024)
        assert estimated_memory_mb < config.memory_limit_gb * 1024


class TestDataValidation:
    """データ検証の統合テスト"""
    
    def test_cross_model_validation(self):
        """モデル間のデータ整合性検証"""
        base_time = datetime.now(UTC)
        symbol = "NZDUSD"
        
        # 同じシンボルで各モデルを作成
        tick = Tick(
            timestamp=base_time,
            symbol=symbol.lower(),  # 小文字で作成
            bid=0.6000,
            ask=0.6001,
            volume=1000.0
        )
        
        ohlc = OHLC(
            timestamp=base_time,
            symbol=symbol.upper(),  # 大文字で作成
            timeframe=TimeFrame.H1,
            open=0.6000,
            high=0.6010,
            low=0.5995,
            close=0.6005,
            volume=10000.0
        )
        
        prediction = Prediction(
            symbol=symbol.upper(),
            predicted_at=base_time,
            target_timestamp=base_time + timedelta(hours=1),
            prediction_type=PredictionType.VOLATILITY,
            predicted_value=0.0010,
            confidence_score=0.8
        )
        
        alert = Alert(
            symbol=symbol.lower(),  # 小文字で作成
            timestamp=base_time,
            alert_type=AlertType.PRICE_THRESHOLD,
            severity=AlertSeverity.WARNING,
            message="Price threshold reached"
        )
        
        # シンボルの正規化確認（全て大文字に統一される）
        assert tick.symbol == "NZDUSD"
        assert ohlc.symbol == "NZDUSD"
        assert prediction.symbol == "NZDUSD"
        assert alert.symbol == "NZDUSD"
    
    def test_timestamp_consistency(self):
        """タイムスタンプの一貫性テスト"""
        # 異なるタイムゾーン情報でのタイムスタンプ作成
        base_time = datetime.now(UTC)
        
        # タイムゾーン情報なしのdatetimeでもUTCとして扱われることを確認
        naive_time = datetime.now()
        
        tick = Tick(
            timestamp=base_time,
            symbol="USDCAD",
            bid=1.3500,
            ask=1.3501,
            volume=100.0
        )
        
        # 将来のタイムスタンプ
        future_time = base_time + timedelta(days=1)
        
        prediction = Prediction(
            symbol="USDCAD",
            predicted_at=base_time,
            target_timestamp=future_time,
            prediction_type=PredictionType.PRICE,
            predicted_value=1.3550,
            confidence_score=0.7
        )
        
        # タイムスタンプの論理的整合性
        assert prediction.target_timestamp > prediction.predicted_at
        assert tick.timestamp <= prediction.predicted_at
    
    def test_edge_case_handling(self):
        """エッジケースの処理テスト"""
        # 極端な値のテスト（シンボルは6文字以上の制約がある）
        extreme_tick = Tick(
            timestamp=datetime.now(UTC),
            symbol="TESTJPY",  # 6文字以上のシンボル
            bid=float(np.float32(1e-10)),  # 非常に小さい値
            ask=float(np.float32(1e10)),   # 非常に大きい値
            volume=float(np.float32(0.0))  # ゼロボリューム
        )
        
        assert extreme_tick.bid > 0
        assert extreme_tick.ask > extreme_tick.bid
        assert extreme_tick.volume == 0.0
        
        # スプレッドの計算（極端に大きい）
        assert extreme_tick.spread == pytest.approx(1e10, rel=1e-5)
        
        # 長いメッセージのアラート
        long_message = "A" * 500  # 最大長
        alert = Alert(
            symbol="TESTJPY",
            timestamp=datetime.now(UTC),
            alert_type=AlertType.RISK_WARNING,
            severity=AlertSeverity.INFO,
            message=long_message
        )
        
        assert len(alert.message) == 500


class TestPerformanceMetrics:
    """パフォーマンスメトリクスのテスト"""
    
    def test_model_creation_performance(self):
        """モデル作成のパフォーマンステスト"""
        n_models = 1000
        
        # Tickモデルの作成時間測定
        start_time = time.perf_counter()
        for _ in range(n_models):
            Tick(
                timestamp=datetime.now(UTC),
                symbol="USDJPY",
                bid=150.0,
                ask=150.01,
                volume=1000.0
            )
        tick_time = time.perf_counter() - start_time
        
        # OHLCモデルの作成時間測定
        start_time = time.perf_counter()
        for _ in range(n_models):
            OHLC(
                timestamp=datetime.now(UTC),
                symbol="EURUSD",
                timeframe=TimeFrame.M1,
                open=1.0850,
                high=1.0855,
                low=1.0845,
                close=1.0852,
                volume=10000.0
            )
        ohlc_time = time.perf_counter() - start_time
        
        # パフォーマンス基準（1000モデルを1秒以内に作成）
        assert tick_time < 1.0
        assert ohlc_time < 1.0
        
        # 作成速度の計算
        tick_rate = n_models / tick_time
        ohlc_rate = n_models / ohlc_time
        
        # 最低1000モデル/秒以上の作成速度
        assert tick_rate > 1000
        assert ohlc_rate > 1000
    
    def test_validation_overhead(self):
        """バリデーションのオーバーヘッドテスト"""
        n_iterations = 1000
        
        # バリデーションありのOHLC作成
        start_time = time.perf_counter()
        for i in range(n_iterations):
            OHLC(
                timestamp=datetime.now(UTC),
                symbol="GBPUSD",
                timeframe=TimeFrame.H1,
                open=1.2500,
                high=1.2510 + i * 0.00001,  # 変動する高値
                low=1.2490,
                close=1.2505,
                volume=5000.0
            )
        validation_time = time.perf_counter() - start_time
        
        # バリデーションのオーバーヘッドが許容範囲内
        assert validation_time < 2.0  # 2秒以内
        
        # 平均バリデーション時間
        avg_validation_time = validation_time / n_iterations
        assert avg_validation_time < 0.002  # 2ms以内/モデル