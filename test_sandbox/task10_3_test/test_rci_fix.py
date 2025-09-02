"""
RCI計算修正のテストスクリプト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rci_calculation():
    """RCI計算の修正をテスト"""
    from src.data_processing.rci import RCICalculatorEngine
    
    # テストデータ作成（価格データ）
    np.random.seed(42)
    prices = 170.0 + np.random.randn(200) * 0.5  # EURJPY価格帯をシミュレート
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(199, -1, -1)]
    
    df = pl.DataFrame({
        "timestamp": timestamps,
        "close": prices.tolist()
    })
    
    logger.info(f"Test data created: {len(df)} rows")
    logger.info(f"Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
    
    # RCI計算
    rci_engine = RCICalculatorEngine()
    periods = [9, 24, 48, 66, 108]
    
    try:
        result = rci_engine.calculate_multiple(
            data=df,
            periods=periods,
            column_name="close",
            mode="batch",
            add_reliability=True
        )
        
        logger.info("RCI calculation successful")
        
        # 結果の検証
        for period in periods:
            rci_col = f"rci_{period}"
            if rci_col in result.columns:
                rci_values = [v for v in result[rci_col].to_list() if v is not None]
                if rci_values:
                    min_val = min(rci_values)
                    max_val = max(rci_values)
                    
                    # RCIは-100〜100の範囲内であるべき
                    if min_val >= -100 and max_val <= 100:
                        logger.info(f"✅ RCI[{period}]: Valid range [{min_val:.2f}, {max_val:.2f}]")
                    else:
                        logger.error(f"❌ RCI[{period}]: Invalid range [{min_val:.2f}, {max_val:.2f}]")
                        
                    # 価格のような値（150以上）が含まれていないかチェック
                    if max_val > 150:
                        logger.error(f"❌ RCI[{period}] contains price-like values!")
                else:
                    logger.warning(f"RCI[{period}]: No valid values")
            else:
                logger.error(f"RCI[{period}] column not found in result")
                
    except Exception as e:
        logger.error(f"RCI calculation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_improved_calculate_rci_history():
    """改善されたcalculate_rci_historyメソッドのテスト"""
    from pipeline_chart_dashboard import PipelineChartManager
    
    # テストデータ作成
    np.random.seed(42)
    prices = 170.0 + np.random.randn(600) * 0.5
    timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(599, -1, -1)]
    
    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": prices.tolist(),
        "high": (prices + np.random.rand(600) * 0.2).tolist(),
        "low": (prices - np.random.rand(600) * 0.2).tolist(),
        "close": prices.tolist(),
        "volume": np.random.randint(100, 1000, 600).tolist()
    })
    
    logger.info(f"Test M5 data created: {len(df)} bars")
    
    # PipelineChartManagerのインスタンスを作成（設定ファイル無しで）
    manager = PipelineChartManager()
    
    # calculate_rci_historyメソッドをテスト
    periods = [24, 33, 48, 66, 108]
    rci_history = manager.calculate_rci_history(df, periods)
    
    # 結果の検証
    logger.info("\n=== RCI History Test Results ===")
    for period, values in rci_history.items():
        if values:
            min_val = min(values)
            max_val = max(values)
            
            if min_val >= -100 and max_val <= 100:
                logger.info(f"✅ RCI[{period}]: {len(values)} values in valid range [{min_val:.2f}, {max_val:.2f}]")
            else:
                logger.error(f"❌ RCI[{period}]: Values outside valid range [{min_val:.2f}, {max_val:.2f}]")
                
            # 価格データ混入チェック
            if max_val > 150:
                logger.error(f"❌ RCI[{period}] still contains price data!")
                logger.error(f"First 5 values: {values[:5]}")
        else:
            logger.warning(f"RCI[{period}]: No values calculated")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing RCI Calculation Fix")
    logger.info("=" * 60)
    
    # 基本的なRCI計算のテスト
    logger.info("\n1. Testing basic RCI calculation:")
    test_rci_calculation()
    
    # 改善されたcalculate_rci_historyメソッドのテスト
    logger.info("\n2. Testing improved calculate_rci_history method:")
    test_improved_calculate_rci_history()
    
    logger.info("\n" + "=" * 60)
    logger.info("Test completed")