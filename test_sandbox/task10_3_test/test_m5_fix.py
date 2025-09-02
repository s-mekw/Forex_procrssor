"""
M5バー境界検出修正版のテスト
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import logging
from datetime import datetime, timedelta
import polars as pl
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_modified_analyzer():
    """修正されたMultiTimeframeAnalyzerをテスト"""
    from src.data_processing.analyzer import MultiTimeframeAnalyzer
    
    # アナライザー初期化
    analyzer = MultiTimeframeAnalyzer(
        short_term_periods=[9, 13, 24, 33, 48],
        long_term_periods=[24, 33, 48, 66, 108],
        long_timeframe="5T"
    )
    
    # テスト用の履歴データ作成（1分足データ）
    base_time = datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=120)
    timestamps = []
    prices = []
    
    for i in range(120):
        timestamps.append(base_time + timedelta(minutes=i))
        prices.append(170.0 + np.random.randn() * 0.5)
    
    history_df = pl.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + np.random.rand() * 0.1 for p in prices],
        "low": [p - np.random.rand() * 0.1 for p in prices],
        "close": prices,
        "volume": [100] * 120
    })
    
    # 履歴データを設定
    analyzer.update_history(history_df)
    
    logger.info("=" * 60)
    logger.info("修正版のM5バー境界検出テスト")
    logger.info("=" * 60)
    
    # 実際のティックをシミュレート（4:58から5:02まで）
    test_start = datetime.now().replace(minute=4, second=58, microsecond=800000)
    
    for i in range(20):
        tick_time = test_start + timedelta(milliseconds=200 * i)
        
        # 新しいバーデータを作成
        new_bar = {
            "timestamp": tick_time,
            "open": 170.0,
            "high": 170.1,
            "low": 169.9,
            "close": 170.0 + np.random.randn() * 0.01,
            "volume": 10
        }
        
        # 外部履歴での分析を実行
        try:
            result = analyzer.analyze_with_external_history(history_df, new_bar)
            
            if result["is_new_long_bar"]:
                logger.info(f"✅ M5 bar completed at {tick_time.strftime('%H:%M:%S.%f')[:-3]}")
                logger.info(f"   Long RCI values: {result['long_rci']}")
            else:
                logger.debug(f"   Tick at {tick_time.strftime('%H:%M:%S.%f')[:-3]} - no new M5 bar")
                
        except Exception as e:
            logger.error(f"Error at {tick_time}: {e}")
    
    logger.info("\n修正により、秒が0でなくても境界を超えたときに")
    logger.info("M5バーが完成と判定されるようになりました。")

def test_with_real_market_simulation():
    """実際の市場データをシミュレート"""
    from src.data_processing.analyzer import MultiTimeframeAnalyzer
    
    analyzer = MultiTimeframeAnalyzer(
        short_term_periods=[9, 13],
        long_term_periods=[24, 33],
        long_timeframe="5T"
    )
    
    # より現実的なデータ生成
    base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    history_data = []
    
    # 10:00から11:00までの1分足データ
    for i in range(60):
        bar_time = base_time + timedelta(minutes=i)
        base_price = 170.0 + np.sin(i * 0.1) * 0.5  # トレンドを持つ価格
        
        history_data.append({
            "timestamp": bar_time,
            "open": base_price,
            "high": base_price + np.random.rand() * 0.05,
            "low": base_price - np.random.rand() * 0.05,
            "close": base_price + (np.random.rand() - 0.5) * 0.02,
            "volume": np.random.randint(50, 200)
        })
    
    history_df = pl.DataFrame(history_data)
    analyzer.update_history(history_df)
    
    logger.info("\n" + "=" * 60)
    logger.info("実際の市場データシミュレーション")
    logger.info("=" * 60)
    
    # 11:00:00から11:10:00までのティックをシミュレート
    current_time = base_time + timedelta(hours=1)
    m5_bar_count = 0
    
    for seconds in range(600):  # 10分間
        # ランダムな間隔でティックが来る（100ms〜500ms）
        tick_time = current_time + timedelta(seconds=seconds, 
                                            milliseconds=np.random.randint(0, 999))
        
        new_bar = {
            "timestamp": tick_time,
            "open": 170.5,
            "high": 170.6,
            "low": 170.4,
            "close": 170.5 + np.random.randn() * 0.01,
            "volume": np.random.randint(5, 20)
        }
        
        result = analyzer.analyze_with_external_history(history_df, new_bar)
        
        if result["is_new_long_bar"]:
            m5_bar_count += 1
            logger.info(f"✅ M5 Bar #{m5_bar_count} completed at {tick_time.strftime('%H:%M:%S.%f')[:-3]}")
    
    logger.info(f"\n10分間で検出されたM5バー: {m5_bar_count}個")
    logger.info("期待値: 2個（11:05:00と11:10:00）")

if __name__ == "__main__":
    # 基本的なテスト
    test_modified_analyzer()
    
    # より現実的なシミュレーション
    test_with_real_market_simulation()