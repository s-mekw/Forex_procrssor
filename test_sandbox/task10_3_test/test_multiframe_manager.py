"""
MultiTimeframeManager動作テストスクリプト

新しいマルチタイムフレーム管理システムの動作を確認します。
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import asyncio
import MetaTrader5 as mt5
from datetime import datetime
import logging
import time

from src.data_processing.multiframe_manager import MultiTimeframeManager
from src.data_processing.analyzer_v2 import MultiTimeframeAnalyzerV2
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_multiframe_manager():
    """MultiTimeframeManagerの基本動作テスト"""
    
    # MT5接続
    mt5_config = {
        'account': 75334547,
        'password': '#Shota1627763',
        'server': 'XMTrading-MT5 3',
        'timeout': 60000,
        'path': 'C:\\Program Files\\XMTrading MT5\\terminal64.exe',
    }
    
    mt5_manager = MT5ConnectionManager(mt5_config)
    if not mt5_manager.connect(mt5_config):
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return
    
    logger.info("MT5 connected successfully")
    
    # MultiTimeframeManagerのテスト
    symbol = "EURJPY#"
    timeframes = ["M1", "M5", "M15"]
    
    manager = MultiTimeframeManager(
        symbol=symbol,
        timeframes=timeframes,
        initial_bars=100,
        max_bars=1000
    )
    
    # 初期データ取得
    if not manager.initialize_data():
        logger.error("Failed to initialize manager")
        return
    
    logger.info("Manager initialized successfully")
    
    # メトリクス表示
    metrics = manager.get_metrics()
    logger.info(f"Initial metrics: {metrics}")
    
    # 各タイムフレームのデータ確認
    for tf in timeframes:
        bars = manager.get_completed_bars(tf, limit=5)
        if bars is not None and not bars.is_empty():
            logger.info(f"{tf} - Latest bars: {len(bars)} bars")
            latest = bars.tail(1)
            logger.info(f"{tf} - Latest bar: {latest['timestamp'][0]} "
                       f"Close: {latest['close'][0]:.5f}")
    
    # リアルタイムティック処理のテスト
    logger.info("\nStarting real-time tick processing...")
    
    tick_count = 0
    bar_completed = {tf: 0 for tf in timeframes}
    start_time = time.time()
    
    while tick_count < 100 and (time.time() - start_time) < 30:  # 最大30秒
        # ティック取得
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            time.sleep(0.1)
            continue
        
        # ティック処理
        results = manager.process_tick(tick)
        tick_count += 1
        
        # 新しいバーが完成したかチェック
        for tf_name, tf_result in results.items():
            if tf_result.get("new_bar"):
                bar_completed[tf_name] += 1
                completed = tf_result.get("completed_bar", {})
                logger.info(f"✅ {tf_name} bar completed: {completed.get('timestamp')} "
                           f"OHLC=[{completed.get('open', 0):.5f}, "
                           f"{completed.get('high', 0):.5f}, "
                           f"{completed.get('low', 0):.5f}, "
                           f"{completed.get('close', 0):.5f}]")
        
        # 現在のバー状態を定期的に表示
        if tick_count % 20 == 0:
            current_bars = manager.get_all_current_bars()
            for tf_name, bar in current_bars.items():
                if bar:
                    logger.info(f"{tf_name} current bar[0]: "
                               f"O:{bar.get('open', 0):.5f} "
                               f"H:{bar.get('high', 0):.5f} "
                               f"L:{bar.get('low', 0):.5f} "
                               f"C:{bar.get('close', 0):.5f}")
        
        time.sleep(0.5)
    
    # 結果サマリー
    logger.info("\n=== Test Summary ===")
    logger.info(f"Ticks processed: {tick_count}")
    for tf in timeframes:
        logger.info(f"{tf} bars completed: {bar_completed[tf]}")
    
    # 最終メトリクス
    final_metrics = manager.get_metrics()
    logger.info(f"Final metrics: {final_metrics}")
    
    mt5.shutdown()
    logger.info("Test completed")


def test_analyzer_v2():
    """MultiTimeframeAnalyzerV2の動作テスト"""
    
    # MT5接続
    mt5_config = {
        'account': 75334547,
        'password': '#Shota1627763',
        'server': 'XMTrading-MT5 3',
        'timeout': 60000,
        'path': 'C:\\Program Files\\XMTrading MT5\\terminal64.exe',
    }
    
    mt5_manager = MT5ConnectionManager(mt5_config)
    if not mt5_manager.connect(mt5_config):
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return
    
    logger.info("MT5 connected successfully")
    
    # MultiTimeframeAnalyzerV2のテスト
    symbol = "EURJPY#"
    timeframes = ["M1", "M5"]
    
    analyzer = MultiTimeframeAnalyzerV2(
        symbol=symbol,
        timeframes=timeframes,
        initial_bars=200,
        max_history_bars=1000
    )
    
    # 初期化
    if not analyzer.initialize():
        logger.error("Failed to initialize analyzer")
        return
    
    logger.info("Analyzer initialized successfully")
    
    # 初期RCI値を表示
    initial_rci = analyzer.get_latest_rci()
    for tf, rci_values in initial_rci.items():
        logger.info(f"{tf} initial RCI values:")
        for period, value in rci_values.items():
            logger.info(f"  RCI[{period}]: {value:.2f}")
    
    # リアルタイム分析テスト
    logger.info("\nStarting real-time analysis...")
    
    tick_count = 0
    start_time = time.time()
    
    while tick_count < 50 and (time.time() - start_time) < 20:  # 最大20秒
        # ティック取得
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            time.sleep(0.1)
            continue
        
        # ティック分析
        result = analyzer.analyze_tick(tick)
        tick_count += 1
        
        # 新しいバーが完成した場合の表示
        for tf_name, tf_data in result["timeframes"].items():
            if tf_data.get("new_bar"):
                logger.info(f"✅ {tf_name} new bar completed")
                # 更新されたRCI値を表示
                for period, value in tf_data.get("rci", {}).items():
                    logger.info(f"  RCI[{period}]: {value:.2f}")
        
        # 定期的にメトリクス表示
        if tick_count % 10 == 0:
            metrics = analyzer.get_metrics()
            logger.info(f"Ticks: {metrics['ticks_processed']}, "
                       f"M1 bars: {metrics['bars_completed']['M1']}, "
                       f"M5 bars: {metrics['bars_completed']['M5']}")
        
        time.sleep(0.5)
    
    # 最終結果
    logger.info("\n=== Analysis Summary ===")
    final_metrics = analyzer.get_metrics()
    logger.info(f"Total ticks processed: {final_metrics['ticks_processed']}")
    logger.info(f"Total RCI calculations: {final_metrics['rci_calculations']}")
    
    # 最終RCI値
    final_rci = analyzer.get_latest_rci()
    for tf, rci_values in final_rci.items():
        logger.info(f"{tf} final RCI values:")
        for period, value in rci_values.items():
            logger.info(f"  RCI[{period}]: {value:.2f}")
    
    mt5.shutdown()
    logger.info("Test completed")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MultiTimeframe Manager Test")
    logger.info("=" * 60)
    
    # 基本動作テスト
    logger.info("\n1. Testing MultiTimeframeManager:")
    test_multiframe_manager()
    
    logger.info("\n" + "=" * 60)
    
    # アナライザーテスト
    logger.info("\n2. Testing MultiTimeframeAnalyzerV2:")
    test_analyzer_v2()
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed")