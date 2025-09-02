"""
基本的なMultiTimeframeManager動作テスト
BTCUSD#でM1とM5のデータ取得とバー更新を確認
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import logging
import time
from datetime import datetime
import toml
import MetaTrader5 as mt5

from src.data_processing.multiframe_manager import MultiTimeframeManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_operations():
    """基本動作テスト"""
    
    # 設定読み込み
    config = toml.load("config.toml")
    mt5_config = config['mt5']
    symbol = config['trading']['symbol']
    
    logger.info("=" * 60)
    logger.info("Basic MultiTimeframe Test")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: M1, M5")
    
    # MT5接続
    logger.info("\n1. MT5接続テスト")
    if not mt5.initialize(
        path=mt5_config['path'],
        login=mt5_config['account'],
        password=mt5_config['password'],
        server=mt5_config['server'],
        timeout=mt5_config['timeout']
    ):
        logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        return
    
    logger.info(f"✅ MT5接続成功: Account {mt5_config['account']}")
    
    # シンボル確認
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Symbol {symbol} not found")
        mt5.shutdown()
        return
    
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            mt5.shutdown()
            return
    
    logger.info(f"✅ Symbol {symbol} selected")
    logger.info(f"   Bid: {symbol_info.bid}, Ask: {symbol_info.ask}")
    
    # MultiTimeframeManager作成
    logger.info("\n2. MultiTimeframeManager初期化")
    manager = MultiTimeframeManager(
        symbol=symbol,
        timeframes=["M1", "M5"],
        initial_bars=100,
        max_bars=500
    )
    
    # 初期データ取得
    if not manager.initialize_data():
        logger.error("Failed to initialize manager")
        mt5.shutdown()
        return
    
    logger.info("✅ Manager初期化成功")
    
    # メトリクス表示
    metrics = manager.get_metrics()
    logger.info(f"   Symbol: {metrics['symbol']}")
    logger.info(f"   Initialized: {metrics['is_initialized']}")
    for tf_name, tf_metrics in metrics['timeframes'].items():
        logger.info(f"   {tf_name}: {tf_metrics['completed_bars']} bars")
    
    # 各タイムフレームのデータ確認
    logger.info("\n3. 初期データ確認")
    for tf in ["M1", "M5"]:
        bars = manager.get_completed_bars(tf, limit=5)
        if bars is not None and not bars.is_empty():
            logger.info(f"\n{tf} - 最新5本:")
            for i in range(len(bars)):
                row = bars[i]
                logger.info(f"   {row['timestamp'][0].strftime('%H:%M')} - "
                          f"O:{row['open'][0]:.2f} H:{row['high'][0]:.2f} "
                          f"L:{row['low'][0]:.2f} C:{row['close'][0]:.2f}")
    
    # リアルタイムティック処理テスト
    logger.info("\n4. リアルタイムティック処理テスト（10秒間）")
    
    tick_count = 0
    m1_bars_completed = 0
    m5_bars_completed = 0
    start_time = time.time()
    test_duration = 10  # 10秒間テスト
    
    logger.info(f"   テスト開始: {datetime.now().strftime('%H:%M:%S')}")
    
    while time.time() - start_time < test_duration:
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
                if tf_name == "M1":
                    m1_bars_completed += 1
                    completed_bar = tf_result.get("completed_bar")
                    if completed_bar:
                        logger.info(f"   ✅ M1 bar completed at {completed_bar['timestamp']}")
                        logger.info(f"      OHLC: {completed_bar['open']:.2f}, "
                                  f"{completed_bar['high']:.2f}, "
                                  f"{completed_bar['low']:.2f}, "
                                  f"{completed_bar['close']:.2f}")
                elif tf_name == "M5":
                    m5_bars_completed += 1
                    completed_bar = tf_result.get("completed_bar")
                    if completed_bar:
                        logger.info(f"   ✅ M5 bar completed at {completed_bar['timestamp']}")
                        logger.info(f"      OHLC: {completed_bar['open']:.2f}, "
                                  f"{completed_bar['high']:.2f}, "
                                  f"{completed_bar['low']:.2f}, "
                                  f"{completed_bar['close']:.2f}")
        
        # 現在のバー情報（デバッグ）
        if tick_count % 10 == 0:
            current_m1 = manager.get_current_bar("M1")
            if current_m1:
                logger.debug(f"   M1 current bar: Close={current_m1.get('close', 0):.2f}")
        
        time.sleep(0.1)  # CPU負荷軽減
    
    logger.info(f"   テスト終了: {datetime.now().strftime('%H:%M:%S')}")
    
    # テスト結果サマリー
    logger.info("\n5. テスト結果サマリー")
    logger.info(f"   処理ティック数: {tick_count}")
    logger.info(f"   M1バー完成数: {m1_bars_completed}")
    logger.info(f"   M5バー完成数: {m5_bars_completed}")
    logger.info(f"   ティック/秒: {tick_count / test_duration:.1f}")
    
    # 最終メトリクス
    final_metrics = manager.get_metrics()
    logger.info("\n6. 最終メトリクス")
    for tf_name, tf_metrics in final_metrics['timeframes'].items():
        logger.info(f"   {tf_name}:")
        logger.info(f"      完成バー数: {tf_metrics['completed_bars']}")
        logger.info(f"      現在のバー: {'あり' if tf_metrics['has_current_bar'] else 'なし'}")
        if tf_metrics['last_bar_time']:
            logger.info(f"      最終バー時刻: {tf_metrics['last_bar_time']}")
    
    # M5バー境界検出のテスト
    logger.info("\n7. M5バー境界検出テスト")
    logger.info("   修正された境界検出ロジックの動作確認:")
    
    # 最新のM5バーを確認
    m5_bars = manager.get_completed_bars("M5", limit=3)
    if m5_bars is not None and not m5_bars.is_empty():
        for i in range(len(m5_bars)):
            row = m5_bars[i]
            timestamp = row['timestamp'][0]
            logger.info(f"   M5 Bar: {timestamp.strftime('%H:%M:%S')} "
                      f"(分: {timestamp.minute}, 秒: {timestamp.second})")
        
        # 境界検出が正しく動作しているか確認
        if m5_bars_completed > 0:
            logger.info("   ✅ M5バー境界検出が正常に動作しています")
        else:
            logger.info("   ℹ️ テスト期間中にM5バーは完成しませんでした（正常）")
    
    # クリーンアップ
    mt5.shutdown()
    logger.info("\n✅ テスト完了")


def test_boundary_detection():
    """M5バー境界検出の詳細テスト"""
    
    logger.info("\n" + "=" * 60)
    logger.info("M5 Boundary Detection Test")
    logger.info("=" * 60)
    
    # 境界検出ロジックの説明
    logger.info("\n修正された境界検出ロジック:")
    logger.info("1. timestamp.second == 0 の条件を削除")
    logger.info("2. 分の境界を超えたかチェック")
    logger.info("3. 例: 10:04:59.800 → 10:05:00.200 で新しいM5バー")
    
    # 設定読み込み
    config = toml.load("config.toml")
    symbol = config['trading']['symbol']
    
    # MT5接続（省略版）
    mt5_config = config['mt5']
    if not mt5.initialize(
        path=mt5_config['path'],
        login=mt5_config['account'],
        password=mt5_config['password'],
        server=mt5_config['server']
    ):
        logger.error("MT5 initialization failed")
        return
    
    # マネージャー作成
    manager = MultiTimeframeManager(
        symbol=symbol,
        timeframes=["M1", "M5"],
        initial_bars=50
    )
    
    if not manager.initialize_data():
        logger.error("Failed to initialize manager")
        mt5.shutdown()
        return
    
    # 現在時刻と次のM5境界を計算
    now = datetime.now()
    current_m5_minute = (now.minute // 5) * 5
    next_m5_minute = current_m5_minute + 5
    
    logger.info(f"\n現在時刻: {now.strftime('%H:%M:%S')}")
    logger.info(f"現在のM5バー: {now.hour:02d}:{current_m5_minute:02d}:00")
    logger.info(f"次のM5バー: {now.hour:02d}:{next_m5_minute:02d}:00")
    
    # 次のM5境界まで待機してテスト
    wait_seconds = (next_m5_minute - now.minute) * 60 - now.second
    if wait_seconds > 0 and wait_seconds < 60:
        logger.info(f"\n次のM5境界まで{wait_seconds}秒待機...")
        
        # 境界前後でティックを処理
        boundary_detected = False
        start_wait = time.time()
        
        while time.time() - start_wait < wait_seconds + 5:  # 境界の5秒後まで
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                results = manager.process_tick(tick)
                
                # M5バー完成をチェック
                if results.get("M5", {}).get("new_bar"):
                    tick_time = datetime.fromtimestamp(tick.time)
                    logger.info(f"✅ M5境界検出! Tick time: {tick_time.strftime('%H:%M:%S.%f')[:-3]}")
                    boundary_detected = True
                    break
            
            time.sleep(0.05)  # 50msごとにチェック
        
        if boundary_detected:
            logger.info("✅ M5バー境界検出テスト成功")
        else:
            logger.info("ℹ️ この期間中にM5境界は検出されませんでした")
    else:
        logger.info(f"ℹ️ 次のM5境界まで{wait_seconds}秒（スキップ）")
    
    mt5.shutdown()
    logger.info("\nテスト完了")


if __name__ == "__main__":
    # 基本テスト実行
    test_basic_operations()
    
    # 境界検出テスト（オプション）
    # test_boundary_detection()