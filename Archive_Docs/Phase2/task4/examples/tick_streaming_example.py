#!/usr/bin/env python3
"""
TickDataStreamer 使用例

このファイルは、TickDataStreamerクラスの基本的な使用方法と
高度な使用例を示します。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

# プロジェクトのインポート
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer, StreamerConfig
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.common.models import Tick


# =============================================================================
# 基本的な使用例
# =============================================================================

async def basic_streaming_example():
    """基本的なティックストリーミングの例"""
    
    # 接続マネージャーの初期化
    connection_manager = MT5ConnectionManager()
    
    # ストリーマーの作成（デフォルト設定）
    streamer = TickDataStreamer("EURUSD", connection_manager)
    
    try:
        # 購読開始
        if await streamer.subscribe_to_ticks():
            print("EURUSD購読開始")
            
            # 100ティック取得
            tick_count = 0
            async for tick in streamer.stream_ticks(max_ticks=100):
                tick_count += 1
                print(f"Tick #{tick_count}: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}")
            
            print(f"合計 {tick_count} ティック取得完了")
        else:
            print("購読開始失敗")
    
    finally:
        # クリーンアップ
        streamer.unsubscribe()
        await streamer.stop_streaming()


# =============================================================================
# カスタム設定での使用例
# =============================================================================

async def custom_config_example():
    """カスタム設定でのストリーミング例"""
    
    # カスタム設定の作成
    config = StreamerConfig(
        buffer_size=5000,           # 小さめのバッファ
        spike_threshold=2.0,         # より厳しいスパイクフィルター
        backpressure_threshold=0.7,  # 早めのバックプレッシャー
        stats_window_size=500,       # 短い統計ウィンドウ
        max_retries=3,              # 再試行回数を減らす
        retry_delay=2.0             # 再試行間隔を長く
    )
    
    connection_manager = MT5ConnectionManager()
    streamer = TickDataStreamer("USDJPY", connection_manager, config=config)
    
    # 統計情報の定期出力
    async def print_stats():
        while True:
            await asyncio.sleep(10)
            stats = streamer.current_stats
            print("\n=== 統計情報 ===")
            print(f"総ティック数: {stats['total_ticks']}")
            print(f"スパイク検出数: {stats['spike_count']}")
            print(f"バッファ使用率: {stats['buffer_usage']:.1%}")
            if 'performance' in stats:
                perf = stats['performance']
                print(f"平均レイテンシ: {perf.get('latency_avg_ms', 0):.2f}ms")
            print("================\n")
    
    # 統計タスクを並行実行
    stats_task = asyncio.create_task(print_stats())
    
    try:
        await streamer.subscribe_to_ticks()
        
        # 30秒間ストリーミング
        timeout = 30
        start_time = asyncio.get_event_loop().time()
        
        async for tick in streamer.stream_ticks():
            # ティック処理（ここでは単純に出力）
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                  f"USDJPY: {tick.bid:.3f}/{tick.ask:.3f}")
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                break
    
    finally:
        stats_task.cancel()
        await streamer.stop_streaming()


# =============================================================================
# エラーハンドリングの例
# =============================================================================

async def error_handling_example():
    """エラーハンドリングを含む堅牢な実装例"""
    
    connection_manager = MT5ConnectionManager()
    streamer = TickDataStreamer("GBPUSD", connection_manager)
    
    # エラーカウンター
    error_count = 0
    max_errors = 10
    
    # カスタムエラーハンドラー
    def handle_error(error_info):
        nonlocal error_count
        error_count += 1
        
        error_type = error_info.get('error_type', 'Unknown')
        message = error_info.get('message', '')
        timestamp = error_info.get('timestamp', '')
        
        logging.error(f"[{timestamp}] エラー #{error_count} ({error_type}): {message}")
        
        # 重大なエラーの場合は通知
        if error_type == 'ConnectionError':
            # ここで通知処理（メール、Slack等）
            print(f"⚠️ 接続エラー発生: {message}")
        
        # エラーが多すぎる場合は停止
        if error_count >= max_errors:
            print(f"❌ エラー数が上限({max_errors})に達しました。停止します。")
            asyncio.create_task(streamer.stop_streaming())
    
    # バックプレッシャーハンドラー
    def handle_backpressure(bp_info):
        level = bp_info.get('level', 'unknown')
        usage = bp_info.get('buffer_usage', 0)
        
        if level == 'critical':
            logging.critical(f"🔴 クリティカルなバックプレッシャー: {usage:.1%}")
            # 処理を一時停止するなどの対策
        elif level == 'warning':
            logging.warning(f"🟡 バックプレッシャー警告: {usage:.1%}")
    
    # リスナー登録
    streamer.add_listener("error", handle_error)
    streamer.add_listener("backpressure", handle_backpressure)
    
    try:
        # 購読開始（リトライ付き）
        for attempt in range(3):
            if await streamer.subscribe_to_ticks():
                print(f"✅ GBPUSD購読開始成功 (試行 {attempt + 1}/3)")
                break
            else:
                print(f"❌ 購読失敗 (試行 {attempt + 1}/3)")
                await asyncio.sleep(2)
        else:
            raise ConnectionError("購読開始に失敗しました")
        
        # ストリーミング
        async for tick in streamer.stream_ticks():
            try:
                # ティック処理
                process_tick_safely(tick)
                
                # エラーカウントが上限に達したら終了
                if error_count >= max_errors:
                    break
                    
            except Exception as e:
                logging.error(f"ティック処理エラー: {e}")
                continue
    
    except Exception as e:
        logging.error(f"致命的エラー: {e}")
    
    finally:
        print(f"終了処理: 合計 {error_count} 個のエラーが発生しました")
        await streamer.stop_streaming()


def process_tick_safely(tick: Tick):
    """安全なティック処理の例"""
    # ここに実際の処理を実装
    # 例: データベース保存、分析、取引シグナル生成等
    pass


# =============================================================================
# カスタムリスナーの実装例
# =============================================================================

class CustomTickProcessor:
    """カスタムティック処理クラスの例"""
    
    def __init__(self):
        self.tick_buffer = []
        self.stats = {
            'total': 0,
            'spikes': 0,
            'min_bid': float('inf'),
            'max_bid': float('-inf')
        }
    
    async def process_tick(self, tick: Tick):
        """ティック処理メソッド"""
        self.stats['total'] += 1
        
        # 統計更新
        if tick.bid < self.stats['min_bid']:
            self.stats['min_bid'] = tick.bid
        if tick.bid > self.stats['max_bid']:
            self.stats['max_bid'] = tick.bid
        
        # バッファに追加
        self.tick_buffer.append(tick)
        
        # 100ティックごとに処理
        if len(self.tick_buffer) >= 100:
            await self.batch_process()
    
    async def batch_process(self):
        """バッチ処理"""
        print(f"バッチ処理: {len(self.tick_buffer)} ティック")
        print(f"  価格レンジ: {self.stats['min_bid']:.5f} - {self.stats['max_bid']:.5f}")
        
        # ここで実際のバッチ処理
        # 例: データベース一括保存、分析実行等
        
        # バッファクリア
        self.tick_buffer.clear()
    
    def on_error(self, error_info):
        """エラーハンドリング"""
        print(f"プロセッサエラー: {error_info}")
    
    def get_summary(self):
        """処理サマリー取得"""
        return {
            'total_processed': self.stats['total'],
            'price_range': (self.stats['min_bid'], self.stats['max_bid']),
            'buffer_size': len(self.tick_buffer)
        }


async def custom_processor_example():
    """カスタムプロセッサを使用した例"""
    
    connection_manager = MT5ConnectionManager()
    streamer = TickDataStreamer("AUDUSD", connection_manager)
    processor = CustomTickProcessor()
    
    # リスナー登録
    streamer.add_listener("tick", processor.process_tick)
    streamer.add_listener("error", processor.on_error)
    
    try:
        await streamer.subscribe_to_ticks()
        
        # 1000ティック処理
        async for tick in streamer.stream_ticks(max_ticks=1000):
            # プロセッサが自動的に処理
            pass
        
        # 最終サマリー
        summary = processor.get_summary()
        print("\n=== 処理サマリー ===")
        print(f"総処理数: {summary['total_processed']}")
        print(f"価格レンジ: {summary['price_range']}")
        print("===================")
    
    finally:
        await streamer.stop_streaming()


# =============================================================================
# 複数通貨ペアの同時ストリーミング
# =============================================================================

async def multi_symbol_streaming():
    """複数通貨ペアの同時ストリーミング例"""
    
    symbols = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
    connection_manager = MT5ConnectionManager()
    
    async def stream_symbol(symbol: str):
        """個別シンボルのストリーミング"""
        streamer = TickDataStreamer(symbol, connection_manager)
        
        try:
            if await streamer.subscribe_to_ticks():
                print(f"✅ {symbol} ストリーミング開始")
                
                tick_count = 0
                async for tick in streamer.stream_ticks(max_ticks=50):
                    tick_count += 1
                    if tick_count % 10 == 0:
                        print(f"{symbol}: {tick_count} ティック処理済み")
                
                print(f"✅ {symbol} 完了: {tick_count} ティック")
            else:
                print(f"❌ {symbol} 購読失敗")
        
        except Exception as e:
            print(f"❌ {symbol} エラー: {e}")
        
        finally:
            await streamer.stop_streaming()
    
    # 全シンボルを並行処理
    tasks = [stream_symbol(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 結果確認
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            print(f"{symbol}: エラー発生 - {result}")
        else:
            print(f"{symbol}: 正常完了")


# =============================================================================
# メイン実行
# =============================================================================

async def main():
    """メイン実行関数"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("TickDataStreamer 使用例デモ")
    print("=" * 50)
    
    examples = [
        ("基本的な使用例", basic_streaming_example),
        ("カスタム設定例", custom_config_example),
        ("エラーハンドリング例", error_handling_example),
        ("カスタムプロセッサ例", custom_processor_example),
        ("複数通貨ペア例", multi_symbol_streaming)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")
        print("-" * 30)
        
        try:
            await func()
        except Exception as e:
            print(f"エラー: {e}")
        
        print("-" * 30)
        
        if i < len(examples):
            print("\n次の例に進みますか？ (Enter to continue, Ctrl+C to exit)")
            try:
                input()
            except KeyboardInterrupt:
                print("\n終了します")
                break
    
    print("\n全ての例が完了しました")


if __name__ == "__main__":
    # 実行
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nプログラムを終了します")