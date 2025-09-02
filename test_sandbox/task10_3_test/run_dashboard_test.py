"""
パイプラインチャートダッシュボードの動作確認
M5チャート更新問題が解決されたか確認
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import logging
from datetime import datetime
import time

# デバッグログを有効化（ファイルにも出力）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug_log.txt', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """ダッシュボードを起動してM5更新を監視"""
    logger.info("=" * 60)
    logger.info("Pipeline Chart Dashboard - M5 Update Test")
    logger.info("=" * 60)
    logger.info(f"現在時刻: {datetime.now()}")
    logger.info("")
    logger.info("修正内容:")
    logger.info("1. analyzer.py の _is_new_long_bar_complete メソッドを修正")
    logger.info("   - timestamp.second == 0 の条件を削除")
    logger.info("   - 境界を超えたかのチェック方式に変更")
    logger.info("")
    
    try:
        # PipelineChartManagerをインポート
        from pipeline_chart_dashboard import PipelineChartManager
        
        logger.info("PipelineChartManager をインポートしました")
        
        # 設定ファイルがあるか確認
        config_file = Path("configs/pipeline_chart_config.toml")
        if config_file.exists():
            logger.info(f"設定ファイルを使用: {config_file}")
            manager = PipelineChartManager(config_path=str(config_file))
        else:
            logger.info("デフォルト設定で起動")
            manager = PipelineChartManager()
        
        # MT5に接続
        logger.info("MT5に接続中...")
        if manager.connect_mt5():
            logger.info("✅ MT5接続成功")
            
            # ダッシュボードを起動
            logger.info("")
            logger.info("ダッシュボードを起動します...")
            logger.info("ブラウザで http://localhost:8050 を開いてください")
            logger.info("")
            logger.info("M5チャートが正しく更新されるか確認してください:")
            logger.info("- 5分ごとに新しいバーが追加される")
            logger.info("- 時刻が00秒ぴったりでなくても更新される")
            logger.info("")
            logger.info("終了するには Ctrl+C を押してください")
            
            # ダッシュボードを実行
            manager.run()
            
        else:
            logger.error("MT5接続に失敗しました")
            
    except KeyboardInterrupt:
        logger.info("\n終了します...")
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("テスト終了")

if __name__ == "__main__":
    main()