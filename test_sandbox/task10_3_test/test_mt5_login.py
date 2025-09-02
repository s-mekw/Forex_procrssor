"""
MT5ログインテストスクリプト
task10_3_config.tomlから認証情報を読み込んでMT5への接続をテストします
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import toml
import os
import MetaTrader5 as mt5
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
import logging

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mt5_login():
    """MT5ログインテスト"""
    print("=" * 60)
    print("MT5 ログインテスト")
    print("=" * 60)
    
    # 設定ファイルのパス
    config_path = os.path.join(os.path.dirname(__file__), 'task10_3_config.toml')
    
    # 設定ファイルの存在確認
    if not os.path.exists(config_path):
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        return False
    
    print(f"✅ 設定ファイルを読み込み: {config_path}")
    
    # 設定ファイル読み込み
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
    except Exception as e:
        print(f"❌ 設定ファイルの読み込みエラー: {e}")
        return False
    
    # MT5設定の確認
    if 'mt5' not in config:
        print("❌ 設定ファイルにmt5セクションがありません")
        return False
    
    mt5_config = config['mt5']
    print("\n📋 MT5設定:")
    print(f"   Login: {mt5_config.get('login')}")
    print(f"   Server: {mt5_config.get('server')}")
    print(f"   Path: {mt5_config.get('path')}")
    print(f"   Timeout: {mt5_config.get('timeout')} ms")
    
    # パスワードは一部のみ表示
    password = mt5_config.get('password', '')
    if password:
        masked_password = password[:2] + '*' * (len(password) - 4) + password[-2:] if len(password) > 4 else '*' * len(password)
        print(f"   Password: {masked_password}")
    
    print("\n🔄 MT5接続を試行中...")
    
    # MT5ConnectionManagerを使用して接続
    connection_config = {
        'account': mt5_config.get('login'),
        'password': mt5_config.get('password'),
        'server': mt5_config.get('server'),
        'timeout': mt5_config.get('timeout', 60000),
        'path': mt5_config.get('path'),
        'max_retries': mt5_config.get('max_retries', 3),
        'retry_delay': mt5_config.get('retry_delay', 1.0)
    }
    
    try:
        # MT5ConnectionManagerのインスタンスを作成
        mt5_manager = MT5ConnectionManager(connection_config)
        
        # 接続試行
        success = mt5_manager.connect(connection_config)
        
        if success:
            print("\n✅ MT5ログイン成功！")
            
            # アカウント情報を取得して表示
            account_info = mt5.account_info()
            if account_info:
                print("\n📊 アカウント情報:")
                print(f"   ログイン: {account_info.login}")
                print(f"   会社: {account_info.company}")
                print(f"   サーバー: {account_info.server}")
                print(f"   残高: ${account_info.balance:,.2f}")
                print(f"   レバレッジ: 1:{account_info.leverage}")
                print(f"   通貨: {account_info.currency}")
            
            # ターミナル情報を取得
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print("\n💻 ターミナル情報:")
                print(f"   会社: {terminal_info.company}")
                print(f"   ビルド: {terminal_info.build}")
                print(f"   接続状態: {'接続中' if terminal_info.connected else '切断'}")
                print(f"   取引許可: {'有効' if terminal_info.trade_allowed else '無効'}")
            
            # 利用可能なシンボルをいくつか表示
            symbols = mt5.symbols_get()
            if symbols:
                print(f"\n📈 利用可能なシンボル数: {len(symbols)}")
                print("   最初の10個のシンボル:")
                for i, symbol in enumerate(symbols[:10]):
                    print(f"   {i+1:2d}. {symbol.name}")
            
            # 切断
            mt5_manager.disconnect()
            print("\n✅ MT5から正常に切断しました")
            
            return True
            
        else:
            print("\n❌ MT5ログイン失敗")
            # エラー情報を取得
            error = mt5.last_error()
            if error:
                print(f"   エラーコード: {error[0]}")
                print(f"   エラーメッセージ: {error[1]}")
            return False
            
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 念のためクリーンアップ
        try:
            mt5.shutdown()
        except:
            pass

if __name__ == "__main__":
    success = test_mt5_login()
    
    print("\n" + "=" * 60)
    if success:
        print("テスト結果: ✅ 成功")
    else:
        print("テスト結果: ❌ 失敗")
    print("=" * 60)
    
    # 終了コード
    sys.exit(0 if success else 1)