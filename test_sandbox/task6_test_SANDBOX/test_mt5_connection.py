"""
MT5接続テスト - 認証情報の確認
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.common.config import BaseConfig
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager

def test_connection():
    """MT5接続をテスト"""
    print("=" * 60)
    print("MT5 Connection Test")
    print("=" * 60)
    
    # 設定を読み込み
    config = BaseConfig()
    
    # 設定内容を確認（パスワード以外）
    print(f"MT5 Login: {config.mt5_login}")
    print(f"MT5 Server: {config.mt5_server}")
    print(f"MT5 Timeout: {config.mt5_timeout}")
    print(f"Password configured: {'Yes' if config.mt5_password else 'No'}")
    
    # 設定の検証
    if config.validate_mt5_settings():
        print("\n✅ MT5 settings are configured")
    else:
        print("\n❌ MT5 settings are incomplete")
        print("Please check your .env file")
        return False
    
    # 接続テスト
    print("\nAttempting to connect to MT5...")
    
    mt5_config = {
        "account": config.mt5_login,
        "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
        "server": config.mt5_server,
        "timeout": config.mt5_timeout,
    }
    
    connection_manager = MT5ConnectionManager()
    
    try:
        if connection_manager.connect(mt5_config):
            print("✅ Successfully connected to MT5!")
            
            # アカウント情報を表示
            if connection_manager._account_info:
                info = connection_manager._account_info
                print(f"\nAccount Info:")
                print(f"  Login: {info.login}")
                print(f"  Server: {info.server}")
                print(f"  Balance: {info.balance}")
                print(f"  Currency: {info.currency}")
                print(f"  Leverage: 1:{info.leverage}")
            
            connection_manager.disconnect()
            return True
        else:
            print("❌ Failed to connect to MT5")
            print("Please check:")
            print("1. MT5 terminal is installed and running")
            print("2. Your login credentials are correct")
            print("3. Internet connection is available")
            return False
            
    except Exception as e:
        print(f"❌ Error during connection: {e}")
        return False

if __name__ == "__main__":
    test_connection()