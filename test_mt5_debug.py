"""MT5接続デバッグスクリプト"""
import MetaTrader5 as mt5
import os

print("=== MT5 Debug Info ===")
print(f"MT5 version: {mt5.__version__}")

# 環境変数を確認
print("\n=== Environment Variables ===")
print(f"FOREX_MT5_LOGIN: {os.getenv('FOREX_MT5_LOGIN', 'NOT SET')}")
print(f"FOREX_MT5_PASSWORD: {'SET' if os.getenv('FOREX_MT5_PASSWORD') else 'NOT SET'}")
print(f"FOREX_MT5_SERVER: {os.getenv('FOREX_MT5_SERVER', 'NOT SET')}")

# 初期化を試行
print("\n=== MT5 Initialization ===")
result = mt5.initialize()
print(f"Initialize result: {result}")

if not result:
    error = mt5.last_error()
    print(f"Error code: {error[0]}")
    print(f"Error message: {error[1]}")
    
    # パスを指定して再試行
    print("\n=== Trying with path ===")
    paths_to_try = [
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        r"C:\Program Files\MetaTrader 5 - Axiory\terminal64.exe",
        r"C:\Program Files\Axiory MetaTrader 5\terminal64.exe",
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"Found MT5 at: {path}")
            result = mt5.initialize(path=path)
            print(f"Initialize with path result: {result}")
            if result:
                print("Success!")
                break
            else:
                error = mt5.last_error()
                print(f"Error: {error}")
        else:
            print(f"Path not found: {path}")
else:
    print("MT5 initialized successfully!")
    
    # ログインを試行
    login = int(os.getenv("FOREX_MT5_LOGIN", "20046505"))
    password = os.getenv("FOREX_MT5_PASSWORD", "")
    server = os.getenv("FOREX_MT5_SERVER", "Axiory-Demo")
    
    print(f"\n=== Login Attempt ===")
    print(f"Login: {login}")
    print(f"Server: {server}")
    
    login_result = mt5.login(login=login, password=password, server=server)
    print(f"Login result: {login_result}")
    
    if not login_result:
        error = mt5.last_error()
        print(f"Login error: {error}")

# シャットダウン
mt5.shutdown()
print("\n=== Shutdown complete ===")