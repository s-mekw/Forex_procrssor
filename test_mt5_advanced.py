"""MT5詳細デバッグスクリプト"""
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv
import time

# 環境変数をロード
load_dotenv()

print("=== MT5 Advanced Debug ===")
print(f"MT5 version: {mt5.__version__}")

# 環境変数確認
login = int(os.getenv("FOREX_MT5_LOGIN", "20046505"))
password = os.getenv("FOREX_MT5_PASSWORD", "")
server = os.getenv("FOREX_MT5_SERVER", "Axiory-Demo")

print(f"\nCredentials:")
print(f"Login: {login}")
print(f"Server: {server}")
print(f"Password is set: {bool(password)}")

# MT5パス
mt5_path = r"C:\Program Files\Axiory MetaTrader 5\terminal64.exe"
print(f"\nMT5 Path: {mt5_path}")
print(f"Path exists: {os.path.exists(mt5_path)}")

# 初期化試行1: パスなし
print("\n=== Attempt 1: Initialize without path ===")
result = mt5.initialize()
print(f"Result: {result}")
if not result:
    print(f"Error: {mt5.last_error()}")
    mt5.shutdown()

# 初期化試行2: パスあり
print("\n=== Attempt 2: Initialize with path ===")
result = mt5.initialize(path=mt5_path)
print(f"Result: {result}")
if not result:
    print(f"Error: {mt5.last_error()}")
    mt5.shutdown()

# 初期化試行3: ログイン情報付き
print("\n=== Attempt 3: Initialize + Login ===")
result = mt5.initialize(path=mt5_path, login=login, password=password, server=server)
print(f"Result: {result}")
if not result:
    print(f"Error: {mt5.last_error()}")
else:
    print("Success!")
    
    # ターミナル情報
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"\nTerminal info:")
        print(f"  Company: {terminal_info.company}")
        print(f"  Name: {terminal_info.name}")
        print(f"  Build: {terminal_info.build}")
        print(f"  Connected: {terminal_info.connected}")
        print(f"  Trade allowed: {terminal_info.trade_allowed}")
    
    # アカウント情報
    account_info = mt5.account_info()
    if account_info:
        print(f"\nAccount info:")
        print(f"  Login: {account_info.login}")
        print(f"  Server: {account_info.server}")
        print(f"  Balance: {account_info.balance}")
        print(f"  Currency: {account_info.currency}")

mt5.shutdown()
print("\n=== Complete ===")