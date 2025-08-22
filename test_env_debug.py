"""環境変数デバッグスクリプト"""
import os
from pathlib import Path
from dotenv import load_dotenv

print("=== .env File Debug ===")

# 現在のディレクトリ
cwd = Path.cwd()
print(f"Current directory: {cwd}")

# .envファイルの存在確認
env_file = cwd / ".env"
print(f".env file path: {env_file}")
print(f".env file exists: {env_file.exists()}")

if env_file.exists():
    print(f".env file size: {env_file.stat().st_size} bytes")
    
    # .envファイルの内容を確認（パスワードはマスク）
    print("\n=== .env Content (masked) ===")
    with open(env_file, 'r') as f:
        for line in f:
            if 'PASSWORD' in line:
                key = line.split('=')[0]
                print(f"{key}=***MASKED***")
            else:
                print(line.strip())

# dotenvをロード
print("\n=== Loading .env ===")
result = load_dotenv(verbose=True, override=True)
print(f"load_dotenv result: {result}")

# 環境変数を再確認
print("\n=== Environment Variables After Load ===")
print(f"FOREX_MT5_LOGIN: {os.getenv('FOREX_MT5_LOGIN', 'NOT SET')}")
print(f"FOREX_MT5_PASSWORD: {'SET' if os.getenv('FOREX_MT5_PASSWORD') else 'NOT SET'}")
print(f"FOREX_MT5_SERVER: {os.getenv('FOREX_MT5_SERVER', 'NOT SET')}")
print(f"FOREX_MT5_TIMEOUT: {os.getenv('FOREX_MT5_TIMEOUT', 'NOT SET')}")

# プロセスの環境変数を直接確認
print("\n=== All FOREX_ Environment Variables ===")
for key, value in os.environ.items():
    if key.startswith('FOREX_'):
        if 'PASSWORD' in key:
            print(f"{key}=***MASKED***")
        else:
            print(f"{key}={value}")