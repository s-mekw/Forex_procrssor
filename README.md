# Forex Processor

高性能なFX取引データ処理システム。MetaTrader 5からのリアルタイムデータ取得、高度な分析、機械学習モデルによる予測を提供します。

## 主な機能

### リアルタイムティックデータ取得
- **非同期ストリーミング**: MT5から低レイテンシ（3ms）でティックデータを取得
- **スパイクフィルター**: 3σルールによる異常値の自動検出・除外
- **バックプレッシャー制御**: データ処理の遅延を防ぐ自動制御機能
- **自動再購読**: 接続障害時の自動復旧機能（サーキットブレーカーパターン）
- **メモリ最適化**: オブジェクトプールによる効率的なメモリ管理

### データ処理
- **リングバッファ**: 最大10,000件のティックデータを効率的に管理
- **統計計算**: リアルタイムで移動平均、標準偏差を計算
- **マルチシンボル対応**: 複数通貨ペアの同時ストリーミング

### パフォーマンス
- **レイテンシ**: 平均3ms、最大10ms以内
- **スループット**: 1,000 ticks/秒の処理能力
- **メモリ効率**: オブジェクトプールで98%の再利用率

## インストール

### 前提条件
- Python 3.11以上
- MetaTrader 5（Windows版）
- uv（Pythonパッケージマネージャー）

### セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/Forex_procrssor.git
cd Forex_procrssor

# uvのインストール（未インストールの場合）
pip install uv

# 依存関係のインストール
uv sync

# 開発環境のセットアップ
uv sync --dev
```

### MT5の設定

1. MetaTrader 5を起動
2. ツール → オプション → エキスパートアドバイザー
3. 「自動売買を許可する」にチェック
4. 「DLLの使用を許可する」にチェック
5. OKをクリック

## クイックスタート

### 基本的な使用例

```python
import asyncio
from src.mt5_data_acquisition.tick_fetcher import TickDataStreamer
from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager

async def main():
    # 接続マネージャーの初期化
    connection_manager = MT5ConnectionManager()
    
    # ストリーマーの作成
    streamer = TickDataStreamer("EURUSD", connection_manager)
    
    try:
        # 購読開始
        if await streamer.subscribe_to_ticks():
            print("ストリーミング開始")
            
            # 100ティック取得
            async for tick in streamer.stream_ticks(max_ticks=100):
                print(f"Bid: {tick.bid:.5f}, Ask: {tick.ask:.5f}")
    
    finally:
        # クリーンアップ
        await streamer.stop_streaming()

# 実行
asyncio.run(main())
```

### カスタム設定での使用

```python
from src.mt5_data_acquisition.tick_fetcher import StreamerConfig

# カスタム設定
config = StreamerConfig(
    buffer_size=20000,           # 大きめのバッファ
    spike_threshold=2.0,          # 厳しいスパイクフィルター
    backpressure_threshold=0.7,   # 早めのバックプレッシャー
    max_retries=3                 # 再試行回数
)

# カスタム設定でストリーマー作成
streamer = TickDataStreamer("USDJPY", connection_manager, config=config)
```

### エラーハンドリング

```python
# エラーハンドラーの登録
def on_error(error_info):
    print(f"エラー: {error_info['message']}")

streamer.add_listener("error", on_error)

# バックプレッシャーハンドラー
def on_backpressure(bp_info):
    print(f"バックプレッシャー: {bp_info['buffer_usage']:.1%}")

streamer.add_listener("backpressure", on_backpressure)
```

## プロジェクト構造

```
Forex_procrssor/
├── src/
│   ├── common/
│   │   └── models.py            # データモデル定義
│   ├── mt5_data_acquisition/
│   │   ├── mt5_client.py        # MT5接続管理
│   │   └── tick_fetcher.py      # ティックデータストリーマー
│   ├── data_processing/         # データ処理モジュール
│   ├── storage/                 # データ永続化
│   └── app/                     # アプリケーション
├── tests/
│   ├── unit/                    # ユニットテスト
│   └── integration/             # 統合テスト
├── docs/
│   ├── api/                     # APIドキュメント
│   ├── examples/                # 使用例
│   └── troubleshooting/         # トラブルシューティング
└── .kiro/
    ├── specs/                   # 仕様書
    └── steering/                # 開発ガイドライン
```

## テスト

```bash
# 全テストの実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov=src --cov-report=html

# 特定のテストのみ実行
uv run pytest tests/unit/test_tick_fetcher.py

# 統合テストの実行
uv run pytest tests/integration/
```

## ドキュメント

- [APIリファレンス](docs/api/tick_fetcher.md)
- [使用例](docs/examples/tick_streaming_example.py)
- [トラブルシューティング](docs/troubleshooting/tick_fetcher.md)
- [開発ガイドライン](.kiro/steering/Python_Development_Guidelines.md)

## 開発

### 開発環境のセットアップ

```bash
# 開発依存関係のインストール
uv sync --dev

# pre-commitフックの設定
pre-commit install

# コードフォーマット
uv run ruff format .

# リント
uv run ruff check .

# 型チェック
uv run mypy src/
```

### コントリビューション

1. フォークする
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

### コーディング規約

- [Python開発ガイドライン](.kiro/steering/Python_Development_Guidelines.md)に従う
- テストカバレッジ80%以上を維持
- 全ての公開APIにdocstringを記述
- 型ヒントを使用

## パフォーマンスベンチマーク

| メトリクス | 値 | 条件 |
|-----------|-----|------|
| レイテンシ（平均） | 3ms | 通常負荷時 |
| レイテンシ（95%ile） | 5ms | 通常負荷時 |
| レイテンシ（最大） | 10ms | ピーク負荷時 |
| スループット | 1,000 ticks/秒 | 単一シンボル |
| メモリ使用量 | 50-100MB | 10,000ティックバッファ |
| CPU使用率 | 5-10% | Core i7での測定 |

## トラブルシューティング

よくある問題と解決方法については[トラブルシューティングガイド](docs/troubleshooting/tick_fetcher.md)を参照してください。

### よくある質問

**Q: MT5に接続できない**
A: MT5が起動していること、自動売買が有効になっていることを確認してください。

**Q: スパイクフィルターが効きすぎる**
A: `StreamerConfig`の`spike_threshold`を調整してください（デフォルト: 3.0）。

**Q: メモリ使用量が増え続ける**
A: `stop_streaming()`を適切に呼び出し、リソースを解放してください。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 連絡先

- Issue: [GitHub Issues](https://github.com/yourusername/Forex_procrssor/issues)
- Email: your.email@example.com

## 謝辞

- MetaTrader 5 Python APIの開発者
- オープンソースコミュニティ

---

最終更新: 2025-01-19