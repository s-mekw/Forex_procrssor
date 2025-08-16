# FX取引支援システム技術設計書

## 概要
Forex_procrssorは、MetaTrader 5との直接統合によるFX取引支援システムです。本設計書では、リアルタイムデータ処理、機械学習予測、時系列データ永続化、可視化ダッシュボードの技術アーキテクチャと詳細設計を定義します。

## システムアーキテクチャ

### 全体アーキテクチャ
```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   MetaTrader 5  │─────▶│  Data Ingestion │─────▶│ Data Processing │
│     Terminal    │      │    Component    │      │    Pipeline     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                  │                         │
                                  ▼                         ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│     InfluxDB    │◀─────│   Time Series   │◀─────│   ML Inference  │
│   (Historical)  │      │     Storage     │      │   (PatchTST)    │
└─────────────────┘      └─────────────────┘      └─────────────────┘
         │                                                  │
         ▼                                                  ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Dash Dashboard │◀─────│    WebSocket    │◀─────│   Prediction    │
│   (Real-time)   │      │     Server      │      │     Results     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

### データフローアーキテクチャ
1. **リアルタイムフロー**: MT5 → WebSocket → 処理 → ダッシュボード
2. **バッチフロー**: MT5 → バッファ → Polars処理 → InfluxDB
3. **MLフロー**: InfluxDB → 特徴生成 → PatchTST → 予測結果

### 非同期処理モデル
```python
# コア非同期アーキテクチャ
class AsyncArchitecture:
    - asyncio Event Loop: メインイベントループ
    - ThreadPoolExecutor: CPU集約的タスク（ML推論）
    - ProcessPoolExecutor: データ処理パイプライン
    - WebSocket Server: リアルタイム通信
    - Background Tasks: 定期実行ジョブ
```

## 詳細コンポーネント設計

### 1. MT5データ取得コンポーネント

#### 1.1 接続管理
```python
class MT5ConnectionManager:
    """MT5接続の生存管理とヘルスチェック"""
    
    # 接続設定
    - login: int  # MT5アカウント番号
    - password: str  # 暗号化パスワード
    - server: str  # ブローカーサーバー
    - timeout: int = 30000  # 接続タイムアウト（ms）
    
    # 接続管理
    async def connect() -> bool:
        """非同期接続確立（指数バックオフ付き）"""
        - MT5ターミナル初期化
        - 認証とセッション確立
        - 接続プール管理
    
    # ヘルスチェック
    async def health_check() -> HealthStatus:
        """10秒間隔の接続状態監視"""
        - ターミナル状態確認
        - ネットワーク遅延測定
        - 自動再接続トリガー
```

#### 1.2 ティックデータストリーミング
```python
class TickDataStreamer:
    """リアルタイムティックデータの非同期ストリーミング"""
    
    # バッファリング設定
    - buffer_size: int = 10000  # リングバッファサイズ
    - batch_interval: float = 0.1  # バッチ処理間隔（秒）
    
    # ストリーミング実装
    async def stream_ticks(symbol: str) -> AsyncIterator[Tick]:
        """非同期ティックストリーム生成"""
        - WebSocketプッシュ
        - バックプレッシャー制御
        - 異常値フィルタリング（3σルール）
    
    # フロー制御
    class BackpressureController:
        - max_queue_size: int = 50000
        - drop_policy: Literal["oldest", "newest"]
        - warning_threshold: float = 0.8
```

#### 1.3 履歴データ取得
```python
class HistoricalDataFetcher:
    """効率的な履歴OHLCデータ取得"""
    
    # バッチ設定
    - chunk_size: int = 10000  # バッチサイズ
    - max_workers: int = 4  # 並列ワーカー数
    
    # データ取得
    async def fetch_ohlc(
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pl.LazyFrame:
        """並列バッチ処理による高速取得"""
        - 日付範囲分割
        - 並列フェッチ
        - Polars LazyFrame統合
```

### 2. 高速データ処理パイプライン

#### 2.1 RealtimePipelineによるマルチタイムフレーム処理
```python
from typing import AsyncIterator
import polars as pl

class RealtimePipeline:
    """リアルタイムデータフローとマルチタイムフレーム分析のオーケストレーション"""

    # RCI期間設定
    RCI_PERIODS_1M = [9, 13, 24, 33, 48, 66, 108]
    RCI_PERIODS_5M = [24, 33, 48, 66, 108]  # 120, 165, 240, 330, 540の代替

    async def process_data(
        self,
        ohlc_1m_stream: AsyncIterator[pl.DataFrame]
    ) -> AsyncIterator[pl.DataFrame]:
        """
        1分足ストリームからマルチタイムフレーム指標を計算する
        要件2.4, 2.5を実装
        """
        
        # 連続した1分足データを保持するバッファ
        data_buffer_1m = pl.DataFrame()

        async for ohlc_1m_chunk in ohlc_1m_stream:
            data_buffer_1m = pl.concat([data_buffer_1m, ohlc_1m_chunk])

            # --- 5分足データの動的リサンプリング (要件2.5.1, 2.5.2) ---
            ohlc_5m = self._resample_to_5m(data_buffer_1m)

            if ohlc_5m.is_empty():
                continue # 5分足バーがまだ確定していない

            # --- RCI計算のオーケストレーション (要件2.5.3, 2.5.4) ---
            # 1分足データで短期RCIを計算
            rci_1m = rci_engine.fast_calculate_multiple(
                data=data_buffer_1m, 
                periods=self.RCI_PERIODS_1M,
                column_name='close'
            )
            
            # 5分足データで長期RCIを計算
            rci_5m = rci_engine.fast_calculate_multiple(
                data=ohlc_5m, 
                periods=self.RCI_PERIODS_5M,
                column_name='close'
            )
            
            # --- 結果の統合 (要件2.5.5) ---
            # 5分足RCIを1分足のタイムスタンプにアップサンプリングして結合
            final_df = self._merge_results(rci_1m, rci_5m)
            
            # バッファ管理: 古いデータを削除
            # 例: 最大RCI期間の2倍を保持
            max_period_1m = max(self.RCI_PERIODS_1M)
            data_buffer_1m = data_buffer_1m.tail(max_period_1m * 2)

            yield final_df

    def _resample_to_5m(self, df_1m: pl.DataFrame) -> pl.DataFrame:
        """1分足から5分足へダウンサンプリング"""
        return (
            df_1m.lazy()
            .group_by_dynamic(
                "timestamp",
                every="5m",
                by="symbol",
                closed="left"
            )
            .agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                # Volumeはモデル入力から除外する方針（集計自体はストレージ用途で継続可）
            ])
            .collect()
        )
        
    def _merge_results(self, rci_1m: pl.DataFrame, rci_5m: pl.DataFrame) -> pl.DataFrame:
        """1分足と5分足のRCI結果を統合"""
        
        # 5分足の結果を1分足のタイムスタンプに結合（前方フィル）
        merged_lazy = (
            rci_1m.lazy()
            .join_asof(
                rci_5m.lazy(),
                on="timestamp",
                by="symbol",
                strategy="forward"
            )
        )
        return merged_lazy.collect()

```

#### 2.2 テクニカル指標計算
```python
class TechnicalIndicatorEngine:
    """polars-ta-extensionによる高速指標計算"""
    
    # 指標設定
    INDICATORS = {
        "sma": {"periods": [20, 50, 100, 200]},
        "ema": {"periods": [20, 50, 100, 200]},
        "rsi": {"period": 14},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bbands": {"period": 20, "std": 2},
        "atr": {"period": 14},
        "adx": {"period": 14},
        "cci": {"period": 20},
        "rci": {"periods": [9, 26, 52]},
    }
    
    # 並列計算
    def calculate_indicators(df: pl.LazyFrame) -> pl.LazyFrame:
        """ベクトル化された並列指標計算"""
        return df.with_columns([
            # SMA計算（ウィンドウ関数）
            pl.col("close").rolling_mean(window_size=period)
            .over("symbol").alias(f"sma_{period}")
            for period in INDICATORS["sma"]["periods"]
        ] + [
            # RSI計算（カスタム実装）
            calculate_rsi(pl.col("close"), period=14)
            .alias("rsi_14")
        ])
```

#### 2.3 メモリ最適化
```python
class MemoryOptimizer:
    """Float32とチャンク処理によるメモリ効率化"""
    
    # メモリプロファイリング
    @memory_profiler
    def optimize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
        """データ型最適化とメモリ削減"""
        return df.with_columns([
            # Float64 → Float32変換（精度要件：小数点以下2桁）
            pl.col(pl.Float64).cast(pl.Float32),
            # 文字列 → カテゴリカル変換
            pl.col(pl.Utf8).cast(pl.Categorical),
            # 不要カラムの削除
        ]).shrink_to_fit()
    
    # チャンク処理
    def process_in_chunks(
        lazy_df: pl.LazyFrame,
        chunk_size: int = 100_000
    ) -> Iterator[pl.DataFrame]:
        """メモリ制限下でのストリーミング処理"""
        for chunk in lazy_df.collect(streaming=True):
            yield optimize_dataframe(chunk)
```

### 3. InfluxDB時系列ストレージ

#### 3.1 データモデル設計
```python
# InfluxDB Line Protocol設計
measurement = "fx_ohlc"
tags = {
    "symbol": "USDJPY",      # 通貨ペア
    "timeframe": "M1",       # 時間足
    "broker": "mt5_demo",    # ブローカー識別子
}
fields = {
    "open": 145.123,         # Float32
    "high": 145.234,         # Float32
    "low": 145.012,          # Float32
    "close": 145.189,        # Float32
    "volume": 12345.0,       # Float32
    "spread": 0.003,         # Float32
}
timestamp = 1704067200000000000  # ナノ秒精度
```

#### 3.2 パフォーマンス最適化
```python
class InfluxDBOptimizer:
    """高速書き込みとクエリ最適化"""
    
    # バッチ書き込み設定
    WRITE_OPTIONS = {
        "batch_size": 5000,
        "flush_interval": 1000,  # ms
        "jitter_interval": 100,  # ms
        "retry_interval": 5000,  # ms
        "max_retries": 3,
        "max_retry_delay": 30000,  # ms
        "exponential_base": 2,
    }
    
    # インデックス戦略
    def create_indexes():
        """効率的なクエリのためのインデックス作成"""
        # タグインデックス（自動）
        # 時間範囲インデックス（自動）
        # カーディナリティ管理
        pass
    
    # 保持ポリシー
    RETENTION_POLICIES = {
        "tick_data": {"duration": "7d", "shard": "1h"},
        "1min_data": {"duration": "90d", "shard": "1d"},
        "1hour_data": {"duration": "2y", "shard": "7d"},
        "1day_data": {"duration": "10y", "shard": "30d"},
    }
```

#### 3.3 ハイブリッドストレージ
```python
class HybridStorage:
    """InfluxDB + Parquet + Redisキャッシュによるハイブリッド戦略"""
    
    # 設定 (config.py から注入)
    - hot_data_retention_days: int
    - cache_ttl_seconds: int

    # 依存性注入
    - influx_handler: InfluxDBHandler
    - parquet_handler: ParquetHandler
    - cache_client: RedisClient

    # 統合クエリインターフェース
    async def query_unified(
        start: datetime,
        end: datetime,
        symbol: str
    ) -> pl.LazyFrame:
        """透過的なホット/コールドデータアクセスとキャッシュ"""
        cache_key = f"query:{symbol}:{start.isoformat()}:{end.isoformat()}"
        
        # 1. キャッシュ確認
        cached_result = await self.cache_client.get(cache_key)
        if cached_result:
            # Prometheusメトリクス更新 (キャッシュヒット)
            query_cache_hits.inc()
            return polars_from_arrow(cached_result)

        # 2. キャッシュミス時はDB/ストレージから取得
        # Prometheusメトリクス更新 (キャッシュミス)
        query_cache_misses.inc()

        # データソース決定ロジック...
        hot_data = await query_influxdb(...)
        cold_data = pl.scan_parquet(...)
        result_df = pl.concat([cold_data, hot_data], how="vertical")

        # 3. 結果をキャッシュに保存
        await self.cache_client.set(cache_key, result_df.to_arrow(), ex=self.cache_ttl_seconds)
        
        return result_df

### 3.4 データ整合性管理
```python
from typing import Protocol

# --- 1. インターフェース定義 (依存性逆転の原則) ---
class ChecksumEngine(Protocol):
    def calculate_batch_checksum(self, data_batch: List[OHLC]) -> Dict[str, Any]: ...

class DuplicateDetector(Protocol):
    async def find_duplicates(self, data_batch: List[OHLC]) -> List[OHLC]: ...
    
# 他のコンポーネントのインターフェースも同様に定義...

# --- 2. 依存性注入を適用したDataIntegrityManager ---
class DataIntegrityManager:
    """統合データ整合性管理システム（DI適用）"""

    # 依存性のコンストラクタ注入
    def __init__(
        self,
        checksum_engine: ChecksumEngine,
        duplicate_detector: DuplicateDetector,
        backup_manager: BackupManager,
        recovery_manager: RecoveryManager,
        repair_engine: RepairEngine,
        alert_manager: AlertManager,
        config_manager: ConfigManager  # 設定も外部から注入
    ):
        self.checksum_engine = checksum_engine
        self.duplicate_detector = duplicate_detector
        # ... 他のコンポーネントを初期化
        self.config = config_manager.get_integrity_settings() # 外部化された設定を取得
    
    # 強化されたロギング
    async def validate_and_process_batch(self, data_batch: List[OHLC]) -> IntegrityValidationResult:
        logger.info("Starting integrity validation for batch", batch_size=len(data_batch))
        try:
            # 各処理ステップで詳細なログを出力
            duplicates = await self.duplicate_detector.find_duplicates(data_batch)
            if duplicates:
                logger.warning("Duplicates found", count=len(duplicates))
            
            # ... 他の処理とロギング ...
            logger.info("Integrity validation completed successfully.")
            return IntegrityValidationResult(...)
        except Exception as e:
            logger.error("Integrity validation failed", error=str(e), exc_info=True)
            raise

# --- 3. 設定ファイル (settings.toml) による外部化 ---
# [data_integrity]
# default_checksum_algorithm = "blake2b"
# supported_algorithms = ["sha256", "blake2b"]
#
# [data_integrity.circuit_breaker]
# failure_threshold = 3
# recovery_timeout_seconds = 120

```
```

### 4. PatchTSTモデル統合

#### 4.1 モデルアーキテクチャ
```python
class PatchTSTPredictor:
    """Transformerベースの時系列予測モデル"""
    
    # モデル設定
    MODEL_CONFIG = {
        "input_size": 30,         # 過去30期間（1分足ベース）
        "forecast_length": 15,    # 1〜15分先（15ステップ）
        "patch_length": 5,        # パッチサイズ
        "stride": 1,              # ストライド
        "hidden_size": 128,       # 隠れ層サイズ
        "n_heads": 8,             # アテンションヘッド数
        "n_layers": 3,            # Transformerレイヤー数
        "dropout": 0.1,           # ドロップアウト率
        "batch_size": 32,         # バッチサイズ
        "num_targets": 1,         # 予測対象数（デフォルト: 終値のみ）
        # num_targets を 3 に設定すると「終値＋1分RCI24＋1分RCI48」の3出力を許容
        "quantile_levels": [0.5, 0.8, 0.9, 0.95],  # 分位出力（中央値0.5含む）
        "use_quantile_heads": True,                 # 分位回帰ヘッドを有効化
    }
    
    # GPU/CPU自動選択
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._build_model().to(self.device)

    # 入力前処理
    # - z-score正規化（学習時統計を保存し推論で使用）
    # - 欠損は前方補完。必要に応じ欠損フラグ列を付加

    # パッチ化/埋め込み
    # - patch_length=5 で系列を分割（30→6トークン）
    # - Linear埋め込み＋学習可能位置埋め込み

    # 出力仕様（形状）
    # - quantiles: [batch, forecast_length(=15), Q] を返す
    # - quantile_levels をメタデータで返却（例: [0.5, 0.8, 0.9, 0.95]）
    # - 単調性担保のため分位出力をソート（lower≤median≤upper）

    # NeuralForecastアダプタ方針
    # - 学習/推論ともに Nixtla NeuralForecast の PatchTST を利用
    # - データ処理は Polars を標準とし、NF呼出直前のみ Pandas へ変換
    # - 公開APIは Polars/tensor を返却し、Pandas型は外部へ露出しない

#### 4.5 NeuralForecast 統合ガイド
```python
# 依存: neuralforecast, pandas, pytorch-lightning
# uv add neuralforecast pandas pytorch-lightning

import torch
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST as NF_PatchTST
from neuralforecast.losses.pytorch import QuantileLoss

def build_nf_model():
    return NF_PatchTST(
        h=15,
        input_size=30,
        patch_len=5,
        stride=1,
        n_head=8,
        n_layers=3,
        hidden_size=128,
        dropout=0.1,
        loss=QuantileLoss(quantiles=[0.5, 0.8, 0.9, 0.95]),
        hist_exog_list=["ema_5", "ema_20", "rci_1m_24", "rci_1m_48"],
        trainer_kwargs={
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'precision': '32-true',
        },
        random_seed=42,
    )

def fit_and_predict(df_pl):
    # df_pl: Polars DataFrame → long-form pandas へ境界で変換
    # 必須列: unique_id, ds, y(=close)
    df_pd: pd.DataFrame = df_pl.to_pandas()
    df_pd = df_pd.rename(columns={'close': 'y'})

    model = build_nf_model()
    nf = NeuralForecast(models=[model], freq='T')
    nf.fit(df=df_pd, val_size=0)
    preds = nf.predict(df=df_pd, h=15)
    return preds  # NF標準の分位列（median/q0.8/q0.9/q0.95 等）
```

実装ポリシー:
- データ処理はPolars優先、NeuralForecast呼び出し直前のみPandasへ変換
- Volumeは入力特徴から除外
- 複数通貨は `unique_id` で管理（例: "USDJPY_M1"）
```

#### 4.2 履歴データ取得インターフェース
```python
class MLDataLoader:
    """InfluxDBからの効率的なML用データ取得"""
    
    async def load_training_data(
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        features: List[str]
    ) -> torch.Tensor:
        """特徴量エンジニアリング済みデータの取得"""
        
        # InfluxDBクエリ
        query = f'''
        from(bucket: "fx_features")
            |> range(start: {start_date}, stop: {end_date})
            |> filter(fn: (r) => r["symbol"] == "{symbol}")
            |> pivot(rowKey: ["_time"], columnKey: ["_field"])
            |> keep(columns: {features})
        '''
        
        # Polars DataFrame変換
        df = await influx_to_polars(query)
        
        # 特徴量正規化
        # Volumeは特徴量から除外（features選択で含めない）。標準化は選択列に限定
        df = df.with_columns([
            (pl.col(feat) - pl.col(feat).mean()) / pl.col(feat).std()
            for feat in features
        ])
        
        # PyTorch Tensor変換
        return torch.tensor(
            df.to_numpy(), 
            dtype=torch.float32,
            device=self.device
        )
```

#### 4.3 推論パイプライン
```python
class InferencePipeline:
    """バッチ推論とストリーミング推論"""
    
    # バッチ推論（定期実行）
    @torch.no_grad()
    async def batch_inference(
        self,
        data_loader: DataLoader
    ) -> pl.DataFrame:
        """GPUバッチ処理による高速推論"""
        predictions = []
        
        for batch in data_loader:
            # GPU推論
            output = self.model(batch.to(self.device))
            predictions.append(output.cpu().numpy())
        
        # 予測結果の構造化
        return self._format_predictions(predictions)
    
    # ストリーミング推論（リアルタイム）
    async def streaming_inference(
        self,
        tick_stream: AsyncIterator[Tick]
    ) -> AsyncIterator[Prediction]:
        """低遅延リアルタイム予測"""
        buffer = deque(maxlen=self.MODEL_CONFIG["input_size"])
        
        async for tick in tick_stream:
            buffer.append(tick)
            if len(buffer) == buffer.maxlen:
                # 推論実行
                features = self._extract_features(buffer)
                prediction = await self._predict_async(features)
                yield prediction
```

#### 4.4 モデル管理
```python
class ModelVersionManager:
    """モデルバージョニングとA/Bテスト"""
    
    # バージョン管理
    models: Dict[str, PatchTSTPredictor] = {}
    active_version: str = "v1.0.0"
    
    # モデル登録
    def register_model(
        version: str,
        model_path: Path,
        metrics: Dict[str, float]
    ):
        """新モデルの登録と検証"""
        # モデルロード
        # パフォーマンステスト
        # メトリクス記録
        pass
    
    # A/Bテスト
    async def ab_test_inference(
        data: torch.Tensor,
        test_ratio: float = 0.1
    ) -> Tuple[Prediction, str]:
        """本番環境でのA/Bテスト実行"""
        if random.random() < test_ratio:
            return await self.models["challenger"].predict(data), "challenger"
        else:
            return await self.models[self.active_version].predict(data), "champion"
```

### 5. Dashダッシュボード

#### 5.1 WebSocketリアルタイム更新
```python
class DashWebSocketServer:
    """双方向リアルタイム通信"""
    
    # WebSocket設定
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.socket_io = SocketIO(self.app.server)
        self.clients: Set[str] = set()
    
    # リアルタイムプッシュ
    @socket_io.on("subscribe")
    async def handle_subscription(sid, data):
        """クライアント購読管理"""
        symbol = data["symbol"]
        self.clients.add(sid)
        
        # ティックデータストリーム購読
        async for tick in tick_stream(symbol):
            await self.socket_io.emit(
                "tick_update",
                {"symbol": symbol, "data": tick},
                room=sid
            )
```

#### 5.2 チャート最適化
```python
class OptimizedChartRenderer:
    """Plotlyチャートのパフォーマンス最適化"""
    
    # データ仮想化
    def create_virtualized_chart(
        df: pl.DataFrame,
        viewport_size: int = 1000
    ) -> go.Figure:
        """大規模データの仮想化レンダリング"""
        # ビューポート内データのみレンダリング
        # ダウンサンプリング（LTTB アルゴリズム）
        # WebGLレンダラー使用
        
        fig = go.Figure()
        fig.add_trace(go.Scattergl(  # WebGL使用
            x=df["timestamp"],
            y=df["close"],
            mode="lines",
            line=dict(width=1),
        ))
        
        # レイアウト最適化
        fig.update_layout(
            uirevision=True,  # ズーム状態保持
            hovermode="x unified",  # ホバー最適化
            xaxis=dict(
                rangeslider=dict(visible=False),  # 軽量化
            )
        )
        
        return fig
    
    # インクリメンタル更新
    def update_chart_data(
        fig: go.Figure,
        new_data: pl.DataFrame
    ) -> go.Figure:
        """差分更新による高速レンダリング"""
        # extendTracesによる追加のみ
        # 古いデータの自動削除
        # アニメーション無効化
        pass
```

#### 5.3 ダッシュボードレイアウト
```python
class DashboardLayout:
    """レスポンシブダッシュボード設計"""
    
    def create_layout(self) -> html.Div:
        return html.Div([
            # ヘッダー
            dbc.Navbar([
                dbc.NavbarBrand("Forex_procrssor"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Real-time", href="/realtime")),
                    dbc.NavItem(dbc.NavLink("Analysis", href="/analysis")),
                    dbc.NavItem(dbc.NavLink("Backtest", href="/backtest")),
                ])
            ]),
            
            # メインコンテンツ
            dbc.Container([
                dbc.Row([
                    # 価格チャート
                    dbc.Col([
                        dcc.Graph(
                            id="price-chart",
                            config={"displayModeBar": False}
                        ),
                    ], width=8),
                    
                    # 予測パネル
                    dbc.Col([
                        html.Div(id="prediction-panel"),
                        html.Div(id="indicators-panel"),
                    ], width=4),
                ]),
                
                # リアルタイムメトリクス
                dbc.Row([
                    dbc.Col([
                        html.Div(id="metrics-grid"),
                    ], width=12),
                ]),
            ], fluid=True),
            
            # WebSocket接続
            html.Div(id="ws-connection", style={"display": "none"}),
        ])
```

#### 5.4 AI予測可視化コンポーネント
```python
class PredictionVisualization:
    """AI予測の可視化コンポーネント（要件5.2.4対応）"""
    
    # 予測線設定
    PREDICTION_LINE_CONFIG = {
        "color": "rgba(255, 99, 71, 0.9)",    # トマト色、90%不透明度
        "width": 3,                           # 線の太さ
        "dash": "solid",                      # 実線
        "name": "AI予測線"
    }
    
    # 信頼区間設定（複数レベル）
    CONFIDENCE_INTERVALS = {
        "95%": {
            "upper_color": "rgba(70, 130, 180, 0.15)",  # 薄いスチールブルー
            "lower_color": "rgba(70, 130, 180, 0.15)",
            "fill_color": "rgba(70, 130, 180, 0.1)",    # 帯の塗りつぶし
            "name": "95% 信頼区間",
            "line_width": 1
        },
        "80%": {
            "upper_color": "rgba(70, 130, 180, 0.25)",  # やや濃いスチールブルー
            "lower_color": "rgba(70, 130, 180, 0.25)", 
            "fill_color": "rgba(70, 130, 180, 0.15)",
            "name": "80% 信頼区間",
            "line_width": 1
        },
        "50%": {
            "upper_color": "rgba(70, 130, 180, 0.35)",  # 濃いスチールブルー
            "lower_color": "rgba(70, 130, 180, 0.35)",
            "fill_color": "rgba(70, 130, 180, 0.25)",
            "name": "50% 信頼区間",
            "line_width": 1
        }
    }
    
    # 描画順序設定（重ね合わせ制御）
    LAYER_ORDER = {
        "confidence_95": 1,    # 最背面
        "confidence_80": 2,
        "confidence_50": 3,
        "prediction_line": 4,  # 最前面
        "actual_price": 5      # 実際の価格（最優先）
    }
```

#### 5.4.1 予測線の描画実装
```python
def add_prediction_line(
    fig: go.Figure,
    prediction_data: pl.DataFrame,
    symbol: str
) -> go.Figure:
    """AI予測線をチャートに追加"""
    
    # 予測線の追加
    fig.add_trace(go.Scattergl(
        x=prediction_data["timestamp"],
        y=prediction_data["predicted_price"],
        mode="lines",
        name=f"{symbol} {PredictionVisualization.PREDICTION_LINE_CONFIG['name']}",
        line=dict(
            color=PredictionVisualization.PREDICTION_LINE_CONFIG["color"],
            width=PredictionVisualization.PREDICTION_LINE_CONFIG["width"],
            dash=PredictionVisualization.PREDICTION_LINE_CONFIG["dash"]
        ),
        opacity=0.9,
        hovertemplate="<b>AI予測</b><br>" +
                     "時刻: %{x}<br>" +
                     "予測価格: %{y:.5f}<br>" +
                     "<extra></extra>",
        # レイヤー順序制御
        layer="above"
    ))
    
    return fig
```

#### 5.4.2 信頼区間の描画実装
```python
def add_confidence_bands(
    fig: go.Figure,
    prediction_data: pl.DataFrame,
    confidence_level: str,
    symbol: str
) -> go.Figure:
    """信頼区間帯をチャートに追加"""
    
    config = PredictionVisualization.CONFIDENCE_INTERVALS[confidence_level]
    
    # 上限線の追加
    fig.add_trace(go.Scattergl(
        x=prediction_data["timestamp"],
        y=prediction_data[f"upper_bound_{confidence_level}"],
        mode="lines",
        name=f"{symbol} {config['name']} 上限",
        line=dict(
            color=config["upper_color"],
            width=config["line_width"]
        ),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    # 下限線の追加（帯の塗りつぶし付き）
    fig.add_trace(go.Scattergl(
        x=prediction_data["timestamp"],
        y=prediction_data[f"lower_bound_{confidence_level}"],
        mode="lines",
        name=config["name"],
        line=dict(
            color=config["lower_color"],
            width=config["line_width"]
        ),
        fill="tonexty",  # 前のトレースとの間を塗りつぶし
        fillcolor=config["fill_color"],
        hovertemplate="<b>" + config["name"] + "</b><br>" +
                     "時刻: %{x}<br>" +
                     "下限: %{y:.5f}<br>" +
                     "<extra></extra>"
    ))
    
    return fig
```

#### 5.4.3 リアルタイム予測更新
```python
class RealTimePredictionUpdater:
    """リアルタイム予測データの更新管理"""
    
    def __init__(self, max_predictions: int = 100):
        self.prediction_buffer = deque(maxlen=max_predictions)
        self.update_interval = 5.0  # 5秒間隔で更新
    
    async def update_prediction_display(
        self,
        fig: go.Figure,
        new_prediction: Dict[str, Any]
    ) -> go.Figure:
        """新しい予測データでチャートを更新"""
        
        # バッファに新しい予測を追加
        self.prediction_buffer.append(new_prediction)
        
        # 古い予測トレースを削除
        fig.data = [
            trace for trace in fig.data 
            if not trace.name.startswith("AI予測")
        ]
        
        # 新しい予測データでトレースを再構築
        current_predictions = pl.DataFrame(list(self.prediction_buffer))
        
        # 予測線を追加
        fig = add_prediction_line(fig, current_predictions, "USDJPY")
        
        # 信頼区間を追加（複数レベル）
        for level in ["95%", "80%", "50%"]:
            fig = add_confidence_bands(fig, current_predictions, level, "USDJPY")
        
        return fig
    
    def configure_prediction_colors(
        self,
        theme: Literal["light", "dark", "auto"] = "auto"
    ) -> Dict[str, Any]:
        """テーマに応じた色設定の調整"""
        
        if theme == "dark":
            # ダークテーマ用の色設定
            return {
                "prediction_line": "rgba(255, 165, 0, 0.9)",  # オレンジ
                "confidence_primary": "rgba(135, 206, 250, 0.2)"  # ライトスカイブルー
            }
        elif theme == "light":
            # ライトテーマ用の色設定
            return {
                "prediction_line": "rgba(220, 20, 60, 0.9)",   # クリムゾン
                "confidence_primary": "rgba(70, 130, 180, 0.2)"  # スチールブルー
            }
        else:
            # 自動選択（システム設定に依存）
            return PredictionVisualization.PREDICTION_LINE_CONFIG
```

#### 5.4.4 パフォーマンス最適化
```python
class PredictionRenderingOptimizer:
    """予測可視化のパフォーマンス最適化"""
    
    # データ間引き設定
    DECIMATION_CONFIG = {
        "viewport_width": 1920,      # 表示領域の幅
        "max_points_per_trace": 500, # トレース当たりの最大点数
        "decimation_threshold": 1000 # 間引き開始閾値
    }
    
    def optimize_prediction_data(
        self,
        prediction_df: pl.DataFrame,
        viewport_start: datetime,
        viewport_end: datetime
    ) -> pl.DataFrame:
        """表示領域に応じたデータの最適化"""
        
        # ビューポート内のデータのみ抽出
        visible_data = prediction_df.filter(
            (pl.col("timestamp") >= viewport_start) &
            (pl.col("timestamp") <= viewport_end)
        )
        
        # データ点数が閾値を超える場合は間引き
        if len(visible_data) > self.DECIMATION_CONFIG["decimation_threshold"]:
            # LTTB (Largest Triangle Three Buckets) アルゴリズムで間引き
            target_points = self.DECIMATION_CONFIG["max_points_per_trace"]
            visible_data = self._apply_lttb_decimation(visible_data, target_points)
        
        return visible_data
    
    def _apply_lttb_decimation(
        self,
        df: pl.DataFrame,
        target_points: int
    ) -> pl.DataFrame:
        """LTTB アルゴリズムによるデータ間引き"""
        # WebGLレンダラーと組み合わせて使用
        # 視覚的品質を保ちながらレンダリング性能を向上
        pass
```

#### 5.4.5 既存システムとの統合
```python
class PredictionChartIntegrator:
    """予測可視化と既存チャートシステムの統合"""
    
    def integrate_with_optimized_chart(
        self,
        base_chart: OptimizedChartRenderer,
        prediction_visualizer: PredictionVisualization
    ) -> go.Figure:
        """最適化チャートレンダラーとの統合"""
        
        # ベースチャートの作成
        fig = base_chart.create_virtualized_chart(price_data)
        
        # AI予測レイヤーの追加
        fig = prediction_visualizer.add_prediction_line(fig, prediction_data)
        
        # 信頼区間レイヤーの追加
        for level in ["95%", "80%", "50%"]:
            fig = prediction_visualizer.add_confidence_bands(
                fig, prediction_data, level, symbol
            )
        
        # レイヤー順序の調整
        fig = self._adjust_layer_order(fig)
        
        return fig
    
    def _adjust_layer_order(self, fig: go.Figure) -> go.Figure:
        """トレースの描画順序を最適化"""
        # 信頼区間 → 予測線 → 実際の価格 の順序で描画
        ordered_traces = []
        
        # 1. 信頼区間トレース（幅の広い順）
        confidence_traces = [t for t in fig.data if "信頼区間" in t.name]
        confidence_traces.sort(key=lambda t: int(t.name.split("%")[0]), reverse=True)
        ordered_traces.extend(confidence_traces)
        
        # 2. 予測線トレース
        prediction_traces = [t for t in fig.data if "予測線" in t.name]
        ordered_traces.extend(prediction_traces)
        
        # 3. 実際の価格トレース
        price_traces = [t for t in fig.data if "価格" in t.name or "OHLC" in t.name]
        ordered_traces.extend(price_traces)
        
        fig.data = ordered_traces
        return fig
```

## 技術的決定事項

### データ型標準化
```python
# Float32統一によるメモリ効率化
STANDARD_DTYPES = {
    "price": pl.Float32,      # 価格データ（精度: 0.01）
    "volume": pl.Float32,     # ボリューム
    "indicator": pl.Float32,  # テクニカル指標
    "timestamp": pl.Datetime("ms"),  # ミリ秒精度
    "symbol": pl.Categorical,  # カテゴリカル変換
}

# メモリ使用量削減効果
# Float64 → Float32: 50%削減
# String → Categorical: 最大90%削減
```

### エラーハンドリング戦略
```python
# 階層的例外設計
class FXSystemError(Exception):
    """基底例外クラス"""
    pass

class MT5ConnectionError(FXSystemError):
    """MT5接続エラー"""
    pass

class DataProcessingError(FXSystemError):
    """データ処理エラー"""
    pass

class ModelInferenceError(FXSystemError):
    """ML推論エラー"""
    pass

# サーキットブレーカー実装
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
```

### パフォーマンスターゲット
```yaml
# システムパフォーマンス目標
latency:
  tick_processing: < 10ms      # ティック処理遅延
  indicator_calculation: < 50ms # 指標計算
  ml_inference: < 100ms        # ML推論
  dashboard_update: < 200ms    # UI更新

throughput:
  tick_ingestion: > 10,000 tps # ティック処理能力
  batch_processing: > 1M rows/s # バッチ処理速度
  
memory:
  tick_buffer: < 1GB           # ティックバッファ
  ml_model: < 500MB            # モデルメモリ
  dashboard_cache: < 2GB       # ダッシュボードキャッシュ
```

## 開発・運用設計

### CI/CDパイプライン
```yaml
# GitHub Actions設定
name: FX System CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest --cov=./ --cov-report=xml
      - name: Run type checking
        run: uv run mypy .
      - name: Run linting
        run: uv run ruff check .
```

### 監視設計
```python
# Prometheusメトリクス
from prometheus_client import Counter, Histogram, Gauge

# メトリクス定義
tick_processed = Counter(
    "fx_ticks_processed_total",
    "Total number of ticks processed",
    ["symbol", "status"]
)

processing_latency = Histogram(
    "fx_processing_latency_seconds",
    "Processing latency in seconds",
    ["operation", "symbol"]
)

active_connections = Gauge(
    "fx_active_connections",
    "Number of active MT5 connections"
)

query_cache_hits = Counter(
    "fx_query_cache_hits_total",
    "Total number of query cache hits"
)

query_cache_misses = Counter(
    "fx_query_cache_misses_total",
    "Total number of query cache misses"
)


# 構造化ログ
import structlog

logger = structlog.get_logger()
logger = logger.bind(
    service="fx-system",
    environment="production"
)
```

## セキュリティ設計

### 認証情報管理
```python
# 環境変数による機密情報管理
from cryptography.fernet import Fernet

class SecureCredentialManager:
    """MT5認証情報の暗号化管理"""
    
    def __init__(self):
        # 環境変数からマスターキー取得
        self.cipher = Fernet(os.environ["MASTER_KEY"])
    
    def encrypt_credentials(self, credentials: dict) -> bytes:
        """認証情報の暗号化"""
        json_creds = json.dumps(credentials)
        return self.cipher.encrypt(json_creds.encode())
    
    def decrypt_credentials(self, encrypted: bytes) -> dict:
        """認証情報の復号化"""
        decrypted = self.cipher.decrypt(encrypted)
        return json.loads(decrypted.decode())
```

### ネットワークセキュリティ
```yaml
# セキュリティ設定
network:
  # TLS設定
  tls:
    enabled: true
    min_version: "1.2"
    cert_file: "/certs/server.crt"
    key_file: "/certs/server.key"
  
  # アクセス制御
  access_control:
    allowed_ips: ["10.0.0.0/8", "172.16.0.0/12"]
    rate_limit: 100  # requests per minute
    
  # CORS設定
  cors:
    allowed_origins: ["https://dashboard.example.com"]
    allowed_methods: ["GET", "POST"]
    allowed_headers: ["Content-Type", "Authorization"]
```

## まとめ

本技術設計書では、Forex_procrssorのFX取引支援システムの包括的な技術アーキテクチャを定義しました。主要な設計決定事項：

1. **非同期アーキテクチャ**: asyncioベースの高性能処理
2. **メモリ最適化**: Float32統一とPolars LazyFrameによる効率化
3. **ハイブリッドストレージ**: InfluxDB（ホット）+ Parquet（コールド）
4. **リアルタイム通信**: WebSocketによる低遅延更新
5. **ML統合**: PatchTSTモデルのGPU最適化推論

この設計により、高頻度取引環境での安定性、拡張性、保守性を実現します。