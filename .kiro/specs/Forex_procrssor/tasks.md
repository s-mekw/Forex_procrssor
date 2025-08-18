# 実装計画

## フェーズ1: プロジェクト構造とコアインターフェース

- [X] 1. プロジェクト構造とテスト環境のセットアップ
  - src/ディレクトリ構造を作成: common、mt5_data_acquisition、data_processing、storage、patchTST_model、app、production
  - tests/ディレクトリ構造を作成: unit、integration、fixtures
  - pytestの設定ファイル（conftest.py）を作成し、共通フィクスチャを定義
  - pyproject.tomlにテストカバレッジ設定を追加（最小カバレッジ80%）
  - _要件: 1.1, 2.1_

- [X] 2. 共通データモデルとインターフェース定義
  - src/common/models.pyにPydanticモデルを作成：Tick、OHLC、Prediction、Alert
  - Float32を標準データ型として定義（メモリ効率最適化）
  - src/common/interfaces.pyに基底クラスを定義：DataFetcher、DataProcessor、StorageHandler、Predictor
  - src/common/config.pyに設定管理クラスを実装（環境変数とTOMLファイル対応）
  - _要件: 2.1, 7.5_

## フェーズ2: MT5データ取得基盤

- [X] 3. MT5接続管理のテスト駆動実装
  - tests/unit/test_mt5_client.pyに接続成功/失敗/再接続のテストケースを作成
  - src/mt5_data_acquisition/mt5_client.pyにMT5ConnectionManagerクラスを実装
  - 指数バックオフによる再接続ロジックを実装（最大5回試行）
  - 接続プール管理とヘルスチェック機能を追加
  - _要件: 1.1_

- [ ] 4. リアルタイムティックデータ取得の実装
  - tests/unit/test_tick_fetcher.pyにティックデータ取得とバリデーションのテストを作成
  - src/mt5_data_acquisition/tick_fetcher.pyにTickDataStreamerクラスを実装
  - 非同期ストリーミングとリングバッファ（10,000件）を実装
  - スパイクフィルター（3σルール）による異常値除外を追加
  - _要件: 1.2_

- [ ] 5. 履歴OHLCデータ取得とバッチ処理
  - tests/unit/test_ohlc_fetcher.pyに履歴データ取得と欠損検出のテストを作成
  - src/mt5_data_acquisition/ohlc_fetcher.pyにHistoricalDataFetcherクラスを実装
  - 10,000バー単位のバッチ処理と並列フェッチ機能を実装
  - 複数時間足（1分〜日足）のサポートを追加
  - _要件: 1.3_

- [ ] 6. ティック→バー変換エンジンの実装
  - tests/unit/test_tick_to_bar.pyにバー生成とタイムスタンプ整合性のテストを作成
  - src/mt5_data_acquisition/tick_to_bar.pyにTickToBarConverterクラスを実装
  - リアルタイム1分足生成と未完成バーの継続更新機能を実装
  - 30秒以上のティック欠損時の警告機能を追加
  - _要件: 1.4_

## フェーズ3: 高速データ処理パイプライン

- [ ] 7. Polarsデータ処理基盤の構築
  - tests/unit/test_data_processor.pyにPolars処理とメモリ最適化のテストを作成
  - src/data_processing/processor.pyにPolarsProcessingEngineクラスを実装
  - LazyFrameによる遅延評価とFloat32統一のスキーマを定義
  - チャンク処理とストリーミング処理の切り替えロジックを実装
  - _要件: 2.1_

- [ ] 8. テクニカル指標計算エンジンの実装
  - tests/unit/test_indicators.pyに各指標の計算精度テストを作成
  - src/data_processing/indicators.pyにTechnicalIndicatorEngineクラスを実装
  - polars-ta-extensionを使用したEMA（5、20、50、100、200期間）計算を実装
  - 増分計算による効率的な更新メカニズムを追加
  - _要件: 2.2_

- [ ] 9. RCI計算エンジンの高速実装
  - tests/unit/test_rci.pyにRCI計算とパラメータ検証のテストを作成
  - src/data_processing/rci.pyにRCICalculatorクラスを実装
  - 設定可能な期間リストのRCI計算を実装（エンジンは汎用）
  - デフォルトは短期RCIのみを1分足で計算（[9, 13, 24, 33, 48, 66, 108]）。長期RCIは5分足にリサンプリング後、置換期間で計算（例: 120→24, 165→33, 240→48, 330→66, 540→108）。エンジン本体の変更は不要
  - Polars Expressionによる高速ランキング処理を実装
  - _要件: 2.3_

- [ ] 10.1. リアルタイム処理パイプライン基盤の構築
  - tests/integration/test_data_pipeline.pyに非同期処理とバックプレッシャーのテストを作成
  - src/data_processing/pipelines.pyにRealtimePipelineクラスの骨格を実装
  - asyncioベースの非同期データフロー処理を実装（1分足データをパススルー）
  - 1秒を超える遅延時のアラート機能を追加
  - _要件: 2.4_

- [ ] 10.2. マルチタイムフレーム分析機能の実装
  - 1分足データから5分足データを動的にリサンプリングする機能をパイプラインに追加
  - 短期RCI（1分足ベース）と長期RCI（5分足ベース）の計算をオーケストレーション
  - 期間セットを明示: 短期（1分足）=[9, 13, 24, 33, 48, 66, 108]、長期（5分足）=[24, 33, 48, 66, 108]（120/165/240/330/540期間の代替）
  - 両方のRCI結果を統合し、後続の処理に渡すデータフローを構築
  - マルチタイムフレーム処理の結合とデータ整合性を検証するテストを作成
  - _要件: 2.5_

- [ ] 10.3. パイプラインのリファクタリングと責務の明確化
  - src/data_processing/analyzer.py に MultiFrameAnalyzer クラスを新設
  - RealtimePipeline からマルチタイムフレーム分析ロジックを MultiFrameAnalyzer に移譲
  - RealtimePipeline はデータフロー管理に専念し、MultiFrameAnalyzer をコンポーネントとして利用する構成に変更
  - Analyzer専門のユニットテスト(tests/unit/test_analyzer.py)を作成し、ロジックの堅牢性を保証
  - _要件: 2.4, 2.5 (リファクタリング)_

## フェーズ4: 時系列データストレージ

- [ ] 11. InfluxDB接続とデータモデル設定
  - tests/unit/test_influx_handler.pyに接続とクエリのテストを作成
  - src/storage/influx_handler.pyにInfluxDBHandlerクラスを実装
  - 環境変数による接続設定と認証を実装
  - タグ（通貨ペア、時間足）とフィールド（OHLC、Volume）のスキーマを定義
  - _要件: 3.1_

- [ ] 12. 高速バッチ書き込みと再試行メカニズム
  - tests/unit/test_influx_writer.pyにバッチ処理とエラー復旧のテストを作成
  - InfluxDBHandlerにバッチサイズ1000での一括書き込み機能を実装
  - 書き込みエラー時の再試行キューとバックオフ戦略を実装
  - 保持ポリシー（ティック7日、1分足90日等）の設定を追加
  - _要件: 3.1_

- [ ] 13. Parquetアーカイブシステムの実装
  - tests/unit/test_parquet_handler.pyにアーカイブと圧縮のテストを作成
  - src/storage/parquet_handler.pyにParquetArchiverクラスを実装
  - 月次自動アーカイブとSnappy圧縮を実装
  - 通貨ペア・年月単位のパーティション戦略を実装
  - _要件: 3.2_

- [ ] 14. 統合クエリエンジンの構築
   - tests/integration/test_query_engine.pyに統合検索と集計のテストを作成
   - src/storage/query_engine.pyにUnifiedQueryEngineクラスを実装
   - InfluxDB（ホット）とParquet（コールド）の透過的アクセスを実装
   - 10ミリ秒以内の時間範囲クエリ最適化を実装
   - _要件: 3.3_

- [ ] 15. データ整合性管理システム
  - tests/unit/test_data_integrity.pyにチェックサムと重複検出のテストを作成
  - src/storage/base.pyにDataIntegrityManagerクラスを実装
  - タイムスタンプとシンボルによるユニーク制約を実装
  - 日次自動バックアップとポイントインタイム復旧機能を追加
  - _要件: 3.4_



## フェーズ5: PatchTST機械学習モデル

- [ ] 16. ML環境とモデル基盤の構築
  - tests/unit/test_ml_environment.pyにGPU検出とフォールバックのテストを作成
  - src/patchTST_model/model.pyにPatchTSTモデルクラスを実装
  - PyTorch + CUDA環境の自動検出とCPUフォールバックを実装
  - モデルバージョン管理とチェックポイント保存機能を追加
  - _要件: 4.1_

- [ ] 17. PatchTSTモデルアーキテクチャの実装
  - tests/unit/test_patchtst_model.pyにモデル構造と推論のテストを作成
  - PatchTSTモデルのTransformerアーキテクチャを実装（パッチサイズ5、8ヘッド、3層）
  - 価格データとテクニカル指標を入力特徴量として統合
  - 1〜12時間先のマルチホライズン予測と信頼区間出力を実装
  - _要件: 4.2_

- [ ] 18. MLデータローダーとInfluxDB統合
  - tests/unit/test_ml_dataloader.pyにデータ取得と前処理のテストを作成
  - src/patchTST_model/datasets.pyにMLDataLoaderクラスを実装
  - InfluxDBからの効率的な特徴量取得を実装
  - 正規化とPyTorch Tensor変換パイプラインを構築
  - _要件: 4.2, 3.3_

- [ ] 19. 自動学習・再学習システムの構築
  - tests/unit/test_auto_training.pyに学習トリガーと性能評価のテストを作成
  - src/patchTST_model/train.pyにAutoTrainerクラスを実装
  - MAPE 5%悪化での自動再学習トリガーを実装
  - 80/20の学習/検証データ分割と増分学習機能を追加
  - _要件: 4.3_

- [ ] 20. リアルタイム推論エンジンの実装
  - tests/integration/test_inference_engine.pyに推論速度とバッチ処理のテストを作成
  - src/patchTST_model/predict.pyにInferencePipelineクラスを実装
  - 200ミリ秒以内の推論レスポンスを実現
  - ホットスワップによる無停止モデル更新を実装
  - _要件: 4.4_

- [ ] 21. モデル評価・監視システム
  - tests/unit/test_model_evaluation.pyに評価指標とドリフト検出のテストを作成
  - src/patchTST_model/evaluate.pyにModelEvaluatorクラスを実装
  - MAPE、RMSE、MAEによる性能測定を実装
  - SHAP値による特徴量重要度の可視化を追加
  - _要件: 4.5_

## フェーズ6: インタラクティブダッシュボード

- [ ] 22. Dashアプリケーション基盤の構築
  - tests/unit/test_dash_app.pyにアプリ初期化とルーティングのテストを作成
  - src/app/main.pyにDashアプリケーションを実装
  - ポート8050でのWebサーバー起動を設定
  - モジュラーレイアウトとマルチページ構造を実装
  - _要件: 5.1_

- [ ] 23. リアルタイムチャートコンポーネント
  - tests/unit/test_chart_components.pyにチャート描画と更新のテストを作成
  - src/app/components/charts.pyにOptimizedChartRendererクラスを実装
  - Plotly WebGLによるローソク足チャートを実装
  - LTTBアルゴリズムによるデータ間引き最適化を追加
  - _要件: 5.2_

- [ ] 24. WebSocketリアルタイム通信の実装
  - tests/integration/test_websocket.pyに接続と再接続のテストを作成
  - src/app/websocket_server.pyにDashWebSocketServerクラスを実装
  - dash-extensionsによるリアルタイムデータ配信を実装
  - 自動再接続とフォールバック機能を追加
  - _要件: 5.3_

- [ ] 25. AI予測可視化コンポーネントの実装
  - tests/unit/test_prediction_viz.pyに予測線と信頼区間描画のテストを作成
  - src/app/components/prediction_viz.pyにPredictionVisualizationクラスを実装
  - AI予測線（トマト色、太さ3）の描画機能を実装
  - 95%、80%、50%の信頼区間帯を異なる透明度で重ね描画
  - _要件: 5.2_

- [ ] 26. カスタマイズ可能UIとテーマ機能
  - tests/unit/test_ui_customization.pyにレイアウト変更とテーマのテストを作成
  - src/app/layouts.pyにカスタマイズ可能なレイアウトシステムを実装
  - ドラッグ&ドロップによるウィジェット配置機能を追加
  - ダーク/ライトテーマの切り替えとローカルストレージ保存を実装
  - _要件: 5.4_

- [ ] 27. ダッシュボードパフォーマンス最適化
  - tests/integration/test_dashboard_performance.pyに描画速度とメモリ使用のテストを作成
  - 仮想スクロールによる大量データ表示の最適化を実装
  - 差分更新による効率的なチャート再描画を実装
  - 30fps以上のフレームレート維持機能を追加
  - _要件: 5.5_

## フェーズ7: 本番運用・監視機能

- [ ] 28. システム監視とヘルスチェック
  - tests/unit/test_monitoring.pyにヘルスチェックとメトリクス収集のテストを作成
  - src/production/monitoring.pyにSystemMonitorクラスを実装
  - 全コンポーネント（MT5、InfluxDB、ML）の状態確認を実装
  - CPU、メモリ、ディスク使用率の監視機能を追加
  - _要件: 6.1_

- [ ] 29. 構造化ログとエラーハンドリング
  - tests/unit/test_logging.pyにログ出力とエラー処理のテストを作成
  - src/common/logger.pyに構造化ログシステムを実装
  - JSON形式のログ出力と5段階のログレベルを設定
  - カスタム例外クラスと自動復旧メカニズムを実装
  - _要件: 6.2, 6.3_

- [ ] 30. アラート・通知システムの構築
  - tests/unit/test_alerts.pyにアラート条件と通知送信のテストを作成
  - src/production/alerts.pyにAlertManagerクラスを実装
  - 価格アラート、テクニカルアラート、AI予測アラートを実装
  - システムエラー時の緊急通知機能を追加
  - _要件: 7.4_

## フェーズ8: システム統合とエンドツーエンド実装

- [ ] 31. マルチ通貨ペア対応とスケーラビリティ
  - tests/integration/test_multi_currency.pyに複数通貨ペア処理のテストを作成
  - 動的な通貨ペア追加機能を実装
  - 通貨ペア単位のワーカープロセス分散を実装
  - 優先度ベースのリソース割り当て機能を追加
  - _要件: 7.1_

- [ ] 32. RESTful APIエンドポイントの実装
  - tests/integration/test_api_endpoints.pyにAPI機能とレート制限のテストを作成
  - src/app/api.pyにOpenAPI準拠のREST APIを実装
  - 時系列データ、予測結果、システム状態のエンドポイントを作成
  - APIキーによる認証とレート制限機能を追加
  - _要件: 7.2_

- [ ] 33. バックテストエンジンの実装
  - tests/integration/test_backtesting.pyにバックテスト実行と結果検証のテストを作成
  - src/backtesting/engine.pyにBacktestEngineクラスを実装
  - 手数料、スプレッド、スリッページを考慮した計算を実装
  - シャープレシオ、最大ドローダウン等の評価指標を追加
  - _要件: 7.3_

- [ ] 34. 統合設定管理システム
  - tests/unit/test_config_management.pyに設定読み込みとバリデーションのテストを作成
  - src/config/settings.tomlに統一設定ファイルを作成
  - 通貨ペア、指標期間、アラート閾値の一元管理を実装
  - 設定変更時の自動リロード機能を追加
  - _要件: 7.5_

- [ ] 35. エンドツーエンドシステム統合
  - tests/e2e/test_system_integration.pyに全機能統合のE2Eテストを作成
  - src/main.pyにシステム全体の起動スクリプトを実装
  - MT5 → 処理 → ML → ストレージ → ダッシュボードの完全統合を実装
  - 全コンポーネントの依存関係と起動順序を管理
  - _要件: 6.1_

- [ ] 36. 自動化エンドツーエンドテストスイート
  - tests/e2e/test_workflows.pyに主要ワークフローの自動テストを作成
  - データ取得 → 処理 → 予測 → 表示の完全フローをテスト
  - 複数通貨ペアでの並列処理をテスト
  - システム障害時の復旧シナリオをテスト
  - _要件: 6.1_