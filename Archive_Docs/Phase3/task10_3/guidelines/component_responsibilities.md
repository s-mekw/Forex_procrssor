# コンポーネント責務ガイドライン

## 概要
このドキュメントは、Task 10.3のリファクタリングで確立された責務分離の原則を文書化し、今後の開発における指針を提供します。

## 設計原則

### 単一責任の原則（SRP）
各コンポーネントは、単一の明確な責務を持つべきです。変更の理由は一つだけであるべきです。

### 依存性逆転の原則（DIP）
高レベルモジュールは低レベルモジュールに依存すべきではありません。両方とも抽象に依存すべきです。

### インターフェース分離の原則（ISP）
クライアントは、使用しないメソッドへの依存を強制されるべきではありません。

## コンポーネント責務定義

### RealtimePipeline

#### 責務範囲
RealtimePipelineは、データフロー管理とオーケストレーションに専念します。

#### 主要責務
1. **データフロー管理**
   - キューからのデータ受信
   - データの順序制御
   - フロー制御とバックプレッシャー

2. **コンポーネント統合**
   - 各分析コンポーネントの呼び出し
   - 結果の集約と配信
   - エラー伝播の管理

3. **メトリクス収集**
   - パフォーマンスメトリクス
   - スループット測定
   - レイテンシー追跡

4. **アラート管理**
   - 閾値監視
   - アラート生成
   - 通知配信

#### 責務外の事項
- ❌ データバッファリング（Analyzerに委譲）
- ❌ 分析ロジック（Analyzerに委譲）
- ❌ データ変換詳細（Analyzerに委譲）

#### コード例
```python
class RealtimePipeline:
    def __init__(self, analyzer: AnalyzerProtocol):
        self._analyzer = analyzer  # 依存性注入
        self._metrics = MetricsCollector()
        self._alerts = AlertManager()
    
    def process(self, data):
        # データフロー管理
        processed = self._preprocess(data)
        
        # 分析の委譲
        self._analyzer.add_new_bar(processed)
        if self._analyzer.is_ready():
            result = self._analyzer.analyze_streaming()
            
            # メトリクスとアラート
            self._metrics.record(result)
            self._alerts.check(result)
```

### MultiTimeframeAnalyzer

#### 責務範囲
MultiTimeframeAnalyzerは、マルチタイムフレーム分析とデータ管理に専念します。

#### 主要責務
1. **データバッファ管理**
   - バッファへのデータ追加
   - バッファサイズ管理
   - データライフサイクル管理

2. **データ変換**
   - DataFrame変換
   - タイムフレーム変換
   - データ正規化

3. **分析ロジック**
   - RCI計算
   - トレンド判定
   - シグナル生成

4. **状態管理**
   - 分析準備状態
   - 内部状態の一貫性
   - エラー状態の管理

#### 責務外の事項
- ❌ データ受信（Pipelineが管理）
- ❌ メトリクス収集（Pipelineが管理）
- ❌ アラート生成（Pipelineが管理）

#### コード例
```python
class MultiTimeframeAnalyzer:
    def __init__(self, config):
        self._buffer = []
        self._config = config
    
    def add_new_bar(self, bar):
        # バッファ管理責務
        self._buffer.append(bar)
        self._manage_buffer_size()
    
    def analyze_streaming(self):
        # 分析責務
        df = self._to_dataframe()
        return self._calculate_rci(df)
```

## インターフェース設計

### AnalyzerProtocol
```python
class AnalyzerProtocol(Protocol):
    """分析コンポーネントの標準インターフェース"""
    
    def add_new_bar(self, bar: dict[str, Any]) -> None:
        """データ追加"""
        ...
    
    def is_ready(self) -> bool:
        """準備状態確認"""
        ...
    
    def analyze_streaming(self) -> dict[str, Any]:
        """分析実行"""
        ...
```

### 利点
1. **テスタビリティ**: モック実装が容易
2. **拡張性**: 新しい分析器の追加が簡単
3. **保守性**: 明確なコントラクト

## データフロー

### 正常フロー
```
1. External Data Source
   ↓
2. RealtimePipeline (receive)
   ↓
3. RealtimePipeline (preprocess)
   ↓
4. MultiTimeframeAnalyzer (add_new_bar)
   ↓
5. MultiTimeframeAnalyzer (analyze_streaming)
   ↓
6. RealtimePipeline (post-process)
   ↓
7. Output (metrics, alerts, storage)
```

### エラーフロー
```
1. Error in Analyzer
   ↓
2. Return error status
   ↓
3. Pipeline handles gracefully
   ↓
4. Log and continue
   ↓
5. Alert if critical
```

## 開発ガイドライン

### 新機能追加時の原則

#### 1. 責務の明確化
新機能を追加する前に、どのコンポーネントが責任を持つべきか明確にする。

```
質問チェックリスト:
□ データフロー関連？ → RealtimePipeline
□ 分析ロジック関連？ → MultiTimeframeAnalyzer
□ 新しい責務？ → 新コンポーネント検討
```

#### 2. インターフェース優先設計
実装前にインターフェースを定義する。

```python
# 良い例
class NewAnalyzerProtocol(Protocol):
    def analyze(self, data: Any) -> Result: ...

# 悪い例
class NewAnalyzer:
    def do_everything(self, *args, **kwargs): ...
```

#### 3. 依存性注入の活用
ハードコーディングではなく、依存性注入を使用する。

```python
# 良い例
def __init__(self, analyzer: AnalyzerProtocol):
    self._analyzer = analyzer

# 悪い例
def __init__(self):
    self._analyzer = MultiTimeframeAnalyzer()
```

### リファクタリング時の指針

#### Step 1: 現状分析
- 責務の重複を特定
- 密結合の箇所を発見
- テストカバレッジを確認

#### Step 2: インターフェース定義
- Protocol定義を作成
- 必要最小限のメソッドのみ
- 後方互換性を考慮

#### Step 3: 段階的移行
- 既存テストを維持
- 一度に一つの責務のみ移動
- 各ステップで動作確認

## アンチパターン

### 避けるべきパターン

#### 1. God Object
```python
# 悪い例
class DoEverything:
    def receive_data(self): ...
    def analyze(self): ...
    def store(self): ...
    def alert(self): ...
    def report(self): ...
```

#### 2. 循環依存
```python
# 悪い例
class A:
    def __init__(self, b: B): ...

class B:
    def __init__(self, a: A): ...
```

#### 3. 責務の漏れ
```python
# 悪い例
class Pipeline:
    def calculate_rci(self):  # 分析責務がPipelineに漏れている
        ...
```

## テスト戦略

### ユニットテスト
各コンポーネントを独立してテスト。

```python
def test_analyzer_buffer_management():
    analyzer = MultiTimeframeAnalyzer()
    analyzer.add_new_bar(bar)
    assert analyzer.get_buffer_size() == 1
```

### 統合テスト
コンポーネント間の連携をテスト。

```python
def test_pipeline_analyzer_integration():
    analyzer = MockAnalyzer()
    pipeline = RealtimePipeline(analyzer)
    pipeline.process(data)
    assert analyzer.called_with_correct_data()
```

## メンテナンス指針

### 定期レビュー項目
1. **責務の純粋性**: 各コンポーネントが単一責務を維持しているか
2. **依存関係**: 不要な依存が追加されていないか
3. **インターフェース安定性**: 破壊的変更がないか
4. **テストカバレッジ**: 新機能がテストされているか

### リファクタリングトリガー
- 責務の重複が3箇所以上
- 循環的複雑度が10を超える
- テストが困難になった
- 同じバグが複数回発生

## まとめ

### 重要原則
1. **責務の明確化**: 各コンポーネントは一つの責務
2. **疎結合**: インターフェースを介した連携
3. **高凝集**: 関連する機能は同一コンポーネント内
4. **テスト可能**: 独立してテスト可能な設計

### 期待される効果
- 保守性の向上
- バグの減少
- 開発速度の向上
- チーム開発の効率化

このガイドラインに従うことで、長期的に保守可能で拡張性の高いシステムを維持できます。