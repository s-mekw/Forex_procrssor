# ソースコードドキュメント自動生成プロンプト

## 概要
このプロンプトは、Forex Processorプロジェクトの`src`ディレクトリ内のPythonファイルから、構造化されたドキュメントを自動生成するためのテンプレートです。

## 実行手順

### 1. 準備フェーズ
- `src_documentations`ディレクトリを作成
- `src`ディレクトリと同じ構造のサブディレクトリを作成

### 2. ディレクトリ構造
```
src_documentations/
├── app/
├── common/
├── data_processing/
├── mt5_data_acquisition/
├── patchTST_model/
├── production/
└── storage/
```

### 3. ドキュメント生成対象ファイル

#### 必須ファイル（重要度: 高）
- `common/models.py` → `common/models.py.md`
- `common/interfaces.py` → `common/interfaces.py.md`
- `common/config.py` → `common/config.py.md`
- `mt5_data_acquisition/mt5_client.py` → `mt5_data_acquisition/mt5_client.py.md`
- `mt5_data_acquisition/tick_fetcher.py` → `mt5_data_acquisition/tick_fetcher.py.md`
- `mt5_data_acquisition/ohlc_fetcher.py` → `mt5_data_acquisition/ohlc_fetcher.py.md`
- `mt5_data_acquisition/tick_to_bar.py` → `mt5_data_acquisition/tick_to_bar.py.md`
- `mt5_data_acquisition/tick_adapter.py` → `mt5_data_acquisition/tick_adapter.py.md`
- `data_processing/processor.py` → `data_processing/processor.py.md`

#### オプションファイル（__init__.pyなど）
- 空または最小限の内容の場合はスキップ可能

## ドキュメントテンプレート

### 基本構造
```markdown
# [ファイル名]

## 概要
[ファイルの目的と主要な機能の説明（2-3文）]

## 依存関係
- 外部ライブラリ: [使用している外部パッケージ]
- 内部モジュール: [プロジェクト内の他モジュールへの依存]

## 主要コンポーネント

### クラス

#### [クラス名]
**目的**: [クラスの役割と責任]

**属性**:
- `属性名` (型): 説明

**メソッド**:
- `メソッド名(パラメータ)`: 簡潔な説明

### 関数

#### [関数名]
**目的**: [関数の機能説明]

**入力**:
- `パラメータ名` (型): 説明
- `パラメータ名` (型): 説明

**出力**:
- 戻り値の型: 説明

**処理フロー**:
1. [処理ステップ1]
2. [処理ステップ2]
3. [処理ステップ3]

**例外**:
- `例外クラス`: 発生条件

### 定数・設定
- `定数名`: 値と用途

## 使用例
\```python
# 基本的な使用方法のコード例
\```

## 注意事項
- [重要な制約や注意点]
- [パフォーマンス考慮事項]
- [セキュリティ考慮事項]
```

## プロジェクト固有の考慮事項

### データ型制約
- **Float32必須**: すべての数値データは`np.float32`を使用（メモリ最適化のため）
- **Polars使用**: DataFrameはPolarsを使用（Pandas禁止）

### 設計パターン
- **Protocol-Based Design**: `typing.Protocol`を使用した柔軟な型システム
- **ABC Enforcement**: 抽象基底クラスによるインターフェース契約
- **Async Pattern**: I/O操作には`async/await`パターンを使用

### ドキュメント記載項目
各ドキュメントには以下を必ず含める：

1. **入力仕様** (Input)
   - パラメータの型と制約
   - 必須/オプションの区別
   - デフォルト値

2. **出力仕様** (Output)
   - 戻り値の型
   - データ形式
   - エラー時の挙動

3. **機能説明** (Functionality)
   - 主要な処理内容
   - アルゴリズムの概要
   - 副作用の有無

## 自動生成プロンプト

以下のプロンプトを使用してドキュメントを生成：

```
指定されたPythonファイル [ファイルパス] を解析し、上記のテンプレートに従ってMarkdown形式のドキュメントを生成してください。

要件：
1. すべてのクラス、関数、重要な定数を文書化
2. 各コンポーネントのInput/Output/機能を明確に記載
3. Forex Processorプロジェクトの設計原則（Float32制約、Polars使用など）を考慮
4. コード例を含める（可能な場合）
5. エラーハンドリングと例外を文書化

生成するドキュメントは src_documentations/[対応するパス]/[ファイル名].md に保存してください。
```

## 検証チェックリスト

### ドキュメント完全性チェック
- [ ] すべての対象ファイルにドキュメントが生成されている
- [ ] 各ドキュメントにInput/Output/機能が記載されている
- [ ] クラス図やフロー図が適切に含まれている（複雑な処理の場合）

### 品質チェック
- [ ] 技術的に正確な説明
- [ ] 一貫した書式とスタイル
- [ ] 実用的なコード例の提供
- [ ] プロジェクト固有の制約の反映

### ファイル構造チェック
- [ ] src_documentations/app/
- [ ] src_documentations/common/
- [ ] src_documentations/data_processing/
- [ ] src_documentations/mt5_data_acquisition/
- [ ] src_documentations/patchTST_model/
- [ ] src_documentations/production/
- [ ] src_documentations/storage/

## 実行後の検証コマンド

```bash
# ディレクトリ構造の確認
tree src_documentations/

# ドキュメントファイル数の確認
find src_documentations -name "*.md" | wc -l

# 各ドキュメントの基本要素チェック
grep -l "## 概要" src_documentations/**/*.md
grep -l "**入力**:" src_documentations/**/*.md
grep -l "**出力**:" src_documentations/**/*.md
```

## 更新履歴
- 2025-08-23: 初版作成
- プロジェクト: Forex Processor
- 対象ディレクトリ: src/
- 出力先: src_documentations/