# Git ワークフロー マニュアル

## ブランチ作成から PR マージまでの一般的な流れ

### 1. 新しいブランチの作成

```bash
# メインブランチから新しいブランチを作成
git checkout main
git pull origin main
git checkout -b feature/新機能名
```

### 2. 開発作業

```bash
# ファイルを編集後、変更を確認
git status

# 変更をステージング
git add .
# または個別ファイル指定
git add ファイル名

# コミット
git commit -m "コミットメッセージ"
```

### 3. リモートへのプッシュ

```bash
# 初回プッシュ（アップストリーム設定）
git push -u origin feature/新機能名

# 以降のプッシュ
git push
```

### 4. プルリクエストの作成

#### GitHub Web UI の場合:
1. GitHubリポジトリページにアクセス
2. "Compare & pull request" ボタンをクリック
3. Base branch: `main` (またはターゲットブランチ)
4. Compare branch: `feature/新機能名`
5. タイトルと説明を記入
6. "Create pull request" をクリック

#### GitHub CLI の場合:
```bash
gh pr create --base main --head feature/新機能名 --title "タイトル" --body "説明"
```

### 5. PR レビューと修正

```bash
# レビューでの修正が必要な場合
git add .
git commit -m "レビュー修正"
git push
```

### 6. PR マージ後のクリーンアップ

```bash
# メインブランチに切り替え
git checkout main

# リモートから最新を取得
git pull origin main

# 作業ブランチを削除
git branch -d feature/新機能名

# リモート追跡ブランチのクリーンアップ
git remote prune origin
```

## 便利なコマンド

### ブランチ管理
```bash
# 全ブランチ確認
git branch -a

# リモートブランチ確認
git branch -r

# ブランチ強制削除（マージされていない場合）
git branch -D ブランチ名
```

### 状態確認
```bash
# 現在の状態確認
git status

# コミット履歴確認
git log --oneline

# 変更差分確認
git diff
```

### リモート管理
```bash
# リモート確認
git remote -v

# リモートブランチの最新状態を取得
git fetch origin

# 削除されたリモートブランチの参照をクリーンアップ
git remote prune origin
```

## トラブルシューティング

### コンフリクトが発生した場合
```bash
# メインブランチの最新を取得してマージ
git checkout main
git pull origin main
git checkout feature/ブランチ名
git merge main

# コンフリクトを手動で解決後
git add .
git commit -m "コンフリクト解決"
git push
```

### 間違ったコミットを取り消したい場合
```bash
# 直前のコミットを取り消し（変更は保持）
git reset --soft HEAD~1

# 直前のコミットを完全に取り消し
git reset --hard HEAD~1
```

### プッシュ済みコミットを修正したい場合
```bash
# 最新コミットのメッセージ修正
git commit --amend -m "新しいメッセージ"

# 強制プッシュ（注意: 共同作業の場合は避ける）
git push --force-with-lease
```