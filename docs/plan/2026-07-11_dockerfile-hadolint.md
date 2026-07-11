# app/Dockerfile に hadolint を導入し、apt-get 自体を撤廃する

## 背景・課題

`app/Dockerfile` のセキュリティ向上のため hadolint を導入し、CI でチェックしたい。
hadolint を実行したところ、警告は次の2件だった。

- **DL3045** (L4): `WORKDIR` 未設定のまま相対パスへ `COPY`
- **DL3008** (L5): `apt-get install` のパッケージバージョン未固定

DL3008 は「バージョン固定 or ルール ignore」が定石だが、Debian はミラーから
古いバージョンが消えるため固定はビルド破損のリスクが高く、ignore は警告の
握りつぶしになる。そもそも **apt-get しないといけない状態を解消する** 方針とした。

## 事前調査

apt でインストールしていた4パッケージの必要性を調査した結果、すべて撤廃可能と判明。

- `wget` / `git`: アプリコード（`app/*.py`, `cmd/*.py`）に使用箇所なし。
  ダウンロードは `urllib.request.urlopen` と `huggingface_hub.snapshot_download` のみ。
  `app/uv.lock` に git 依存もなし。
- `libgl1-mesa-glx` / `libglib2.0-0`: 必要としているのは `invisible-watermark` が
  依存する **`opencv-python`（GUI 版）だけ**。`controlnet-aux` は既に
  `opencv-python-headless` を使用しており、GUI 版を headless に置き換えれば不要
  （headless の manylinux wheel は必要な共有ライブラリを同梱している）。

前提の裏付けとして、素の `python:3.14-slim`（apt パッケージ追加なし）で
`opencv-python-headless` の `import cv2` が成功することを確認済み。

## 変更内容

### 1. `app/pyproject.toml` — opencv-python を headless に置換

- `dependencies` に `opencv-python-headless` を追加
- `[tool.uv]` の `override-dependencies` で GUI 版を解決から除外

```toml
override-dependencies = [
  # Exclude the GUI build of opencv pulled in by invisible-watermark;
  # opencv-python-headless provides cv2 instead.
  "opencv-python; sys_platform == 'never'",
]
```

`cd app && uv lock` で `app/uv.lock` を再生成（`opencv-python` は
`sys_platform == 'never'` マーカー付きとなり、インストール対象から外れる）。

### 2. `app/Dockerfile` — apt-get 撤廃 + WORKDIR 追加

```dockerfile
FROM python:3.14.6-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/usr/local
WORKDIR /app
COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --frozen --no-dev --no-cache
```

- apt-get の RUN 行が消えるため DL3008 は根本解消（`.hadolint.yaml` も不要）
- `WORKDIR /app` で DL3045 を解消。パッケージは `UV_PROJECT_ENVIRONMENT=/usr/local`
  によりシステム環境へ入るため、WORKDIR の場所は実行時に影響しない
  （Modal は実行時に自前の workdir を設定する）

### 3. `.github/workflows/docker_image_ci.yml` — hadolint ジョブ追加

既存の `build` ジョブと並列に `hadolint` ジョブを追加（トリガー paths は現状のまま）。
アクションは既存ワークフローのスタイルに合わせて commit SHA で固定
（`hadolint/hadolint-action` v3.3.0）。

## リスクとフォールバック

- 万一実行環境で `import cv2` が失敗する場合は、`libglib2.0-0` のみ apt-get で残し、
  その行に `# hadolint ignore=DL3008` のインラインコメントを付けるフォールバックを取る。

## 検証

1. `hadolint app/Dockerfile` → 警告ゼロ（確認済み）
2. `docker run --rm --platform linux/amd64 python:3.14-slim` 上で
   `pip install opencv-python-headless` → `import cv2` 成功（確認済み）
3. `cd app && uv lock --check` でロックの整合確認（確認済み）
4. フルビルドは torch cu126（x86_64 のみ・数 GB）のためローカルでは重い →
   push 後に既存の Docker image CI（`docker build`）で検証
   （paths トリガーに `app/Dockerfile`・`pyproject.toml`・`uv.lock` が含まれる）
