# app/ の依存関係管理を uv lockfile 化して dependabot を整合性のある一括更新にする

## 背景・課題

`app/requirements.txt` は Modal コンテナイメージ用の依存定義で、`app/Dockerfile` が
`--extra-index-url https://download.pytorch.org/whl/cu126` 付きで `pip install` していた。

dependabot（pip ecosystem・グループなし）は torch / xformers / accelerate などを
**個別の PR** で上げてくるが、xformers は特定の torch バージョンに対してビルドされ
`torch==X.Y.*` を厳密にピンするため、単独バンプは原理的に壊れる。毎回手動で
足並みを揃える運用コストが高かった。

## 方針

リポジトリのルートは既に uv 管理（`pyproject.toml` + `uv.lock`、Makefile も `uv run`）
なので、`app/` も uv プロジェクト化し、dependabot を `uv` ecosystem + グループ更新に
切り替える。uv のリゾルバが xformers の torch ピンを尊重して整合の取れた組み合わせに
再ロックするため、「足並みを揃える」問題が根本的に解決する。

- xformers 削除（SDPA 移行）も足並み問題の元凶除去として検討したが、
  動作確認を先に行いたいとの判断で今回は見送り（保留）

## 変更内容

### 1. `app/pyproject.toml` + `app/uv.lock` を新規作成、`app/requirements.txt` を削除

- `requires-python = "==3.11.*"`（Dockerfile の `python:3.11.11-slim-bookworm` に合わせる）
- dependencies は旧 requirements.txt の内容を `>=` の下限指定で移植。
  正確なピンは uv.lock が持ち、dependabot は lockfile のみの更新で済む
- PyTorch cu126 インデックスは uv 公式パターンで設定:
  `[[tool.uv.index]]`（`explicit = true`）+ `[tool.uv.sources]` で
  torch / torchvision / xformers をインデックスに向ける
- 初回ロックは旧 requirements.txt と同じバージョンに固定し、移行自体では挙動を変えない

### 2. `app/Dockerfile` を uv インストールに書き換え

- `ghcr.io/astral-sh/uv:0.11.24` から uv バイナリを COPY（docker ecosystem の
  dependabot が今後更新）
- `UV_PROJECT_ENVIRONMENT=/usr/local` を設定して `uv sync --frozen --no-dev --no-cache`
  で venv を作らずシステム Python に直接インストール（Modal がそのまま import できる）

### 3. `.github/dependabot.yml` を uv ecosystem に変更

- `pip`（`./app`）→ `uv`（`/app`）に変更し、全依存を 1 つの週次 PR にグループ化
  （`groups.app-dependencies.patterns: ["*"]`）
- 今まで監視外だったルートの `pyproject.toml` / `uv.lock`（modal クライアント側）にも
  同様の `uv` エントリを追加

### 4. CI・Makefile・README の追随

- `docker_image_ci.yml` のトリガーパスを `app/pyproject.toml` / `app/uv.lock` に変更。
  この CI が docker build（= `uv sync` の実行）まで通すので dependabot PR の検証ゲートになる
- Makefile のデプロイ系ターゲットは `uv run --project ..` に変更
  （app/ に pyproject.toml ができたため、ルート環境の modal を明示的に解決する）

## 移行時に踏んだ問題と修正

### torchvision の ABI 不整合（0065759）

`explicit = true` のインデックスは `[tool.uv.sources]` に載せたパッケージにしか
適用されないため、間接依存の torchvision が PyPI の汎用ビルドに解決され、
cu126 の torch と ABI 不整合を起こした
（`operator torchvision::nms does not exist` → `CLIPImageProcessor` import 失敗）。

torchvision を直接依存に昇格させ、sources で cu126 インデックスに向けて解決。
旧 pip では `0.27.0+cu126` がローカルバージョン規則で優先されていたため
顕在化していなかった。

### モデル変更時にイメージレイヤキャッシュが陳腐化（9f37620）

Modal は `run_function(build_image)` のレイヤキャッシュを COPY したファイルの
内容変更では無効化しないため、config.yml のモデル変更が反映されず
`The directory '/vol/cache/<model>' does not exist.` で起動失敗していた。

config.yml の sha256 を `run_function(build_image, args=(config_hash,))` で渡し、
キャッシュキーに含めることで config 変更時のみモデルダウンロードを再実行する。
ハッシュ計算はコンテナ内 import で失敗しないよう `modal.is_local()` ガード付き。

## 検証

- `uv lock --check` で lock と pyproject の整合を確認
- `docker build --platform linux/amd64` がローカルで成功し、コンテナ内で
  torchvision（`0.27.0+cu126`）/ `CLIPImageProcessor` / SDXL パイプラインの
  import が全て成功することを確認
- `uv run --project ..` で modal クライアントが解決できることを確認
- リンタ: ruff / prettier / tombi パス

## 運用後のフロー（期待効果）

- 毎週、torch / xformers / accelerate 等が**整合の取れた組み合わせで再ロックされた
  1 つの PR** として届く（uv がリゾルバレベルで互換性を保証）
- docker build CI がその PR 上でインストールまで検証
- モデル変更は config.yml の編集 + `make app_img` だけで反映される
  （`MODAL_FORCE_BUILD` 不要）

## 残タスク

- xformers 削除（SDPA 移行）の可否判断: 動作確認の結果を見て決める。
  削除する場合は `app_img.py` の `enable_xformers_memory_efficient_attention()`
  2 箇所の削除と依存除去 + 再ロックのみ
- モデル切り替えが頻繁になる場合は、app_vid.py と同様の `modal.Volume` 方式への
  移行を検討（複数モデル併存・再ダウンロード不要・`modal volume` CLI で管理可能）
