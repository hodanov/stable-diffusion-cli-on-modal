# テストコードの段階導入と CI 組み込み（PR #301, #303, #302, #308, #309, #310）

## 背景・課題

リポジトリにテストが 1 本も無く、CI は ruff の lint だけだった。Python は 6 ファイル
約 1,500 行（`cmd/` が CLI 側、`app/` が Modal サーバ側）で、変更のたびに壊れていないかを
機械的に確かめる手段が無い状態だった。

制約は 2 つある。

1. **GPU と torch が無い環境でしか CI を回せない**: `app/` は Python 3.14 + CUDA 版 torch
   （`pytorch-cu130` インデックス）で、CI に載せるのは非現実的
2. **`cmd/` と `app/` はパッケージではない**: flat import（`from domain import ...`）で
   書かれており、`modal run` / `modal deploy` が各ディレクトリで実行される前提

調査の結果、6 モジュールとも root の環境（Python 3.12、torch なし）で import でき、
import 時にネットワークへ出ない（`Volume.from_name` / `Image.from_dockerfile` /
`Secret.from_dotenv` はすべて遅延評価）ことが分かったため、GPU 無しで回せる範囲を
テスト対象とする方針にした。

## 方針

テスト対象を「GPU 無しで実行できる範囲」に限定し、1 段階 1 PR で積み上げる。

- **対象**: 値オブジェクトの検証、画像の正規化、出力ファイル名、Modal 呼び出しの引数、
  config 検証、ダウンロード分岐（モック）、純粋な計算（サイズ・flow_shift・URL 処理）
- **対象外**: GPU 推論（`run_inference`）、モデル読み込み、後処理。実機確認はユーザーに依頼する
- **テストを書くためのリファクタは挙動不変に限る**: private メソッドを名前マングリング
  （`_Cls__method`）経由で叩くのではなく、モジュール関数へ切り出してからテストする

実行環境は root プロジェクト（`uv run`）に統一した。Makefile が `modal deploy` を
`uv run --project ..` で動かしているのと同じ構成になる。

## 変更内容

### Stage 0: テスト基盤と CI（PR #301）

- root の dev グループに pytest / pytest-cov を追加
- `[tool.pytest.ini_options]` で `testpaths = ["tests"]`、`pythonpath = ["cmd", "app"]`、
  `--import-mode=importlib` を設定。flat import のモジュールを解決するため
- CI（`lint_python.yml`）に `test` ジョブを追加。lint ジョブと同じく setup-uv を
  コミット SHA で pin し、`uv run --frozen pytest` を実行
- `make test` を追加、`.coverage` と `.pytest_cache/` を gitignore

### ruff.toml のコミットと既存違反の解消（PR #303）

ローカルにだけあった `ruff.toml`（`lint.select=["ALL"]`）をコミットし、CI の
`ruff check .` にも同じルールを効かせた。合わせて既存 47 件を解消した。

- `app/**` の PLC0415（30 件）は per-file-ignores で除外。Modal アプリはコンテナ専用の
  重い依存を関数内 import しており、ローカルでモジュールを読み込むための意図的な設計
- S310（6 件）は `ensure_http_url()` を追加し、urlopen 前に http(s) 以外のスキームを
  ValueError で拒否したうえで理由付き noqa
- PTH / FBT001 / DTZ011 はコード修正（pathlib 化、bool 引数のキーワード専用化、
  タイムゾーンを明示した日付取得）。T201 は既存の流儀に合わせて noqa
- `tests/**` では S101 と PLR2004 を、フォーマッタと衝突する COM812 は全体で無効化

### Stage 1〜4: テスト追加

- **Stage 1（PR #302）**: `cmd/domain.py`。handler にインラインで書かれていた
  `== "True"` と `-1 → None` の変換を `parse_bool_flag` / `unset_if_negative` として
  domain へ移設
- **Stage 2（PR #308）**: `cmd/infrastructure.py` と両 handler。プロダクトコードの
  変更なし。`local_entrypoint` は `info.raw_f` で元の関数を取り出してテストした
- **Stage 3（PR #309）**: `app/app_vid.py`。2 クラスで重複していた
  `normalize_hf_url` / `filename_from_url` を 1 つにまとめ、`resolve_flow_shift` と
  `target_size_for_image`（mod_value を引数化）を関数へ切り出し
- **Stage 4（PR #310）**: `app/app_img.py`。`double_image_size` を関数へ切り出し

Modal 依存は `modal.Cls.from_name` を fake に差し替え、`.remote()` に渡る kwargs を固定した。
ダウンロード系は `snapshot_download` / `urlopen` / diffusers のモデルクラス / volume を
モックし、config の内容で ignore/allow パターンと呼び出し回数がどう変わるかを確認している。

### Stage 5: 仕上げ（この PR）

- CI の pytest に `--cov-fail-under=65` を追加（実測 67% に対する下限）
- README / README_ja にテストの実行方法と `tests/` の構成を追記

## 検証

- `uv run pytest`: 115 件成功。カバレッジは `cmd/` が 95〜100%、`app_img.py` 57%、
  `app_vid.py` 36%、全体 67%
- `uv run ruff check .` と `uv run ruff format --check`
- 各段階で**ミューテーション確認**を実施。実装を一時的に壊して該当テストが落ちることを
  確かめてから元に戻した（`parse_bool_flag` の比較、ループ回数、flow_shift のしきい値、
  ダウンロードのパターン名、拡大処理など）
- Stage 3 の関数切り出しは、旧実装と新関数へ乱数 5000 ケースを流して出力が完全一致する
  ことを確認（画像サイズ・要求サイズ・mod_value・フラグの組み合わせ）
- 各 PR で CI（test / lint / CodeQL / docker build）がグリーンであることを確認

## 運用メモ

- **`tests/cli/conftest.py` は modal の内部構造に依存する**。`local_entrypoint` の
  `info.raw_f` から元の関数を取り出しているため、modal のアップグレードで壊れうる。
  その場合は黙って素通りせず AssertionError で落ちる。壊れたら handler 本体を
  decorator なしの関数へ切り出す方針に切り替える
- **カバレッジ下限は 65%**。GPU 経路を除いた現実的な水準で、`app/` に未テストのコードを
  足すと相対的に下がる。大きく下がるときは下限ではなくテストを足す方向で対応する
- **未テストの領域**: `SDXLTxt2Img` / `WanTI2V` の `setup` と `run_inference`、後処理
  （アップスケール・顔補正）、`dequantize_comfy_scaled_fp8`（torch が必要なため
  importorskip 扱い）。ここは実機の `make img_by_sdxl_txt2img` / `make vid_by_wan_ti2v`
  での確認に頼る
- **テストで固定した既存挙動**: `save_prompts` が名前マングリングされたキー
  （`_Prompts__prompt` など）で書き出すこと、vae が `vae.name` ではなく
  `model.name` のディレクトリへ保存されること。どちらも意図的かは未確認で、
  変更するならテストの更新が必要
