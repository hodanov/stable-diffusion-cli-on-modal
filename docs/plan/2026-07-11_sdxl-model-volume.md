# app_img.py のモデル置き場を modal.Volume に移行する

## ステータス

実装済み（2026-07-11）。Modal 上での検証（下記「検証」の 1〜5）は未実施。

## 背景・課題

app_img.py（SDXL）は `run_function(build_image)` でモデルをイメージレイヤに
焼き込む構造になっている。config.yml のハッシュをキャッシュキーに含める対策
（コミット 9f37620）でモデル変更は反映されるようになったが、構造上の制約が残る:

- モデルを切り替えるたびにイメージレイヤの再ビルド（= モデルの再ダウンロード）が走る
- 複数モデルを併存できない（1 イメージ 1 構成）。行き来するたびに再ビルド
- イメージ内のモデルは不可視で、個別に消したり確認したりできない

一方 app_vid.py（Wan I2V）は既に `modal.Volume` 方式で、この問題がない。
app_img.py も同じパターンに揃えることで、モデル管理を統一する。

- Volume の料金は $0.09/GiB/月・毎月 1 TiB まで無料（2026-07 時点）。
  SDXL モデル（fp16 で 7GB 前後）なら無料枠で 100 個以上置ける

## 方針

app_vid.py のパターンをそのまま SDXL 側に写像する:

- **Volume 定義**: `Volume.from_name("wan-i2v-models", ...)` に対応する
  `Volume.from_name("sdxl-models", create_if_missing=True)` を新設
- **マウント**: `App("sdxl-cli", volumes={"/vol/models": volume})`
  （app_vid.py の `App("wan-i2v-cli", volumes=...)` と同形）
- **ダウンロード**: `prepare_wan_i2v` に対応する `@app.function` の
  `prepare_sdxl` を新設
- **Makefile**: `make prep_wan_i2v` に対応する `make prep_sdxl` を新設
- **setup()**: `volume.reload()` → 存在チェック → `from_pretrained` の流れも同じ

## 変更内容

### 1. `app/app_img.py`

- モジュールレベル:
  - `model_volume = Volume.from_name("sdxl-models", create_if_missing=True)` を追加し、
    `App("sdxl-cli", volumes={MODEL_VOLUME_PATH: model_volume})` でマウント
  - `BASE_CACHE_PATH` 系の定数を `/vol/cache` → Volume マウント先（`/vol/models`）配下に変更。
    lora / textual_inversion / controlnet / upscaler のサブパスも同様に配下へ移す
  - `run_function(build_image, args=(config_hash,), ...)` と config_hash の計算を削除。
    イメージは「依存 + `COPY config.yml /`」だけの純粋なレイヤになる
    （9f37620 のハッシュ機構は Volume 化により不要になる）
- `build_image()` を `@app.function(timeout=3600, secrets=[...])` の
  `prepare_sdxl()` に改名・変更:
  - 冒頭で `model_volume.reload()`
  - 既存の `StableDiffusionCLISetupSDXL.download_model()` / `CommonSetup.download_setup_files()`
    をそのまま再利用（保存先が Volume 配下になるだけ）
  - ダウンロード完了後に `model_volume.commit()`（app_vid.py と同じく永続化に必須）
  - 冪等化: `/vol/models/<model名>` が既に存在すればモデル本体のダウンロードをスキップ
    （LoRA 等の小物は毎回上書きで構わない）。強制再取得したい場合は
    `modal volume rm sdxl-models /<model名>` で消してから再実行する運用とする
- `setup()`（`@enter`）:
  - 存在チェックの前に `model_volume.reload()` を追加（app_vid.py:160 と同じ）
  - 存在チェックのエラーメッセージに「`make prep_sdxl` を先に実行してほしい」旨を含める

### 2. `Makefile`

```makefile
prep_sdxl:
 cd ./app && uv run --project .. modal run app_img.py::prepare_sdxl
```

### 3. ドキュメント

- README.md / README_ja.md のセットアップ手順に
  「デプロイ後、初回とモデル変更時に `make prep_sdxl` を実行する」を追記
- docs/plan/2026-07-11_uv-lockfile-dependabot.md の残タスク項が本プランで消化される

## 運用フローの変化

- **初回セットアップ**: 現在は `make app_img`（ビルド中にDL）。
  移行後は `make app_img` → `make prep_sdxl` の2段階
- **モデル変更**: 現在は config.yml 編集 → `make app_img`（レイヤ再ビルドでDL）。
  移行後は config.yml 編集 → `make prep_sdxl`（初回のみDL）→ `make app_img`\*
- **過去モデルに戻す**: 現在はレイヤ再ビルド。移行後は config.yml 編集 →
  `make app_img` のみ（DLなし、prep 不要）
- **モデルの確認/削除**: 現在は不可。移行後は `modal volume ls / rm sdxl-models`

\* config.yml は `COPY config.yml /` でイメージに入るため、モデル名の変更を
コンテナに反映するには `make app_img` の再実行は必要（モデルDLは走らない、
数十秒のレイヤ更新のみ）。config も Volume や Secret に逃がせばデプロイ不要に
できるが、スコープ外とする。

## 実装時の注意

- `download_model()` の `cache_dir`（HF のダウンロードキャッシュ）が Volume 配下だと
  中間ファイルも Volume に残る。app_vid.py 同様、最終的な `save_pretrained` 先だけ
  Volume にし、`cache_dir` はコンテナローカル（例: `/tmp/hf_cache`）に逃がすとクリーン
- 旧イメージに焼き込まれたモデルの掃除は不要（新デプロイで参照されなくなり、
  Modal 側で GC される）
- `enable_xformers_memory_efficient_attention()` の扱い（SDPA 移行の保留判断）は
  本プランとは独立。どちらが先でも衝突しない

## 検証

1. `make app_img` でデプロイ（この時点ではモデル未取得）
2. `make prep_sdxl` で Volume にモデルが入る（`modal volume ls sdxl-models` で確認）
3. `make img_by_sdxl_txt2img` で推論成功
4. config.yml のモデルを変更 → `make prep_sdxl` → `make app_img` → 推論成功
5. 元のモデルに戻す → `make app_img` のみ（prep 不要・DLが走らないこと）→ 推論成功
6. ruff / 既存 CI がグリーン
