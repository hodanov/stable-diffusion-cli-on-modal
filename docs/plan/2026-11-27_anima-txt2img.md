# Anima 画像生成対応プラン

## Context

`circlestone-labs/Anima` は Comfy Org と CircleStone Labs が作った約 2B のアニメ特化 T2I モデル。中身は次の組み合わせ:

- DiT: Cosmos-Predict2-2B の派生
- テキストエンコーダ: Qwen3-0.6B
- LLM adapter（T5 トークン埋め込み ↔ Qwen hidden states のクロスアテンション）
- VAE: Qwen-Image

今のリポは SDXL（txt2img + img2img 再描画）と Wan I2V だけなので、Anima でも同じ CLI 体験（`make` → Modal → `outputs/`）で画像を作れるようにする。

### 調査結果（プロがやるならこう）

- **diffusers が Anima を公式サポート済み**（PR #13732）。ただし modular のみで、`AnimaModularPipeline` / `AnimaAutoBlocks` を使う（標準の `AnimaPipeline` は無い）。ロック中の `diffusers==0.40.0` の wheel に `modular_pipelines/anima/*`、`AnimaTextConditioner`、`AnimaLoraLoaderMixin` が入ってるのを確認したので、**依存の更新は不要**。
- 公式 Diffusers リポは `circlestone-labs/Anima-Base-v1.0-Diffusers`（5.6GB、bf16）だけ。Turbo / Aesthetic は `circlestone-labs/Anima` の `split_files/diffusion_models/anima-{turbo,aesthetic}-v1.1.safetensors`（ComfyUI 用の単一ファイル、4.18GB）しかない。
- 単一ファイルの中身は `net.*`（Cosmos DiT）と `net.llm_adapter.*`（text_conditioner）が混在してる。変換手順は次の通り:
  - DiT 側は diffusers 内蔵の `diffusers.loaders.single_file_utils.convert_cosmos_transformer_checkpoint_to_diffusers` で変換できる（`net.` は自動で剥がれる）。
  - adapter 側は prefix を剥がせばそのまま `AnimaTextConditioner` の state_dict になる。公式の `scripts/convert_anima_to_diffusers.py` も同じロジック。
  - `CosmosTransformer3DModel.from_single_file` を直接使うと、adapter キーが混ざってるのと nvidia の gated リポに config を取りに行くのとで詰まる。なので自前で分割して、Base リポの `transformer/` と `text_conditioner/` の config で組み立てる。
- 推奨パラメータ（モデルカードより）:
  - 解像度: 512²〜1536²
  - Base / Aesthetic: 30〜50 steps、CFG 4〜5
  - Turbo: CFG 1、8〜12 steps
  - プロンプト: Danbooru タグ（小文字・スペース区切り）と自然文の混在 OK。アーティストは `@name`。
  - 推奨 prefix: `masterpiece, best quality, score_7, safe,`
  - 推奨 negative: `worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration`
  - Aesthetic では `score_*` を使わない。
- CFG は呼び出し引数じゃなくて guider コンポーネントで設定する（`ClassifierFreeGuidance(guidance_scale=...)`）。scale 1.0 なら `_is_cfg_enabled` が False になって uncond パスを省略するので、Turbo は約 2 倍速い。
- スケジューラは `FlowMatchEulerDiscreteScheduler(shift=3.0)` だけ（Comfy の euler 相当）。er_sde / beta57 は diffusers に無いので、そこは割り切る。
- img2img（`image`、`strength`）が `AnimaAutoBlocks` に入ってるので、SDXL と同じ「2 倍リサイズ → img2img 再描画」の upscaler をそのまま再現できる。
- ライセンス: CircleStone Non-Commercial と NVIDIA Open Model License。モデルの商用利用は不可で、生成物の商用利用は可。README に明記する。

### 決めた方針

1. 推論基盤は **diffusers Modular**。ComfyUI ヘッドレスは採用しない。
2. デフォルトは **Turbo v1.1**。config の `transformer_url` で Aesthetic / Base に切り替えられるようにする。
3. **別アプリ `app/app_anima.py`**（Modal App `anima-cli`、Volume `anima-models`）。Wan と同じ構成。

## 実装

### 1. `app/app_anima.py`（新規）

- 定数:
  - `DEFAULT_ANIMA_REPO_ID = "circlestone-labs/Anima-Base-v1.0-Diffusers"`
  - `MODEL_VOLUME_NAME = "anima-models"`
  - `DEFAULT_NEGATIVE_PROMPT`（モデルカードの推奨値）
- `ensure_http_url`、`normalize_hf_url`、`filename_from_url` は `app_vid.py` の実装を流用する。各アプリは単体で deploy する（Modal がファイル単位でマウントする）ので、既存方針どおりコピーで持つ。
- 純粋関数（テスト対象）:
  - `split_anima_checkpoint(state_dict) -> (transformer_sd, text_conditioner_sd)`: `net.llm_adapter.` で分割する。
  - `resolve_size(height, width)`: 16 の倍数への丸めと、512²〜1536² の面積チェックを行う。
- `AnimaSetup`: config の `anima.model` を読む。
  - `name`（必須）、`repo_id`（任意、既定は Base）、`transformer_url`（任意、Turbo / Aesthetic の単一ファイル）、`loras`（任意）。
  - `download_model()`:
    1. `snapshot_download(repo_id, local_dir=<vol>/<name>)` を実行する。`transformer_url` を指定した場合は `transformer/*.safetensors` と `text_conditioner/*.safetensors` を ignore する（Wan の `safetensors_url` と同じパターン）。
    2. `transformer_url` を指定した場合は `hf_hub_download` / urlopen で HF_CACHE_DIR に落とす。次に `safetensors.torch.load_file` → `split_anima_checkpoint` → `convert_cosmos_transformer_checkpoint_to_diffusers` を通す。そのうえで、`init_empty_weights` 下で `CosmosTransformer3DModel.from_config(<name>/transformer の config)` と `AnimaTextConditioner.from_config(...)` を作り、`load_state_dict(strict=True, assign=True)` → bf16 → `save_pretrained` する。キー不一致は missing / unexpected を列挙して例外にする。
- `prepare_anima()`（`@app.function`、timeout 3600）: 既存のモデルがあればスキップし、最後に `model_volume.commit()` する。`prepare_sdxl` と同じ流れ。
- `AnimaTxt2Img`（`@app.cls`、gpu="L4"。2B bf16 なので 24GB で十分。timeout と startup_timeout は SDXL と同じ）:
  - `@enter setup`:

    ```python
    pipe = ModularPipeline.from_pretrained(path)
    pipe.load_components(dtype=torch.bfloat16)
    pipe.to("cuda")
    ```

    config の `loras` があれば `pipe.load_lora_weights(...)` と `set_adapters` を呼ぶ（`AnimaLoraLoaderMixin`）。

  - `@method run_inference(prompt, n_prompt, height, width, steps, seed, guidance_scale, output_format, use_upscaler)`:
    - `pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=...))` で CFG を設定する。
    - `images = pipe(prompt=..., negative_prompt=n_prompt or DEFAULT_NEGATIVE_PROMPT, height, width, num_inference_steps=steps, generator=..., output="images")`
    - `use_upscaler` のときは 2 倍リサイズ → `pipe(image=..., strength=0.3, ...)` で再描画する。Anima の推奨上限 1536² を超えるので、出力が 1024² 超えなら長辺 1536 にクランプした「1.5 倍」にする。
    - 出力のバイト列化は `app_img.py` と同じ。
- 実装時に分かったこと:
  - `modular_model_index.json` は全コンポーネントの参照先が Hub のリポ ID になっている。そのため `load_components(pretrained_model_name_or_path=<volume のパス>, dtype=torch.bfloat16)` で参照先を上書きして、Volume 上の変換済み重みを読むようにした。
  - 戻り値は `pipe(..., output="images")` で PIL のリストとして受け取る。

### 2. `app/config.sample.yml`

`anima:` セクションを追加する。

```yaml
anima:
  model:
    name: anima-turbo-v1.1
    # repo_id: circlestone-labs/Anima-Base-v1.0-Diffusers
    transformer_url: https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-turbo-v1.1.safetensors
  # loras: [{name: x.safetensors, url: ..., scale: 1.0}]
```

### 3. CLI 側（`cmd/`）

- `cmd/infrastructure.py`:
  - `AnimaTxt2Img(Txt2ImgInterface)` を追加する。`modal.Cls.from_name("anima-cli", "AnimaTxt2Img")` と `spawn` + `wait_for_result` は SDXL 版と同じ形。
  - `new_txt2img` の match に `"anima"` を足す。
  - `guidance_scale` を渡すため、`new_txt2img` に `guidance_scale: float | None` を追加する。SDXL では無視する。
- `cmd/txt2img_handler.py`: `--guidance-scale`（既定 -1 → `unset_if_negative`）を追加する。未指定ならサーバ側の既定値（Turbo なら 1.0）を使う。
  - サーバ側の既定値は config の `anima.model.guidance_scale` で持つ。既定は 1.0、Aesthetic / Base を使う人は 4.5 に書き換える。サンプルにコメントで書く。

### 4. `Makefile`

- `app_anima`（deploy）を追加する。
- `prep_anima`（`modal run app_anima.py::prepare_anima`）を追加する。
- `img_by_anima_txt2img` を追加する。`--version "anima"` で、prompt 例は推奨 prefix 付き、`--steps 10`、`--height 1024 --width 1024`。
- `app:` に `app_anima` を足す。

### 5. テスト（`tests/app/test_app_anima.py` 新規、`tests/cli/` 追記）

torch / GPU 不要の範囲で、既存の `test_app_vid.py` の流儀に合わせる。

- `split_anima_checkpoint`: adapter と DiT が正しく分かれる。prefix が剥がれる。
- `AnimaSetup`: config のバリデーション（name 欠落、`transformer_url` 有無で snapshot の ignore パターンが変わる）。`snapshot_download` は monkeypatch する。
- `resolve_size`、upscale サイズ計算の境界値。
- CLI: `new_txt2img("anima", ...)` が `AnimaTxt2Img` を返す。`guidance_scale` が `spawn` に渡る（既存の `test_handlers.py` / `test_infrastructure.py` のモックを流用）。

### 6. ドキュメント

- `README.md` と `README_ja.md` に Anima の手順を追加する。
  - 手順: `app_anima` → `prep_anima` → `img_by_anima_txt2img`
  - プロンプトの書き方の要点と推奨パラメータ表（Turbo / Aesthetic / Base）
  - **非商用ライセンス**の注意
- `docs/plan/2026-11-27_anima-txt2img.md` にこのプランを残す（既存の慣例）。

## 検証

1. `uv run pytest --rootdir=tests --confcutdir=tests -p no:cacheprovider tests` と `uv run ruff check .` を回す（sandbox 下の .env 対策つき）。CI の coverage 65% を維持する。
2. Modal 実機での確認（deploy 系はユーザーが実行する方針）:
   - `make app_anima`
   - `make prep_anima`: Volume に transformer / text_conditioner が保存され、キー不一致の例外が出ないこと。
   - `make img_by_anima_txt2img`: `outputs/` にアニメ調の 1024² 画像が出ること。
   - `--use-upscaler True` で 2 枚目が出ること。
   - config を Aesthetic と `guidance_scale: 4.5` に変えて、30 steps でも生成できること。
3. 初回実行のログで、ロード時間と 1 枚あたりの生成時間を確認する（L4 で Turbo 10 steps なら数秒の想定）。
