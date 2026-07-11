# Stable Diffusion CLI on Modal

[Modal](https://modal.com/)上でStable Diffusionを動かすためのDiffusersベースのスクリプトです。WebUIは無く、CLIでのみ動作します。txt2imgの推論と、img2imgとUpscalerを利用した高解像度化の機能を備えています。

## このスクリプトでできること

1. txt2imgまたはimt2imgによる画像生成ができます。
   ![txt2imgでの生成画像例](assets/20230902_tile_imgs.png)
   利用可能なバージョン:
   - SDXL（のみ）

2. アップスケーラーとControlNet Tileを利用した高解像度な画像を生成することができます。

| ベース画像                                                       | アップスケール後                                                 |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| <img src="assets/20230708204347_1172778945_0_0.png" width="300"> | <img src="assets/20230708204347_1172778945_0_2.png" width="300"> |

1. その他、LoRAとTextual inversionを利用できます。

## 必須項目

このスクリプトを実行するには最低限下記のツールが必要です:

- python: >= 3.12
- [uv](https://docs.astral.sh/uv/)（仮想環境と依存ライブラリの管理に使用）
- ModalのAPIトークン
- Hagging FaceのAPIトークン（非公開のリポジトリのモデルを参照したい場合に必須）

本プロジェクトは`uv`でローカルの依存ライブラリ（主に`modal` CLI）を`pyproject.toml`と`uv.lock`で固定管理します。`uv`をインストールした上で、下記で環境を作成します:

```bash
# uv のインストール（https://docs.astral.sh/uv/getting-started/installation/ を参照）
brew install uv

# .venv を作成し、ロックされた依存関係をインストール
uv sync
```

torchやdiffusersなどの重いMLライブラリはModalコンテナのイメージ側（`app/pyproject.toml` + `app/uv.lock`）に入るため、ローカルにはインストールされません。CLIの実行に必要なのは`modal`のみです。

Modalのトークンも必要です:

```bash
uv run modal token new
```

`make`の各ターゲットは`uv run`経由で実行されるため、venvを手動でアクティベートする必要はありません。詳細は[Modalのドキュメント](https://modal.com/docs/guide)を参照してください。

## クイックスタート

下記の手順で画像が生成され、outputs ディレクトリに出力されます。

1. リポジトリをgit clone
2. ./app/config.example.yml を ./app/config.ymlにコピー
3. Makefile を開いてプロンプトを設定
4. `make app_img` を実行（SDXL用アプリをModal上にデプロイ）
5. `make prep_sdxl` を実行（SDXLモデルをModal Volumeに保存。config.ymlのモデル変更時も再実行）
6. `make app_vid` を実行（Wan I2V用アプリをModal上にデプロイ）
7. `make prep_wan_i2v` を実行（Wan I2VモデルをModal Volumeに保存）
8. `make img_by_sdxl_txt2img` を実行（スクリプトが起動）
9. `make vid_by_wan_ti2v` を実行（TI2Vの動画生成）

## ディレクトリ構成

```txt
.
├── .env                        # Secrets manager
├── Makefile
├── README.md
├── cmd/                      # A directory with scripts to run inference.
│   ├── outputs/                # Images are outputted this directory.
...
│   └── txt2img_handler.py         # A script to run txt2img inference.
│   └── ti2v_handler.py            # A script to run TI2V inference.
└── app/                # コンフィグとModalアプリ
    ├── app_img.py              # Modalアプリ本体（SDXL）
    ├── app_vid.py              # Modalアプリ本体（Wan I2V）
    ├── Dockerfile              # ベースイメージビルド用
    ├── config.yml              # モデル/VAE等の設定
    ├── pyproject.toml          # コンテナイメージに入る依存の定義
    └── uv.lock                 # イメージ依存のロックファイル
```

## 使い方の詳細

### 1. リポジトリをgit cloneする

```bash
git clone https://github.com/hodanov/stable-diffusion-modal.git
cd stable-diffusion-modal
```

### 2. .envファイルを設定する

Hugging FaceのトークンをHUGGING_FACE_TOKENに記入します。

このスクリプトはHuggingFaceからモデルをダウンロードして使用しますが、プライベートリポジトリにあるモデルを参照する場合、この環境変数の設定が必要です。

```txt
HUGGING_FACE_TOKEN="ここにHuggingFaceのトークンを記載する"
```

### 3. ./app/config.ymlを設定する

推論に使うモデルを設定します。Safetensorsファイルをそのまま利用します。VAE、LoRA、Textual Inversionも設定可能です。

下記のように、nameにモデル名、urlにSafetensorsファイルがあるURLを指定します。

```yml
# 設定例（SDXL のみ対応）
version: "sdxl"
model:
  name: stable-diffusion-xl
  url: https://huggingface.co/replace/with/your/sdxl/model.safetensors
vae:
  # 任意（カスタムVAEを使う場合のみ）
  name: your-sdxl-vae
  url: https://huggingface.co/replace/with/your/sdxl/vae.safetensors
```

Wan I2V を使う場合は `wan_i2v` にモデルを設定します。

```yml
wan_i2v:
  model:
    name: wan-i2v-a14b
    # safetensors_url を指定しない時に使う
    # 省略時は Wan-AI/Wan2.2-I2V-A14B-Diffusers が使われる
    repo_id: Wan-AI/Wan2.2-I2V-A14B-Diffusers
    # 任意: 指定した場合は repo_id の重みより safetensors を優先
    # safetensors_url: https://huggingface.co/user/repo/resolve/main/your.safetensors
```

LoRAは下記のように指定します。

```yml
# 設定例
loras:
  - name: mecha.safetensors # ファイル名を指定。任意の名前で良いが、拡張子`.safetensors`は必須。
    url: https://civitai.com/api/download/models/150907?type=Model&format=SafeTensor # ダウンロードリンクを指定
```

SDXLを使いたい場合は`version`に`sdxl`を指定し、urlに使いたいsdxlのモデルを指定します。

```yml
version: "sdxl"
model:
  name: stable-diffusion-xl
  url: https://huggingface.co/xxxx/xxxx
```

### 4. Makefileの設定（プロンプトの設定）

プロンプトをMakefileに設定します。

```makefile
# 設定例
img_by_sdxl_txt2img:
  cd ./cmd && uv run modal run txt2img_handler.py::main \
  --version "sdxl" \
  --prompt "A dog is running on the grass" \
  --n-prompt "" \
  --height 1024 \
  --width 1024 \
  --samples 1 \
  --steps 30 \
  --use-upscaler "True" \
  --output-format "avif"
```

- prompt: プロンプトを指定します。
- n-prompt: ネガティブプロンプトを指定します。
- height: 画像の高さを指定します。
- width: 画像の幅を指定します。
- samples: 生成する画像の数を指定します。
- steps: ステップ数を指定します。
- seed: seedを指定します。
- use-upscaler: 画像の解像度を上げるためのアップスケーラーを有効にします。
- output-format: 出力フォーマットを指定します。avifとpngのみ対応。

### 5. アプリケーションをデプロイする

下記のコマンドでModal上にアプリケーションが構築されます。

```bash
make app
```

初回デプロイ後は、下記のコマンドでモデルをModal Volumeにダウンロードします。

```bash
make prep_sdxl
make prep_wan_i2v
```

### 6. 推論を実行する

下記のコマンドでtxt2img推論が実行されます。

```bash
make img_by_sdxl_txt2img
```

### 7. モデルを追加・切り替えする

新しいモデルをVolumeに追加する手順:

1. `./app/config.yml` の `model.name` と `model.url` を新しいモデルに書き換える
2. `make prep_sdxl` を実行（新しいモデルが `sdxl-models` Volumeにダウンロードされます。既存のモデルは消えないため、複数モデルを併存できます）
3. `make app_img` を実行（コンテナに新しい `config.yml` を反映するための再デプロイ。モデルのダウンロードは走らず、数十秒のレイヤ更新のみ）

手順2〜3は `make switch_sdxl` で一括実行できます。

Volumeに既にあるモデルへ切り替える場合は、`./app/config.yml` を書き換えて `make app_img` を実行するだけです。`make prep_sdxl` を実行しても、モデルが既に存在する場合はダウンロードがスキップされるだけなので無害です。

Volumeの中身の確認・削除はModal CLIで行えます。

```bash
uv run modal volume ls sdxl-models
uv run modal volume rm sdxl-models /<モデル名>
```

モデル名が同じままURLだけ変わった場合など、強制的に再ダウンロードしたいときは、先にVolumeからモデルを削除してから `make prep_sdxl` を実行してください。
