# Stable Diffusion CLI on Modal

[Modal](https://modal.com/)上でStable Diffusionを動かすためのDiffusersベースのスクリプトだよ。WebUIは無く、CLIでのみ動作。txt2imgの推論と、アップスケール/リファインでの高解像度化に対応してる。

## このスクリプトでできること

1. txt2imgまたはimt2imgによる画像生成ができます。
  ![txt2imgでの生成画像例](assets/20230902_tile_imgs.png)
  利用可能なバージョン:
    - SDXL（のみ）

2. アップスケーラーを使って高解像度化できるよ（SDXL）。

| ベース画像                                                       | アップスケール後                                                 |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| <img src="assets/20230708204347_1172778945_0_0.png" width="300"> | <img src="assets/20230708204347_1172778945_0_2.png" width="300"> |

3. その他、LoRAとTextual inversionを利用できます。

## 必須項目

このスクリプトを実行するには最低限下記のツールが必要です:

- python: >= 3.11
- modal: >= 1.0.3
- ModalのAPIトークン
- Hagging FaceのAPIトークン（非公開のリポジトリのモデルを参照したい場合に必須）

`modal`はModalをCLIから操作するためのPythonライブラリです。下記のようにインストールします:

```bash
pip install modal
```

And you need a modal token to use this script:

```bash
modal token new
```

詳細は[Modalのドキュメント](https://modal.com/docs/guide)を参照してください。

## クイックスタート

下記の手順で画像が生成され、outputs ディレクトリに出力されます。

1. リポジトリをgit clone
2. ./app/config.example.yml を ./app/config.ymlにコピー
3. Makefile を開いてプロンプトを設定
4. `make app` を実行（Modal上にアプリケーションをデプロイ）
5. `make img_by_sdxl_txt2img` を実行（スクリプトが起動）

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
└── app/                # コンフィグとModalアプリ
    ├── app.py                  # Modalアプリ本体（SDXL）
    ├── Dockerfile              # ベースイメージビルド用
    ├── config.yml              # モデル/VAE等の設定
    └── requirements.txt
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
  cd ./cmd && modal run txt2img_handler.py::main \
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

### 6. 推論を実行する

下記のコマンドでtxt2img推論が実行されるよ。

```bash
make img_by_sdxl_txt2img
```
