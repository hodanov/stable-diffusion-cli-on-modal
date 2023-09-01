# Stable Diffusion Modal

[Modal](https://modal.com/)上でStable Diffusionを動かすためのスクリプトです。txt2imgの推論を実行することができます。ControlNet TileとUpscalerを利用した高解像度化に対応しており、モデルの取り替えも自由に行えます（再ビルドは必要ですが、マルチステージビルド的なコードを実装して効率よく取り替えできるようにしています）。

## このスクリプトでできること

1. txt2imgによる画像生成ができます。

![](assets/20230902_tile_imgs.png)

2. アップスケーラーとControlNet Tileを利用した高解像度な画像生成ができます。

Stable Diffusion 1.5、または2系では、生成画像のサイズは標準で512x512~1024程度までで、それ以上の解像度の画像を作ろうとすると、人物や背景が崩れてしまいます。

アップスケーラーとControlNet Tileを組み合わせることで、3072x2048pxまでの高解像度画像を生成することができます。

ベース画像
![](assets/20230708204347_1172778945_0_0.png)

アップスケール後
![](assets/20230708204347_1172778945_0_2.png)

## 必須項目

このスクリプトを実行するには最低限下記のツールが必要です:

- python: > 3.10
- modal-client
- ModalのAPIトークン
- Hagging FaceのAPIトークン（非公開のリポジトリのモデルを参照したい場合に必須）

`modal-client`はModalをCLIから操作するためのPythonライブラリです。下記のようにインストールします:

```
pip install modal-client
```

And you need a modal token to use this script:

```
modal token new
```

詳細は[Modalのドキュメント](https://modal.com/docs/guide)を参照してください。

## クイックスタート

下記の手順で画像が生成され、outputs ディレクトリに出力されます。

1. リポジトリをgit clone
2. .envファイルを作成し、.env.example を参考に huggingface の API トークンとモデルを設定
3. ./setup_files/config.example.yml を ./setup_files/config.ymlにコピー
4. Makefile を開いてプロンプトを設定
5. make deployをコマンドラインで実行(Modal上にアプリケーションが構築されます)
6. make run(スクリプトが起動します)

## ディレクトリ構成

```
.
├── .env                    # Secrets manager
├── Makefile
├── README.md
├── sdcli/                  # A directory with scripts to run inference.
│   ├── outputs/            # Images are outputted this directory.
│   ├── txt2img.py          # A script to run txt2img inference.
│   └── util.py
└── setup_files/            # A directory with config files.
    ├── __main__.py         # A main script to run inference.
    ├── Dockerfile          # To build a base image.
    ├── config.yml          # To set a model, vae and some tools.
    ├── requirements.txt
    ├── setup.py            # Build an application to deploy on Modal.
    └── txt2img.py          # There is a class to run inference.
```

## 使い方の詳細

### 1. リポジトリをgit cloneする

```
git clone https://github.com/hodanov/stable-diffusion-modal.git
cd stable-diffusion-modal
```

### 2. .envファイルを設定する

Hugging FaceのトークンをHUGGING_FACE_TOKENに記入します。

このスクリプトはHuggingFaceからモデルをダウンロードして使用しますが、プライベートリポジトリにあるモデルを参照する場合、この環境変数の設定が必要です。

```
HUGGING_FACE_TOKEN="ここにHuggingFaceのトークンを記載する"
```

### 3. ./setup_files/config.ymlを設定する

推論に使うモデルを設定します。VAE、Controlnet、LoRA、Textual Inversionも設定可能です。

```
# 設定例
model:
  name: stable-diffusion-2-1
  repo_id: stabilityai/stable-diffusion-2-1
vae:
  name: sd-vae-ft-mse
  repo_id: stabilityai/sd-vae-ft-mse
loras:
  - name: hogehoge.safetensors
    download_url: https://hogehoge/xxxx
  - name: fugafuga.safetensors
    download_url: https://fugafuga/xxxx
textual_inversions:
  - name: hogehoge
    download_url: https://hogehoge/xxxx
  - name: fugafuga
    download_url: https://fugafuga/xxxx
controlnets:
  - name: control_v11f1e_sd15_tile
    repo_id: lllyasviel/control_v11f1e_sd15_tile
```

ModelとVAEは[こちらのリポジトリ](https://huggingface.co/stabilityai/stable-diffusion-2-1)にあるような、Diffusersのために構成されたモデルを利用します。

Civitaiなどで共有されているsafetensors形式のファイルは変換が必要です（diffusersの公式リポジトリにあるスクリプトで変換できます）。

[変換スクリプト](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py)

```
# 変換スクリプトの使用例
python ./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --from_safetensors \
--checkpoint_path="ここに変換したいsafetensors形式のファイルを指定" \
--dump_path="出力先を指定" \
--device='cuda:0'
```

### 4. Makefileの設定（プロンプトの設定）

プロンプトをMakefileに設定します。

```
# 設定例
run:
 cd ./sdcli && modal run txt2img.py \
 --prompt "hogehoge" \
 --n-prompt "mogumogu" \
 --height 768 \
 --width 512 \
 --samples 20 \
 --steps 30 \
 --upscaler "RealESRGAN_x2plus" \
 --use-face-enhancer "False" \
 --fix-by-controlnet-tile "True"
```

- prompt: プロンプトを指定します。
- n-prompt: ネガティブプロンプトを指定します。
- height: 画像の高さを指定します。
- width: 画像の幅を指定します。
- samples: 生成する画像の数を指定します。
- steps: ステップ数を指定します。
- upscaler: 画像の解像度を上げるためのアップスケーラーを指定します。
- fix-by-controlnet-tile: ControlNet 1.1 Tileの利用有無を指定します。有効にすると、崩れた画像を修復しつつ、高解像度な画像を生成します。

### 5. make deployの実行

下記のコマンでModal上にアプリケーションが構築されます。

```
make deploy
```

### 6. make runの実行

下記のコマンドでtxt2img推論が実行されます。

```
make run
```

## Author

[Hoda](https://hodalog.com)
