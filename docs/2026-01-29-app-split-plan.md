# Modal App 分離（img / vid）実装プラン

## 目的

- 画像生成と動画生成を同一リポジトリで管理しつつ、Modal App を分けてデプロイ単位を軽くする。
- requirements.txt / config.yml は共用のまま維持する。

## 方針

- `App("sdxl-cli")` と `App("wan-i2v-cli")` の2つに分割する。
- CLIは既存の `cmd/txt2img_handler.py` と `cmd/ti2v_handler.py` を維持。
- 依存と設定は共通のままにして、分割はあくまで App のみ。

## 変更内容

### 1. app を分割

- `app/app_img.py` を追加して SDXL 関連だけを移動。
- `app/app_vid.py` を追加して Wan I2V 関連だけを移動。
- それぞれで `App("sdxl-cli")`, `App("wan-i2v-cli")` を定義。

### 2. build_image の切り分け

- `build_image()` を各 app に持たせる。
- どちらも `requirements.txt` と `config.yml` を共通利用。

### 3. cmd 側の参照先変更

- `cmd/infrastructure.py` の `modal.Cls.from_name` を修正。
  - 画像: `modal.Cls.from_name("sdxl-cli", "SDXLTxt2Img")`
  - 動画: `modal.Cls.from_name("wan-i2v-cli", "WanTI2V")`

### 4. Makefile 更新

- `make app_img` → `modal deploy app_img.py`
- `make app_vid` → `modal deploy app_vid.py`
- 既存の `make app` は残すなら両方デプロイに変更。

### 5. README 更新

- デプロイ手順を `app_img` / `app_vid` に分けて明記。
- どの CLI がどの App を参照するかを追記。

## 期待される効果

- 画像・動画それぞれ必要なときだけデプロイできる。
- デプロイ時間とコストの無駄が減る。
- 将来的に依存や設定を分けるときの足場になる。

## 留意点

- requirements / config の共用は維持するため、依存衝突があれば将来分割を検討。
- モデルDLの設計（build時DL vs setup時DL）は別タスクで扱う。
