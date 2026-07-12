# lint CI の Python ハードコード撤廃と ruff のバージョン固定（PR #282）

## 背景・課題

mise 導入（バージョン管理とタスクランナーの一元化）を検討した結果、Python 1本の
プロジェクトでは uv と役割が重複するため見送り、uv に寄せ切る方針とした。
その検討過程で、バージョン管理の一元化ができていない箇所が2つ見つかった。

1. **Python バージョンの不整合**: `lint_python.yml` が `python-version: "3.14"` を
   ハードコードしている一方、root プロジェクトは `requires-python = ">=3.12"`。
   なお app/ の `==3.14.*` は Docker ベースイメージとの厳密一致が必須な意図的設計
   （2026-07-11_python-3.14-upgrade.md 参照）のため、不整合には該当しない
2. **ruff のバージョンが未固定**: CI は `pip install ruff` で毎回最新を取得し、
   pyproject の依存にも入っていないため、ローカルと CI で ruff の挙動が食い違い得る

## 方針

CI から Python バージョン指定を消し、ruff を root の dev 依存として uv.lock で
固定する。バージョンの単一ソースは各 pyproject の `requires-python` になる。

全体を 3.14 に統一する案（root の `requires-python` も上げる）は、root（modal CLI
クライアント）に機能上の必要がなく、今後 app 側がバンプするたびに root も手動追随が
必要になるため不採用とした。

## 変更内容

1. **ruff を root の dev 依存に追加**: `uv add --dev ruff` で
   `[dependency-groups] dev` に追加し、uv.lock で `0.15.21` に固定。
   dependabot は uv ecosystem（directory "/"）を監視済みのため、以後の更新は
   自動で PR が来る
2. **`.github/workflows/lint_python.yml` を uv 経由に書き換え**:
   `actions/setup-python`（3.14 ハードコード）+ `pip install ruff` を削除し、
   `astral-sh/setup-uv`（リポジトリの慣例に合わせコミット SHA で pin、v8.3.2、
   `enable-cache: true`）+ `uv run --frozen ruff check --output-format=github .`
   に置き換え。lock どおりの ruff でローカルと CI が同条件になる

## 検証

- `uv lock --check` で lock と pyproject の整合を確認
- `uv run --frozen ruff check --output-format=github --isolated .` が exit 0
  （`--isolated` はリポジトリに ruff 設定ファイルが無い CI 相当の条件）
- workflow YAML は prettier / actionlint パス
- push 後、lint CI（`CI` workflow）がグリーンになることを確認済み

## 運用メモ

- ローカルに未トラックの `ruff.toml`（`lint.select=["ALL"]`）がある場合、CI
  （設定ファイルなし = デフォルトルール）と lint 結果が食い違う。CI にも同じ
  ルールを適用するなら `ruff.toml` のコミットと指摘の解消が別途必要
- mise 再検討の条件: 複数言語のツールチェーンが増えたとき、またはチーム開発に
  なったとき
