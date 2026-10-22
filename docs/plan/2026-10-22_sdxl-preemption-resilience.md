# SDXL 推論の preemption 対策

## 背景・課題

`make img_by_sdxl_txt2img` を実行すると、ときどき Modal コンソールに次のログが出る。

```
Container terminated due to preemption. Your Function will be restarted with the same input.
```

そのあいだ手元の CLI は何もログを出さないまま待ち続け、止まったように見える。

調査で分かったこと:

1. **preemption は Modal 側のイベントで、コードからは防げない**: Modal の関数はすべて
   preemptible で、`nonpreemptible=True` は GPU 関数では使えない
   （<https://modal.com/docs/guide/preemption>）。コンテナが長く動くほど当たりやすい。
2. **無言で待つ仕組み**: `.remote()` は結果が返るまでタイムアウトなしで待つ
   （modal 1.6.1 の `_get_single_output(timeout=None)`）。preemption が起きると、Modal は
   サーバー側で同じ input を新しいコンテナに入れ直す。新しいコンテナは GPU の確保、image pull、
   `@enter` をやり直す。ところが `modal run` が手元に流すのは ephemeral app
   `run-stable-diffusion-cli` のログだけで、deploy 済みの `sdxl-cli` のログは流れてこない。
   そのため、再起動の進み具合が手元からは見えない。
3. **再起動のコストを上げていたところ**: `@enter` で同じ `cache_path` から SDXL のフル重みを
   2 回 load していた（`StableDiffusionXLPipeline` と `StableDiffusionXLImg2ImgPipeline`）。
   `.to("cuda")` と `vae.enable_tiling()` は推論のたびに呼んでいた。tokenizer も推論のたびに
   Volume から読み直していた。`timeout` を指定していないので、起動時間の上限は既定の 300 秒だった。

## 方針

preemption は起きる前提で考える。そのうえで、**再起動を軽くして、手元から見えるようにして、
無限には待たない**ようにする。「1 サンプル = 1 input」という既存の作りは、処理を細かく分けて
進捗を保存するという Modal の推奨にすでに合っているので、そのまま残す。

## 変更内容

### クライアント（`cmd/`）

- `infrastructure.wait_for_result()` を追加した。`.spawn()` で投げた呼び出しに対して
  `FunctionCall.get(timeout=30)` を繰り返す。
  - 投げた時点で call_id（`fc-...`）をログに出し、その後は 30 秒ごとに経過時間をログに出す。
  - 15 分待っても結果が返らなければ `cancel()` して、call_id 付きの `TimeoutError` を投げる。
  - Ctrl+C（`KeyboardInterrupt`）でも `cancel()` する。spawn した呼び出しは、手元が終了しても
    GPU 上で動き続けるため。
  - modal の `get()` が「まだ結果が無い」ときに投げるのは組み込みの `TimeoutError`。
    リモート側のタイムアウト（`modal.exception.FunctionTimeoutError`）は組み込みの
    `TimeoutError` を継承していないので、待ち続けずにそのまま伝わる。
- `SDXLTxt2Img.run_inference` は `.remote()` をやめ、`.spawn()` と `wait_for_result()` を使う。
- `txt2img_handler.main` はサンプルのループを `try/finally` で囲んだ。途中で失敗しても
  prompts を保存する。各サンプルの開始時にもログを出す。

### コンテナ（`app/app_img.py`）

- refiner を `StableDiffusionXLImg2ImgPipeline.from_pipe(self.__pipe, torch_dtype=torch.float16)`
  で作り、base とコンポーネントを共有する。これで Volume からの load は 1 回になる。
  - `torch_dtype` は省略できない。`from_pipe` は既定で `float32` にキャストするので、省略すると
    共有している base 側のコンポーネントまで `float32` になる。
- `.to("cuda")` と `vae.enable_tiling()` は `@enter` で 1 回だけ呼ぶ。
- tokenizer は `self.__pipe.tokenizer` を使い回す。
- `@enter` で load にかかった時間を print する。コンソールで再起動の進み具合を見るため。
- `@app.cls` に `timeout=600, startup_timeout=900` を明示した。

## 検証

- `uv run pytest`（カバレッジ下限 65% を含む CI と同じ条件）と `uv run ruff check .`。
- GPU 推論と `@enter` はユニットテストの対象外なので、実機で確認する。
  `make app_img` → `make img_by_sdxl_txt2img` を実行し、次の点を見る。
  - 手元に call_id と「Still waiting」が出ること。
  - コンソールに load 時間が出ること。

## 後続（今回はやらない）

- 今回の変更のあとでも再起動が遅ければ、memory snapshot を入れる
  （`enable_memory_snapshot=True` を指定し、`@enter(snap=True/False)` で CPU への load と GPU への転送を分ける）。
- A10G の確保待ちが長いとログで分かったら、GPU の fallback リスト（例: `gpu=["A10G", "L4"]`）を検討する。
- `WanTI2V`（B200、長時間）にも、動画向けの `max_wait` で `wait_for_result()` を使う。
