.PHONY: app test

test:
	uv run pytest

app: app_img app_vid app_anima

app_img:
	cd ./app && uv run --project .. modal deploy app_img.py

app_vid:
	cd ./app && uv run --project .. modal deploy app_vid.py

app_anima:
	cd ./app && uv run --project .. modal deploy app_anima.py

prep_sdxl:
	cd ./app && uv run --project .. modal run app_img.py::prepare_sdxl

prep_wan_i2v:
	cd ./app && uv run --project .. modal run app_vid.py::prepare_wan_i2v

prep_anima:
	cd ./app && uv run --project .. modal run app_anima.py::prepare_anima

switch_sdxl: prep_sdxl app_img

img_by_sdxl_txt2img:
	cd ./cmd && uv run modal run txt2img_handler.py::main \
	--version "sdxl" \
    --prompt "A dog is running on the grass" \
    --n-prompt "" \
	--height 1024 \
	--width 1024 \
	--samples 1 \
	--steps 28 \
	--use-upscaler "True" \
	--output-format "avif"

img_by_anima_txt2img:
	cd ./cmd && uv run modal run txt2img_handler.py::main \
	--version "anima" \
	--prompt "masterpiece, best quality, score_7, safe, 1girl, solo, silver hair, long hair, school uniform, cherry blossoms, smile, looking at viewer" \
	--n-prompt "" \
	--height 1024 \
	--width 1024 \
	--samples 1 \
	--steps 10 \
	--use-upscaler "False" \
	--output-format "avif"
