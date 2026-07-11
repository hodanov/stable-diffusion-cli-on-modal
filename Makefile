.PHONY: app

app: app_img app_vid

app_img:
	cd ./app && uv run modal deploy app_img.py

app_vid:
	cd ./app && uv run modal deploy app_vid.py

prep_wan_i2v:
	cd ./app && uv run modal run app_vid.py::prepare_wan_i2v

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
