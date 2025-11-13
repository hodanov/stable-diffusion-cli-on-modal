.PHONY: app

app:
	cd ./app && modal deploy app.py

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
