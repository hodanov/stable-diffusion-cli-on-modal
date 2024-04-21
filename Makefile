.PHONY: all app clean

app:
	cd ./app && modal deploy __main__.py

img_by_sd15_txt2img:
	cd ./cmd && modal run sd15_txt2img.py \
	--prompt "a photograph of an astronaut riding a horse" \
	--n-prompt "" \
	--height 512 \
	--width 768 \
	--samples 1 \
	--steps 30 \
	--use-upscaler "True" \
	--fix-by-controlnet-tile "True" \
	--output-format "avif"

img_by_sd15_img2img:
	cd ./cmd && modal run sd15_img2img.py \
	--prompt "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k" \
	--n-prompt "" \
	--samples 1 \
	--steps 30 \
	--use-upscaler "True" \
	--fix-by-controlnet-tile "True" \
	--output-format "avif" \
	--base-image-url "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"

img_by_sdxl_txt2img:
	cd ./cmd && modal run sdxl_txt2img.py \
	--prompt "A dog is running on the grass" \
	--height 1024 \
	--width 1024 \
	--samples 1 \
	--output-format "avif"