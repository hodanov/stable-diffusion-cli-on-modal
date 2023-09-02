deploy:
	cd ./setup_files && modal deploy __main__.py

# `--upscaler` is a name of upscaler you want to use.
# You can use upscalers the below:
#   - `RealESRGAN_x4plus`
#   - `RealESRNet_x4plus`
#   - `RealESRGAN_x4plus_anime_6B`
#   - `RealESRGAN_x2plus`
run:
	cd ./sdcli && modal run txt2img.py \
	--prompt "a photograph of an astronaut riding a horse" \
	--n-prompt "" \
	--height 512 \
	--width 768 \
	--samples 1 \
	--steps 30 \
	--upscaler "RealESRGAN_x2plus" \
	--use-face-enhancer "False" \
	--fix-by-controlnet-tile "True"
