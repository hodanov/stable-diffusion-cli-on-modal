deploy:
	modal deploy ./setup_files/setup.py

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
	--width 512 \
	--samples 1 \
	--steps 50 \
	--upscaler "" \
	--use-face-enhancer "False" \
	--use-hires-fix "False"
