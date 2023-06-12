run:
	modal run sd_cli.py \
	--prompt "A woman with bob hair" \
	--n-prompt "" \
	--height 768 \
	--width 512 \
	--samples 5 \
	--steps 50 \
	--upscaler "RealESRGAN_x4plus_anime_6B"
