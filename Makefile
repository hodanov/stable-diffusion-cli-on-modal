run:
	modal run sd_cli.py \
	--prompt "A woman with bob hair" \
	--n-prompt "" \
	--upscaler "RealESRGAN_x4plus_anime_6B" \
	--height 768 \
	--width 512 \
	--samples 5 \
	--steps 50
