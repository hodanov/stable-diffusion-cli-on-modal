run:
	modal run sd_cli.py \
	--prompt "a woman with bob hair" \
	--n-prompt "" \
	--height 768 \
	--width 512 \
	--samples 5 \
	--steps 20 \
	--upscaler "sd_x2_latent_upscaler"
