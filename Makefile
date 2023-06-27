deploy:
	modal deploy sdcli.py

run:
	modal run entrypoint.py \
	--prompt "a photograph of an astronaut riding a horse" \
	--n-prompt "" \
	--height 512 \
	--width 512 \
	--samples 1 \
	--steps 50
