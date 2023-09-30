[日本語版 README はこちら](README_ja.md)

# Stable Diffusion Modal

This is a Diffusers-based script for running Stable Diffusion on [Modal](https://modal.com/). It can perform txt2img inference and has the ability to increase resolution using ControlNet Tile and Upscaler.

## Features

1. Image generation using txt2img
   ![](assets/20230902_tile_imgs.png)

2. Upscaling

| Before upscaling                                                 | After upscaling                                                  |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| <img src="assets/20230708204347_1172778945_0_0.png" width="300"> | <img src="assets/20230708204347_1172778945_0_2.png" width="300"> |

## Requirements

The app requires the following to run:

- python: > 3.10
- modal-client
- A token for Modal.

The `modal-client` is the Python library. In order to install that:

```
pip install modal-client
```

And you need a modal token to use this script:

```
modal token new
```

Please see [the documentation of Modal](https://modal.com/docs/guide) for modals and tokens.

## Getting Started

To use the script, execute the below.

1. git clone the repository.
2. Copy `./setup_files/config.sample.yml` to `./setup_files/config.yml`
3. Open the Makefile and set prompts.
4. Execute `make deploy` command. An application will be deployed to Modal.
5. Execute `make run` command.

Images are generated and output to the `outputs/` directory.

## Directory structure

```
.
├── .env                    # Secrets manager
├── Makefile
├── README.md
├── sdcli/                  # A directory with scripts to run inference.
│   ├── outputs/            # Images are outputted this directory.
│   ├── txt2img.py          # A script to run txt2img inference.
│   └── util.py
└── setup_files/            # A directory with config files.
    ├── __main__.py         # A main script to run inference.
    ├── Dockerfile          # To build a base image.
    ├── config.yml          # To set a model, vae and some tools.
    ├── requirements.txt
    ├── setup.py            # Build an application to deploy on Modal.
    └── txt2img.py          # There is a class to run inference.
```

## How to use

### 1. `git clone` the repository

```
git clone https://github.com/hodanov/stable-diffusion-modal.git
cd stable-diffusion-modal
```

### 2. Add hugging_face_token to .env file

Hugging Add hugging_face_token to .env file.

This script downloads and uses a model from HuggingFace, but if you want to use a model in a private repository, you will need to set this environment variable.

```
HUGGING_FACE_TOKEN="Write your hugging face token here."
```

### 3. Add the model to ./setup_files/config.yml

Add the model used for inference. VAE, LoRA, and Textual Inversion are also configurable.

```
# ex)
model:
  name: stable-diffusion-2-1
  repo_id: stabilityai/stable-diffusion-2-1
vae:
  name: sd-vae-ft-mse
  repo_id: stabilityai/sd-vae-ft-mse
controlnets:
  - name: control_v11f1e_sd15_tile
    repo_id: lllyasviel/control_v11f1e_sd15_tile
```

Use a model configured for Diffusers, such as the one found in [this repository](https://huggingface.co/stabilityai/stable-diffusion-2-1). Files in safetensor format shared by Civitai etc. need to be converted (you can do so with a script in the diffusers official repository).

[https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py)

```
# Example of using conversion script
python ./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --from_safetensors \
--checkpoint_path="Write the filename of safetensor format here" \
--dump_path="Write the output path here" \
--device='cuda:0'
```

LoRA and Textual Inversion don't require any conversion and can directly use safetensors files. Add the download link to config.yml as below.

```
# Example
loras:
  - name: lora_name.safetensors # Specify the LoRA file name. Any name is fine, but the extension `.safetensors` is required.
    download_url: download_link_here # Specify the download link for the safetensor file.
```

### 4. Setting prompts

Set the prompt to Makefile.

```
# ex)
run:
 cd ./sdcli && modal run txt2img.py \
 --prompt "hogehoge" \
 --n-prompt "mogumogu" \
 --height 768 \
 --width 512 \
 --samples 1 \
 --steps 30 \
 --seed 12321 |
 --upscaler "RealESRGAN_x2plus" \
 --use-face-enhancer "False" \
 --fix-by-controlnet-tile "True"
```

### 5. make deploy

Execute the below command. An application will be deployed on Modal.

```
make deploy
```

### 6. make run

The txt2img inference is executed with the following command.

```
make run
```

Thank you.
