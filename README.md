[日本語版 README はこちら](README_ja.md)

# Stable Diffusion CLI on Modal

This is a Diffusers-based script for running Stable Diffusion on [Modal](https://modal.com/). This script has no WebUI and only works with CLI. It performs txt2img inference and can upscale/refine outputs.

## Features

1. Image generation using txt2img or img2img.
  ![example for txt2img](assets/20230902_tile_imgs.png)
  Available version:
    - SDXL (only)

2. Upscaling

| Before upscaling                                                 | After upscaling                                                  |
| ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| <img src="assets/20230708204347_1172778945_0_0.png" width="300"> | <img src="assets/20230708204347_1172778945_0_2.png" width="300"> |

## Requirements

The app requires the following to run:

- python: >= 3.11
- modal: >= 1.0.3
- A token for Modal.

The `modal` is the Python library. In order to install that:

```bash
pip install modal
```

And you need a modal token to use this script:

```bash
modal token new
```

Please see [the documentation of Modal](https://modal.com/docs/guide) for modals and tokens.

## Getting Started

To use the script, execute the below.

1. git clone the repository.
2. Copy `./app/config.sample.yml` to `./app/config.yml`
3. Open the Makefile and set prompts.
4. Execute `make app` command. An application will be deployed to Modal.
5. Execute `make img_by_sdxl_txt2img` command.

Images are generated and output to the `outputs/` directory.

## Directory structure

```txt
.
├── .env                        # Secrets manager
├── Makefile
├── README.md
├── cmd/                      # A directory with scripts to run inference.
│   ├── outputs/                # Images are outputted this directory.
...
│   └── txt2img_handler.py         # A script to run txt2img inference.
└── app/                # A directory with config and Modal app.
    ├── app.py                  # Modal app and inference implementation (SDXL)
    ├── Dockerfile              # To build a base image.
    ├── config.yml              # To set a model, VAE and optional tools.
    └── requirements.txt
```

## How to use

### 1. `git clone` the repository

```bash
git clone https://github.com/hodanov/stable-diffusion-modal.git
cd stable-diffusion-modal
```

### 2. Add hugging_face_token to .env file

Add hugging_face_token to .env file.

This script downloads and uses a model from HuggingFace, but if you want to use a model in a private repository, you will need to set this environment variable.

```txt
HUGGING_FACE_TOKEN="Write your hugging face token here."
```

### 3. Add the model to ./app/config.yml

Add the model used for inference. Use the Safetensors file as is. VAE, LoRA, and Textual Inversion are also configurable.

```yml
# ex)
version: "sdxl"
model:
  name: stable-diffusion-xl
  url: https://huggingface.co/replace/with/your/sdxl/model.safetensors # Safetensors file URL.
vae:
  # Optional for SDXL; keep if you provide a custom VAE.
  name: your-sdxl-vae
  url: https://huggingface.co/replace/with/your/sdxl/vae.safetensors
```

If you want to use LoRA and Textual Inversion, configure as follows.

```yml
# Example
loras:
  - name: lora_name.safetensors # Specify the LoRA file name. Any name is fine, but the extension `.safetensors` is required.
    url: download_link_here # Specify the download link for the safetensor file.
```

If you want to use SDXL:

```yml
version: "sdxl"
model:
  name: stable-diffusion-xl
  url: https://huggingface.co/xxxx/xxxx
```

### 4. Setting prompts

Set the prompt to Makefile.

```makefile
# ex)
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
```

- prompt: Specifies the prompt.
- n-prompt: Specifies a negative prompt.
- height: Specifies the height of the image.
- width: Specifies the width of the image.
- samples: Specifies the number of images to generate.
- steps: Specifies the number of steps.
- seed: Specifies the seed.
- use-upscaler: Enables the upscaler to increase the image resolution.
- output-format: Specifies the output format. Only avif and png are supported.

### 5. Deploy an application

Execute the below command. An application will be deployed on Modal.

```bash
make app
```

### 6. Run inference

The txt2img inference is executed with the following command.

```bash
make img_by_sdxl_txt2img
```

Thank you.
