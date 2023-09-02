[日本語版 README はこちら](README_ja.md)

# Stable Diffusion Modal

This is a Diffusers-based script for running Stable Diffusion on [Modal](https://modal.com/). It can perform txt2img inference and has the ability to increase resolution using ControlNet Tile and Upscaler.

## Features

1. Image generation using txt2img
![](assets/20230902_tile_imgs.png)

2. Upscaling

| Before upscaling | After upscaling |
| ---- | ---- |
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

Thank you.

## Author

[Hoda](https://hodalog.com)
