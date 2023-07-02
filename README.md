# Stable Diffusion Modal

This is the script to execute Stable Diffusion on [Modal](https://modal.com/).

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
2. Create the `.env` file and set a huggingface API token with reference to `.env.example`.
3. Copy `./setup_files/config.sample.yml` to `./setup_files/config.yml`
4. Open the Makefile and set prompts.
5. Execute `make deploy` command. An application will be deployed to Modal.
6. Execute `make run` command.

Images are generated and output to the `outputs/` directory.

## Directory structure

```
.
├── .env                    # Secrets manager
├── Makefile
├── README.md
├── sdcli/                  # A directory with scripts to run inference.
│   ├── __init__.py
│   ├── outputs/            # Images are outputted this directory.
│   ├── txt2img.py          # A script to run txt2img inference.
│   └── util.py
└── setup_files/            # A directory with config files.
    ├── Dockerfile          # To build a base image.
    ├── config.yml          # To set a model, vae and some tools.
    ├── requirements.txt
    └── setup.py            # Build an application to deploy on Modal.
```

Thank you.

## Author

[Hoda](https://hodalog.com)
