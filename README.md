# Stable Diffusion Modal

This is the script to execute Stable Diffusion on [Modal](https://modal.com/).

## Requirements

The app requires the following to run:

- python: v3.10 >
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

```
1. git clone the repository.
2. Create the `.env` file and set a huggingface API token and a model with reference to `.env.example`.
3. Open the Makefile and set prompts.
4. Execute `make run` command.
```

Images are generated and output to the `outputs/` directory.

Thank you.

## Author

[Hoda](https://hodalog.com)
