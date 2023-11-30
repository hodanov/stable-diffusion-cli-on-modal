from __future__ import annotations

from setup import stub
from txt2img import StableDiffusion


@stub.function(gpu="A10G")
def main():
    StableDiffusion


if __name__ == "__main__":
    main.local()
